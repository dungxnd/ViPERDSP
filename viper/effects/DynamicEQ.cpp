#include "DynamicEQ.h"
#include "../utils/FastAudioMath.h"
#include <algorithm>
#include <cmath>
#include <numbers>

namespace {
constexpr float kOvershootRangeDb = 12.0f;
} // namespace

DynamicEQ::DynamicEQ() {
    Reset();
}

void DynamicEQ::Process(std::span<float> samples) noexcept {
    if (!enable_ || band_count_ == 0u || samples.empty()) return;
    [[assume(samples.size() % 2 == 0)]];

    const size_t frame_count = samples.size() / 2u;
    // C++23 mdspan: audio[frame, channel] — layout_right maps to samples[frame*2 + ch]
    StereoView audio(samples.data(), frame_count, 2u);

    for (uint32_t b = 0u; b < band_count_; ++b) {
        auto&       p  = params_[b];
        auto&       st = state_[b];
        const auto& fc = coeffs_[b];

        for (size_t f = 0u; f < frame_count; f += kControlPeriod) {
            const size_t chunk = std::min(static_cast<size_t>(kControlPeriod),
                                          frame_count - f);

            // ----------------------------------------------------------------
            // 1. Audio-rate envelope tracking (one MAC per channel per frame)
            // ----------------------------------------------------------------
            for (size_t i = 0u; i < chunk; ++i) {
                const float l   = audio[f + i, 0u];
                const float r   = audio[f + i, 1u];
                const float p_l = l * l;
                const float p_r = r * r;

                st.env_l += (p_l > st.env_l ? st.attack_coeff : st.release_coeff)
                             * (p_l - st.env_l);
                st.env_r += (p_r > st.env_r ? st.attack_coeff : st.release_coeff)
                             * (p_r - st.env_r);
            }

            // ----------------------------------------------------------------
            // 2. Sub-block control (once per 16 samples):
            //    Power-domain threshold gate skips FastLog2 when signal silent.
            // ----------------------------------------------------------------
            const float max_env_sq = std::max(st.env_l, st.env_r);
            float desired_gain_db  = 0.0f;

            if (max_env_sq > p.linear_threshold_sq) {
                const float env_db    = viper::dsp::FastPowerToDb(max_env_sq);
                const float overshoot = env_db - p.threshold_db;
                const float ratio     = std::min(overshoot / kOvershootRangeDb, 1.0f);
                desired_gain_db       = p.target_gain_db * ratio;
            }

            const float g_coeff = (std::abs(desired_gain_db) > std::abs(st.current_gain_db))
                                      ? st.attack_coeff : st.release_coeff;
            st.current_gain_db += g_coeff * (desired_gain_db - st.current_gain_db);

            // Biquad coefficients recomputed at sub-block rate, NOT per sample.
            // No sin/cos here — trig_cache_ was computed on the last parameter change.
            FastUpdateBandCoeffs(b, st.current_gain_db);

            // ----------------------------------------------------------------
            // 3. TDF-II biquad over the sub-block (vectorizable inner loop)
            // ----------------------------------------------------------------
#pragma clang loop vectorize(enable)
            for (size_t i = 0u; i < chunk; ++i) {
                for (size_t ch = 0u; ch < 2u; ++ch) {
                    const float in = audio[f + i, ch];
                    auto& fs = filter_state_[ch][b];

                    const float out = fc.b0 * in + fs.s1;
                    fs.s1 = fc.b1 * in - fc.a1 * out + fs.s2;
                    fs.s2 = fc.b2 * in - fc.a2 * out;

                    audio[f + i, ch] = out;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FastUpdateBandCoeffs — zero trig ops in the dynamic audio path.
// cos_w0, sin_w0, alpha are read from trig_cache_[band] which was filled
// once during PrecomputeTrigConstants() on frequency/Q parameter changes.
//
// Peak filter:  A = 10^(gain/40) via FastExp2 (3 cycles, no std::pow)
//               b1 == a1 == neg_2_cos_w0 — one coefficient shared
// Shelf filters: still require std::sqrt(A) but that's unavoidable math.
// ---------------------------------------------------------------------------
void DynamicEQ::FastUpdateBandCoeffs(const uint32_t band, const float gain_db) noexcept {
    const auto& p = params_[band];
    const auto& t = trig_cache_[band];

    const float A = viper::dsp::FastDbToSqrtLinear(gain_db); // 10^(gain/40)

    float a0, a1, a2, b0, b1, b2;

    switch (p.filter_type) {
        case 1: { // Low Shelf
            const float sqrtA  = std::sqrt(A);
            const float S_term = sqrtA * 2.0f
                                 * (t.sin_w0 * 0.5f
                                    * std::sqrt((A + 1.0f / A) * (1.0f / p.q - 1.0f) + 2.0f));
            a0 =  (A + 1.0f) + (A - 1.0f) * t.cos_w0 + S_term;
            a1 = -2.0f * ((A - 1.0f) + (A + 1.0f) * t.cos_w0);
            a2 =  (A + 1.0f) + (A - 1.0f) * t.cos_w0 - S_term;
            b0 =  A * ((A + 1.0f) - (A - 1.0f) * t.cos_w0 + S_term);
            b1 =  2.0f * A * ((A - 1.0f) - (A + 1.0f) * t.cos_w0);
            b2 =  A * ((A + 1.0f) - (A - 1.0f) * t.cos_w0 - S_term);
            break;
        }
        case 2: { // High Shelf
            const float sqrtA  = std::sqrt(A);
            const float S_term = sqrtA * 2.0f
                                 * (t.sin_w0 * 0.5f
                                    * std::sqrt((A + 1.0f / A) * (1.0f / p.q - 1.0f) + 2.0f));
            a0 =  (A + 1.0f) - (A - 1.0f) * t.cos_w0 + S_term;
            a1 =  2.0f * ((A - 1.0f) - (A + 1.0f) * t.cos_w0);
            a2 =  (A + 1.0f) - (A - 1.0f) * t.cos_w0 - S_term;
            b0 =  A * ((A + 1.0f) + (A - 1.0f) * t.cos_w0 + S_term);
            b1 = -2.0f * A * ((A - 1.0f) + (A + 1.0f) * t.cos_w0);
            b2 =  A * ((A + 1.0f) + (A - 1.0f) * t.cos_w0 - S_term);
            break;
        }
        default: { // Peak — zero sin/cos, a1 == b1 == neg_2_cos_w0 (identity)
            const float alpha_A   = t.alpha * A;
            const float alpha_inv = t.alpha / A;
            a0 = 1.0f + alpha_inv;
            a1 = t.neg_2_cos_w0;
            a2 = 1.0f - alpha_inv;
            b0 = 1.0f + alpha_A;
            b1 = t.neg_2_cos_w0; // b1 == a1 always in Peak biquad
            b2 = 1.0f - alpha_A;
            break;
        }
    }

    const float inv_a0 = 1.0f / a0;
    coeffs_[band] = {
        .b0 = b0 * inv_a0,
        .b1 = b1 * inv_a0,
        .b2 = b2 * inv_a0,
        .a1 = a1 * inv_a0,
        .a2 = a2 * inv_a0
    };
}

// ---------------------------------------------------------------------------
// PrecomputeTrigConstants — called ONLY on frequency or Q parameter changes.
// Uses double precision for angular frequency accuracy; stores float results.
// ---------------------------------------------------------------------------
void DynamicEQ::PrecomputeTrigConstants(const uint32_t band) noexcept {
    const auto& p = params_[band];
    const double w0 = 2.0 * std::numbers::pi_v<double>
                      * static_cast<double>(p.frequency)
                      / static_cast<double>(sampling_rate_);
    const float cs = static_cast<float>(std::cos(w0));
    const float sn = static_cast<float>(std::sin(w0));

    trig_cache_[band] = {
        .cos_w0       = cs,
        .sin_w0       = sn,
        .alpha        = sn / (2.0f * p.q),
        .neg_2_cos_w0 = -2.0f * cs
    };
}

void DynamicEQ::Reset() noexcept {
    for (uint32_t i = 0u; i < kMaxBands; ++i) {
        state_[i]           = BandState{};
        filter_state_[0][i] = BiquadState{};
        filter_state_[1][i] = BiquadState{};
        PrecomputeTrigConstants(i);
        RecalcAttackRelease(i);
        FastUpdateBandCoeffs(i, 0.0f);
    }
}

void DynamicEQ::SetEnable(const bool enable) noexcept {
    if (enable_ != enable) {
        if (enable) Reset();
        enable_ = enable;
    }
}

void DynamicEQ::SetBandCount(const uint32_t count) noexcept {
    band_count_ = std::min(count, kMaxBands);
}

void DynamicEQ::SetSamplingRate(const uint32_t sampling_rate) noexcept {
    if (sampling_rate_ != sampling_rate && sampling_rate > 0u) {
        sampling_rate_ = sampling_rate;
        Reset();
    }
}

void DynamicEQ::SetBandFrequency(const uint32_t band, const float value) noexcept {
    if (band < kMaxBands) {
        params_[band].frequency = value;
        PrecomputeTrigConstants(band);
        FastUpdateBandCoeffs(band, state_[band].current_gain_db);
    }
}

void DynamicEQ::SetBandGain(const uint32_t band, const float value) noexcept {
    if (band < kMaxBands) params_[band].target_gain_db = value;
}

void DynamicEQ::SetBandQ(const uint32_t band, const float value) noexcept {
    if (band < kMaxBands) {
        params_[band].q = std::max(value, 0.1f);
        PrecomputeTrigConstants(band);
        FastUpdateBandCoeffs(band, state_[band].current_gain_db);
    }
}

void DynamicEQ::SetBandThreshold(const uint32_t band, const float value) noexcept {
    if (band < kMaxBands) {
        params_[band].threshold_db        = value;
        // 10^(threshold_db/10): FastDbToPower = FastExp2(db * log2(10)/10)
        // NOT FastDbToLinear(value*0.5) which would give 10^(db/40) — wrong by 4×
        params_[band].linear_threshold_sq = viper::dsp::FastDbToPower(value);
    }
}

void DynamicEQ::SetBandAttack(const uint32_t band, const float value) noexcept {
    if (band < kMaxBands) {
        params_[band].attack_ms = value;
        RecalcAttackRelease(band);
    }
}

void DynamicEQ::SetBandRelease(const uint32_t band, const float value) noexcept {
    if (band < kMaxBands) {
        params_[band].release_ms = value;
        RecalcAttackRelease(band);
    }
}

void DynamicEQ::SetBandFilterType(const uint32_t band, const int value) noexcept {
    if (band < kMaxBands) {
        params_[band].filter_type = std::clamp(value, 0, 2);
        FastUpdateBandCoeffs(band, state_[band].current_gain_db);
    }
}

void DynamicEQ::RecalcAttackRelease(const uint32_t band) noexcept {
    const auto sr      = static_cast<float>(sampling_rate_);
    const auto att_sec = params_[band].attack_ms  / 1000.0f;
    const auto rel_sec = params_[band].release_ms / 1000.0f;

    state_[band].attack_coeff  = att_sec > 0.0f
                                     ? 1.0f - std::exp(-1.0f / (att_sec * sr)) : 1.0f;
    state_[band].release_coeff = rel_sec > 0.0f
                                     ? 1.0f - std::exp(-1.0f / (rel_sec * sr)) : 1.0f;
}
