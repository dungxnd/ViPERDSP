#include "DynamicEQ.h"
#include <algorithm>
#include <array>
#include <cmath>

namespace {
constexpr float kGainChangeThreshold = 0.1f;
constexpr float kOvershootRangeDb    = 12.0f;
constexpr double kMinEnvelope        = 1e-20;
} // namespace

DynamicEQ::DynamicEQ() {
    Reset();
}

void DynamicEQ::Process(float *samples, const uint32_t size) {
    if (!enable_ || band_count_ == 0 || size == 0) return;

    const uint32_t frame_count = size * 2;

    for (uint32_t b = 0; b < band_count_; ++b) {
        auto attack_coeff  = state_[b].attack_coeff;
        auto release_coeff = state_[b].release_coeff;
        auto env_l         = state_[b].envelope_l;
        auto env_r         = state_[b].envelope_r;
        auto smoothed_gain = state_[b].smoothed_gain_db;
        auto last_applied  = state_[b].last_applied_gain_db;
        const float  target_gain = params_[b].target_gain_db;
        const auto   threshold   = static_cast<double>(params_[b].threshold_db);

        for (uint32_t i = 0; i < frame_count; i += 2) {
            const auto sample_l = static_cast<double>(samples[i]);
            const auto sample_r = static_cast<double>(samples[i + 1]);

            const double power_l = sample_l * sample_l;
            const double power_r = sample_r * sample_r;

            env_l += (power_l > env_l ? attack_coeff : release_coeff) * (power_l - env_l);
            env_r += (power_r > env_r ? attack_coeff : release_coeff) * (power_r - env_r);

            const double rms = std::max(std::sqrt(std::max(env_l, env_r)), kMinEnvelope);
            const double envelope_db   = 20.0 * std::log10(rms);
            const double overshoot     = envelope_db - threshold;

            double desired_gain_db = 0.0;
            if (overshoot > 0.0) {
                const double ratio = std::min(overshoot / kOvershootRangeDb, 1.0);
                desired_gain_db = static_cast<double>(target_gain) * ratio;
            }

            const double gain_coeff = (std::abs(desired_gain_db) > std::abs(smoothed_gain))
                                          ? attack_coeff : release_coeff;
            smoothed_gain += gain_coeff * (desired_gain_db - smoothed_gain);

            if (const auto current_gain_db = static_cast<float>(smoothed_gain);
                std::abs(current_gain_db - last_applied) > kGainChangeThreshold) {
                ConfigureApplicationFilter(b, current_gain_db);
                last_applied = current_gain_db;
            }

            samples[i]     = static_cast<float>(apply_l_[b].ProcessSample(sample_l));
            samples[i + 1] = static_cast<float>(apply_r_[b].ProcessSample(sample_r));
        }

        state_[b].envelope_l           = env_l;
        state_[b].envelope_r           = env_r;
        state_[b].smoothed_gain_db      = smoothed_gain;
        state_[b].last_applied_gain_db  = last_applied;
    }
}

void DynamicEQ::Reset() {
    for (uint32_t i = 0; i < kMaxBands; ++i) {
        state_[i] = BandState{};

        apply_l_[i].Reset();
        apply_r_[i].Reset();

        RecalcAttackRelease(i);
        ConfigureApplicationFilter(i, 0.0f);
    }
}

void DynamicEQ::SetEnable(const bool enable) {
    if (enable_ != enable) {
        if (enable) Reset();
        enable_ = enable;
    }
}

void DynamicEQ::SetBandCount(uint32_t count) {
    count = std::min(count, kMaxBands);
    if (band_count_ != count) {
        band_count_ = count;
        Reset();
    }
}

void DynamicEQ::SetSamplingRate(const uint32_t sampling_rate) {
    if (sampling_rate_ != sampling_rate) {
        sampling_rate_ = sampling_rate;
        Reset();
    }
}

void DynamicEQ::SetBandFrequency(const uint32_t band, const float value) {
    params_[band].frequency = value;
    ConfigureApplicationFilter(band, 0.0f);
    state_[band].last_applied_gain_db = 0.0f;
}

void DynamicEQ::SetBandQ(const uint32_t band, const float value) {
    params_[band].q = value;
    ConfigureApplicationFilter(band, 0.0f);
    state_[band].last_applied_gain_db = 0.0f;
}

void DynamicEQ::SetBandGain(const uint32_t band, const float value) {
    params_[band].target_gain_db = value;
}

void DynamicEQ::SetBandThreshold(const uint32_t band, const float value) {
    params_[band].threshold_db = value;
}

void DynamicEQ::SetBandAttack(const uint32_t band, const float value) {
    params_[band].attack_ms = value;
    RecalcAttackRelease(band);
}

void DynamicEQ::SetBandRelease(const uint32_t band, const float value) {
    params_[band].release_ms = value;
    RecalcAttackRelease(band);
}

void DynamicEQ::SetBandFilterType(const uint32_t band, const int value) {
    static constexpr std::array<MultiBiquad::FilterType, 3> kTypes{{
        MultiBiquad::FilterType::Peak,
        MultiBiquad::FilterType::LowShelf,
        MultiBiquad::FilterType::HighShelf,
    }};
    params_[band].filter_type = (value >= 0 && value <= 2)
        ? kTypes[value]
        : MultiBiquad::FilterType::Peak;
    ConfigureApplicationFilter(band, 0.0f);
    state_[band].last_applied_gain_db = 0.0f;
}

void DynamicEQ::RecalcAttackRelease(const uint32_t band) {
    const auto sr = static_cast<double>(sampling_rate_);
    const auto attack_sec  = static_cast<double>(params_[band].attack_ms)  / 1000.0;
    const auto release_sec = static_cast<double>(params_[band].release_ms) / 1000.0;

    state_[band].attack_coeff  = (attack_sec  > 0.0) ? 1.0 - std::exp(-1.0 / (attack_sec  * sr)) : 1.0;
    state_[band].release_coeff = (release_sec > 0.0) ? 1.0 - std::exp(-1.0 / (release_sec * sr)) : 1.0;
}

void DynamicEQ::ConfigureApplicationFilter(const uint32_t band, const float gain_db) {
    const auto& p = params_[band];
    apply_l_[band].RefreshFilter(p.filter_type, gain_db, p.frequency, sampling_rate_, p.q, false);
    apply_r_[band].RefreshFilter(p.filter_type, gain_db, p.frequency, sampling_rate_, p.q, false);
}
