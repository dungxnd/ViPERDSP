#include "IIRFilter.h"
#include <algorithm>
#include <cmath>

IIRFilter::IIRFilter(const uint32_t bands) {
    target_gains_.fill(kDefaultLevelQ);
    current_gains_.fill(kDefaultLevelQ);
    UpdateCoeffConstants();
    SetBandCount(bands);
    // SetBandCount already calls Reset(); no second Reset() needed.
}

void IIRFilter::Process(std::span<float> samples) noexcept {
    // bands_ == 0 means no valid coefficients; leave the buffer untouched.
    if (!enable_ || bands_ == 0 || samples.empty()) return;
    [[assume(samples.size() % 2 == 0)]];

    const auto coeffs = min_phase_iir_coeffs_.GetCoefficients();
    if (coeffs.empty()) return;

    const size_t frame_count = samples.size() / 2u;
    StereoView audio(samples.data(), frame_count, 2u);

    for (size_t f = 0; f < frame_count; ++f) {
        // ── 1. Gain smoothing (τ = 20 ms) ──────────────────────────────────────
        // Only runs while a slider is moving; snaps to target once converged to
        // avoid ~4 000 CPU cycles/frame during steady-state playback.
        if (gains_dirty_) [[unlikely]] {
            bool still_moving = false;
            for (uint32_t k = 0; k < bands_; ++k) {
                const float diff = target_gains_[k] - current_gains_[k];
                if (std::abs(diff) > 1e-5f) {
                    current_gains_[k] += diff * gain_smooth_coeff_;
                    still_moving = true;
                } else {
                    current_gains_[k] = target_gains_[k];
                }
            }
            gains_dirty_ = still_moving;
        }

        // ── 2. Transposed Direct Form II per stereo channel ────────────────────
        // The parallel bank processes all bands against the same dry input, so
        // the inner band loop has no loop-carried dependency and is
        // auto-vectorizable across NEON / AVX2 lanes.
        for (size_t ch = 0; ch < 2; ++ch) {
            const float in = audio[f, ch];
            float accumulated = 0.0f;

            #pragma clang loop vectorize(enable)
            for (uint32_t k = 0; k < bands_; ++k) {
                const auto& c  = coeffs[k];
                auto&       st = state_[ch][k];

                // TDF-II (b1=0, b2=-b0):
                //   y    = b0·x + s1
                //   s1   = -a1·y + s2
                //   s2   = -b0·x - a2·y
                const float y = c.b0 * in + st.s1;
                st.s1 = -c.a1 * y + st.s2;
                st.s2 = -c.b0 * in - c.a2 * y;

                // ── 3. State sanitation ───────────────────────────────────────
                // Flush subnormals first (ARM without FZ bit can stall 10–100×).
                // Then catch any surviving NaN/Inf from coefficient pathology.
                if (std::abs(st.s1) < kAntiDenormal) st.s1 = 0.0f;
                else if (!std::isfinite(st.s1))       st.s1 = 0.0f;
                if (std::abs(st.s2) < kAntiDenormal) st.s2 = 0.0f;
                else if (!std::isfinite(st.s2))       st.s2 = 0.0f;

                accumulated += y * current_gains_[k];
            }

            // Leave sample untouched if the whole sum is non-finite rather than
            // zeroing the stream (passthrough is always safer than silence).
            if (std::isfinite(accumulated)) [[likely]] {
                // ── 4. Soft fade-in on first enable (prevents click/pop) ──────
                if (fade_in_gain_ < 1.0f) [[unlikely]] {
                    audio[f, ch] = std::lerp(in, accumulated, fade_in_gain_);
                } else {
                    audio[f, ch] = accumulated;
                }
            }
        }

        // Advance fade-in ramp once per frame (same for both channels).
        if (fade_in_gain_ < 1.0f) [[unlikely]] {
            fade_in_gain_ = std::min(fade_in_gain_ + fade_in_step_, 1.0f);
        }
    }
}

void IIRFilter::Reset() noexcept {
    for (auto& ch : state_) ch.fill(BiquadState{});
    // Snap current gains to target so re-enabling produces no zipper artifact.
    current_gains_ = target_gains_;
    gains_dirty_   = false;
    // Reset fade-in ramp so the next enable() triggers a 10 ms onset crossfade.
    fade_in_gain_  = 0.0f;
}

void IIRFilter::SetEnable(const bool enable) noexcept {
    if (enable_ != enable) {
        if (enable) Reset();  // Reset() sets fade_in_gain_ = 0 → triggers ramp
        enable_ = enable;
    }
}

void IIRFilter::SetBandCount(const uint32_t bands) {
    if (bands != 10 && bands != 15 && bands != 25 && bands != 31) return;
    if (bands_ == bands) return;
    bands_ = bands;
    [[maybe_unused]] const auto ok = min_phase_iir_coeffs_.UpdateCoeffs(bands_, sampling_rate_);
    target_gains_.fill(kDefaultLevelQ);
    Reset();
}

void IIRFilter::SetBandLevel(const uint32_t band, const float level) noexcept {
    // ── 1. Input sanitization ─────────────────────────────────────────────────
    // Reject out-of-range band index or non-finite values from UI / IPC layer.
    // std::pow(10, NaN/20) → NaN → would silently corrupt target_gains_.
    if (band >= kMaxBands || !std::isfinite(level)) return;
    // Clamp to realistic graphic EQ boundaries so pow() stays well-behaved.
    const float  clamped      = std::clamp(level, -60.0f, 30.0f);
    const double linear_gain  = std::pow(10.0, static_cast<double>(clamped) / 20.0);
    target_gains_[band]       = static_cast<float>(linear_gain * kDefaultLevelQ);
    gains_dirty_              = true;
}

void IIRFilter::SetBandLevels(std::span<const float> levels) noexcept {
    const uint32_t count = std::min(static_cast<uint32_t>(levels.size()), bands_);
    for (uint32_t i = 0; i < count; ++i) {
        SetBandLevel(i, levels[i]);
    }
}

void IIRFilter::SetSamplingRate(const uint32_t sampling_rate) {
    if (sampling_rate_ != sampling_rate && sampling_rate > 0) {
        sampling_rate_ = sampling_rate;
        UpdateCoeffConstants();
        if (bands_ != 0) {
            [[maybe_unused]] const auto ok =
                min_phase_iir_coeffs_.UpdateCoeffs(bands_, sampling_rate_);
        }
        Reset();
    }
}

void IIRFilter::UpdateCoeffConstants() noexcept {
    const auto sr_f    = static_cast<float>(sampling_rate_);
    // τ = 20 ms → coefficient = 1 − e^(−1 / (0.02 · Fs))
    gain_smooth_coeff_ = 1.0f - std::exp(-1.0f / (0.020f * sr_f));
    // 10 ms linear fade-in ramp on enable; step = 1 frame / (0.010 · Fs frames).
    fade_in_step_      = 1.0f / (0.010f * sr_f);
}
