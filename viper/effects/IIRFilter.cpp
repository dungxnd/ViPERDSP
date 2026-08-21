#include "IIRFilter.h"
#include <algorithm>
#include <cmath>

IIRFilter::IIRFilter(const uint32_t bands) {
    target_gains_.fill(kDefaultLevelQ);
    current_gains_.fill(kDefaultLevelQ);
    UpdateCoeffConstants();
    SetBandCount(bands);
}

void IIRFilter::Process(std::span<float> samples) noexcept {
    // bands_ == 0 means no valid coefficients; leave the buffer untouched (passthrough).
    if (!enable_ || bands_ == 0 || samples.empty()) return;
    [[assume(samples.size() % 2 == 0)]];

    const auto coeffs = min_phase_iir_coeffs_.GetCoefficients();
    if (coeffs.empty()) return;

    const size_t frame_count = samples.size() / 2u;
    StereoView audio(samples.data(), frame_count, 2u);

    for (size_t f = 0; f < frame_count; ++f) {
        // ── 1. Gain smoothing (τ = 20 ms) ──────────────────────────────────────
        // Only updates while sliders are in motion; snaps once converged.
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
        for (size_t ch = 0; ch < 2; ++ch) {
            const float in = audio[f, ch];
            float accumulated = 0.0f;

            // Clean, branchless inner loop that auto-vectorizes naturally on SIMD targets
            for (uint32_t k = 0; k < bands_; ++k) {
                const auto& c  = coeffs[k];
                auto&       st = state_[ch][k];

                // TDF-II (b1 = 0, b2 = -b0):
                //   y    = b0·x + s1
                //   s1   = -a1·y + s2
                //   s2   = -b0·x - a2·y
                const float y = c.b0 * in + st.s1;
                st.s1 = -c.a1 * y + st.s2;
                st.s2 = -c.b0 * in - c.a2 * y;

                accumulated += y * current_gains_[k];
            }

            // ── 3. Passthrough safety & Soft fade-in ───────────────────────────
            // If the output sum is finite, write it; otherwise keep dry input (safe passthrough).
            if (std::isfinite(accumulated)) [[likely]] {
                if (fade_in_gain_ < 1.0f) [[unlikely]] {
                    audio[f, ch] = std::lerp(in, accumulated, fade_in_gain_);
                } else {
                    audio[f, ch] = accumulated;
                }
            }
        }

        // Advance fade-in ramp once per frame
        if (fade_in_gain_ < 1.0f) [[unlikely]] {
            fade_in_gain_ = std::min(fade_in_gain_ + fade_in_step_, 1.0f);
        }
    }
}

void IIRFilter::Reset() noexcept {
    for (auto& ch : state_) ch.fill(BiquadState{});
    current_gains_ = target_gains_;
    gains_dirty_   = false;
    fade_in_gain_  = 0.0f;
}

void IIRFilter::SetEnable(const bool enable) noexcept {
    if (enable_ != enable) {
        if (enable) Reset();
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
    if (band >= kMaxBands || !std::isfinite(level)) return;
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
    gain_smooth_coeff_ = 1.0f - std::exp(-1.0f / (0.020f * sr_f));
    fade_in_step_      = 1.0f / (0.010f * sr_f);
}
