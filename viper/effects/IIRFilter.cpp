#include "IIRFilter.h"
#include <algorithm>
#include <cmath>

IIRFilter::IIRFilter(const uint32_t bands) {
    target_gains_.fill(kDefaultLevelQ);
    current_gains_.fill(kDefaultLevelQ);
    UpdateGainSmoothCoeff();
    SetBandCount(bands);
    // SetBandCount already calls Reset(); no second Reset() needed.
}

void IIRFilter::Process(std::span<float> samples) noexcept {
    // bands_ == 0 means no valid coefficients; leave the buffer untouched so
    // audio passes through unmodified rather than being zeroed.
    if (!enable_ || bands_ == 0 || samples.empty()) return;
    [[assume(samples.size() % 2 == 0)]];

    const auto coeffs = min_phase_iir_coeffs_.GetCoefficients();
    if (coeffs.empty()) return;

    const size_t frame_count = samples.size() / 2u;
    StereoView audio(samples.data(), frame_count, 2u);

    for (size_t f = 0; f < frame_count; ++f) {
        // Smooth all gains toward their targets once per frame.
        // gain_smooth_coeff_ ≈ 1 − e^(−1/(0.02·Fs)), gives τ ≈ 20 ms.
        for (uint32_t k = 0; k < bands_; ++k) {
            current_gains_[k] += (target_gains_[k] - current_gains_[k]) * gain_smooth_coeff_;
        }

        // Transposed Direct Form II per stereo channel.
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

                // Clamp diverged state so one bad band cannot poison subsequent
                // frames through the recursive feedback path.
                if (!std::isfinite(st.s1)) st.s1 = 0.0f;
                if (!std::isfinite(st.s2)) st.s2 = 0.0f;

                accumulated += y * current_gains_[k];
            }

            // Leave the sample untouched if the whole sum is non-finite
            // (e.g. NaN from a corrupt band level) rather than zeroing the stream.
            if (std::isfinite(accumulated)) [[likely]] {
                audio[f, ch] = accumulated;
            }
        }
    }
}

void IIRFilter::Reset() noexcept {
    for (auto& ch : state_) ch.fill(BiquadState{});
    // Snap current gains to target so re-enabling produces no zipper artifact.
    current_gains_ = target_gains_;
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
    if (band >= kMaxBands) return;
    const double linear_gain = std::pow(10.0, static_cast<double>(level) / 20.0);
    target_gains_[band] = static_cast<float>(linear_gain * 0.636);
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
        UpdateGainSmoothCoeff();
        if (bands_ != 0) {
            [[maybe_unused]] const auto ok =
                min_phase_iir_coeffs_.UpdateCoeffs(bands_, sampling_rate_);
        }
        Reset();
    }
}

void IIRFilter::UpdateGainSmoothCoeff() noexcept {
    // τ = 20 ms → coefficient = 1 − e^(−1 / (0.02 · Fs))
    gain_smooth_coeff_ =
        1.0f - std::exp(-1.0f / (0.020f * static_cast<float>(sampling_rate_)));
}
