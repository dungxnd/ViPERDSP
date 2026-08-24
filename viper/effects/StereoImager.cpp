#include "StereoImager.h"
#include <array>

StereoImager::StereoImager() {
    ConfigureCrossovers();
}

void StereoImager::Process(float *samples, const uint32_t size) noexcept {
    if (!enable_ || size == 0) return;
    if (size > kMaxFrames) return;

    for (uint32_t b = 0; b < kNumBands; ++b) {
        // Deinterleave input into planar scratch
        for (uint32_t f = 0; f < size; ++f) {
            scratch_l_[f] = samples[f * 2u];
            scratch_r_[f] = samples[f * 2u + 1u];
        }
        // Block crossover filter for band b (in-place on scratch)
        crossover_.ProcessBand(b, kNumBands, scratch_l_.data(), scratch_r_.data(), size);
        // Apply stereo width; accumulate directly into output — no band_buffers_
        const float w = band_widths_[b];
        if (b == 0u) {
            for (uint32_t f = 0; f < size; ++f) {
                const float mid  = (scratch_l_[f] + scratch_r_[f]) * 0.5f;
                const float side = (scratch_l_[f] - scratch_r_[f]) * 0.5f * w;
                samples[f * 2u]      = mid + side;
                samples[f * 2u + 1u] = mid - side;
            }
        } else {
            for (uint32_t f = 0; f < size; ++f) {
                const float mid  = (scratch_l_[f] + scratch_r_[f]) * 0.5f;
                const float side = (scratch_l_[f] - scratch_r_[f]) * 0.5f * w;
                samples[f * 2u]      += mid + side;
                samples[f * 2u + 1u] += mid - side;
            }
        }
    }
}

void StereoImager::Reset() noexcept {
    crossover_.Reset();
    ConfigureCrossovers();
}

void StereoImager::SetEnable(const bool enable) noexcept {
    if (enable_ != enable) {
        if (enable) Reset();
        enable_ = enable;
    }
}

void StereoImager::SetLowWidth(const float value) noexcept {
    band_widths_[0] = value / 100.0f;
}

void StereoImager::SetMidWidth(const float value) noexcept {
    band_widths_[1] = value / 100.0f;
}

void StereoImager::SetHighWidth(const float value) noexcept {
    band_widths_[2] = value / 100.0f;
}

void StereoImager::SetLowCrossover(const float value) noexcept {
    if (crossover_freqs_[0] != value) {
        crossover_freqs_[0] = value;
        ConfigureCrossovers();
    }
}

void StereoImager::SetHighCrossover(const float value) noexcept {
    if (crossover_freqs_[1] != value) {
        crossover_freqs_[1] = value;
        ConfigureCrossovers();
    }
}

void StereoImager::SetSamplingRate(const uint32_t sampling_rate) noexcept {
    if (sampling_rate_ != sampling_rate) {
        sampling_rate_ = sampling_rate;
        ConfigureCrossovers();
    }
}

void StereoImager::ConfigureCrossovers() noexcept {
    crossover_.Configure(crossover_freqs_.data(), kNumCrossovers, sampling_rate_);
}

void StereoImager::ProcessPlanar(std::span<float> L, std::span<float> R) noexcept {
    if (!IsEnabled() || L.empty()) return;
    if (L.size() > kMaxFrames) return;

    const auto size = static_cast<uint32_t>(L.size());

    for (uint32_t b = 0u; b < kNumBands; ++b) {
        // Copy L/R into planar scratch — crossover filters the scratch in-place
        for (uint32_t f = 0u; f < size; ++f) {
            scratch_l_[f] = L[f];
            scratch_r_[f] = R[f];
        }
        crossover_.ProcessBand(b, kNumBands, scratch_l_.data(), scratch_r_.data(), size);
        // Apply stereo width; accumulate directly into L/R — no band_buffers_
        const float w = band_widths_[b];
        if (b == 0u) {
            #pragma clang loop vectorize(enable)
            for (uint32_t f = 0u; f < size; ++f) {
                const float mid  = (scratch_l_[f] + scratch_r_[f]) * 0.5f;
                const float side = (scratch_l_[f] - scratch_r_[f]) * 0.5f * w;
                L[f] = mid + side;
                R[f] = mid - side;
            }
        } else {
            #pragma clang loop vectorize(enable)
            for (uint32_t f = 0u; f < size; ++f) {
                const float mid  = (scratch_l_[f] + scratch_r_[f]) * 0.5f;
                const float side = (scratch_l_[f] - scratch_r_[f]) * 0.5f * w;
                L[f] += mid + side;
                R[f] += mid - side;
            }
        }
    }
}
