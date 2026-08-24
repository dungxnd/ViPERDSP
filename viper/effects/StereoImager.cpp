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
        // Block crossover filter for band b
        crossover_.ProcessBand(b, kNumBands, scratch_l_.data(), scratch_r_.data(), size);
        // Apply stereo width and re-interleave into band_buffers_[b]
        const float w = band_widths_[b];
        for (uint32_t f = 0; f < size; ++f) {
            const float fl   = scratch_l_[f];
            const float fr   = scratch_r_[f];
            const float mid  = (fl + fr) * 0.5f;
            const float side = (fl - fr) * 0.5f * w;
            band_buffers_[b][f * 2u]      = mid + side;
            band_buffers_[b][f * 2u + 1u] = mid - side;
        }
    }

    for (uint32_t i = 0; i < size * 2u; i += 2) {
        float sum_l = 0.0f, sum_r = 0.0f;
        for (uint32_t b = 0; b < kNumBands; ++b) {
            sum_l += band_buffers_[b][i];
            sum_r += band_buffers_[b][i + 1];
        }
        samples[i]     = sum_l;
        samples[i + 1] = sum_r;
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

void StereoImager::ProcessPlanar(float* __restrict L, float* __restrict R, const size_t frames) noexcept {
    if (!IsEnabled() || frames == 0u) return;
    if (frames > kMaxFrames) return;

    const auto size = static_cast<uint32_t>(frames);

    for (uint32_t b = 0u; b < kNumBands; ++b) {
        // scratch_l_/scratch_r_ already planar — copy L/R directly
        for (uint32_t f = 0u; f < size; ++f) {
            scratch_l_[f] = L[f];
            scratch_r_[f] = R[f];
        }
        crossover_.ProcessBand(b, kNumBands, scratch_l_.data(), scratch_r_.data(), frames);
        // Apply stereo width and store interleaved into band_buffers_[b]
        const float w = band_widths_[b];
        for (uint32_t f = 0u; f < size; ++f) {
            const float fl   = scratch_l_[f];
            const float fr   = scratch_r_[f];
            const float mid  = (fl + fr) * 0.5f;
            const float side = (fl - fr) * 0.5f * w;
            band_buffers_[b][f * 2u]      = mid + side;
            band_buffers_[b][f * 2u + 1u] = mid - side;
        }
    }

    for (uint32_t f = 0u; f < size; ++f) {
        float sum_l = 0.0f, sum_r = 0.0f;
        for (uint32_t b = 0u; b < kNumBands; ++b) {
            sum_l += band_buffers_[b][f * 2u];
            sum_r += band_buffers_[b][f * 2u + 1u];
        }
        L[f] = sum_l;
        R[f] = sum_r;
    }
}
