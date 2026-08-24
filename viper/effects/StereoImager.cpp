#include "StereoImager.h"
#include <array>

StereoImager::StereoImager() {
    ConfigureCrossovers();
}

void StereoImager::Process(float *samples, const uint32_t size) noexcept {
    if (!enable_ || size == 0) return;
    if (size > kMaxFrames) return;

    const uint32_t frame_count = size * 2;

    for (uint32_t b = 0; b < kNumBands; b++) {
        for (uint32_t i = 0; i < frame_count; i += 2) {
            double sample_l = samples[i];
            double sample_r = samples[i + 1];

            crossover_.ProcessSampleStereo(b, kNumBands, sample_l, sample_r);

            const auto f_l = static_cast<float>(sample_l);
            const auto f_r = static_cast<float>(sample_r);

            const float mid  = (f_l + f_r) * 0.5f;
            const float side = (f_l - f_r) * 0.5f * band_widths_[b];

            band_buffers_[b][i]     = mid + side;
            band_buffers_[b][i + 1] = mid - side;
        }
    }

    for (uint32_t i = 0; i < frame_count; i += 2) {
        float sum_l = 0.0f;
        float sum_r = 0.0f;
        for (uint32_t b = 0; b < kNumBands; b++) {
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

    for (uint32_t b = 0u; b < kNumBands; ++b) {
        for (size_t i = 0u; i < frames; ++i) {
            double sl = L[i];
            double sr = R[i];
            crossover_.ProcessSampleStereo(b, kNumBands, sl, sr);

            const float fl = static_cast<float>(sl);
            const float fr = static_cast<float>(sr);
            const float mid  = (fl + fr) * 0.5f;
            const float side = (fl - fr) * 0.5f * band_widths_[b];
            band_buffers_[b][i * 2u]      = mid + side;
            band_buffers_[b][i * 2u + 1u] = mid - side;
        }
    }

    for (size_t i = 0u; i < frames; ++i) {
        float sum_l = 0.0f;
        float sum_r = 0.0f;
        for (uint32_t b = 0u; b < kNumBands; ++b) {
            sum_l += band_buffers_[b][i * 2u];
            sum_r += band_buffers_[b][i * 2u + 1u];
        }
        L[i] = sum_l;
        R[i] = sum_r;
    }
}
