#include "ColorfulMusic.h"

ColorfulMusic::ColorfulMusic() {
    stereo_3d_surround_.SetStereoWiden(0.0f);
    depth_surround_.SetSamplingRate(sampling_rate_);
    depth_surround_.SetStrength(0);
}

void ColorfulMusic::Process(float* const samples, const uint32_t size) {
    if (!enabled_) return;
    depth_surround_.Process(samples, size);
    stereo_3d_surround_.Process(samples, size);
}

void ColorfulMusic::Reset() {
    depth_surround_.SetSamplingRate(sampling_rate_);
}

void ColorfulMusic::SetEnable(const bool enable) {
    if (enabled_ != enable) {
        if (!enabled_) {
            Reset();
        }
        enabled_ = enable;
    }
}

void ColorfulMusic::SetDepthValue(const uint32_t value) {
    depth_surround_.SetStrength(value);
}

void ColorfulMusic::SetMidImageValue(const float value) {
    stereo_3d_surround_.SetMiddleImage(value);
}

void ColorfulMusic::SetWidenValue(const float value) {
    stereo_3d_surround_.SetStereoWiden(value);
}

void ColorfulMusic::SetSamplingRate(const uint32_t sampling_rate) {
    if (sampling_rate_ != sampling_rate) {
        sampling_rate_ = sampling_rate;
        depth_surround_.SetSamplingRate(sampling_rate_);
    }
}

void ColorfulMusic::ProcessPlanar(float* __restrict L, float* __restrict R, const size_t frames) noexcept {
    if (!IsEnabled() || frames == 0) return;
    const auto n = static_cast<uint32_t>(frames);
    for (size_t i = 0; i < frames; ++i) {
        pp_scratch_[2u * i]      = L[i];
        pp_scratch_[2u * i + 1u] = R[i];
    }
    Process(pp_scratch_.data(), n);
    for (size_t i = 0; i < frames; ++i) {
        L[i] = pp_scratch_[2u * i];
        R[i] = pp_scratch_[2u * i + 1u];
    }
}
