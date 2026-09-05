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

void ColorfulMusic::SetConfig(const Config& config) noexcept {
    config_ = config;
    SetEnable(config.enable);
    SetWidenValue(config.widening);
    SetMidImageValue(config.mid_image);
    SetDepthValue(config.depth);
}

void ColorfulMusic::SetEnable(const bool enable) {
    config_.enable = enable;
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

void ColorfulMusic::ProcessPlanar(std::span<float> L, std::span<float> R) noexcept {
    if (!IsEnabled() || L.empty()) return;
    depth_surround_.ProcessPlanar(L.data(), R.data(), L.size());
    stereo_3d_surround_.ProcessPlanar(L.data(), R.data(), L.size());
}
