#include "DynamicSystem.h"

DynamicSystem::DynamicSystem() {
    dynamic_bass_.SetSamplingRate(sampling_rate_);
    dynamic_bass_.Reset();
}

void DynamicSystem::Process(float *samples, const uint32_t size) noexcept {
    if (!enable_) return;
    dynamic_bass_.FilterSamples(samples, size);
}

void DynamicSystem::Reset() noexcept {
    dynamic_bass_.SetSamplingRate(sampling_rate_);
    dynamic_bass_.Reset();
}

void DynamicSystem::SetEnable(const bool enable) noexcept {
    if (enable_ != enable) {
        if (enable) Reset();
        enable_ = enable;
    }
}

void DynamicSystem::SetBassGain(const float gain) noexcept {
    dynamic_bass_.SetBassGain(gain);
}

void DynamicSystem::SetSamplingRate(const uint32_t sampling_rate) noexcept {
    if (sampling_rate_ != sampling_rate) {
        sampling_rate_ = sampling_rate;
        dynamic_bass_.SetSamplingRate(sampling_rate);
    }
}

void DynamicSystem::SetSideGain(const float gain_low, const float gain_high) noexcept {
    if (gain_low  >= 0.0f) side_gain_low_  = gain_low;
    if (gain_high >= 0.0f) side_gain_high_ = gain_high;
    dynamic_bass_.SetSideGain(side_gain_low_, side_gain_high_);
}

void DynamicSystem::SetXCoeffs(const int low, const int high) noexcept {
    if (low  >= 0) x_low_  = low;
    if (high >= 0) x_high_ = high;
    dynamic_bass_.SetFilterXPassFrequency(
        static_cast<uint32_t>(x_low_), static_cast<uint32_t>(x_high_));
}

void DynamicSystem::SetYCoeffs(const int low, const int high) noexcept {
    if (low  >= 0) y_low_  = low;
    if (high >= 0) y_high_ = high;
    dynamic_bass_.SetFilterYPassFrequency(
        static_cast<uint32_t>(y_low_), static_cast<uint32_t>(y_high_));
}

void DynamicSystem::ProcessPlanar(float* __restrict L, float* __restrict R, const size_t frames) noexcept {
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
