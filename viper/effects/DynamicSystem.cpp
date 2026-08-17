#include "DynamicSystem.h"
#include "../constants.h"

DynamicSystem::DynamicSystem()
    : sampling_rate_(VIPER_DEFAULT_SAMPLING_RATE) {
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
