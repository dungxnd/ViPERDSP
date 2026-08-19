#include "DepthSurround.h"
#include <cmath>

DepthSurround::DepthSurround() noexcept {
    SetSamplingRate(44100u);
    RefreshStrength(strength_);
}

void DepthSurround::Process(float* const samples, const uint32_t size) noexcept {
    if (!enable_) return;

    const float gain_r = strength_at_least500_ ? -gain_ : gain_;

    for (uint32_t i = 0; i < size; ++i) {
        const float sample_l = samples[2 * i];
        const float sample_r = samples[2 * i + 1];

        prev_[0] = gain_  * time_const_delay_[0].ProcessSample(sample_l + prev_[1]);
        prev_[1] = gain_r * time_const_delay_[1].ProcessSample(sample_r + prev_[0]);

        const float l    = prev_[0] + sample_l;
        const float r    = prev_[1] + sample_r;
        const float diff = (l - r) / 2.0f;
        const float avg  = (l + r) / 2.0f;
        const auto  avg_out = static_cast<float>(highpass_.ProcessSample(diff));

        samples[2 * i]     = avg + (diff - avg_out);
        samples[2 * i + 1] = avg - (diff - avg_out);
    }
}

void DepthSurround::SetStrength(const uint32_t value) noexcept {
    strength_ = value;
    RefreshStrength(value);
}

void DepthSurround::RefreshStrength(const uint32_t strength) noexcept {
    strength_at_least500_ = strength >= 500;
    enable_ = (strength != 0);
    if (strength != 0) {
        const auto gain = static_cast<float>(
            std::pow(10.0, (strength / 1000.0 * 10.0 - 15.0) / 20.0)
        );
        gain_ = std::min(gain, 1.0f);
    } else {
        gain_ = 0.0f;
    }
}

void DepthSurround::SetSamplingRate(const uint32_t sampling_rate) noexcept {
    time_const_delay_[0].SetParameters(sampling_rate, 0.02f);
    time_const_delay_[1].SetParameters(sampling_rate, 0.014f);
    highpass_.SetHighPassParameter(800.0f, sampling_rate, -11.0, 0.72f);
    prev_.fill(0.0f);
}
