#pragma once

#include <cstdint>

class IIR_1st {
public:
    IIR_1st() = default;

    void Mute() noexcept;

    void SetCoefficients(float b0, float b1, float a1) noexcept;
    void SetHighPassFilterBW(float frequency, uint32_t sampling_rate) noexcept;
    void SetLowPassFilterBW(float frequency, uint32_t sampling_rate) noexcept;

    float b0_{0.0f};
    float b1_{0.0f};
    float a1_{0.0f};
    float prev_sample_{0.0f};
};

[[nodiscard]] inline float Filter(IIR_1st *filter, float sample) noexcept {
    const float hist = sample * filter->b1_;
    sample = filter->prev_sample_ + sample * filter->b0_;
    filter->prev_sample_ = sample * filter->a1_ + hist;
    return sample;
}
