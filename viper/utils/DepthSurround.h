#pragma once

#include "Biquad.h"
#include "TimeConstDelay.h"
#include <array>
#include <cstdint>

class DepthSurround {
public:
    DepthSurround() noexcept;

    void Process(float* samples, uint32_t size) noexcept;

    void SetStrength(uint32_t value) noexcept;
    void SetSamplingRate(uint32_t sampling_rate) noexcept;
    void RefreshStrength(uint32_t strength) noexcept;

private:
    bool     enable_               = false;
    bool     strength_at_least500_ = false;
    uint32_t strength_             = 0;
    float    gain_                 = 0.0f;

    std::array<float, 2>          prev_{};
    std::array<TimeConstDelay, 2> time_const_delay_;
    Biquad                        highpass_;
};
