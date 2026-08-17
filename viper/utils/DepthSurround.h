#pragma once

#include "Biquad.h"
#include "TimeConstDelay.h"
#include <array>
#include <cstdint>

class DepthSurround {
public:
    DepthSurround();

    void Process(float* samples, uint32_t size);

    void SetStrength(uint32_t value);
    void SetSamplingRate(uint32_t sampling_rate);
    void RefreshStrength(uint32_t strength);

private:
    bool     enable_              = false;
    bool     strength_at_least500_ = false;
    uint32_t strength_            = 0;
    float    gain_                = 0.0f;

    std::array<float, 2>         prev_{};
    std::array<TimeConstDelay, 2> time_const_delay_;
    Biquad                        highpass_;
};
