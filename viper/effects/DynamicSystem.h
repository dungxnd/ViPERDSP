#pragma once

#include "../utils/DynamicBass.h"
#include <cstdint>

class DynamicSystem {
public:
    DynamicSystem();

    void Process(float *samples, uint32_t size) noexcept;
    void Reset() noexcept;

    void SetEnable(bool enable) noexcept;
    void SetBassGain(float gain) noexcept;
    void SetSamplingRate(uint32_t sampling_rate) noexcept;
    void SetSideGain(float gain_low, float gain_high) noexcept;
    void SetXCoeffs(int low, int high) noexcept;
    void SetYCoeffs(int low, int high) noexcept;

private:
    bool enable_{false};

    uint32_t sampling_rate_{44100u};

    int x_low_{0};
    int x_high_{0};
    int y_low_{0};
    int y_high_{0};
    float side_gain_low_{0.0f};
    float side_gain_high_{0.0f};

    DynamicBass dynamic_bass_{};
};
