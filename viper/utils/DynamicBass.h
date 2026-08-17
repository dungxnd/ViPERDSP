#pragma once

#include "Biquad.h"
#include "PolesFilter.h"
#include <cstdint>

class DynamicBass {
public:
    DynamicBass();

    void FilterSamples(float *samples, uint32_t size) noexcept;
    void Reset() noexcept;

    void SetBassGain(float value) noexcept;
    void SetFilterXPassFrequency(uint32_t low, uint32_t high) noexcept;
    void SetFilterYPassFrequency(uint32_t low, uint32_t high) noexcept;
    void SetSideGain(float gain_x, float gain_y) noexcept;
    void SetSamplingRate(uint32_t sampling_rate) noexcept;

private:
    uint32_t low_freq_x_{120};
    uint32_t high_freq_x_{80};
    uint32_t low_freq_y_{40};
    uint32_t high_freq_y_{0};
    uint32_t sampling_rate_;

    float q_peak_{0.0f};
    float bass_gain_{1.0f};
    float side_gain_x_{1.0f};
    float side_gain_y_{1.0f};

    PolesFilter filter_x_{};
    PolesFilter filter_y_{};
    Biquad low_pass_{};

    void RecalcLowPass() noexcept;
};
