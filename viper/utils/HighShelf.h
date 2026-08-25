#pragma once

#include <cstdint>

class HighShelf {
public:
    HighShelf() noexcept = default;

    double Process(double sample) noexcept;

    void SetFrequency(float value) noexcept;
    void SetGain(float value) noexcept;
    void SetSamplingRate(uint32_t sampling_rate) noexcept;

private:
    float  frequency_     = 0.0f;
    double sampling_rate_ = 44100.0;

    double gain_ = 0.0;
    double x1_   = 0.0;
    double x2_   = 0.0;
    double y1_   = 0.0;
    double y2_   = 0.0;
    double b0_   = 0.0;
    double b1_   = 0.0;
    double b2_   = 0.0;
    double a0_   = 0.0;
    double a1_   = 0.0;
    double a2_   = 0.0;

    // Recompute filter coefficients from the current frequency_/gain_/
    // sampling_rate_.  Does NOT clear delay state, so gain/frequency changes
    // during playback don't click.
    void UpdateCoefficients() noexcept;
};
