#pragma once

#include <cstdint>

class Biquad {
public:
    Biquad() noexcept;

    double ProcessSample(double sample) noexcept;
    void   Reset() noexcept;

    void SetCoeffs(double a0, double a1, double a2, double b0, double b1, double b2) noexcept;
    void SetBandPassParameter(float frequency, uint32_t sampling_rate, float q_factor);
    void SetHighPassParameter(float frequency, uint32_t sampling_rate, double db_gain, float q_factor);
    void SetLowPassParameter(float frequency, uint32_t sampling_rate, float q_factor);

private:
    double x1_ = 0.0;
    double x2_ = 0.0;
    double y1_ = 0.0;
    double y2_ = 0.0;
    double a1_ = 0.0;
    double a2_ = 0.0;
    double b0_ = 0.0;
    double b1_ = 0.0;
    double b2_ = 0.0;
};
