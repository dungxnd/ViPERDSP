#include "HighShelf.h"
#include <cmath>
#include <numbers>

double HighShelf::Process(const double sample) noexcept {
    const double out =
        (x1_ * b1_ + sample * b0_ + b2_ * x2_ - y1_ * a1_ - a2_ * y2_) * a0_;
    y2_ = y1_;
    y1_ = out;
    x2_ = x1_;
    x1_ = sample;
    return out;
}

void HighShelf::SetFrequency(const float value) noexcept {
    if (frequency_ == value) return;
    frequency_ = value;
    UpdateCoefficients();
}

void HighShelf::SetGain(const float value) noexcept {
    gain_ = 20.0 * std::log10(static_cast<double>(value));
    UpdateCoefficients();
}

void HighShelf::SetSamplingRate(const uint32_t sampling_rate) noexcept {
    sampling_rate_ = sampling_rate;
    x1_ = 0.0;
    x2_ = 0.0;
    y1_ = 0.0;
    y2_ = 0.0;
    UpdateCoefficients();
}

// Coefficients are a pure function of (frequency_, gain_, sampling_rate_).
// Recompute without touching delay state so live gain/frequency changes are
// click-free; SetSamplingRate() separately clears the delay registers.
void HighShelf::UpdateCoefficients() noexcept {
    const double x     = 2.0 * std::numbers::pi_v<double> * frequency_ / sampling_rate_;
    const double sin_x = std::sin(x);
    const double cos_x = std::cos(x);
    const double y     = std::exp(gain_ * std::log(10.0) / 40.0);

    const double z = std::sqrt(y * 2.0) * sin_x;
    const double a = (y - 1.0) * cos_x;
    const double b = y + 1.0 - a;
    const double c = z + b;
    const double d = (y + 1.0) * cos_x;
    const double e = y + 1.0 + a;
    const double f = y - 1.0 - d;

    a0_ = 1.0 / c;
    a1_ = f * 2.0;
    a2_ = b - z;
    b0_ = (e + z) * y;
    b1_ = -y * 2.0 * (y - 1.0 + d);
    b2_ = (e - z) * y;
}
