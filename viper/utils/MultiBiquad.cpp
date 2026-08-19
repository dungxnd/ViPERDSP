#include "MultiBiquad.h"
#include <cmath>
#include <numbers>

MultiBiquad::MultiBiquad() noexcept = default;

double MultiBiquad::ProcessSample(const double sample) noexcept {
    const double out = sample * b0_ + x1_ * b1_ + x2_ * b2_ + y1_ * a1_ + y2_ * a2_;
    x2_ = x1_;
    x1_ = std::isfinite(sample) ? sample : 0.0;
    y2_ = y1_;
    y1_ = std::isfinite(out)    ? out    : 0.0;
    return std::isfinite(out)   ? out    : 0.0;
}

void MultiBiquad::Reset() noexcept {
    a1_ = 0.0; a2_ = 0.0;
    b0_ = 0.0; b1_ = 0.0; b2_ = 0.0;
    x1_ = 0.0; x2_ = 0.0;
    y1_ = 0.0; y2_ = 0.0;
}

void MultiBiquad::RefreshFilter(const FilterType type, const FilterParams& p) {
    using FT = FilterType;

    const auto gain_db     = p.gain_db;
    const auto frequency   = p.frequency;
    const auto sample_rate = static_cast<double>(p.sample_rate);
    const auto q_factor    = p.q_factor;

    double gain;
    if (type == FT::Peak || type == FT::LowShelf || type == FT::HighShelf) {
        gain = std::pow(10.0, gain_db / 40.0);
    } else {
        gain = std::pow(10.0, gain_db / 20.0);
    }

    const double omega     = 2.0 * std::numbers::pi * frequency / sample_rate;
    const double sin_omega = std::sin(omega);
    const double cos_omega = std::cos(omega);

    double y;
    double z;
    if (type == FT::LowShelf || type == FT::HighShelf) {
        y = sin_omega / 2.0
            * std::sqrt((1.0 / gain + gain) * (1.0 / q_factor - 1.0) + 2.0);
        z = std::sqrt(gain) * 2.0 * y;
    } else if (p.is_bandwidth) {
        y = std::sinh(q_factor * std::log(2.0) * omega / 2.0 / sin_omega) * sin_omega;
        z = -1.0;
    } else {
        y = sin_omega / (q_factor + q_factor);
        z = -1.0;
    }

    struct Coeffs { double a0, a1, a2, b0, b1, b2; };
    Coeffs c{};

    switch (type) {
        case FT::LowPass: {
            c.a0 = 1.0 + y;
            c.a1 = -2.0 * cos_omega;
            c.a2 = 1.0 - y;
            c.b0 = (1.0 - cos_omega) / 2.0;
            c.b1 = 1.0 - cos_omega;
            c.b2 = (1.0 - cos_omega) / 2.0;
            break;
        }
        case FT::HighPass: {
            c.a0 = 1.0 + y;
            c.a1 = -2.0 * cos_omega;
            c.a2 = 1.0 - y;
            c.b0 = (1.0 + cos_omega) / 2.0;
            c.b1 = -(1.0 + cos_omega);
            c.b2 = (1.0 + cos_omega) / 2.0;
            break;
        }
        case FT::BandPass: {
            c.a0 = 1.0 + y;
            c.a1 = -2.0 * cos_omega;
            c.a2 = 1.0 - y;
            c.b0 = y;
            c.b1 = 0.0;
            c.b2 = -y;
            break;
        }
        case FT::BandStop: {
            c.a0 = 1.0 + y;
            c.a1 = -2.0 * cos_omega;
            c.a2 = 1.0 - y;
            c.b0 = 1.0;
            c.b1 = -2.0 * cos_omega;
            c.b2 = 1.0;
            break;
        }
        case FT::AllPass: {
            c.a0 = 1.0 + y;
            c.a1 = -2.0 * cos_omega;
            c.a2 = 1.0 - y;
            c.b0 = 1.0 - y;
            c.b1 = -2.0 * cos_omega;
            c.b2 = 1.0 + y;
            break;
        }
        case FT::Peak: {
            c.a0 = 1.0 + y / gain;
            c.a1 = -2.0 * cos_omega;
            c.a2 = 1.0 - y / gain;
            c.b0 = 1.0 + y * gain;
            c.b1 = -2.0 * cos_omega;
            c.b2 = 1.0 - y * gain;
            break;
        }
        case FT::LowShelf: {
            const double t1 = gain + 1.0 - (gain - 1.0) * cos_omega;
            const double t2 = gain + 1.0 + (gain - 1.0) * cos_omega;
            c.a0 = t2 + z;
            c.a1 = (gain - 1.0 + (gain + 1.0) * cos_omega) * -2.0;
            c.a2 = t2 - z;
            c.b0 = (t1 + z) * gain;
            c.b1 = gain * 2.0 * (gain - 1.0 - (gain + 1.0) * cos_omega);
            c.b2 = (t1 - z) * gain;
            break;
        }
        case FT::HighShelf: {
            const double t1 = gain + 1.0 + (gain - 1.0) * cos_omega;
            const double t2 = gain + 1.0 - (gain - 1.0) * cos_omega;
            c.a0 = t2 + z;
            c.a1 = (gain - 1.0 - (gain + 1.0) * cos_omega) * 2.0;
            c.a2 = t2 - z;
            c.b0 = (t1 + z) * gain;
            c.b1 = gain * -2.0 * (gain - 1.0 + (gain + 1.0) * cos_omega);
            c.b2 = (t1 - z) * gain;
            break;
        }
    }

    // Normalize by a0
    a1_ = -(c.a1 / c.a0);
    a2_ = -(c.a2 / c.a0);
    b0_ = c.b0 / c.a0;
    b1_ = c.b1 / c.a0;
    b2_ = c.b2 / c.a0;
}

void MultiBiquad::RefreshFilter(
    const FilterType type,
    const float gain_db,
    const float frequency,
    const uint32_t sample_rate,
    const float q_factor,
    const bool is_bandwidth
) {
    RefreshFilter(type, FilterParams{
        .gain_db      = static_cast<double>(gain_db),
        .frequency    = static_cast<double>(frequency),
        .sample_rate  = sample_rate,
        .q_factor     = static_cast<double>(q_factor),
        .is_bandwidth = is_bandwidth,
    });
}
