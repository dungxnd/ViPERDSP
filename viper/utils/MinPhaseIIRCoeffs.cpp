#include "MinPhaseIIRCoeffs.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

namespace {

constexpr std::array<float, 10> kFreq10 = {
    31.0f, 62.0f, 125.0f, 250.0f, 500.0f,
    1000.0f, 2000.0f, 4000.0f, 8000.0f, 16000.0f
};

constexpr std::array<float, 15> kFreq15 = {
    25.0f, 40.0f, 63.0f, 100.0f, 160.0f,
    250.0f, 400.0f, 630.0f, 1000.0f, 1600.0f,
    2500.0f, 4000.0f, 6300.0f, 10000.0f, 16000.0f
};

constexpr std::array<float, 25> kFreq25 = {
    20.0f,   31.5f,   40.0f,   50.0f,    80.0f,
    100.0f,  125.0f,  160.0f,  250.0f,   315.0f,
    400.0f,  500.0f,  800.0f,  1000.0f,  1250.0f,
    1600.0f, 2500.0f, 3150.0f, 4000.0f,  5000.0f,
    8000.0f, 10000.0f,12500.0f,16000.0f, 20000.0f
};

constexpr std::array<float, 31> kFreq31 = {
    20.0f,   25.0f,   31.5f,   40.0f,    50.0f,
    63.0f,   80.0f,   100.0f,  125.0f,   160.0f,
    200.0f,  250.0f,  315.0f,  400.0f,   500.0f,
    630.0f,  800.0f,  1000.0f, 1250.0f,  1600.0f,
    2000.0f, 2500.0f, 3150.0f, 4000.0f,  5000.0f,
    6300.0f, 8000.0f, 10000.0f,12500.0f, 16000.0f,
    20000.0f
};

} // anonymous namespace

float MinPhaseIIRCoeffs::GetIndexFrequency(const uint32_t index) const noexcept {
    switch (bands_) {
        case 10: return index < kFreq10.size() ? kFreq10[index] : 0.0f;
        case 15: return index < kFreq15.size() ? kFreq15[index] : 0.0f;
        case 25: return index < kFreq25.size() ? kFreq25[index] : 0.0f;
        case 31: return index < kFreq31.size() ? kFreq31[index] : 0.0f;
        default: return 0.0f;
    }
}

bool MinPhaseIIRCoeffs::UpdateCoeffs(const uint32_t bands, const uint32_t sampling_rate) noexcept {
    if (bands != 10 && bands != 15 && bands != 25 && bands != 31) return false;
    if (sampling_rate == 0) return false;

    bands_ = bands;
    coeffs_.assign(bands, BiquadBandCoeffs{});

    std::span<const float> band_freqs;
    double bandwidth = 0.0;

    switch (bands) {
        case 10: band_freqs = kFreq10; bandwidth = 1.0;       break;
        case 15: band_freqs = kFreq15; bandwidth = 2.0 / 3.0; break;
        case 25: band_freqs = kFreq25; bandwidth = 1.0 / 3.0; break;
        case 31: band_freqs = kFreq31; bandwidth = 1.0 / 3.0; break;
        default: return false;
    }

    // Clamp usable bandwidth to 49 % of Nyquist so bands approaching or
    // exceeding Nyquist get a safely reduced center rather than degenerate
    // (NaN / Inf) coefficients.
    const double nyquist_safe = static_cast<double>(sampling_rate) * 0.49;
    const double inv_sr       = 1.0 / static_cast<double>(sampling_rate);

    for (uint32_t i = 0; i < bands; ++i) {
        const double center      = std::min(static_cast<double>(band_freqs[i]), nyquist_safe);
        const auto [lower_raw, upper] = Find_F1_F2(center, bandwidth);
        // Keep the lower edge below 95 % of Nyquist so cos/sin arguments stay valid.
        const double lower = std::min(lower_raw, nyquist_safe * 0.95);

        const double x = 2.0 * std::numbers::pi_v<double> * center * inv_sr;
        const double y = 2.0 * std::numbers::pi_v<double> * lower  * inv_sr;

        const double cos_x = std::cos(x);
        const double cos_y = std::cos(y);
        const double sin_y = std::sin(y);

        // Exact closed-form solution — avoids catastrophic cancellation that
        // occurs at low frequencies or high sample rates (≥48 kHz) where the
        // discriminant Δ = sin²(y)·(cos x − cos y)² drops below double
        // precision.  Derivation: factor Δ = (e−d)(e+d) algebraically:
        //   e − d = sin²(y),   e + d = (cos x − cos y)²
        // The stable root |r| < 0.5 is then:
        //   r = 0.5 · (sin y − |cos x − cos y|) / (sin y + |cos x − cos y|)
        // Map to TDF-II form (b1=0, b2=−b0):
        //   b0 = 0.5 − r,  a1 = −(r+0.5)·cos(x)·2,  a2 = 2r
        const double u = std::abs(cos_x - cos_y);
        const double v = sin_y; // = sin(y) ≥ 0 for y ∈ [0, π]
        const double denom = v + u;

        if (denom > 1e-12) {
            const double r = 0.5 * (v - u) / denom;
            coeffs_[i].b0 = static_cast<float>(0.5 - r);
            coeffs_[i].a1 = static_cast<float>(-(r + 0.5) * cos_x * 2.0);
            coeffs_[i].a2 = static_cast<float>(r + r);
        }
    }

    return true;
}

MinPhaseIIRCoeffs::FreqPair MinPhaseIIRCoeffs::Find_F1_F2(
    const double center_freq,
    const double bandwidth_octaves
) noexcept {
    const double x = std::pow(2.0, bandwidth_octaves * 0.5);
    return { center_freq / x, center_freq * x };
}
