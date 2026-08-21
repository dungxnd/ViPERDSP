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

const double* MinPhaseIIRCoeffs::GetCoefficients() const noexcept {
    return coeffs_.data();
}

float MinPhaseIIRCoeffs::GetIndexFrequency(const uint32_t index) const noexcept {
    switch (bands_) {
        case 10: return kFreq10[index];
        case 15: return kFreq15[index];
        case 25: return kFreq25[index];
        case 31: return kFreq31[index];
        default: return 0.0f;
    }
}

int MinPhaseIIRCoeffs::UpdateCoeffs(const uint32_t bands, const uint32_t sampling_rate) {
    if (bands != 10 && bands != 15 && bands != 25 && bands != 31) {
        return 0;
    }

    bands_ = bands;
    coeffs_.assign(bands * 4, 0.0);

    const float* band_freqs = nullptr;
    double bandwidth = 0.0;

    switch (bands) {
        case 10:
            band_freqs = kFreq10.data();
            bandwidth  = 1.0;
            break;
        case 15:
            band_freqs = kFreq15.data();
            bandwidth  = 2.0 / 3.0;
            break;
        case 25:
            band_freqs = kFreq25.data();
            bandwidth  = 1.0 / 3.0;
            break;
        case 31:
            band_freqs = kFreq31.data();
            bandwidth  = 1.0 / 3.0;
            break;
        default:;
    }

    // Clamp usable bandwidth to 49 % of Nyquist so bands that approach or exceed
    // the Nyquist frequency get a safe reduced center rather than producing
    // degenerate (NaN / Inf) coefficients.
    const double nyquist_safe = static_cast<double>(sampling_rate) * 0.49;
    const double inv_sr       = 1.0 / static_cast<double>(sampling_rate);

    for (uint32_t i = 0; i < bands; ++i) {
        const double center = std::min(static_cast<double>(band_freqs[i]), nyquist_safe);
        const auto [lower_raw, upper] = Find_F1_F2(center, bandwidth);
        // Keep the lower edge below 95 % of Nyquist so cos/sin arguments stay valid.
        const double lower = std::min(lower_raw, nyquist_safe * 0.95);

        const double x = 2.0 * std::numbers::pi_v<double> * center * inv_sr;
        const double y = 2.0 * std::numbers::pi_v<double> * lower  * inv_sr;

        const double cos_x = std::cos(x);
        const double cos_y = std::cos(y);
        const double sin_y = std::sin(y);

        const double b_half = cos_x * cos_x * 0.5;
        const double sin_y2 = sin_y * sin_y;
        const double cos_y2 = cos_y * cos_y;

        const double d = b_half - cos_x * cos_y + 0.5 - sin_y2;
        const double e = sin_y2 + (b_half + cos_y2 - cos_x * cos_y - 0.5);
        const double f = b_half * 0.25 - cos_x * cos_y * 0.25 + 0.125 - sin_y2 * 0.25;

        if (const auto root = SolveRoot(d, e, f); root.has_value()) {
            const double r = *root;
            coeffs_[i * 4]     = r + r;
            coeffs_[i * 4 + 1] = 0.5 - r;
            coeffs_[i * 4 + 2] = (r + 0.5) * cos_x * 2.0;
        }
    }

    return 1;
}

MinPhaseIIRCoeffs::FreqPair MinPhaseIIRCoeffs::Find_F1_F2(
    const double center_freq,
    const double bandwidth_octaves
) noexcept {
    const double x = std::pow(2.0, bandwidth_octaves * 0.5);
    return { center_freq / x, center_freq * x };
}

std::optional<double> MinPhaseIIRCoeffs::SolveRoot(
    const double coeff_a,
    const double coeff_b,
    const double coeff_c
) noexcept {
    // Guard against d ≈ 0: would produce ±Inf / NaN coefficients that instantly
    // destabilise the IIR state variables.
    if (std::abs(coeff_a) < 1e-12) return std::nullopt;

    const double x = (coeff_c - coeff_b * coeff_b / (coeff_a * 4.0)) / coeff_a;
    if (x >= 0.0) return std::nullopt;

    const double z = std::sqrt(-x);
    const double y = coeff_b / (coeff_a + coeff_a);
    const double a = -y - z;
    const double b = z  - y;

    // The product of the two roots equals 0.25, so exactly one root has |r| < 0.5
    // (stable pole) and the other has |r| > 0.5 (unstable pole).  Always pick the
    // stable one; previously the code unconditionally returned 'a' which is the
    // unstable root whenever d > 0 (high-frequency bands).
    if (std::abs(a) < 0.5 && std::abs(b) >= 0.5) return a;
    if (std::abs(b) < 0.5 && std::abs(a) >= 0.5) return b;
    // Both roots inside unit circle (rare): prefer smaller magnitude.
    return (std::abs(a) < std::abs(b)) ? a : b;
}
