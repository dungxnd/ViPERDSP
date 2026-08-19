#include "IIRFilter.h"
#include <algorithm>
#include <cmath>

IIRFilter::IIRFilter(const uint32_t bands) {
    if (bands == 10 || bands == 15 || bands == 25 || bands == 31) {
        bands_ = bands;
        [[maybe_unused]] const auto ok = min_phase_iir_coeffs_.UpdateCoeffs(bands_, sampling_rate_);
    }
    band_levels_with_q_.fill(kDefaultLevelQ);
    // buf_ and buf_index* already initialized by in-class member defaults.
}

void IIRFilter::Process(float* const samples, const uint32_t size) noexcept {
    if (!enable_) return;

    const double* const coeffs = min_phase_iir_coeffs_.GetCoefficients();
    if (coeffs == nullptr || size == 0) return;

    for (uint32_t i = 0; i < size; ++i) {
        for (uint32_t j = 0; j < 2; ++j) {
            const double sample = samples[i * 2 + j];
            double accumulated  = 0.0;

            for (uint32_t k = 0; k < bands_; ++k) {
                const uint32_t buf_idx = buf_index0_ + j * 8 + k * 16;
                buf_[buf_idx] = sample;

                const double coeff1 = coeffs[k * 4];
                const double coeff2 = coeffs[k * 4 + 1];
                const double coeff3 = coeffs[k * 4 + 2];

                const double a = coeff3 * buf_[buf_idx + (buf_index1_ + 3 - buf_index0_)];
                const double b = coeff2 * (sample - buf_[buf_idx + (buf_index2_ - buf_index0_)]);
                const double c = coeff1 * buf_[buf_idx + (buf_index2_ - buf_index0_ + 3)];

                const double tmp = a + b - c;
                buf_[buf_idx + 3] = tmp;
                accumulated += tmp * band_levels_with_q_[k];
            }

            samples[i * 2 + j] = static_cast<float>(accumulated);
        }

        buf_index0_ = (buf_index0_ + 1) % 3;
        buf_index1_ = (buf_index1_ + 1) % 3;
        buf_index2_ = (buf_index2_ + 1) % 3;
    }
}

void IIRFilter::Reset() noexcept {
    buf_.fill(0.0);
    buf_index0_ = 2;
    buf_index1_ = 1;
    buf_index2_ = 0;
}

void IIRFilter::SetEnable(const bool enable) noexcept {
    if (enable_ != enable) {
        if (enable) Reset();
        enable_ = enable;
    }
}

void IIRFilter::SetBandCount(const uint32_t bands) {
    if (bands != 10 && bands != 15 && bands != 25 && bands != 31) return;
    if (bands_ == bands) return;
    bands_ = bands;
    [[maybe_unused]] const auto ok = min_phase_iir_coeffs_.UpdateCoeffs(bands_, sampling_rate_);
    band_levels_with_q_.fill(kDefaultLevelQ);
    Reset();
}

void IIRFilter::SetBandLevel(const uint32_t band, const float level) noexcept {
    if (band > 30) return;
    const double band_level = std::pow(10.0, static_cast<double>(level) / 20.0);
    band_levels_with_q_[band] = static_cast<float>(band_level * 0.636);
}

void IIRFilter::SetBandLevels(const float* const levels, const uint32_t count) noexcept {
    const uint32_t n = std::min(count, 31u);
    for (uint32_t i = 0; i < n; ++i) {
        SetBandLevel(i, levels[i]);
    }
}

void IIRFilter::SetSamplingRate(const uint32_t sampling_rate) {
    if (sampling_rate_ != sampling_rate) {
        sampling_rate_ = sampling_rate;
        if (bands_ != 0) {
            [[maybe_unused]] const auto ok = min_phase_iir_coeffs_.UpdateCoeffs(bands_, sampling_rate);
        }
        Reset();
    }
}
