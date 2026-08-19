#include "Harmonic.h"
#include <algorithm>
#include <cmath>
#include <ranges>

static constexpr double kPrevOutDecay = 0.999;

static constexpr std::array<float, 10> kHarmonicDefault{
    1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
};

Harmonic::Harmonic() {
    UpdateCoeffs(kHarmonicDefault);
    Reset();
}

double Harmonic::Process(const double sample) noexcept {
    const double prev_last = last_processed_;

    const double x = sample;
    const auto* const c = coeffs_.data();
    auto y       = static_cast<double>(c[10]);

    y = std::fma(x, y, static_cast<double>(c[9]));
    y = std::fma(x, y, static_cast<double>(c[8]));
    y = std::fma(x, y, static_cast<double>(c[7]));
    y = std::fma(x, y, static_cast<double>(c[6]));
    y = std::fma(x, y, static_cast<double>(c[5]));
    y = std::fma(x, y, static_cast<double>(c[4]));
    y = std::fma(x, y, static_cast<double>(c[3]));
    y = std::fma(x, y, static_cast<double>(c[2]));
    y = std::fma(x, y, static_cast<double>(c[1]));
    y = std::fma(x, y, static_cast<double>(c[0]));

    last_processed_ = y;
    prev_out_ = last_processed_ + prev_out_ * kPrevOutDecay - prev_last;

    if (sample_counter_ < biggest_coeff_) {
        ++sample_counter_;
        return 0.0;
    }

    return prev_out_;
}

void Harmonic::Reset() noexcept {
    last_processed_ = 0.0;
    sample_counter_ = 0;
    prev_out_       = 0.0;
}

void Harmonic::SetHarmonics(const std::span<const float, 10> coeffs) {
    UpdateCoeffs(coeffs);
    Reset();
}

void Harmonic::UpdateCoeffs(const std::span<const float, 10> coeffs) {
    std::array<float, 11> arr1{};
    std::array<float, 11> arr2{};

    float biggest_coeff = 0.0f;
    float abs_coeff_sum = 0.0f;
    for (uint32_t i = 0; i < 10; i++) {
        const float abs_coeff = std::abs(coeffs[i]);
        abs_coeff_sum += abs_coeff;
        if (abs_coeff > biggest_coeff) {
            biggest_coeff = abs_coeff;
        }
    }
    biggest_coeff_ = static_cast<uint32_t>(biggest_coeff * 10000.0f);

    std::ranges::copy(coeffs, arr1.begin() + 1);

    const float norm_factor = (abs_coeff_sum > 1.0f) ? (1.0f / abs_coeff_sum) : 1.0f;
    for (uint32_t i = 1; i < 11; i++) {
        arr1[i] *= norm_factor;
    }

    coeffs_.fill(0.0f);

    coeffs_[10] = arr1[10];

    for (uint32_t i = 2; i < 11; i++) {
        for (uint32_t j = 0; j < i; j++) {
            const float tmp = arr2[i - j];
            arr2[i - j]     = coeffs_[i - j];
            coeffs_[i - j]  = coeffs_[i - j - 1] * 2.0f - tmp;
        }
        const float tmp = arr1[10 - i + 1] - arr2[0];
        arr2[0]         = coeffs_[0];
        coeffs_[0]      = tmp;
    }

    for (uint32_t i = 1; i < 11; i++) {
        coeffs_[10 - i + 1] = coeffs_[10 - i] - arr2[10 - i + 1];
    }

    coeffs_[0] = arr1[0] / 2.0f - arr2[0];
}
