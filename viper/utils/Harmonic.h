#pragma once

#include <array>
#include <cstdint>
#include <span>

class Harmonic {
public:
    Harmonic();

    double Process(double sample) noexcept;
    void   Reset() noexcept;

    void SetHarmonics(std::span<const float, 10> coeffs);
    void UpdateCoeffs(std::span<const float, 10> coeffs);

private:
    uint32_t biggest_coeff_  = 0;
    uint32_t sample_counter_ = 0;

    std::array<float, 11> coeffs_{};

    double last_processed_ = 0.0;
    double prev_out_       = 0.0;
};
