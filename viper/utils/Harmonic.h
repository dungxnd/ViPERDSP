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
    std::array<float, 11> coeffs_{};

    double last_processed_ = 0.0;
    double prev_out_       = 0.0;
};
