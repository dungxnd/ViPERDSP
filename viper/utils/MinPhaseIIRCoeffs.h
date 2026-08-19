#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

class MinPhaseIIRCoeffs {
public:
    MinPhaseIIRCoeffs() noexcept = default;

    // Rule of Zero — unique_ptr handles lifetime; copy is deleted implicitly.
    MinPhaseIIRCoeffs(const MinPhaseIIRCoeffs&)            = delete;
    MinPhaseIIRCoeffs& operator=(const MinPhaseIIRCoeffs&) = delete;
    MinPhaseIIRCoeffs(MinPhaseIIRCoeffs&&)                 = default;
    MinPhaseIIRCoeffs& operator=(MinPhaseIIRCoeffs&&)      = default;

    [[nodiscard]] const double* GetCoefficients() const noexcept;
    [[nodiscard]] float         GetIndexFrequency(uint32_t index) const noexcept;

    [[nodiscard]] int UpdateCoeffs(uint32_t bands, uint32_t sampling_rate);

private:
    struct FreqPair {
        double lower;
        double upper;
    };

    uint32_t             bands_{0};
    std::vector<double>  coeffs_;

    [[nodiscard]] static FreqPair        Find_F1_F2(double center_freq, double bandwidth_octaves) noexcept;
    [[nodiscard]] static std::optional<double> SolveRoot(double coeff_a, double coeff_b, double coeff_c) noexcept;
};
