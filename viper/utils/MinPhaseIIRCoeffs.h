#pragma once

#include <cstdint>
#include <span>
#include <vector>

class MinPhaseIIRCoeffs {
public:
    // TDF-II coefficients for one bandpass band.
    // b0 = 0.5 - r          (b2 = -b0 implicitly, b1 = 0 algebraically)
    // a1 = -coeff3  = -(r + 0.5) * cos(x) * 2.0
    // a2 =  coeff1  = 2r
    struct alignas(16) BiquadBandCoeffs {
        float b0{0.0f};
        float a1{0.0f};
        float a2{0.0f};
    };

    MinPhaseIIRCoeffs() noexcept = default;

    // Rule of Zero — vector handles lifetime; copy is deleted implicitly.
    MinPhaseIIRCoeffs(const MinPhaseIIRCoeffs&)            = delete;
    MinPhaseIIRCoeffs& operator=(const MinPhaseIIRCoeffs&) = delete;
    MinPhaseIIRCoeffs(MinPhaseIIRCoeffs&&)                 = default;
    MinPhaseIIRCoeffs& operator=(MinPhaseIIRCoeffs&&)      = default;

    [[nodiscard]] std::span<const BiquadBandCoeffs> GetCoefficients() const noexcept {
        return coeffs_;
    }

    [[nodiscard]] float GetIndexFrequency(uint32_t index) const noexcept;

    // Returns true on success.  coeffs_ is left empty on failure so the
    // caller can test via GetCoefficients().empty().
    [[nodiscard]] bool UpdateCoeffs(uint32_t bands, uint32_t sampling_rate) noexcept;

private:
    struct FreqPair {
        double lower;
        double upper;
    };

    uint32_t                      bands_{0};
    std::vector<BiquadBandCoeffs> coeffs_;

    // Not constexpr: std::pow is not guaranteed constexpr in C++23.
    [[nodiscard]] static FreqPair Find_F1_F2(double center_freq, double bandwidth_octaves) noexcept;
};
