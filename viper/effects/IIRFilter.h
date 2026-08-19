#pragma once

#include "../utils/MinPhaseIIRCoeffs.h"
#include <array>
#include <cstdint>

class IIRFilter {
public:
    explicit IIRFilter(uint32_t bands);

    void Process(float* samples, uint32_t size) noexcept;
    void Reset() noexcept;

    void SetEnable(bool enable) noexcept;
    void SetBandCount(uint32_t bands);
    void SetBandLevel(uint32_t band, float level) noexcept;
    void SetBandLevels(const float* levels, uint32_t count) noexcept;
    void SetSamplingRate(uint32_t sampling_rate);

private:
    static constexpr float kDefaultLevelQ = 0.636f;

    bool     enable_{false};
    uint32_t bands_{0};
    uint32_t sampling_rate_{44100};
    uint32_t buf_index0_{2};
    uint32_t buf_index1_{1};
    uint32_t buf_index2_{0};

    std::array<double, 496>  buf_{};
    std::array<float,   31>  band_levels_with_q_{};
    MinPhaseIIRCoeffs        min_phase_iir_coeffs_;
};
