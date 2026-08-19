#pragma once

#include "../utils/Harmonic.h"
#include "../utils/MultiBiquad.h"
#include <array>
#include <cstdint>

class PsychoacousticBass {
public:
    PsychoacousticBass();

    void Process(float* samples, uint32_t size) noexcept;
    void Reset() noexcept;

    void SetEnable(bool enable) noexcept;
    void SetCutoff(uint32_t value) noexcept;
    void SetIntensity(uint32_t value) noexcept;
    void SetHarmonicOrder(uint32_t value) noexcept;
    void SetOriginalBassLevel(uint32_t value) noexcept;
    void SetSamplingRate(uint32_t sampling_rate) noexcept;

private:
    bool     enable_{false};
    uint32_t sampling_rate_{44100};
    uint32_t cutoff_{80};
    uint32_t harmonic_order_{3};

    float  intensity_{0.5f};
    float  original_bass_level_{1.0f};
    double envelope_{1e-10};

    std::array<MultiBiquad, 2> lowpass_;
    std::array<MultiBiquad, 2> highpass_;
    std::array<Harmonic,    2> harmonics_;

    void RefreshFilters() noexcept;
    void ApplyHarmonicCoeffs() noexcept;
};
