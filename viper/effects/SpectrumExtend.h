#pragma once

#include "../utils/Harmonic.h"
#include "../utils/MultiBiquad.h"
#include <array>
#include <cstdint>

class SpectrumExtend {
public:
    SpectrumExtend();

    void ProcessPlanar(float* __restrict L, float* __restrict R, size_t frames) noexcept;
    void Reset() noexcept;

    [[nodiscard]] bool IsEnabled() const noexcept { return enable_; }
    void SetEnable(bool enable) noexcept;
    void SetExciter(float value) noexcept;
    void SetReferenceFrequency(uint32_t value) noexcept;
    void SetSamplingRate(uint32_t sampling_rate) noexcept;

private:
    void Process(float *samples, uint32_t size) noexcept;

    bool enable_{false};

    uint32_t sampling_rate_{44100u};
    uint32_t reference_freq_{7600u};

    float exciter_{0.0f};

    std::array<MultiBiquad, 2> highpass_{};
    std::array<MultiBiquad, 2> lowpass_{};
    std::array<Harmonic, 2>    harmonics_{};
};
