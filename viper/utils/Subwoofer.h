#pragma once

#include "MultiBiquad.h"
#include <array>
#include <cstdint>

class Subwoofer {
public:
    Subwoofer() noexcept;

    void Process(float* samples, uint32_t size) noexcept;

    void SetBassGain(uint32_t sampling_rate, float gain_db) noexcept;

private:
    uint32_t sampling_rate_{44100u};
    float    gain_{0.0f};
    float    gain_lower_{0.0f};

    std::array<MultiBiquad, 2> peak_;
    std::array<MultiBiquad, 2> peak_low_;
    std::array<MultiBiquad, 2> lowpass_;
};
