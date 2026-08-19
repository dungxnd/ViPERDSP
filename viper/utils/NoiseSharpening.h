#pragma once

#include "IIR_1st.h"
#include <array>
#include <cstdint>

class NoiseSharpening {
public:
    NoiseSharpening() noexcept;

    void Process(float* buffer, uint32_t size) noexcept;
    void Reset() noexcept;
    void SetGain(float gain) noexcept;
    void SetSamplingRate(uint32_t sampling_rate) noexcept;

private:
    uint32_t sampling_rate_{44100u};
    float    gain_{0.0f};

    std::array<float,   2> in_{};
    std::array<IIR_1st, 2> filters_;
};
