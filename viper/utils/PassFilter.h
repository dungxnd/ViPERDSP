#pragma once

#include "IIR_NOrder_BW_LH.h"
#include <array>
#include <cstdint>

class PassFilter {
public:
    PassFilter();

    void ProcessFrames(float *buffer, uint32_t size) noexcept;
    void ProcessPlanar(float* __restrict L, float* __restrict R, size_t frames) noexcept;
    void Reset() noexcept;

    void SetSamplingRate(uint32_t sampling_rate) noexcept;

private:
    uint32_t sampling_rate_{44100u};

    std::array<IIR_NOrder_BW_LH, 4> filters_{
        IIR_NOrder_BW_LH(3), IIR_NOrder_BW_LH(3),
        IIR_NOrder_BW_LH(1), IIR_NOrder_BW_LH(1)
    };
};
