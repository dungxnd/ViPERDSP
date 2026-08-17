#pragma once

#include "IIR_NOrder_BW_LH.h"
#include <array>
#include <cstdint>

class PassFilter {
public:
    PassFilter();

    void ProcessFrames(float *buffer, uint32_t size) noexcept;
    void Reset() noexcept;

    void SetSamplingRate(uint32_t sampling_rate) noexcept;

private:
    uint32_t sampling_rate_;

    std::array<IIR_NOrder_BW_LH, 4> filters_;
};
