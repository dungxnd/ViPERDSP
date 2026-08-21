#pragma once

#include "../utils/PConvSingle.h"
#include <cstdint>

class VHE {
public:
    VHE();

    uint32_t Process(const float *source, float *dest, uint32_t frame_size);
    void Reset();

    [[nodiscard]] bool GetEnable() const noexcept;

    void SetEnable(bool enable) noexcept;
    void SetEffectLevel(uint32_t value) noexcept;
    void SetSamplingRate(uint32_t sampling_rate) noexcept;

private:
    bool enable_{false};

    uint32_t sampling_rate_{44100u};
    uint32_t effect_level_{0u};

    PConvSingle conv_left_;
    PConvSingle conv_right_;
};
