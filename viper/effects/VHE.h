#pragma once

#include <span>


#include "../utils/PConvZeroLatency.h"
#include <cstdint>

// Virtual Headphone Engine — applies per-level HRIR pairs via zero-latency
// hybrid convolution (Gardner OLS, K=128 head + uniform OLS tail).
// Supports 5 effect levels × {44100, 48000} Hz; kernels are baked as int16
// quantised arrays and dequantised into PConvZeroLatency on every Reset().
class VHE {
public:
    VHE();

    uint32_t Process(const float *source, float *dest, uint32_t frame_size);

    // Planar processing: L and R are separate contiguous arrays.
    // Eliminates the ProcessInterleaved stride overhead.
    void ProcessPlanar(std::span<float> L, std::span<float> R) noexcept;
    [[nodiscard]] bool IsEnabled() const noexcept { return GetEnable(); }

    // Reloads HRIR kernels for the current (effect_level_, sampling_rate_) pair;
    // called automatically by SetEnable/SetEffectLevel/SetSamplingRate on change.
    void Reset();

    [[nodiscard]] bool GetEnable() const noexcept;

    void SetEnable(bool enable) noexcept;
    void SetEffectLevel(uint32_t value) noexcept;
    void SetSamplingRate(uint32_t sampling_rate) noexcept;

private:
    bool     enable_        {false};
    uint32_t sampling_rate_ {44100u};
    uint32_t effect_level_  {0u};

    PConvZeroLatency conv_left_;
    PConvZeroLatency conv_right_;
};
