#pragma once

#include "../utils/MultiBiquad.h"
#include "../utils/WaveBuffer.h"
#include <array>
#include <cstdint>

class DiffSurround {
public:
    DiffSurround();

    void ProcessPlanar(float* __restrict L, float* __restrict R, size_t frames) noexcept;
    void Reset();

    [[nodiscard]] bool IsEnabled() const noexcept { return enable_; }
    void SetEnable(bool enable);
    void SetDelayTime(float value);
    void SetReverse(bool value);
    void SetWetDryMix(float value);
    void SetLPCutoff(float value);
    void SetSamplingRate(uint32_t sampling_rate);

private:
    void Process(float *samples, uint32_t size);

    bool enable_{false};
    bool reverse_{false};

    uint32_t sampling_rate_{44100u};

    float delay_time_{0.0f};
    float wet_dry_mix_{1.0f};
    float lp_cutoff_{0.0f};

    std::array<WaveBuffer, 2> buffers_{WaveBuffer(1, 0x1000), WaveBuffer(1, 0x1000)};
    MultiBiquad lp_filter_{};
    alignas(64) std::array<float, 4096u * 2u> scratch_{};
};
