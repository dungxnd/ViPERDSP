#pragma once

#include "../utils/MultiBiquad.h"
#include "../utils/WaveBuffer.h"
#include <array>
#include <cstdint>

class DiffSurround {
public:
    DiffSurround();

    void Process(float *samples, uint32_t size);
    void Reset();

    void SetEnable(bool enable);
    void SetDelayTime(float value);
    void SetReverse(bool value);
    void SetWetDryMix(float value);
    void SetLPCutoff(float value);
    void SetSamplingRate(uint32_t sampling_rate);

private:
    bool enable_{false};
    bool reverse_{false};

    uint32_t sampling_rate_{44100u};

    float delay_time_{0.0f};
    float wet_dry_mix_{1.0f};
    float lp_cutoff_{0.0f};

    std::array<WaveBuffer, 2> buffers_;
    MultiBiquad lp_filter_{};
};
