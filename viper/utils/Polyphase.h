#pragma once

#include "FIR.h"
#include "WaveBuffer.h"
#include <array>

class Polyphase {
public:
    explicit Polyphase(int coeff_type);

    uint32_t Process(float *samples, uint32_t size);
    void Reset();

    [[nodiscard]] uint32_t GetLatency() const;

    void SetSamplingRate(uint32_t sampling_rate);

private:
    uint32_t sampling_rate_{44100u};
    uint32_t latency_{63u};

    std::array<float, 0x7e0> buffer_{};

    FIR fir1_;
    FIR fir2_;
    WaveBuffer wave_buffer1_;
    WaveBuffer wave_buffer2_;
};
