#pragma once

#include "FIR.h"
#include "WaveBuffer.h"
#include <array>
#include <span>

class Polyphase {
public:
    explicit Polyphase(int coeff_type);

    uint32_t Process(float *samples, uint32_t size);
    // span overload: size = frames (not interleaved samples).
    uint32_t Process(std::span<float> samples) {
        return Process(samples.data(),
                       static_cast<uint32_t>(samples.size() / 2u));
    }
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
