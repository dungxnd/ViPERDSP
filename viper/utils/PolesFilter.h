#pragma once

#include <array>
#include <cstdint>

class PolesFilter {
public:
    struct Channel {
        float lower_angle{0.0f};
        float upper_angle{0.0f};
        float in[3]{};
        float x[4]{};
        float y[4]{};
    };

    PolesFilter();

    void FilterLeft(float sample, float *out1, float *out2, float *out3) noexcept;
    void FilterRight(float sample, float *out1, float *out2, float *out3) noexcept;
    void Reset() noexcept;

    void SetPassFilter(uint32_t lower_freq, uint32_t upper_freq) noexcept;
    void SetSamplingRate(uint32_t sampling_rate) noexcept;

    void UpdateCoeff() noexcept;

private:
    uint32_t lower_freq_{160};
    uint32_t upper_freq_{8000};
    uint32_t sampling_rate_;

    std::array<Channel, 2> channels_{};
};
