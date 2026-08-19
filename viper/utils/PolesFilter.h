#pragma once

#include <array>
#include <cstdint>

class PolesFilter {
public:
    struct Channel {
        float lower_angle{0.0f};
        float upper_angle{0.0f};
        std::array<float, 3> in{};
        std::array<float, 4> x{};
        std::array<float, 4> y{};
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
    uint32_t sampling_rate_{44100u};

    std::array<Channel, 2> channels_{};
};
