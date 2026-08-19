#pragma once

#include <array>
#include <cstdint>

class SoftwareLimiter {
public:
    SoftwareLimiter();

    [[nodiscard]] float Process(float sample) noexcept;
    void Reset() noexcept;

    void SetGate(float gate) noexcept;

private:
    bool ready_{false};

    uint32_t write_index_{0};

    float gate_{0.999999f};
    float target_gain_{1.0f};
    float gain_envelope_{1.0f};
    float smoothed_gain_{1.0f};
    std::array<float, 256> arr256_{};
    std::array<float, 512> arr512_{};
};
