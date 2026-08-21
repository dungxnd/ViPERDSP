#pragma once

#include <array>
#include <cstdint>

class SoftwareLimiter {
public:
    SoftwareLimiter();

    [[nodiscard]] float Process(float sample) noexcept;
    void Reset() noexcept;

    void SetGate(float gate) noexcept;
    // Must be called before Reset() whenever the host sampling rate changes.
    // Recomputes the one-pole release coefficient so the 80 ms time constant is
    // sample-rate-accurate at any supported rate (8 kHz–384 kHz).
    void SetSamplingRate(uint32_t sampling_rate) noexcept;

private:
    bool ready_{false};

    uint32_t write_index_{0};
    uint32_t sampling_rate_{48000};  // current host rate; updated by SetSamplingRate()

    float gate_{0.999999f};
    float target_gain_{1.0f};
    float gain_envelope_{1.0f};
    float smoothed_gain_{1.0f};
    // One-pole release coefficient: 1 − exp(−1 / (τ · fs)).
    // Initialised for 48 kHz; recomputed by SetSamplingRate().
    float release_coeff_{0.0f};
    std::array<float, 256> arr256_{};
    std::array<float, 512> arr512_{};
};
