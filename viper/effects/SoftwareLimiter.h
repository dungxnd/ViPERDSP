#pragma once

#include <array>
#include <cstdint>

class SoftwareLimiter {
public:
    SoftwareLimiter();

    [[nodiscard]] float Process(float sample) noexcept;
    void ProcessBlock(float* __restrict x, size_t frames) noexcept;
    void Reset() noexcept;

    void SetGate(float gate) noexcept;
    // Must be called before Reset() whenever the host sampling rate changes.
    // Recomputes the one-pole attack and release coefficients so time constants
    // are sample-rate-accurate at any supported rate (8 kHz–384 kHz).
    void SetSamplingRate(uint32_t sampling_rate) noexcept;

private:
    bool ready_{false};

    uint32_t write_index_{0};
    uint32_t sampling_rate_{48000};  // current host rate; updated by SetSamplingRate()

    float gate_{0.999999f};
    float target_gain_{1.0f};
    float gain_envelope_{1.0f};
    float smoothed_gain_{1.0f};
    // One-pole coefficients: 1 − exp(−1 / (τ · fs)).
    // Attack: fast (5 ms) — ramps gain down over the 256-sample lookahead horizon.
    // Release: slow (80 ms) — exponential recovery after the transient.
    // Both initialised for 48 kHz; recomputed by SetSamplingRate().
    float attack_coeff_{0.0f};
    float release_coeff_{0.0f};
    std::array<float, 256> arr256_{};
    std::array<float, 512> arr512_{};
};
