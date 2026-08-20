#pragma once

#include "MultiBiquad.h"
#include <array>
#include <cstdint>
#include <span>

class Subwoofer {
public:
    Subwoofer() noexcept;

    void Process(float* samples, uint32_t size) noexcept;
    // span overload: size = frames (not interleaved samples).
    void Process(std::span<float> samples) noexcept {
        Process(samples.data(), static_cast<uint32_t>(samples.size() / 2u));
    }

    // Clears all biquad delay-state registers to prevent inter-session DC
    // offsets and filter ringing from leaking into a new audio stream.
    void Reset() noexcept {
        for (auto& p : peak_)     p.Reset();
        for (auto& p : peak_low_) p.Reset();
        for (auto& lp : lowpass_) lp.Reset();
    }

    void SetBassGain(uint32_t sampling_rate, float linear_gain) noexcept;

private:
    uint32_t sampling_rate_{44100u};
    float    gain_{0.0f};
    float    gain_lower_{0.0f};
    // True when the effective linear gain is zero: Process() becomes a no-op,
    // preserving unity-gain pass-through instead of applying a -6 dB shelf.
    bool     bypassed_{false};

    std::array<MultiBiquad, 2> peak_;
    std::array<MultiBiquad, 2> peak_low_;
    std::array<MultiBiquad, 2> lowpass_;
};
