#pragma once

#include <array>
#include <cstdint>
#include <span>

class Polyphase {
public:
    static constexpr uint32_t kNumTaps  = 63u;
    static constexpr uint32_t kLatency  = 31u; // Group delay = (kNumTaps - 1) / 2

    explicit Polyphase(int coeff_type) noexcept;

    // Process interleaved stereo in-place.  size = frame count (not sample count).
    // Operates sample-accurately at any block size; no internal accumulation.
    void Process(float* samples, uint32_t size) noexcept;
    void Process(std::span<float> samples) noexcept {
        Process(samples.data(), static_cast<uint32_t>(samples.size() / 2u));
    }

    void Reset() noexcept;

    [[nodiscard]] static constexpr uint32_t GetLatency() noexcept { return kLatency; }
    void SetSamplingRate(uint32_t sampling_rate) noexcept { sampling_rate_ = sampling_rate; }

private:
    uint32_t sampling_rate_{44100u};
    const std::array<float, kNumTaps>* coeffs_{nullptr};

    // Per-channel circular delay-line.  Power-of-two size (64) enables cheap
    // index masking.  kNumTaps (63) always fits within 64 slots.
    static constexpr uint32_t kHistorySize = 64u;
    static constexpr uint32_t kHistoryMask = kHistorySize - 1u;

    std::array<std::array<float, kHistorySize>, 2> history_{};
    uint32_t history_idx_{0u};
};
