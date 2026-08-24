#pragma once

#include <array>
#include <cstdint>
#include <span>

class Polyphase {
public:
    static constexpr uint32_t kNumTaps = 63u;
    static constexpr uint32_t kLatency = 31u; // Group delay = (kNumTaps - 1) / 2

    // coeff_type param is accepted for API compatibility but ignored —
    // coefficients are now designed at runtime via DesignLinearPhaseFilter().
    explicit Polyphase(int coeff_type = 2) noexcept;

    // Process interleaved stereo in-place.  size = frame count (not sample count).
    // Operates sample-accurately at any block size; no internal accumulation.
    void Process(float* samples, uint32_t size) noexcept;
    void Process(std::span<float> samples) noexcept {
        Process(samples.data(), static_cast<uint32_t>(samples.size() / 2u));
    }

    // Process separate L/R planar arrays in-place (zero interleave bounce).
    // in_l/in_r are the input arrays; out_l/out_r receive the FIR output.
    // in_l/out_l and in_r/out_r may alias (in-place is safe).
    void ProcessPlanar(const float* __restrict in_l, const float* __restrict in_r,
                       float* __restrict out_l,       float* __restrict out_r,
                       size_t frames) noexcept;

    void Reset() noexcept;

    [[nodiscard]] static constexpr uint32_t GetLatency() noexcept { return kLatency; }

    // Redesigns coefficients and resets history if the rate actually changed.
    void SetSamplingRate(uint32_t sampling_rate) noexcept;

    // Redesigns coefficients for a new cutoff (does not reset history).
    void SetCutoffFrequency(float cutoff_hz) noexcept;

private:
    uint32_t sampling_rate_{44100u};
    float    cutoff_hz_{100.0f};

    // Coefficients aligned for 256-bit SIMD loads.
    alignas(32) std::array<float, kNumTaps> coeffs_{};

    // Capacity must be power of two (64) so that (idx - 1) & kHistIdxMask wraps
    // correctly across 0..kHistCap-1 without ever skipping an index.
    // (62 = 0x3E is NOT a valid mask: it zeroes bit-0, skipping all odd indices.)
    //
    // kHistIdxMask limits history_idx_ to [0, 63].  Any read slice
    //   history_[ch][history_idx_ .. history_idx_ + kNumTaps - 1]
    // is contiguous (max index = 63 + 62 = 125 < kHistBufSize=128),
    // enabling auto-vectorisation without a masked index inside the inner loop.
    static constexpr uint32_t kHistCap     = 64u;             // power-of-two head range
    static constexpr uint32_t kHistIdxMask = kHistCap - 1u;  // 63 = 0b00111111
    static constexpr uint32_t kHistBufSize = 128u;            // must be >= kHistCap + kNumTaps - 1

    // Double-buffered (128 floats per channel) for contiguous SIMD pointer reads.
    alignas(32) std::array<std::array<float, kHistBufSize>, 2> history_{};
    uint32_t history_idx_{0u};

    void DesignLinearPhaseFilter() noexcept;
};
