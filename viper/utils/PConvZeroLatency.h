#pragma once

#include "AudioFIFO.h"
#include "PFFFTRegistry.h"
#include <cstdint>
#include <span>
#include <vector>

using PFFFT_Setup = struct PFFFT_Setup;

/// ---------------------------------------------------------------------------
/// PConvZeroLatency — Gardner Hybrid Zero-Latency Convolution Engine
///
/// Architecture (Standard OLS Hybrid, zero-latency):
///   • Stage 0 (Head) — Direct-form FIR on h[0..K-1], K=64 taps. 0 delay.
///   • Stage 1 (Tail) — Uniform Overlap-Save on h[K..L-1]:
///                      Block size B = K = 64, FFT size R = 2B = 128.
///
/// OLS correctness invariant:
///   For R=128, B=64, overlap=B=64:  first B output samples are circularly
///   aliased and discarded; the last B samples are alias-free linear conv.
///   The tail's one-block delay (B samples) is the exact causal offset needed
///   for y_tail[n] = conv(x[n-B], h_tail), so summing head + tail is
///   sample-accurate with zero added latency.
///
/// Usage:
///   LoadKernel(span<const float>, gain) — allocates all buffers.
///   Process(in, out, n)                 — real-time safe, no allocation.
///   ProcessInterleaved(...)             — same semantics, interleaved buffer.
/// ---------------------------------------------------------------------------

class PConvZeroLatency {
public:
    PConvZeroLatency() noexcept = default;
    ~PConvZeroLatency();

    PConvZeroLatency(const PConvZeroLatency&)            = delete;
    PConvZeroLatency& operator=(const PConvZeroLatency&) = delete;
    PConvZeroLatency(PConvZeroLatency&&)                 = delete;
    PConvZeroLatency& operator=(PConvZeroLatency&&)      = delete;

    /// Load IR from a std::span. Returns true on success.
    bool LoadKernel(std::span<const float> kernel, float gain = 1.0f);

    /// Convenience overload — raw pointer + length.
    bool LoadKernel(const float* kernel, uint32_t size, float gain = 1.0f);

    void Reset()            noexcept;
    void UnloadKernel()     noexcept;
    void ReleaseResources() noexcept;

    [[nodiscard]] bool InstanceUsable() const noexcept { return instance_usable_; }

    /// Process `n` non-interleaved samples. in/out may alias.
    void Process(const float* input, float* output, uint32_t n) noexcept;

    /// Process one channel of a stride-interleaved buffer.
    void ProcessInterleaved(const float* input, float* output,
                            uint32_t channel, uint32_t stride,
                            uint32_t n) noexcept;

private:
    // --- Head / Tail sizing ---
    // B = K = 64: block size equals head length so OLS discard == head length.
    // R = 2B = 128: smallest valid PFFFT size; first B samples discarded, last B clean.
    static constexpr uint32_t kHeadLen   = 64u;   // K  — head FIR tap count
    static constexpr uint32_t kHeadMask  = 63u;   // power-of-two wrap mask
    static constexpr uint32_t kTailBlock = 64u;   // B  — OLS new-input block size
    static constexpr uint32_t kTailFFTSz = 128u;  // R  = 2*B

    // --- Runtime state ---
    bool     instance_usable_{false};
    uint32_t tail_partition_count_{0};

    // Head: double-buffer circular history for contiguous SIMD access.
    //   head_history_[i] and head_history_[i + kHeadLen] always hold the same
    //   sample, so any K-sample window is contiguous without modulo indexing.
    //   head_coeffs_rev_[k] = h[K-1-k] (reversed) lets the FIR inner loop read
    //   oldest-to-newest through a plain pointer increment.
    float    head_history_[kHeadLen * 2u]{};
    float    head_coeffs_[kHeadLen]{};
    float    head_coeffs_rev_[kHeadLen]{};
    uint32_t head_idx_{0u};

    // Tail UPOLS state
    PFFFT_Setup* tail_fft_setup_{nullptr};

    float* fft_in_{nullptr};    // R floats — overlap-save input window
    float* fft_out_{nullptr};   // R floats — IFFT output
    float* fft_work_{nullptr};  // R floats — PFFFT scratch
    float* accum_{nullptr};     // R floats — frequency-domain accumulator

    std::vector<float*> filter_segs_; // J × R — IR partitions in freq domain
    std::vector<float*> fdl_;         // J × R — frequency delay line
    uint32_t fdl_idx_{0u};

    // B-sample historical input for the 2B overlap-save window
    std::vector<float> prev_input_;  // B floats

    // Decoupling FIFOs
    AudioFIFO input_fifo_;
    AudioFIFO output_fifo_;

    void ProcessTailBlock() noexcept;
};
