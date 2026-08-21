#pragma once

#include "PFFFTRegistry.h"
#include <array>
#include <cstdint>
#include <span>
#include <vector>

using PFFFT_Setup = struct PFFFT_Setup;

/// ---------------------------------------------------------------------------
/// PConvNUPC — Non-Uniform Partitioned Convolution Engine
///
/// Architecture (Paper 2, Stefani/Farina/Turchet, JAES Oct 2025):
///
///   Stage 0  — Direct-form FIR head, K=64 taps (0 latency).
///   Group 1  — 4 partitions, block 64,   FFT 128,  trigger every 64 samples.
///   Group 2  — 4 partitions, block 128,  FFT 256,  trigger every 128 samples.
///   Group 3  — 4 partitions, block 256,  FFT 512,  trigger every 256 samples.
///   Group 4  — 4 partitions, block 512,  FFT 1024, trigger every 512 samples.
///   Group 5  — 4 partitions, block 1024, FFT 2048, trigger every 1024 samples.
///   Group 6+ — N partitions, block 2048, FFT 4096, trigger every 2048 samples.
///
/// The inherent block-size delay of each group provides the exact causal offset
/// required by linear convolution mathematics — zero algorithmic latency at the
/// output.
///
/// Real-time safety:
///   LoadKernel() performs all allocations and FFT precompute.
///   Process() / ProcessInterleaved() are strictly allocation-free.
/// ---------------------------------------------------------------------------

class PConvNUPC {
public:
    PConvNUPC() noexcept = default;
    ~PConvNUPC();

    PConvNUPC(const PConvNUPC&)            = delete;
    PConvNUPC& operator=(const PConvNUPC&) = delete;
    PConvNUPC(PConvNUPC&&)                 = delete;
    PConvNUPC& operator=(PConvNUPC&&)      = delete;

    /// Load IR from a std::span. Returns true on success.
    bool LoadKernel(std::span<const float> kernel, float gain = 1.0f);

    /// Convenience overload — raw pointer + length.
    bool LoadKernel(const float* kernel, uint32_t size, float gain = 1.0f);

    void Reset()            noexcept;
    void UnloadKernel()     noexcept;
    void ReleaseResources() noexcept;

    [[nodiscard]] bool InstanceUsable() const noexcept { return instance_usable_; }

    void Process(const float* input, float* output, uint32_t n) noexcept;
    void ProcessInterleaved(const float* input, float* output,
                            uint32_t channel, uint32_t stride,
                            uint32_t n) noexcept;

private:
    // Direct-form FIR head ------------------------------------------------
    static constexpr uint32_t kHeadLen  = 64u;
    static constexpr uint32_t kHeadMask = 63u;

    // Double-buffer circular history: head_history_[i] == head_history_[i+kHeadLen]
    // so any K-sample window is always contiguous — no modulo in the inner loop.
    // head_coeffs_rev_[k] = h[K-1-k] to read oldest→newest via plain pointer walk.
    float    head_history_[kHeadLen * 2u]{};
    float    head_coeffs_[kHeadLen]{};
    float    head_coeffs_rev_[kHeadLen]{};
    uint32_t head_idx_{0u};

    // NUPC stage descriptor -----------------------------------------------
    struct Stage {
        uint32_t block_size{0};       // Bs
        uint32_t fft_size{0};         // 2*Bs
        uint32_t num_partitions{0};   // P_s
        uint32_t fdl_index{0};        // circular write head
        uint32_t sample_counter{0};   // counts [0..Bs-1]
        uint32_t output_offset{0};    // read-ahead offset into accum_ring_

        PFFFT_Setup* fft_setup{nullptr};

        float* fft_in{nullptr};       // 2*Bs: prev_Bs | curr_Bs
        float* fft_out{nullptr};      // 2*Bs
        float* fft_work{nullptr};     // 2*Bs
        float* accum_spectrum{nullptr}; // 2*Bs

        std::vector<float*> filter_spectra; // num_partitions × 2*Bs
        std::vector<float*> fdl;            // num_partitions × 2*Bs
        std::vector<float>  prev_input;     // Bs historical samples
    };

    // Master accumulation ring --------------------------------------------
    // Ring size = 2 * max_block_size to hold the longest active overlap.
    static constexpr uint32_t kRingSize = 8192u;   // ≥ 2 × 2048
    static constexpr uint32_t kRingMask = kRingSize - 1u;

    float    accum_ring_[kRingSize]{};
    uint32_t ring_read_idx_{0u};

    std::vector<Stage> stages_;
    bool               instance_usable_{false};

    void ExecuteStageFFT(Stage& stage) noexcept;
    void ProcessOneSample(float in, float& out) noexcept;

    // NUPC group table (static): {block_size, max_partitions_in_group}
    // The tail group has unbounded partitions; the last entry is a sentinel.
    struct GroupDesc {
        uint32_t block_size;
        uint32_t max_partitions;  // UINT32_MAX = "all remaining"
    };
    static constexpr std::array<GroupDesc, 6> kGroups{{
        {   64u, 4u },
        {  128u, 4u },
        {  256u, 4u },
        {  512u, 4u },
        { 1024u, 4u },
        { 2048u, UINT32_MAX },
    }};
};
