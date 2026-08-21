#pragma once

#include "PFFFTRegistry.h"
#include <array>
#include <cstdint>
#include <span>
#include <vector>

// AArch64 FP16 support: halves tail filter RAM on arm64 devices.
// vcvt_f16_f32 / vcvt_f32_f16 are zero-latency on Cortex-A5x/A7x.
#if defined(__aarch64__) && defined(__ARM_FP16_FORMAT_IEEE)
#  include <arm_neon.h>
#  define VIPER_USE_FP16_TAIL 1
#else
#  define VIPER_USE_FP16_TAIL 0
#endif

using PFFFT_Setup = struct PFFFT_Setup;

/// ---------------------------------------------------------------------------
/// PConvNUPC — Non-Uniform Partitioned Convolution Engine
///
/// Architecture (Paper 2, Stefani/Farina/Turchet, JAES Oct 2025):
///
///   Stage 0  — Direct-form FIR head, K=128 taps (0 latency).
///   Group 0  — 4 partitions, block 128,  FFT 256,  trigger every 128 samples.
///   Group 1  — 4 partitions, block 256,  FFT 512,  trigger every 256 samples.
///   Group 2  — 4 partitions, block 512,  FFT 1024, trigger every 512 samples.
///   Group 3  — 4 partitions, block 1024, FFT 2048, trigger every 1024 samples.
///   Group 4+ — N partitions, block 2048, FFT 4096, trigger every 2048 samples.
///             (time-sliced over 4 × 512-sample sub-intervals for CPU smoothing)
///
/// Starting groups at B=128 aligns with the widened head (K=128) and removes
/// the B=64 group, reducing overall partition count for medium-length IRs.
///
/// The inherent block-size delay of each group provides the exact causal offset
/// required by linear convolution mathematics — zero algorithmic latency.
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
    // -------------------------------------------------------------------------
    // Direct-form FIR head
    // -------------------------------------------------------------------------
    // K=128: doubles the direct SIMD slice vs K=64, keeping ~32 NEON vmlaq_f32
    // instructions/sample when vectorized by clang.  Aligned with Group 0's
    // block size so each group's first partition starts at a power-of-2 offset.
    static constexpr uint32_t kHeadLen  = 128u;
    static constexpr uint32_t kHeadMask = 127u;

    // Double-buffer circular history: head_history_[i] == head_history_[i+kHeadLen]
    // so any K-sample window is always contiguous — no modulo in the inner loop.
    // head_coeffs_rev_[k] = h[K-1-k] to read oldest→newest via plain pointer walk.
    float    head_history_[kHeadLen * 2u]{};
    float    head_coeffs_[kHeadLen]{};
    float    head_coeffs_rev_[kHeadLen]{};
    uint32_t head_idx_{0u};

    // -------------------------------------------------------------------------
    // NUPC stage descriptor
    // -------------------------------------------------------------------------
    struct Stage {
        uint32_t block_size{0};         // Bs
        uint32_t fft_size{0};           // 2*Bs
        uint32_t num_partitions{0};     // P_s
        uint32_t fdl_index{0};          // circular write head into fdl[]
        uint32_t sample_counter{0};     // counts [0 .. Bs-1]
        uint32_t output_offset{0};      // read-ahead offset into accum_ring_

        PFFFT_Setup* fft_setup{nullptr};

        float* fft_in{nullptr};         // 2*Bs: [prev_Bs | curr_Bs]
        float* fft_out{nullptr};        // 2*Bs
        float* fft_work{nullptr};       // 2*Bs
        float* accum_spectrum{nullptr}; // 2*Bs

        std::vector<float*> filter_spectra; // num_partitions × 2*Bs (FP32)
        std::vector<float*> fdl;            // num_partitions × 2*Bs
        std::vector<float>  prev_input;     // Bs historical samples

        // ----- Time-sliced tail fields (only used when is_time_sliced=true) -----
        // The B=2048 tail stage spreads its FFT+MAC+IFFT work across 4 sub-intervals
        // of 512 samples each to eliminate periodic CPU micro-spikes in the callback.
        bool     is_time_sliced{false};
        uint32_t slice_phase{0};          // 0=idle, 1=fwd_fft, 2=mac_a, 3=mac_b, 4=ifft
        uint32_t phase_sample_count{0};   // samples elapsed since last phase boundary
        uint32_t saved_ring_read_idx{0};  // ring position when ForwardFFT was captured

#if VIPER_USE_FP16_TAIL
        // FP16 filter spectra for AArch64: halves RAM, L2 bandwidth for Stage 4.
        // filter_spectra[p] is freed and set to nullptr when FP16 is active.
        std::vector<__fp16*> filter_spectra_fp16;
#endif
    };

    // -------------------------------------------------------------------------
    // Master accumulation ring
    // -------------------------------------------------------------------------
    // kRingSize = 2 × 2048 + headroom.  Maximum write distance:
    //   Group 4 (B=2048, output_offset = ir_offset - 2048).
    //   For a 5-group schedule, max ir_offset before tail = 128+512+1024+2048+4096 = 7808.
    //   output_offset_max = 7808 - 2048 = 5760.  Write end: 5760 + 2048 = 7808 < 8192. ✓
    static constexpr uint32_t kRingSize = 8192u;
    static constexpr uint32_t kRingMask = kRingSize - 1u;

    float    accum_ring_[kRingSize]{};
    uint32_t ring_read_idx_{0u};

    std::vector<Stage> stages_;
    bool               instance_usable_{false};

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------
    void ExecuteStageFFT(Stage& stage) noexcept;
    void ExecuteStageSlice(Stage& stage, uint32_t samples_advanced) noexcept;

    // NUPC group table: {block_size, max_partitions_in_group}.
    // Group 4 (last): UINT32_MAX = "all remaining IR partitions".
    // Starting at B=128 (aligned with K=128 head) gives 5 groups vs the previous 6.
    struct GroupDesc {
        uint32_t block_size;
        uint32_t max_partitions;
    };
    static constexpr std::array<GroupDesc, 5> kGroups{{
        {  128u, 4u },          // Group 0: FFT 256,  covers taps [128..639]
        {  256u, 4u },          // Group 1: FFT 512,  covers taps [640..1663]
        {  512u, 4u },          // Group 2: FFT 1024, covers taps [1664..3711]
        { 1024u, 4u },          // Group 3: FFT 2048, covers taps [3712..7807]
        { 2048u, UINT32_MAX },  // Group 4: FFT 4096 (tail, time-sliced, FP16 on aarch64)
    }};

    // Sub-interval at which each time-slice phase fires (samples within a 2048-block).
    static constexpr uint32_t kSliceInterval = 512u;

    // -------------------------------------------------------------------------
    // L2 cache budget
    // -------------------------------------------------------------------------
    // The active working set (Head FIR + master ring + all FDLs + active filter
    // spectra) is kept below f=0.50 of a 512 KB L2 cache to prevent background
    // OS threads from evicting hot audio state.
    //
    // kMaxAllowedL2Bytes = 512 KB × 0.50 = 256 KB.
    //
    // When the tail group's filter spectra in FP32 would exceed this budget,
    // FP16 compression is enabled automatically (AArch64 only).  Short reverbs
    // that fit in budget remain in FP32 for full numerical precision.
    static constexpr size_t kL2CacheSizeBytes  = 512uz * 1024uz;
    static constexpr float  kCacheSafetyFactor = 0.50f;
    static constexpr size_t kMaxAllowedL2Bytes =
        static_cast<size_t>(
            static_cast<float>(kL2CacheSizeBytes) * kCacheSafetyFactor); // 256 KB

    /// Returns the estimated active L2 working-set size in bytes for a given
    /// tail partition count and FP16 mode.  Called in LoadKernel() before any
    /// allocation to determine whether FP16 compression should be activated.
    [[nodiscard]] size_t CalculateWorkingSetBytes(uint32_t tail_partitions,
                                                   bool     use_fp16_tail) const noexcept;

#if VIPER_USE_FP16_TAIL
    // On-the-fly FP16→FP32 widening + complex multiply-accumulate.
    // Processes the PFFFT "unordered" spectrum layout (Ncvec pairs of v4sf).
    // __restrict__ is the portable C++ spelling of the restrict hint
    // (RESTRICT is only defined inside pffft.c's translation unit).
    static void ZconvolveAccumulateFP16(const float* __restrict__ fdl,
                                        const __fp16* __restrict__ filter_fp16,
                                        float* __restrict__ accum,
                                        int ncvec) noexcept;
#endif
};
