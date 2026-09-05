#include "PConvNUPC.h"
#include "pffft.h"
#include <algorithm>
#include <cstring>
#include <ranges>
#include <span>

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

PConvNUPC::~PConvNUPC() {
    ReleaseResources();
}

void PConvNUPC::ReleaseResources() noexcept {
    auto free_a = [](float*& p) noexcept {
        if (p) { pffft_aligned_free(p); p = nullptr; }
    };

    for (Stage& s : stages_) {
        for (float* p : s.filter_spectra) if (p) pffft_aligned_free(p);
        for (float* p : s.fdl)            if (p) pffft_aligned_free(p);
        free_a(s.fft_in);
        free_a(s.fft_out);
        free_a(s.fft_work);
        free_a(s.accum_spectrum);
        if (s.fft_setup) {
            PFFFTRegistry::Instance().Release(s.fft_setup);
            s.fft_setup = nullptr;
        }
#if VIPER_USE_FP16_TAIL
        for (__fp16* p : s.filter_spectra_fp16) {
            if (p) { pffft_aligned_free(p); }
        }
        s.filter_spectra_fp16.clear();
#endif
    }
    stages_.clear();
    instance_usable_ = false;
}

void PConvNUPC::UnloadKernel() noexcept {
    ReleaseResources();
}

// ---------------------------------------------------------------------------
// Reset — zero all runtime state
// ---------------------------------------------------------------------------

void PConvNUPC::Reset() noexcept {
    if (!instance_usable_) return;

    // Clear both halves of the double-buffer history.
    head_history_.fill(0.0f);
    head_idx_ = 0u;

    accum_ring_.fill(0.0f);
    ring_read_idx_ = 0u;

    for (Stage& s : stages_) {
        s.fdl_index           = 0u;
        s.sample_counter      = 0u;
        s.slice_phase         = 0u;
        s.phase_sample_count  = 0u;
        s.saved_ring_read_idx = 0u;
        std::ranges::fill(s.prev_input, 0.0f);
        for (uint32_t p = 0; p < s.num_partitions; ++p) {
            std::fill_n(s.fdl[p], s.fft_size, 0.0f);
        }
    }
}

// ---------------------------------------------------------------------------
// CalculateWorkingSetBytes — L2 cache budget estimator
//
// Estimates bytes hot during Process(): head FIR arrays, accum_ring, all
// early-group FDLs + filter_spectra (FP32), and the tail FDL + spectra
// (FP32 or FP16).  Per-stage scratch (fft_in/out/work) is excluded — only
// one stage's scratch is hot at a time.
// ---------------------------------------------------------------------------

size_t PConvNUPC::CalculateWorkingSetBytes(const uint32_t tail_partitions,
                                            const bool     use_fp16_tail) const noexcept {
    size_t total = 0u;

    // Head FIR: 3×K floats (double-buffer history + reversed coefficients).
    total += static_cast<size_t>(kHeadLen * 3u) * sizeof(float);

    total += static_cast<size_t>(kRingSize) * sizeof(float);

    // Early groups: FDL + filter_spectra, both FP32.  Iterate until tail sentinel.
    for (const auto& [bs, max_p] : kGroups) {
        if (max_p == UINT32_MAX) break;
        const uint32_t fft_sz = bs * 2u;
        total += static_cast<size_t>(max_p) * static_cast<size_t>(fft_sz) * 2u * sizeof(float);
    }

    // Tail group (B=2048, FFT=4096): FDL always FP32; spectra FP32 or FP16.
    constexpr uint32_t kTailFFTSize = 2048u * 2u;
    total += static_cast<size_t>(tail_partitions) * kTailFFTSize * sizeof(float);
    const size_t filter_elem = use_fp16_tail ? sizeof(uint16_t) : sizeof(float);
    total += static_cast<size_t>(tail_partitions) * kTailFFTSize * filter_elem;

    return total;
}

// ---------------------------------------------------------------------------
// LoadKernel — internal helpers
// ---------------------------------------------------------------------------

// Returns the number of B=2048 tail partitions needed for a given IR size,
// or 0 if the IR fits entirely in early groups.  No allocation.
static uint32_t EstimateTailPartitions(const uint32_t size,
                                        const uint32_t head_len,
                                        const std::array<PConvNUPC::GroupDesc, 5>& groups) noexcept {
    uint32_t scan_offset = head_len;
    for (const auto& [bs, max_p] : groups) {
        if (scan_offset >= size) return 0u;
        const uint32_t remaining = size - scan_offset;
        const uint32_t by_size   = (remaining + bs - 1u) / bs;
        if (max_p == UINT32_MAX) {
            return by_size;
        }
        const uint32_t used = std::min(max_p, by_size);
        scan_offset += used * bs;
    }
    return 0u;
}

// Allocate and FFT-transform all partition filter spectra for one stage.
// Returns false on allocation failure; on failure all locally allocated
// memory is freed and the caller must call ReleaseResources().
static bool AllocateStagePartitions(PConvNUPC::Stage& s,
                                     const std::span<const float> kernel,
                                     uint32_t& ir_offset,
                                     const float gain) noexcept {
    const uint32_t fft_size    = s.fft_size;
    const uint32_t block_size  = s.block_size;
    const uint32_t num_parts   = s.num_partitions;
    const uint32_t kernel_size = static_cast<uint32_t>(kernel.size());

    s.filter_spectra.resize(num_parts, nullptr);
    s.fdl.resize(num_parts, nullptr);

    // Temporary aligned scratch for zero-padded partition input.
    // Allocated once here and freed at the end — never survives this function.
    float* const part_scratch =
        static_cast<float*>(pffft_aligned_malloc(fft_size * sizeof(float)));
    if (!part_scratch) return false;

    bool ok = true;
    for (uint32_t p = 0; p < num_parts && ok; ++p) {
        s.filter_spectra[p] =
            static_cast<float*>(pffft_aligned_malloc(fft_size * sizeof(float)));
        s.fdl[p] =
            static_cast<float*>(pffft_aligned_malloc(fft_size * sizeof(float)));

        if (!s.filter_spectra[p] || !s.fdl[p]) {
            // Clean up partially allocated pair.
            if (s.filter_spectra[p]) { pffft_aligned_free(s.filter_spectra[p]); s.filter_spectra[p] = nullptr; }
            if (s.fdl[p])            { pffft_aligned_free(s.fdl[p]);            s.fdl[p] = nullptr; }
            ok = false;
            break;
        }

        std::fill_n(s.fdl[p], fft_size, 0.0f);

        // Zero-pad partition into scratch[0..block_size-1].
        std::fill_n(part_scratch, fft_size, 0.0f);
        const uint32_t copy_len = std::min(kernel_size - ir_offset, block_size);
        for (uint32_t i = 0; i < copy_len; ++i) {
            part_scratch[i] = kernel[ir_offset + i] * gain;
        }

        pffft_transform(s.fft_setup,
                        part_scratch,
                        s.filter_spectra[p],
                        s.fft_work,
                        PFFFT_FORWARD);

        ir_offset += block_size;
        if (ir_offset > kernel_size) ir_offset = kernel_size;
    }

    pffft_aligned_free(part_scratch);

    if (!ok) {
        // Free successfully allocated partitions.
        for (float* p : s.filter_spectra) if (p) pffft_aligned_free(p);
        for (float* p : s.fdl)            if (p) pffft_aligned_free(p);
        s.filter_spectra.clear();
        s.fdl.clear();
    }
    return ok;
}

// ---------------------------------------------------------------------------
// LoadKernel
// ---------------------------------------------------------------------------

// std::span primary overload
bool PConvNUPC::LoadKernel(const std::span<const float> kernel,
                            const float                  gain) {
    if (kernel.empty()) return false;

    ReleaseResources();

    const auto size = static_cast<uint32_t>(kernel.size());

    // ---- Stage 0: head FIR (first 128 taps) ---------------------------------
    const uint32_t head_tap_count = std::min(size, kHeadLen);
    head_coeffs_.fill(0.0f);
    head_coeffs_rev_.fill(0.0f);
    for (uint32_t k = 0; k < head_tap_count; ++k) {
        head_coeffs_[k] = kernel[k] * gain;
    }
    // head_coeffs_rev_[k] = h[K-1-k]: reversed for oldest→newest contiguous read.
    for (uint32_t k = 0; k < kHeadLen; ++k) {
        head_coeffs_rev_[k] = head_coeffs_[kHeadLen - 1u - k];
    }

    if (size <= kHeadLen) {
        instance_usable_ = true;
        Reset();
        return true;
    }

    // Pre-compute tail count for the L2 budget check (no allocation).
    const uint32_t tail_partitions_estimate =
        EstimateTailPartitions(size, kHeadLen, kGroups);

    // FP16 tail: AArch64 only; enabled when FP32 working set > 256 KB L2 budget.
    // Short IRs that fit in budget stay FP32 for full numerical precision.
    [[maybe_unused]] bool enable_fp16_tail = false;
#if VIPER_USE_FP16_TAIL
    {
        const size_t ws_fp32 = CalculateWorkingSetBytes(tail_partitions_estimate, false);
        enable_fp16_tail = (ws_fp32 > kMaxAllowedL2Bytes);
    }
#endif

    // ---- NUPC stages: build from IR offset kHeadLen onwards ----------------
    auto alloc = [](uint32_t n) {
        return static_cast<float*>(pffft_aligned_malloc(n * sizeof(float)));
    };

    uint32_t ir_offset = kHeadLen;  // Samples consumed so far.

    for (const auto& [block_size, max_parts] : kGroups) {
        if (ir_offset >= size) break;

        const uint32_t fft_size      = block_size * 2u;
        const uint32_t remaining_ir  = size - ir_offset;
        const uint32_t max_p_by_size = (remaining_ir + block_size - 1u) / block_size;
        const uint32_t num_parts =
            (max_parts == UINT32_MAX)
            ? max_p_by_size
            : std::min(max_parts, max_p_by_size);

        if (num_parts == 0u) break;

        Stage s{};
        s.block_size     = block_size;
        s.fft_size       = fft_size;
        s.num_partitions = num_parts;
        // Precompute IFFT normalisation factor: 1/(2*block_size).
        // Storing in Stage eliminates the division-in-hot-path pattern and
        // makes the value statically provably non-zero (block_size >= 128).
        s.norm           = 1.0f / static_cast<float>(fft_size);

        // Causal offset: OLS delay for this stage = block_size samples.
        // Partition 0 covers taps [ir_offset..+Bs-1]; those results are valid
        // at output sample ir_offset.  Ring write: read_idx + (ir_offset - Bs).
        s.output_offset  = ir_offset - block_size;
        s.is_time_sliced = (block_size >= 2048u);

        s.fft_setup = PFFFTRegistry::Instance().Acquire(fft_size);
        if (!s.fft_setup) { ReleaseResources(); return false; }

        s.fft_in         = alloc(fft_size);
        s.fft_out        = alloc(fft_size);
        s.fft_work       = alloc(fft_size);
        s.accum_spectrum = alloc(fft_size);

        if (!s.fft_in || !s.fft_out || !s.fft_work || !s.accum_spectrum) {
            pffft_aligned_free(s.fft_in);
            pffft_aligned_free(s.fft_out);
            pffft_aligned_free(s.fft_work);
            pffft_aligned_free(s.accum_spectrum);
            PFFFTRegistry::Instance().Release(s.fft_setup);
            ReleaseResources();
            return false;
        }

        // Allocate + FFT all partition filter spectra.
        // AllocateStagePartitions advances ir_offset as it works.
        if (!AllocateStagePartitions(s, kernel, ir_offset, gain)) {
            pffft_aligned_free(s.fft_in);
            pffft_aligned_free(s.fft_out);
            pffft_aligned_free(s.fft_work);
            pffft_aligned_free(s.accum_spectrum);
            PFFFTRegistry::Instance().Release(s.fft_setup);
            ReleaseResources();
            return false;
        }

#if VIPER_USE_FP16_TAIL
        // Convert FP32 filter spectra to FP16 when the L2 budget check triggered.
        if (s.is_time_sliced && enable_fp16_tail) {
            s.filter_spectra_fp16.resize(num_parts, nullptr);
            for (uint32_t p = 0; p < num_parts; ++p) {
                // pffft_aligned_malloc: 64-byte aligned, available on Android NDK
                // bionic (::aligned_alloc is not).
                __fp16* buf = static_cast<__fp16*>(
                    pffft_aligned_malloc(fft_size * sizeof(__fp16)));
                if (!buf) {
                    for (__fp16* allocated : s.filter_spectra_fp16) {
                        if (allocated) pffft_aligned_free(allocated);
                    }
                    s.filter_spectra_fp16.clear();
                    pffft_aligned_free(s.fft_in);
                    pffft_aligned_free(s.fft_out);
                    pffft_aligned_free(s.fft_work);
                    pffft_aligned_free(s.accum_spectrum);
                    PFFFTRegistry::Instance().Release(s.fft_setup);
                    ReleaseResources();
                    return false;
                }
                const float* src = s.filter_spectra[p];
                __fp16*      dst = buf;
                for (uint32_t i = 0; i < fft_size; i += 4u) {
                    float32x4_t v32 = vld1q_f32(src + i);
                    float16x4_t v16 = vcvt_f16_f32(v32);
                    vst1_f16(dst + i, v16);
                }
                s.filter_spectra_fp16[p] = buf;

                pffft_aligned_free(s.filter_spectra[p]);  // FP16 is now sole source
                s.filter_spectra[p] = nullptr;
            }
        }
#endif

        s.prev_input.assign(block_size, 0.0f);
        stages_.push_back(std::move(s));
    }

    instance_usable_ = true;
    Reset();
    return true;
}

// Raw-pointer convenience overload — delegates to span overload.
bool PConvNUPC::LoadKernel(const float* const kernel,
                            const uint32_t     size,
                            const float        gain) {
    return LoadKernel(std::span<const float>{kernel, size}, gain);
}

// ---------------------------------------------------------------------------
// Process — chunk-based non-interleaved DSP path.
//
// Each iteration computes the largest `chunk` that fits before any stage fires,
// then runs the head FIR and ring read over the full chunk (SIMD-eligible).
// Input is batch-copied into stage windows with one std::copy_n per stage.
// This eliminates per-sample counter branch predictions and enables the
// head FIR inner loop to be auto-vectorized without loop-carried counters.
// ---------------------------------------------------------------------------

void PConvNUPC::Process(const float* const input,
                         float* const       output,
                         const uint32_t     n) noexcept {
    if (!instance_usable_ || n == 0) {
        if (input != output) std::copy_n(input, n, output);
        return;
    }

    uint32_t pos = 0;
    while (pos < n) {
        // 1. Determine max chunk until the next stage boundary.
        //    For time-sliced stages we also consider their slice phase boundary.
        uint32_t chunk = n - pos;
        for (const Stage& s : stages_) {
            if (!s.is_time_sliced) {
                chunk = std::min(chunk, s.block_size - s.sample_counter);
            } else {
                // Time-sliced: limit by samples until next kSliceInterval boundary.
                const uint32_t til_phase =
                    kSliceInterval - (s.phase_sample_count % kSliceInterval);
                chunk = std::min(chunk, til_phase);
                // Also limit by block boundary (triggers ForwardFFT).
                const uint32_t til_block = s.block_size - s.sample_counter;
                chunk = std::min(chunk, til_block);
            }
        }
        // chunk is at least 1 (counters can never be == their limit at this point).

        // 2. Head FIR + ring read over `chunk` samples.
        for (uint32_t i = 0; i < chunk; ++i) {
            const float in = input[pos + i];

            head_history_[head_idx_]            = in;
            head_history_[head_idx_ + kHeadLen] = in;
            const float* const hist_ptr = &head_history_[head_idx_ + 1u];

            float head_out = 0.0f;
            #pragma clang loop vectorize(enable)
            for (uint32_t k = 0; k < kHeadLen; ++k) {
                head_out += head_coeffs_rev_[k] * hist_ptr[k];
            }
            head_idx_ = (head_idx_ + 1u) & kHeadMask;

            const float tail_out        = accum_ring_[ring_read_idx_];
            accum_ring_[ring_read_idx_] = 0.0f;
            ring_read_idx_              = (ring_read_idx_ + 1u) & kRingMask;

            output[pos + i] = head_out + tail_out;
        }

        // 3. Advance stage counters and fire.
        for (Stage& s : stages_) {
            std::copy_n(input + pos,
                        chunk,
                        s.fft_in + s.block_size + s.sample_counter);
            s.sample_counter += chunk;

            if (!s.is_time_sliced) {
                if (s.sample_counter == s.block_size) {
                    s.sample_counter = 0u;
                    ExecuteStageFFT(s);
                }
            } else {
                ExecuteStageSlice(s, chunk);
                if (s.sample_counter == s.block_size) {
                    s.sample_counter = 0u;
                }
            }
        }

        pos += chunk;
    }
}

// ---------------------------------------------------------------------------
// ProcessInterleaved
// ---------------------------------------------------------------------------

void PConvNUPC::ProcessInterleaved(const float* const input,
                                    float* const       output,
                                    const uint32_t     channel,
                                    const uint32_t     stride,
                                    const uint32_t     n) noexcept {
    if (!instance_usable_ || n == 0) return;

    // Deinterleave → Process → reinterleave using two stack-allocated scratch buffers.
    //
    // Stack headroom analysis:
    //   • kMaxStack = 4096 floats × 2 buffers × 4 bytes = 32 KiB per call.
    //   • Host frames are bounded to ≤4096 samples by the Android AudioFlinger ABI.
    //   • Android NDK threads have ≥1 MB of stack (bionic default); POSIX minimum is 64 KiB.
    //   • 32 KiB ≪ 64 KiB: safe on all supported targets (arm64-v8a, x86_64, armv7).
    //   • The over-size branch re-uses the same kMaxStack buffers in chunks, so peak
    //     stack consumption is always ≤ 32 KiB regardless of n.
    constexpr uint32_t kMaxStack = 4096u;
    if (n <= kMaxStack) {
        alignas(64) std::array<float, kMaxStack> in_buf{};
        alignas(64) std::array<float, kMaxStack> out_buf{};

        for (uint32_t i = 0; i < n; ++i) {
            in_buf[i] = input[i * stride + channel];
        }
        Process(in_buf.data(), out_buf.data(), n);
        for (uint32_t i = 0; i < n; ++i) {
            output[i * stride + channel] = out_buf[i];
        }
    } else {
        // Rare over-size path: process in kMaxStack-sized chunks.
        for (uint32_t i = 0; i < n; ) {
            const uint32_t chunk = std::min(n - i, kMaxStack);
            alignas(64) std::array<float, kMaxStack> in_buf{};
            alignas(64) std::array<float, kMaxStack> out_buf{};
            for (uint32_t j = 0; j < chunk; ++j) {
                in_buf[j] = input[(i + j) * stride + channel];
            }
            Process(in_buf.data(), out_buf.data(), chunk);
            for (uint32_t j = 0; j < chunk; ++j) {
                output[(i + j) * stride + channel] = out_buf[j];
            }
            i += chunk;
        }
    }
}

// ---------------------------------------------------------------------------
// ExecuteStageFFT — triggered once per block_size samples for non-sliced stages
// ---------------------------------------------------------------------------

void PConvNUPC::ExecuteStageFFT(Stage& stage) noexcept {
    // 1. Copy prev_input into the first half of fft_in.
    std::copy_n(stage.prev_input.data(), stage.block_size, stage.fft_in);

    // 2. Save the current block as the next cycle's prev_input.
    std::copy_n(stage.fft_in + stage.block_size,
                stage.block_size,
                stage.prev_input.data());

    // 3. Forward FFT (2*block_size real → PFFFT freq domain), stored in FDL.
    pffft_transform(stage.fft_setup,
                    stage.fft_in,
                    stage.fdl[stage.fdl_index],
                    stage.fft_work,
                    PFFFT_FORWARD);

    // 4. Frequency-domain multiply-accumulate over all partitions.
    //    For exactly-4-partition groups, pffft_zconvolve_4x fuses all 4 MACs
    //    into a single pass over accum_spectrum, cutting L1 write-backs by ~75%.
    if (stage.num_partitions == 4u) {
        std::array<const float*, 4> fdl_ptrs{};
        std::array<const float*, 4> filter_ptrs{};
        for (uint32_t p = 0; p < 4u; ++p) {
            const uint32_t slot = (stage.fdl_index - p + 4u) % 4u;
            fdl_ptrs[p]    = stage.fdl[slot];
            filter_ptrs[p] = stage.filter_spectra[p];
        }
        // pffft_zconvolve_4x writes ab directly (not accumulate):
        // no need to zero accum_spectrum first.
        pffft_zconvolve_4x(stage.fft_setup,
                           fdl_ptrs.data(),
                           filter_ptrs.data(),
                           stage.accum_spectrum,
                           1.0f);
    } else {
        // General path (tail stage with arbitrary partition count).
        // num_partitions is always > 0: LoadKernel breaks before push_back when num_parts==0.
        [[assume(stage.num_partitions > 0u)]];
        std::fill_n(stage.accum_spectrum, stage.fft_size, 0.0f);
        for (uint32_t p = 0; p < stage.num_partitions; ++p) {
            const uint32_t slot =
                (stage.fdl_index - p + stage.num_partitions) % stage.num_partitions;
            pffft_zconvolve_accumulate(
                stage.fft_setup,
                stage.fdl[slot],
                stage.filter_spectra[p],
                stage.accum_spectrum,
                1.0f
            );
        }
    }

    // 5. Inverse FFT → fft_out.
    pffft_transform(stage.fft_setup,
                    stage.accum_spectrum,
                    stage.fft_out,
                    stage.fft_work,
                    PFFFT_BACKWARD);

    // 6. Overlap-save: discard first block_size (aliased), scatter the second
    //    block_size into the master ring normalised by 1/(2*block_size).
    //    stage.norm is precomputed in LoadKernel — no division here.
    //    Denormal flush: prevents CPU subnormal stalls on silent reverb tails.
    [[assume(stage.num_partitions > 0u)]];
    const float denorm = 1e-25f;
    for (uint32_t i = 0; i < stage.block_size; ++i) {
        const uint32_t ring_idx =
            (ring_read_idx_ + stage.output_offset + i) & kRingMask;
        const float v = stage.fft_out[stage.block_size + i] * stage.norm + denorm - denorm;
        accum_ring_[ring_idx] += v;
    }

    // 7. Advance the FDL ring.
    stage.fdl_index = (stage.fdl_index + 1u) % stage.num_partitions;
}

// ---------------------------------------------------------------------------
// AccumulateStageMac — shared MAC helper for ExecuteStageSlice
//
// Accumulates partitions [p_start, p_end) of the given stage into its
// accum_spectrum.  Extracted to reduce nesting depth in ExecuteStageSlice's
// switch cases and to centralise the FP16/FP32 dispatch.
// ---------------------------------------------------------------------------

void PConvNUPC::AccumulateStageMac(Stage& stage,
                                    const uint32_t p_start,
                                    const uint32_t p_end) noexcept {
    // num_partitions is always > 0: Stages are only created with num_parts > 0 in LoadKernel.
    [[assume(stage.num_partitions > 0u)]];
    for (uint32_t p = p_start; p < p_end; ++p) {
        const uint32_t slot =
            (stage.fdl_index - p + stage.num_partitions) % stage.num_partitions;
#if VIPER_USE_FP16_TAIL
        if (!stage.filter_spectra_fp16.empty() && stage.filter_spectra_fp16[p]) {
            ZconvolveAccumulateFP16(stage.fdl[slot],
                                   stage.filter_spectra_fp16[p],
                                   stage.accum_spectrum,
                                   static_cast<int>(stage.fft_size / 8u));
        } else
#endif
        {
            pffft_zconvolve_accumulate(stage.fft_setup,
                                       stage.fdl[slot],
                                       stage.filter_spectra[p],
                                       stage.accum_spectrum,
                                       1.0f);
        }
    }
}

// ---------------------------------------------------------------------------
// ExecuteStageSlice — time-distributed execution for the B=2048 tail stage
//
// Phases (each kSliceInterval=512 samples apart within a 2048-sample block):
//   Phase 1 (at block boundary)  — ForwardFFT: capture input → FDL slot.
//   Phase 2 (at +512 samples)    — MAC first half of partitions.
//   Phase 3 (at +1024 samples)   — MAC second half of partitions.
//   Phase 4 (at +1536 samples)   — InverseFFT + scatter into ring.
//
// saved_ring_read_idx is captured at Phase 1 so the Phase 4 ring scatter uses
// the CORRECT ring position (not the position 1536 samples later).
// ---------------------------------------------------------------------------

void PConvNUPC::ExecuteStageSlice(Stage& stage, const uint32_t samples_advanced) noexcept {
    // Case: block boundary hit this call — trigger Phase 1 immediately.
    if (stage.sample_counter == stage.block_size) {
        // Phase 1: Forward FFT.
        std::copy_n(stage.prev_input.data(), stage.block_size, stage.fft_in);
        std::copy_n(stage.fft_in + stage.block_size,
                    stage.block_size,
                    stage.prev_input.data());
        pffft_transform(stage.fft_setup,
                        stage.fft_in,
                        stage.fdl[stage.fdl_index],
                        stage.fft_work,
                        PFFFT_FORWARD);
        std::fill_n(stage.accum_spectrum, stage.fft_size, 0.0f);

        // Save ring position for use in the IFFT scatter phase.
        stage.saved_ring_read_idx = ring_read_idx_;
        stage.slice_phase         = 1u;
        stage.phase_sample_count  = 0u;
        // Do NOT accumulate samples_advanced on the boundary frame: the chunk
        // that triggered Phase 1 belongs to the block that just completed, not
        // to the new phase interval.  Accumulation resumes on the next call.
        return;
    }

    // Advance phase counter.
    stage.phase_sample_count += samples_advanced;

    // Fire deferred phases at kSliceInterval boundaries.
    while (stage.phase_sample_count >= kSliceInterval && stage.slice_phase >= 1u) {
        stage.phase_sample_count -= kSliceInterval;
        stage.slice_phase        += 1u;

        switch (stage.slice_phase) {
            case 2u:
                // Phase 2: MAC first half of partitions.
                AccumulateStageMac(stage, 0u, stage.num_partitions / 2u);
                break;

            case 3u:
                // Phase 3: MAC second half of partitions.
                AccumulateStageMac(stage, stage.num_partitions / 2u, stage.num_partitions);
                break;

            case 4u: {
                // Phase 4: Inverse FFT + scatter into ring.
                pffft_transform(stage.fft_setup,
                                stage.accum_spectrum,
                                stage.fft_out,
                                stage.fft_work,
                                PFFFT_BACKWARD);

                // stage.norm is precomputed in LoadKernel — no runtime division.
                const float denorm = 1e-25f;
                for (uint32_t i = 0; i < stage.block_size; ++i) {
                    // Use saved_ring_read_idx: the ring position when ForwardFFT
                    // fired, NOT the current (advanced-by-1536-samples) position.
                    const uint32_t ring_idx =
                        (stage.saved_ring_read_idx + stage.output_offset + i) & kRingMask;
                    const float v =
                        stage.fft_out[stage.block_size + i] * stage.norm + denorm - denorm;
                    accum_ring_[ring_idx] += v;
                }

                // Advance FDL ring and reset phase state.
                [[assume(stage.num_partitions > 0u)]];
                stage.fdl_index   = (stage.fdl_index + 1u) % stage.num_partitions;
                stage.slice_phase = 0u;
                break;
            }

            default:
                break;
        }
    }
}

// ---------------------------------------------------------------------------
// ZconvolveAccumulateFP16 — AArch64 only
//
// On-the-fly FP16 → FP32 widening then complex-multiply-accumulate against fdl.
// Processes the PFFFT "unordered" spectrum layout: Ncvec pairs of (v4sf real,
// v4sf imag).  This mirrors the inner loop of pffft_zconvolve_accumulate.
//
// Note: DC and Nyquist bins (stored as scalars at fdl[0] and fdl[N-1] in the
// unordered PFFFT format) are handled correctly by the PFFFT internal layout —
// the first and last float values in the interleaved vectors hold those bins.
// The full complex loop covers them naturally; no special-case scalar path is
// needed here because the FP16 conversion preserves their representation.
// ---------------------------------------------------------------------------

#if VIPER_USE_FP16_TAIL
void PConvNUPC::ZconvolveAccumulateFP16(const float* __restrict__ fdl,
                                         const __fp16* __restrict__ filter_fp16,
                                         float* __restrict__ accum,
                                         const int            ncvec) noexcept {
    // Iterate over Ncvec v4sf-pair bins (each pair = real + imag vectors of 4).
    // Process 2 pairs per iteration (matches the stride in pffft_zconvolve_accumulate).
    const float32x4_t* va  = reinterpret_cast<const float32x4_t*>(fdl);
    const __fp16*      vb  = filter_fp16;
    float32x4_t*       vab = reinterpret_cast<float32x4_t*>(accum);

    // The PFFFT real-transform layout stores DC and Nyquist as REAL SCALARS at
    // fdl[0] and fdl[4] (lane 0 of the first two v4sf).  The SIMD loop below
    // treats every v4sf pair as a complex bin, which cross-contaminates those
    // scalar slots (their remaining lanes hold real bins 1..7).  Mirror
    // pffft_zconvolve_accumulate exactly: save the PRE-LOOP accumulator values,
    // run the full complex loop, then OVERWRITE the two scalar slots with
    // prev + correct_product (a += after the loop would add to the corrupted
    // complex cross-product residue).
    const float prev_dc = accum[0];
    const float prev_ny = accum[4];
    const float dc_prod = fdl[0] * static_cast<float>(filter_fp16[0]);
    const float ny_prod = fdl[4] * static_cast<float>(filter_fp16[4]);

    for (int i = 0; i < ncvec; i += 2) {
        // Load 8 FP16 values as two float16x4_t (4 FP16 each), widen to float32x4_t.
        float32x4_t br0 = vcvt_f32_f16(vld1_f16(vb + 4 * (2*i+0)));
        float32x4_t bi0 = vcvt_f32_f16(vld1_f16(vb + 4 * (2*i+1)));
        float32x4_t br1 = vcvt_f32_f16(vld1_f16(vb + 4 * (2*i+2)));
        float32x4_t bi1 = vcvt_f32_f16(vld1_f16(vb + 4 * (2*i+3)));

        float32x4_t ar0 = va[2*i+0];
        float32x4_t ai0 = va[2*i+1];
        float32x4_t ar1 = va[2*i+2];
        float32x4_t ai1 = va[2*i+3];

        // Complex multiply: (ar + j·ai)(br + j·bi) = (ar·br - ai·bi) + j(ar·bi + ai·br)
        float32x4_t wr0 = vmulq_f32(ar0, br0);
        wr0              = vmlsq_f32(wr0,  ai0, bi0);
        float32x4_t wi0 = vmulq_f32(ai0, br0);
        wi0              = vmlaq_f32(wi0,  ar0, bi0);

        float32x4_t wr1 = vmulq_f32(ar1, br1);
        wr1              = vmlsq_f32(wr1,  ai1, bi1);
        float32x4_t wi1 = vmulq_f32(ai1, br1);
        wi1              = vmlaq_f32(wi1,  ar1, bi1);

        // Accumulate.
        vab[2*i+0] = vaddq_f32(vab[2*i+0], wr0);
        vab[2*i+1] = vaddq_f32(vab[2*i+1], wi0);
        vab[2*i+2] = vaddq_f32(vab[2*i+2], wr1);
        vab[2*i+3] = vaddq_f32(vab[2*i+3], wi1);
    }

    // Overwrite the corrupted SIMD scalar slots with the exact products.
    accum[0] = prev_dc + dc_prod;
    accum[4] = prev_ny + ny_prod;
}
#endif
