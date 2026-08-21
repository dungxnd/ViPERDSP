#include "PConvNUPC.h"
#include "pffft.h"
#include <algorithm>
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
    std::fill_n(head_history_, kHeadLen * 2u, 0.0f);
    head_idx_ = 0u;

    std::ranges::fill(accum_ring_, 0.0f);
    ring_read_idx_ = 0u;

    for (Stage& s : stages_) {
        s.fdl_index      = 0u;
        s.sample_counter = 0u;
        std::ranges::fill(s.prev_input, 0.0f);
        for (uint32_t p = 0; p < s.num_partitions; ++p) {
            std::fill_n(s.fdl[p], s.fft_size, 0.0f);
        }
    }
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

    // ---- Stage 0: head FIR (first 64 taps) ---------------------------------
    const uint32_t head_tap_count = std::min(size, kHeadLen);
    std::fill_n(head_coeffs_,     kHeadLen, 0.0f);
    std::fill_n(head_coeffs_rev_, kHeadLen, 0.0f);
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

    // ---- NUPC stages: build from IR offset kHeadLen onwards ----------------
    auto alloc = [](uint32_t n) -> float* {
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

        // ── Causal output offset (Bug 3 fix) ──────────────────────────────
        // When this stage fires (after block_size new input samples have
        // accumulated), its partition 0 covers IR taps [ir_offset .. ir_offset+Bs-1].
        // The linear convolution result y[n] for these taps is valid starting at
        // sample n = ir_offset.  The OLS block delay is exactly Bs samples, so the
        // output lands at ring position:
        //   ring_read_idx_ + (ir_offset - block_size)
        // which is (ir_offset - block_size) samples ahead of the current read head.
        // This is the head-relative look-ahead distance the ring must receive.
        s.output_offset = ir_offset - block_size;

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

        s.filter_spectra.resize(num_parts, nullptr);
        s.fdl.resize(num_parts, nullptr);

        // NOTE: no temporary heap buffer needed — reuse s.fft_in, which is already
        // allocated via pffft_aligned_malloc (64-byte aligned, satisfying PFFFT's
        // VALIGNED assertion on all ABIs including armeabi-v7a and 32-bit x86).

        for (uint32_t p = 0; p < num_parts; ++p) {
            s.filter_spectra[p] = alloc(fft_size);
            s.fdl[p]            = alloc(fft_size);
            if (!s.filter_spectra[p] || !s.fdl[p]) {
                for (uint32_t q = 0; q <= p; ++q) {
                    if (s.filter_spectra[q]) pffft_aligned_free(s.filter_spectra[q]);
                    if (s.fdl[q])            pffft_aligned_free(s.fdl[q]);
                }
                pffft_aligned_free(s.fft_in);
                pffft_aligned_free(s.fft_out);
                pffft_aligned_free(s.fft_work);
                pffft_aligned_free(s.accum_spectrum);
                PFFFTRegistry::Instance().Release(s.fft_setup);
                ReleaseResources();
                return false;
            }

            std::fill_n(s.fdl[p], fft_size, 0.0f);

            // Zero-pad into the first block_size elements of s.fft_in; the second
            // half (block_size..fft_size-1) is already zeroed from the alloc call
            // above (pffft_aligned_malloc is not required to zero — we do it here).
            std::fill_n(s.fft_in, fft_size, 0.0f);
            const uint32_t copy_len = std::min(size - ir_offset, block_size);
            for (uint32_t i = 0; i < copy_len; ++i) {
                s.fft_in[i] = kernel[ir_offset + i] * gain;
            }

            pffft_transform(s.fft_setup,
                            s.fft_in,
                            s.filter_spectra[p],
                            s.fft_work,
                            PFFFT_FORWARD);

            ir_offset += block_size;
            if (ir_offset > size) ir_offset = size;
        }

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
// Process — non-interleaved
// ---------------------------------------------------------------------------

void PConvNUPC::Process(const float* const input,
                        float* const       output,
                        const uint32_t     n) noexcept {
    if (!instance_usable_ || n == 0) {
        if (input != output) std::copy_n(input, n, output);
        return;
    }

    for (uint32_t i = 0; i < n; ++i) {
        float out = 0.0f;
        ProcessOneSample(input[i], out);
        output[i] = out;
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

    for (uint32_t i = 0; i < n; ++i) {
        float out = 0.0f;
        ProcessOneSample(input[i * stride + channel], out);
        output[i * stride + channel] = out;
    }
}

// ---------------------------------------------------------------------------
// ProcessOneSample — core hot path: branchless per-sample logic
// ---------------------------------------------------------------------------

void PConvNUPC::ProcessOneSample(const float in, float& out) noexcept {
    // 1. Head FIR (K=64, 0 latency).
    // Double-buffer write: head_history_[j] == head_history_[j+K] so the window
    // [head_idx_+1 .. head_idx_+K] is always contiguous (no modulo in inner loop).
    head_history_[head_idx_]            = in;
    head_history_[head_idx_ + kHeadLen] = in;
    // hist_ptr[0] = x[n-K+1] (oldest) .. hist_ptr[K-1] = x[n] (newest).
    const float* const hist_ptr = &head_history_[head_idx_ + 1u];
    float head_out = 0.0f;
    // head_coeffs_rev_[k] = h[K-1-k]: sum == sum_{k} h[k]*x[n-k].
    #pragma clang loop vectorize(enable)
    for (uint32_t k = 0; k < kHeadLen; ++k) {
        head_out += head_coeffs_rev_[k] * hist_ptr[k];
    }
    head_idx_ = (head_idx_ + 1u) & kHeadMask;

    // 2. Read one previously-accumulated tail sample from the ring.
    const float tail_out = accum_ring_[ring_read_idx_];
    accum_ring_[ring_read_idx_] = 0.0f;  // Clear for future accumulation.
    ring_read_idx_ = (ring_read_idx_ + 1u) & kRingMask;

    out = head_out + tail_out;

    // 3. Advance each NUPC stage clock.
    for (Stage& s : stages_) {
        // Append to the second half of fft_in (curr window).
        s.fft_in[s.block_size + s.sample_counter] = in;
        if (++s.sample_counter == s.block_size) {
            s.sample_counter = 0u;
            ExecuteStageFFT(s);
        }
    }
}

// ---------------------------------------------------------------------------
// ExecuteStageFFT — triggered once per block_size samples for a given stage
// ---------------------------------------------------------------------------

void PConvNUPC::ExecuteStageFFT(Stage& stage) noexcept {
    // 1. Copy prev_input into the first half of fft_in.
    std::copy_n(stage.prev_input.data(), stage.block_size, stage.fft_in);

    // 2. Save the current block as the next cycle's prev_input.
    std::copy_n(stage.fft_in + stage.block_size,
                stage.block_size,
                stage.prev_input.data());

    // 3. Forward FFT (2*block_size real → freq domain), stored in current FDL slot.
    pffft_transform(stage.fft_setup,
                    stage.fft_in,
                    stage.fdl[stage.fdl_index],
                    stage.fft_work,
                    PFFFT_FORWARD);

    // 4. Frequency-domain multiply-accumulate over all partitions.
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

    // 5. Inverse FFT → fft_out.
    pffft_transform(stage.fft_setup,
                    stage.accum_spectrum,
                    stage.fft_out,
                    stage.fft_work,
                    PFFFT_BACKWARD);

    // 6. Overlap-save: discard first block_size (aliased), accumulate the
    //    second block_size into the master ring, normalised by 1/(2*block_size).
    //    Write starts at ring_read_idx_ + output_offset (the look-ahead
    //    distance that the stage's block delay exactly fills).
    //    Denormal flush applied to prevent CPU subnormal stalls on silent tails.
    const float norm   = 1.0f / static_cast<float>(stage.fft_size);
    const float denorm = 1e-25f;
    for (uint32_t i = 0; i < stage.block_size; ++i) {
        const uint32_t ring_idx =
            (ring_read_idx_ + stage.output_offset + i) & kRingMask;
        const float v = stage.fft_out[stage.block_size + i] * norm + denorm - denorm;
        accum_ring_[ring_idx] += v;
    }

    // 7. Advance the FDL ring.
    stage.fdl_index = (stage.fdl_index + 1u) % stage.num_partitions;
}
