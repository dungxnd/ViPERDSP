#include "PConvZeroLatency.h"
#include "pffft.h"
#include <algorithm>
#include <cmath>
#include <ranges>
#include <span>

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

PConvZeroLatency::~PConvZeroLatency() {
    ReleaseResources();
}

void PConvZeroLatency::ReleaseResources() noexcept {
    auto free_a = [](float*& p) noexcept {
        if (p) { pffft_aligned_free(p); p = nullptr; }
    };

    for (float* p : filter_segs_) if (p) pffft_aligned_free(p);
    filter_segs_.clear();

    for (float* p : fdl_)         if (p) pffft_aligned_free(p);
    fdl_.clear();

    free_a(fft_in_);
    free_a(fft_out_);
    free_a(fft_work_);
    free_a(accum_);

    if (tail_fft_setup_) {
        PFFFTRegistry::Instance().Release(tail_fft_setup_);
        tail_fft_setup_ = nullptr;
    }

    prev_input_.clear();
    tail_partition_count_ = 0u;
    fdl_idx_              = 0u;
    instance_usable_      = false;
}

void PConvZeroLatency::UnloadKernel() noexcept {
    ReleaseResources();
}

// ---------------------------------------------------------------------------
// Reset — zero all delay-state to silence.
//
// Output FIFO starts EMPTY (no pre-fill): the OLS math guarantees that the
// tail's first B clean output samples arrive exactly when the head has
// processed B samples, so head + tail sum is causal with zero extra delay.
// ---------------------------------------------------------------------------

void PConvZeroLatency::Reset() noexcept {
    if (!instance_usable_) return;

    head_history_.fill(0.0f);
    head_idx_ = 0u;
    fdl_idx_  = 0u;
    std::ranges::fill(prev_input_, 0.0f);

    for (uint32_t j = 0; j < tail_partition_count_; ++j) {
        std::fill_n(fdl_[j], kTailFFTSz, 0.0f);
    }

    input_fifo_.Reset();
    output_fifo_.Reset();
}

// ---------------------------------------------------------------------------
// LoadKernel
// ---------------------------------------------------------------------------

bool PConvZeroLatency::LoadKernel(const std::span<const float> kernel,
                                   const float                  gain) {
    if (kernel.empty()) return false;

    ReleaseResources();

    const auto size = static_cast<uint32_t>(kernel.size());

    // ---- Stage 0: direct-form FIR head (h[0..K-1]) -------------------------
    const uint32_t head_tap_count = std::min(size, kHeadLen);
    head_coeffs_.fill(0.0f);
    head_coeffs_rev_.fill(0.0f);
    for (uint32_t k = 0; k < head_tap_count; ++k) {
        head_coeffs_[k] = kernel[k] * gain;
    }
    // Reversed copy: head_coeffs_rev_[k] = h[K-1-k].
    // Enables oldest→newest contiguous pointer read in the FIR inner loop.
    for (uint32_t k = 0; k < kHeadLen; ++k) {
        head_coeffs_rev_[k] = head_coeffs_[kHeadLen - 1u - k];
    }

    // If the entire IR fits in the head, no tail stage needed.
    if (size <= kHeadLen) {
        instance_usable_ = true;
        Reset();
        return true;
    }

    // ---- Stage 1: Uniform OLS tail on h[K..L-1] ----------------------------
    // B=128, R=256, J = ceil((L-K)/B). For 4096-tap HRIRs: J = 31 (was 63 at K=64).
    const uint32_t tail_len = size - kHeadLen;
    tail_partition_count_   = (tail_len + kTailBlock - 1u) / kTailBlock;

    tail_fft_setup_ = PFFFTRegistry::Instance().Acquire(kTailFFTSz);
    if (!tail_fft_setup_) { ReleaseResources(); return false; }

    auto alloc = [](uint32_t n) {
        return static_cast<float*>(pffft_aligned_malloc(n * sizeof(float)));
    };

    fft_in_   = alloc(kTailFFTSz);
    fft_out_  = alloc(kTailFFTSz);
    fft_work_ = alloc(kTailFFTSz);
    accum_    = alloc(kTailFFTSz);

    if (!fft_in_ || !fft_out_ || !fft_work_ || !accum_) {
        ReleaseResources(); return false;
    }

    filter_segs_.resize(tail_partition_count_, nullptr);
    fdl_.resize(tail_partition_count_, nullptr);

    // Load-path only scratch (1 KiB); alignas(64) satisfies PFFFT SIMD alignment.
    alignas(64) std::array<float, kTailFFTSz> part_buf{};

    for (uint32_t j = 0; j < tail_partition_count_; ++j) {
        filter_segs_[j] = alloc(kTailFFTSz);
        fdl_[j]         = alloc(kTailFFTSz);
        if (!filter_segs_[j] || !fdl_[j]) { ReleaseResources(); return false; }

        std::fill_n(fdl_[j], kTailFFTSz, 0.0f);

        // Zero-pad second half so circular ↔ linear conv holds for output[B..2B-1].
        const uint32_t offset   = kHeadLen + j * kTailBlock;
        const uint32_t copy_len = std::min(size - offset, kTailBlock);
        part_buf.fill(0.0f);
        for (uint32_t i = 0; i < copy_len; ++i) {
            part_buf[i] = kernel[offset + i] * gain;
        }
        pffft_transform(tail_fft_setup_, part_buf.data(), filter_segs_[j], fft_work_, PFFFT_FORWARD);
    }

    prev_input_.assign(kTailBlock, 0.0f);  // B historical samples for 2B OLS window

    const std::size_t fifo_cap =
        std::max<std::size_t>(static_cast<std::size_t>(kTailBlock) * 8u, 4096u);
    input_fifo_.Init(fifo_cap);
    output_fifo_.Init(fifo_cap);

    instance_usable_ = true;
    Reset();
    return true;
}

// Raw-pointer convenience overload — delegates to span overload.
bool PConvZeroLatency::LoadKernel(const float* const kernel,
                                   const uint32_t     size,
                                   const float        gain) {
    return LoadKernel(std::span<const float>{kernel, size}, gain);
}

// ---------------------------------------------------------------------------
// Process — non-interleaved
// ---------------------------------------------------------------------------

void PConvZeroLatency::Process(const float* const input,
                                float* const       output,
                                const uint32_t     n) noexcept {
    if (!instance_usable_ || n == 0) {
        if (input != output) std::copy_n(input, n, output);
        return;
    }

    for (uint32_t i = 0; i < n; ++i) {
        const float in = input[i];

        head_history_[head_idx_]            = in;
        head_history_[head_idx_ + kHeadLen] = in;
        const float* const hist_ptr = &head_history_[head_idx_ + 1u];
        float head_out = 0.0f;
        #pragma clang loop vectorize(enable)
        for (uint32_t k = 0; k < kHeadLen; ++k) {
            head_out += head_coeffs_rev_[k] * hist_ptr[k];
        }
        head_idx_ = (head_idx_ + 1u) & kHeadMask;

        float tail_out = 0.0f;
        if (tail_partition_count_ > 0u && output_fifo_.AvailableRead() > 0u) {
            output_fifo_.Read(&tail_out, 1u);
        }

        output[i] = head_out + tail_out;

        if (tail_partition_count_ > 0u) {
            input_fifo_.Write(&in, 1u);
            while (input_fifo_.AvailableRead() >= kTailBlock) {
                ProcessTailBlock();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ProcessInterleaved
// ---------------------------------------------------------------------------

void PConvZeroLatency::ProcessInterleaved(const float* const input,
                                           float* const       output,
                                           const uint32_t     channel,
                                           const uint32_t     stride,
                                           const uint32_t     n) noexcept {
    if (!instance_usable_ || n == 0) return;

    for (uint32_t i = 0; i < n; ++i) {
        const float in = input[i * stride + channel];

        head_history_[head_idx_]            = in;
        head_history_[head_idx_ + kHeadLen] = in;
        const float* const hist_ptr = &head_history_[head_idx_ + 1u];
        float head_out = 0.0f;
        #pragma clang loop vectorize(enable)
        for (uint32_t k = 0; k < kHeadLen; ++k) {
            head_out += head_coeffs_rev_[k] * hist_ptr[k];
        }
        head_idx_ = (head_idx_ + 1u) & kHeadMask;

        float tail_out = 0.0f;
        if (tail_partition_count_ > 0u && output_fifo_.AvailableRead() > 0u) {
            output_fifo_.Read(&tail_out, 1u);
        }

        output[i * stride + channel] = head_out + tail_out;

        if (tail_partition_count_ > 0u) {
            input_fifo_.Write(&in, 1u);
            while (input_fifo_.AvailableRead() >= kTailBlock) {
                ProcessTailBlock();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ProcessTailBlock — OLS block, R=256, B=128.
//
//   fft_in_ = [prev_B | curr_B].  Each partition is B-tap, so circular
//   aliasing affects output[0..B-2]; discarding the first B samples is
//   sufficient.  The clean output[B..2B-1] corresponds to the input block
//   B samples ago — the exact causal offset already provided by the head FIR.
// ---------------------------------------------------------------------------

void PConvZeroLatency::ProcessTailBlock() noexcept {
    // 1. Build 2B overlap-save window: [prev_B(128) | curr_B(128)]
    std::copy_n(prev_input_.data(), kTailBlock, fft_in_);
    input_fifo_.Read(fft_in_ + kTailBlock, kTailBlock);
    // Save the new block as the next cycle's overlap.
    std::copy_n(fft_in_ + kTailBlock, kTailBlock, prev_input_.data());

    // 2. Forward FFT → current FDL slot.
    pffft_transform(tail_fft_setup_, fft_in_, fdl_[fdl_idx_], fft_work_, PFFFT_FORWARD);

    // 3. Frequency-domain multiply-accumulate over all J partitions.
    std::fill_n(accum_, kTailFFTSz, 0.0f);
    for (uint32_t j = 0; j < tail_partition_count_; ++j) {
        const uint32_t slot = (fdl_idx_ - j + tail_partition_count_) % tail_partition_count_;
        pffft_zconvolve_accumulate(
            tail_fft_setup_,
            fdl_[slot],
            filter_segs_[j],
            accum_,
            1.0f
        );
    }

    // 4. Inverse FFT → time domain.
    pffft_transform(tail_fft_setup_, accum_, fft_out_, fft_work_, PFFFT_BACKWARD);

    // 5. OLS: discard first B=128 (aliased), keep last B=128 (clean).
    //    PFFFT IFFT is unnormalised — apply 1/R. Denormal flush for silent tails.
    const float norm   = 1.0f / static_cast<float>(kTailFFTSz);
    const float denorm = 1e-25f;
    for (uint32_t i = 0; i < kTailBlock; ++i) {
        fft_out_[kTailBlock + i] = fft_out_[kTailBlock + i] * norm + denorm - denorm;
    }
    output_fifo_.Write(fft_out_ + kTailBlock, kTailBlock);

    // 6. Advance FDL ring.
    fdl_idx_ = (fdl_idx_ + 1u) % tail_partition_count_;
}
