#include "PConvSingle.h"
#include "pffft.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <ranges>

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

PConvSingle::~PConvSingle() {
    ReleaseResources();
}

void PConvSingle::ReleaseResources() noexcept {
    for (float* p : filter_segments_) {
        if (p) pffft_aligned_free(p);
    }
    filter_segments_.clear();

    for (float* p : fdl_segments_) {
        if (p) pffft_aligned_free(p);
    }
    fdl_segments_.clear();

    auto free_aligned = [](float*& p) noexcept {
        if (p) { pffft_aligned_free(p); p = nullptr; }
    };
    free_aligned(fft_in_);
    free_aligned(fft_out_);
    free_aligned(fft_work_);
    free_aligned(accum_spectrum_);

    if (fft_setup_) {
        pffft_destroy_setup(fft_setup_);
        fft_setup_ = nullptr;
    }

    instance_usable_ = false;
    segment_count_   = 0;
    segment_size_    = 0;
    fft_size_        = 0;
    fdl_index_       = 0;
}

void PConvSingle::UnloadKernel() noexcept {
    ReleaseResources();
}

// ---------------------------------------------------------------------------
// Reset — clear all state, pre-fill output FIFO to establish pipeline delay
// ---------------------------------------------------------------------------

void PConvSingle::Reset() noexcept {
    if (!instance_usable_) return;

    fdl_index_ = 0;
    std::ranges::fill(prev_input_, 0.0f);

    for (uint32_t p = 0; p < segment_count_; ++p) {
        std::fill_n(fdl_segments_[p], fft_size_, 0.0f);
    }

    input_fifo_.Reset();
    output_fifo_.Reset();

    // Pre-load the output FIFO with one block of silence so that the first
    // call to Process/ProcessInterleaved can immediately pop L samples back.
    // This implements the mandatory L-sample pipeline latency of UPOLS.
    output_fifo_.WriteZeros(segment_size_);
}

// ---------------------------------------------------------------------------
// Kernel loading
// ---------------------------------------------------------------------------

uint32_t PConvSingle::LoadKernel(const float* const kernel, const uint32_t kernel_size,
                                 const uint32_t segment_size) {
    return LoadKernel(kernel, 1.0f, kernel_size, segment_size);
}

uint32_t PConvSingle::LoadKernel(const float* const kernel, const float gain,
                                 const uint32_t kernel_size, const uint32_t segment_size) {
    if (!kernel || kernel_size < 2 || segment_size < 32 || !std::has_single_bit(segment_size)) {
        return 0;
    }

    ReleaseResources();

    segment_size_  = segment_size;
    fft_size_      = segment_size * 2u;
    segment_count_ = (kernel_size + segment_size - 1u) / segment_size;

    fft_setup_ = pffft_new_setup(static_cast<int>(fft_size_), PFFFT_REAL);
    if (!fft_setup_) return 0;

    fft_in_         = static_cast<float*>(pffft_aligned_malloc(fft_size_ * sizeof(float)));
    fft_out_        = static_cast<float*>(pffft_aligned_malloc(fft_size_ * sizeof(float)));
    fft_work_       = static_cast<float*>(pffft_aligned_malloc(fft_size_ * sizeof(float)));
    accum_spectrum_ = static_cast<float*>(pffft_aligned_malloc(fft_size_ * sizeof(float)));

    if (!fft_in_ || !fft_out_ || !fft_work_ || !accum_spectrum_) {
        ReleaseResources();
        return 0;
    }

    prev_input_.assign(segment_size_, 0.0f);
    filter_segments_.resize(segment_count_, nullptr);
    fdl_segments_.resize(segment_count_, nullptr);

    for (uint32_t p = 0; p < segment_count_; ++p) {
        filter_segments_[p] = static_cast<float*>(pffft_aligned_malloc(fft_size_ * sizeof(float)));
        fdl_segments_[p]    = static_cast<float*>(pffft_aligned_malloc(fft_size_ * sizeof(float)));
        if (!filter_segments_[p] || !fdl_segments_[p]) {
            ReleaseResources();
            return 0;
        }
        std::fill_n(fdl_segments_[p], fft_size_, 0.0f);

        // Build one IR partition: [h[p*L .. p*L+L-1], 0 ... 0]
        std::fill_n(fft_in_, fft_size_, 0.0f);
        const uint32_t offset     = p * segment_size_;
        const uint32_t copy_count = std::min(kernel_size - offset, segment_size_);
        for (uint32_t i = 0; i < copy_count; ++i) {
            fft_in_[i] = kernel[offset + i] * gain;
        }
        pffft_transform(fft_setup_, fft_in_, filter_segments_[p], fft_work_, PFFFT_FORWARD);
    }

    // Allocate FIFOs with at least 4× partition size capacity
    const std::size_t fifo_cap = std::max<std::size_t>(segment_size_ * 4u, 8192u);
    input_fifo_.Init(fifo_cap);
    output_fifo_.Init(fifo_cap);

    instance_usable_ = true;
    Reset();
    return segment_count_;
}

// ---------------------------------------------------------------------------
// Core Processing
// ---------------------------------------------------------------------------

void PConvSingle::Process(const float* const input, float* const output,
                          const uint32_t n) noexcept {
    if (!instance_usable_ || n == 0) {
        if (input != output) std::copy_n(input, n, output);
        return;
    }

    // Chunk input in segment_size_ steps so the FIFO never overflows
    // regardless of how large n is (offline / batch hosts may pass n >> L).
    uint32_t processed = 0;
    while (processed < n) {
        const uint32_t chunk = std::min(n - processed, segment_size_);
        input_fifo_.Write(input + processed, chunk);

        while (input_fifo_.AvailableRead() >= segment_size_) {
            ProcessBlock();
        }

        const std::size_t avail   = output_fifo_.AvailableRead();
        const std::size_t to_read = std::min<std::size_t>(chunk, avail);
        output_fifo_.Read(output + processed, to_read);

        if (to_read < chunk) {
            std::fill_n(output + processed + to_read, chunk - to_read, 0.0f);
        }

        processed += chunk;
    }
}

void PConvSingle::ProcessInterleaved(const float* const input, float* const output,
                                     const uint32_t channel, const uint32_t stride,
                                     const uint32_t n) noexcept {
    if (!instance_usable_ || n == 0) return;

    // Deinterleave channel into input FIFO
    for (uint32_t i = 0; i < n; ++i) {
        const float s = input[i * stride + channel];
        input_fifo_.Write(&s, 1u);
    }

    while (input_fifo_.AvailableRead() >= segment_size_) {
        ProcessBlock();
    }

    // Reinterleave output FIFO back into the output buffer
    for (uint32_t i = 0; i < n; ++i) {
        if (output_fifo_.AvailableRead() > 0u) {
            float s = 0.0f;
            output_fifo_.Read(&s, 1u);
            output[i * stride + channel] = s;
        } else {
            output[i * stride + channel] = 0.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// Single UPOLS block — called when input_fifo_ has >= L samples
// ---------------------------------------------------------------------------

void PConvSingle::ProcessBlock() noexcept {
    // 1. Construct the 2L time-domain block: [x_prev(L), x_curr(L)]
    std::copy_n(prev_input_.data(), segment_size_, fft_in_);
    input_fifo_.Read(fft_in_ + segment_size_, segment_size_);
    // Save current block as next iteration's "previous" L samples
    std::copy_n(fft_in_ + segment_size_, segment_size_, prev_input_.data());

    // 2. Forward FFT → store in current FDL slot
    pffft_transform(fft_setup_, fft_in_, fdl_segments_[fdl_index_], fft_work_, PFFFT_FORWARD);

    // 3. Frequency-domain multiply-accumulate across all P partitions
    std::fill_n(accum_spectrum_, fft_size_, 0.0f);
    for (uint32_t p = 0; p < segment_count_; ++p) {
        const uint32_t slot = (fdl_index_ - p + segment_count_) % segment_count_;
        pffft_zconvolve_accumulate(
            fft_setup_,
            fdl_segments_[slot],
            filter_segments_[p],
            accum_spectrum_,
            1.0f
        );
    }

    // 4. Inverse FFT → time domain
    pffft_transform(fft_setup_, accum_spectrum_, fft_out_, fft_work_, PFFFT_BACKWARD);

    // 5. Overlap-Save: discard first L (aliased), keep second L [L .. 2L-1],
    //    applying 1/N normalisation required by PFFFT unnormalised IFFT.
    const float norm = 1.0f / static_cast<float>(fft_size_);
    for (uint32_t i = segment_size_; i < fft_size_; ++i) {
        fft_out_[i] *= norm;
    }
    output_fifo_.Write(fft_out_ + segment_size_, segment_size_);

    // 6. Advance the FDL circular index
    fdl_index_ = (fdl_index_ + 1u) % segment_count_;
}
