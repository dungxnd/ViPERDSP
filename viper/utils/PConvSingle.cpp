#include "PConvSingle.h"
#include "pffft.h"
#include <algorithm>
#include <bit>
#include <cstring>

PConvSingle::~PConvSingle() {
    ReleaseResources();
}

void PConvSingle::Reset() {
    if (!instance_usable_) return;

    for (uint32_t i = 0; i < segment_count_; ++i) {
        std::fill_n(input_history_[i], fft_size_, 0.0f);
    }
    std::fill_n(overlap_buffer_, segment_size_, 0.0f);
    std::fill_n(fft_buffer_,     fft_size_,     0.0f);
    std::fill_n(accum_buffer_,   fft_size_,     0.0f);
    std::fill_n(mono_buffer_,    segment_size_, 0.0f);
    std::fill_n(fft_work_,       fft_size_,     0.0f);
    delay_line_index_ = 0;
    input_fill_       = 0;
}

uint32_t PConvSingle::GetFFTSize()      const noexcept { return segment_size_ * 2; }
uint32_t PConvSingle::GetSegmentCount() const noexcept { return segment_count_;    }
uint32_t PConvSingle::GetSegmentSize()  const noexcept { return segment_size_;     }
bool     PConvSingle::InstanceUsable()  const noexcept { return instance_usable_;  }

void PConvSingle::Convolve(float* const buffer, const uint32_t n) {
    ConvSegment(buffer, false, 0, n);
}

void PConvSingle::ConvolveInterleaved(float* const buffer, const int channel, const uint32_t n) {
    ConvSegment(buffer, true, channel, n);
}

void PConvSingle::ConvSegment(
    float* const buffer, const bool interleaved, const int channel, const uint32_t n
) {
    if (!instance_usable_ || n == 0) return;

    const uint32_t stride = interleaved ? 2u : 1u;
    uint32_t done = 0;
    while (done < n) {
        const uint32_t room  = segment_size_ - input_fill_;
        const uint32_t chunk = std::min(n - done, room);
        ConvChunk(buffer + done * stride, interleaved, channel, chunk);
        done += chunk;
    }
}

void PConvSingle::ConvChunk(
    float* const buffer, const bool interleaved, const int channel, const uint32_t n
) {
    for (uint32_t i = 0; i < n; ++i) {
        mono_buffer_[input_fill_ + i] =
            interleaved ? buffer[i * 2 + channel] : buffer[i];
    }

    std::copy_n(overlap_buffer_,     segment_size_,         fft_buffer_);
    std::copy_n(mono_buffer_,        input_fill_ + n,       fft_buffer_ + segment_size_);
    std::fill_n(fft_buffer_ + segment_size_ + input_fill_ + n,
                segment_size_ - (input_fill_ + n), 0.0f);

    pffft_transform(fft_setup_, fft_buffer_, input_history_[delay_line_index_], fft_work_, PFFFT_FORWARD);

    std::fill_n(accum_buffer_, fft_size_, 0.0f);
    for (uint32_t k = 0; k < segment_count_; ++k) {
        const uint32_t idx = (delay_line_index_ - k + segment_count_) % segment_count_;
        pffft_zconvolve_accumulate(fft_setup_, input_history_[idx], filter_segments_[k], accum_buffer_, 1.0f);
    }

    pffft_transform(fft_setup_, accum_buffer_, fft_buffer_, fft_work_, PFFFT_BACKWARD);

    const float scale  = 1.0f / static_cast<float>(fft_size_);
    const float* output = fft_buffer_ + segment_size_ + input_fill_;

    if (interleaved) {
        for (uint32_t i = 0; i < n; ++i) {
            buffer[i * 2 + channel] = output[i] * scale;
        }
    } else {
        for (uint32_t i = 0; i < n; ++i) {
            buffer[i] = output[i] * scale;
        }
    }

    input_fill_ += n;

    if (input_fill_ == segment_size_) {
        std::copy_n(mono_buffer_, segment_size_, overlap_buffer_);
        delay_line_index_ = (delay_line_index_ + 1) % segment_count_;
        input_fill_ = 0;
    }
}

uint32_t PConvSingle::LoadKernel(
    const float* const kernel, const uint32_t kernel_size, const uint32_t segment_size
) {
    if (kernel && kernel_size >= 2 && segment_size >= 2
        && std::has_single_bit(segment_size)) {
        instance_usable_ = false;
        ReleaseResources();
        segment_size_ = segment_size;
        const uint32_t n = ProcessKernel(kernel, kernel_size);
        if (n != 0) {
            instance_usable_ = true;
            return n;
        }
        ReleaseResources();
    }
    return 0;
}

uint32_t PConvSingle::LoadKernel(
    const float* const kernel, const float gain,
    const uint32_t kernel_size, const uint32_t segment_size
) {
    if (kernel && kernel_size >= 2 && segment_size >= 2
        && std::has_single_bit(segment_size)) {
        instance_usable_ = false;
        ReleaseResources();
        segment_size_ = segment_size;
        const uint32_t n = ProcessKernel(kernel, gain, kernel_size);
        if (n != 0) {
            instance_usable_ = true;
            return n;
        }
        ReleaseResources();
    }
    return 0;
}

uint32_t PConvSingle::ProcessKernel(const float* const kernel, const uint32_t kernel_size) {
    fft_size_      = segment_size_ * 2;
    segment_count_ = (kernel_size + segment_size_ - 1) / segment_size_;

    fft_setup_ = pffft_new_setup(static_cast<int>(fft_size_), PFFFT_REAL);
    if (!fft_setup_) return 0;

    fft_work_       = static_cast<float*>(pffft_aligned_malloc(fft_size_     * sizeof(float)));
    fft_buffer_     = static_cast<float*>(pffft_aligned_malloc(fft_size_     * sizeof(float)));
    accum_buffer_   = static_cast<float*>(pffft_aligned_malloc(fft_size_     * sizeof(float)));
    overlap_buffer_ = static_cast<float*>(pffft_aligned_malloc(segment_size_ * sizeof(float)));
    mono_buffer_    = static_cast<float*>(pffft_aligned_malloc(segment_size_ * sizeof(float)));

    if (!fft_work_ || !fft_buffer_ || !accum_buffer_ || !overlap_buffer_ || !mono_buffer_) {
        return 0;
    }

    std::fill_n(overlap_buffer_, segment_size_, 0.0f);
    std::fill_n(mono_buffer_,    segment_size_, 0.0f);

    filter_segments_.resize(segment_count_, nullptr);
    input_history_.resize(segment_count_, nullptr);
    for (uint32_t i = 0; i < segment_count_; ++i) {
        filter_segments_[i] = static_cast<float*>(pffft_aligned_malloc(fft_size_ * sizeof(float)));
        input_history_[i]   = static_cast<float*>(pffft_aligned_malloc(fft_size_ * sizeof(float)));
        std::fill_n(input_history_[i], fft_size_, 0.0f);
    }

    for (uint32_t i = 0; i < segment_count_; ++i) {
        std::fill_n(fft_buffer_, fft_size_, 0.0f);
        const uint32_t offset = i * segment_size_;
        const uint32_t count  = std::min(kernel_size - offset, segment_size_);
        std::copy_n(kernel + offset, count, fft_buffer_);
        pffft_transform(fft_setup_, fft_buffer_, filter_segments_[i], fft_work_, PFFFT_FORWARD);
    }

    delay_line_index_ = 0;
    return segment_count_;
}

uint32_t PConvSingle::ProcessKernel(
    const float* const kernel, const float gain, const uint32_t kernel_size
) {
    auto* const scaled = static_cast<float*>(pffft_aligned_malloc(kernel_size * sizeof(float)));
    if (!scaled) return 0;
    for (uint32_t i = 0; i < kernel_size; ++i) {
        scaled[i] = kernel[i] * gain;
    }
    const uint32_t result = ProcessKernel(scaled, kernel_size);
    pffft_aligned_free(scaled);
    return result;
}

void PConvSingle::ReleaseResources() {
    for (float* p : filter_segments_) { pffft_aligned_free(p); }
    filter_segments_.clear();

    for (float* p : input_history_) { pffft_aligned_free(p); }
    input_history_.clear();

    auto free_if = [](float*& p) {
        if (p) { pffft_aligned_free(p); p = nullptr; }
    };
    free_if(overlap_buffer_);
    free_if(fft_buffer_);
    free_if(accum_buffer_);
    free_if(mono_buffer_);
    free_if(fft_work_);

    if (fft_setup_) {
        pffft_destroy_setup(fft_setup_);
        fft_setup_ = nullptr;
    }

    instance_usable_   = false;
    segment_count_     = 0;
    segment_size_      = 0;
    fft_size_          = 0;
    delay_line_index_  = 0;
}

void PConvSingle::UnloadKernel() {
    instance_usable_ = false;
    ReleaseResources();
}
