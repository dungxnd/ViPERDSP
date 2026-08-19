#include "Convolver.h"
#include "../utils/Crc32.h"
#include "../utils/WavReader.h"
#include <algorithm>
#include <cstddef>
#include <vector>

static constexpr int kConvSegmentSize = 0x1000;

static void ApplyCrossChannel(float* const buf, const uint32_t n, const float cc) noexcept {
    for (uint32_t i = 0; i < n; ++i) {
        const float L = buf[i * 2];
        const float R = buf[i * 2 + 1];
        buf[i * 2]     = L + cc * (R - L);
        buf[i * 2 + 1] = R + cc * (L - R);
    }
}

uint32_t Convolver::Process(
    const float* const source, float* const dest, const uint32_t frame_size
) {
    if (enable_ && kernel_ch1_.InstanceUsable() && kernel_ch2_.InstanceUsable()) {
        constexpr auto seg = static_cast<uint32_t>(kConvSegmentSize);
        for (uint32_t off = 0; off < frame_size; off += seg) {
            const uint32_t n = std::min(frame_size - off, seg);

            if (source != dest) {
                std::copy_n(source + off * 2, n * 2, dest + off * 2);
            }
            float* const buf_ptr = dest + off * 2;

            kernel_ch1_.ConvolveInterleaved(buf_ptr, 0, n);
            kernel_ch2_.ConvolveInterleaved(buf_ptr, 1, n);

            if (is_valid_cross_channel_) {
                ApplyCrossChannel(buf_ptr, n, cross_channel_);
            }
        }
    }
    return frame_size;
}

void Convolver::Reset() {
    kernel_ch1_.Reset();
    kernel_ch2_.Reset();
}

bool     Convolver::GetEnable()   const noexcept { return enable_;    }
uint32_t Convolver::GetKernelID() const noexcept { return kernel_id_; }

void Convolver::SetEnable(const bool enable) {
    if (enable_ != enable) {
        if (enable) Reset();
        enable_ = enable;
    }
}

void Convolver::SetKernel(const char* const path) {
    if (!path || path[0] == '\0') return;
    if (kernel_file_path_ == path) return;

    kernel_ch1_.Reset(); kernel_ch2_.Reset();
    kernel_ch3_.Reset(); kernel_ch4_.Reset();
    kernel_ch1_.UnloadKernel(); kernel_ch2_.UnloadKernel();
    kernel_ch3_.UnloadKernel(); kernel_ch4_.UnloadKernel();
    is_quad_channel_           = 0;
    current_kernel_buffer_crc_ = 0;

    WavData wav;
    if (!ReadWavFile(path, wav)) {
        kernel_file_path_.clear();
        kernel_id_ = 0;
        Reset();
        return;
    }

    if (wav.frame_count < 16 || wav.channels == 0 || wav.channels > 2) {
        kernel_file_path_.clear();
        kernel_id_ = 0;
        Reset();
        return;
    }

    bool success;
    if (wav.channels == 1) {
        const uint32_t ret1 = kernel_ch1_.LoadKernel(wav.samples.get(), wav.frame_count, kConvSegmentSize);
        const uint32_t ret2 = kernel_ch2_.LoadKernel(wav.samples.get(), wav.frame_count, kConvSegmentSize);
        success = ret1 != 0 && ret2 != 0;
    } else {
        auto ch1 = std::make_unique<float[]>(wav.frame_count);
        auto ch2 = std::make_unique<float[]>(wav.frame_count);
        for (uint32_t i = 0; i < wav.frame_count; i++) {
            ch1[i] = wav.samples[i * 2];
            ch2[i] = wav.samples[i * 2 + 1];
        }
        const uint32_t ret1 = kernel_ch1_.LoadKernel(ch1.get(), wav.frame_count, kConvSegmentSize);
        const uint32_t ret2 = kernel_ch2_.LoadKernel(ch2.get(), wav.frame_count, kConvSegmentSize);
        success = ret1 != 0 && ret2 != 0;
    }

    if (success) {
        is_quad_channel_           = 1;
        kernel_file_path_          = path;
        kernel_id_                 = 0;
        current_kernel_buffer_crc_ = 0;
        Reset();
    } else {
        kernel_ch1_.UnloadKernel(); kernel_ch2_.UnloadKernel();
        kernel_ch3_.UnloadKernel(); kernel_ch4_.UnloadKernel();
        kernel_file_path_.clear();
        kernel_id_ = 0;
        Reset();
    }
}

void Convolver::SetKernel(const float* const buf, const uint32_t size) {
    if (size < 16) return;
    kernel_ch1_.Reset();
    kernel_ch2_.Reset();
    const uint32_t ret1 = kernel_ch1_.LoadKernel(buf, size, kConvSegmentSize);
    const uint32_t ret2 = kernel_ch2_.LoadKernel(buf, size, kConvSegmentSize);
    if (ret1 == 0 || ret2 == 0) {
        kernel_ch1_.UnloadKernel();
        kernel_ch2_.UnloadKernel();
    }
    kernel_id_                 = 0;
    current_kernel_buffer_crc_ = 0;
    Reset();
}

void Convolver::SetKernelBuffer(const float* const buf, uint32_t size) {
    if (!buf || size == 0 || kernel_buffer_.empty() || expected_size_ == 0) return;
    if (current_size_ >= expected_size_) return;
    size = std::min(size, expected_size_ - current_size_);
    std::copy_n(buf, size, kernel_buffer_.data() + current_size_);
    current_size_ += size;
}

void Convolver::SetKernelStereo(
    const float* const ch_l, const float* const ch_r, const uint32_t frame_count
) {
    if (frame_count < 16) return;
    kernel_ch1_.Reset();
    kernel_ch2_.Reset();
    const uint32_t ret1 = kernel_ch1_.LoadKernel(ch_l, frame_count, kConvSegmentSize);
    const uint32_t ret2 = kernel_ch2_.LoadKernel(ch_r, frame_count, kConvSegmentSize);
    if (ret1 == 0 || ret2 == 0) {
        kernel_ch1_.UnloadKernel();
        kernel_ch2_.UnloadKernel();
    }
    kernel_id_                 = 0;
    current_kernel_buffer_crc_ = 0;
    Reset();
}

void Convolver::SetCrossChannel(float value) {
    if (value <= 0.0f) {
        is_valid_cross_channel_ = false;
        return;
    }
    cross_channel_          = std::min(value, 1.0f);
    is_valid_cross_channel_ = true;
}

void Convolver::SetSamplingRate(const uint32_t sampling_rate) {
    sampling_rate_ = sampling_rate;
}

void Convolver::PrepareKernelBuffer(
    const uint32_t buf_size, const uint32_t ch_count, const bool reset
) {
    if (!reset) {
        if (ch_count - 1 < 2 && buf_size > 0) {
            kernel_buffer_.assign(buf_size, 0.0f);
            expected_size_  = buf_size;
            current_size_   = 0;
            channel_count_  = ch_count;
        }
    } else {
        kernel_buffer_.clear();
        kernel_buffer_.shrink_to_fit();
        expected_size_             = 0;
        current_size_              = 0;
        channel_count_             = 0;
        current_kernel_buffer_crc_ = 0;
        kernel_ch1_.Reset(); kernel_ch1_.UnloadKernel();
        kernel_ch2_.Reset(); kernel_ch2_.UnloadKernel();
        kernel_file_path_.clear();
        kernel_id_ = 0;
    }
}

void Convolver::CommitKernelBuffer(
    const uint32_t expected_size, const uint32_t expected_crc, const uint32_t kernel_id
) {
    if (kernel_buffer_.empty() || expected_size_ != expected_size || current_size_ == 0) {
        ClearKernelBuffer();
        return;
    }

    const uint32_t calculated_crc =
        Crc32(reinterpret_cast<const uint8_t*>(  // NOLINT: Crc32 requires uint8_t*
              reinterpret_cast<const std::byte*>(kernel_buffer_.data())),
              current_size_ * 4);
    if (channel_count_ - 1 > 1 || calculated_crc != expected_crc
        || calculated_crc == current_kernel_buffer_crc_) {
        ClearKernelBuffer();
        return;
    }

    current_kernel_buffer_crc_ = calculated_crc;

    const uint32_t frames_per_channel = current_size_ / channel_count_;
    bool loaded;

    if (channel_count_ == 1) {
        const uint32_t ret1 = kernel_ch1_.LoadKernel(kernel_buffer_.data(), frames_per_channel, kConvSegmentSize);
        const uint32_t ret2 = kernel_ch2_.LoadKernel(kernel_buffer_.data(), frames_per_channel, kConvSegmentSize);
        loaded = ret1 != 0 && ret2 != 0;
    } else {
        auto ch1 = std::make_unique<float[]>(frames_per_channel);
        auto ch2 = std::make_unique<float[]>(frames_per_channel);
        for (uint32_t i = 0; i < frames_per_channel; ++i) {
            ch1[i] = kernel_buffer_[i * 2];
            ch2[i] = kernel_buffer_[i * 2 + 1];
        }
        const uint32_t ret1 = kernel_ch1_.LoadKernel(ch1.get(), frames_per_channel, kConvSegmentSize);
        const uint32_t ret2 = kernel_ch2_.LoadKernel(ch2.get(), frames_per_channel, kConvSegmentSize);
        loaded = ret1 != 0 && ret2 != 0;
    }

    if (!loaded) {
        kernel_ch1_.UnloadKernel();
        kernel_ch2_.UnloadKernel();
        current_kernel_buffer_crc_ = 0;
        kernel_id_ = 0;
        ClearKernelBuffer();
        Reset();
        return;
    }

    kernel_file_path_.clear();
    kernel_id_ = kernel_id;
    ClearKernelBuffer();
    Reset();
}

void Convolver::ClearKernelBuffer() noexcept {
    kernel_buffer_.clear();
    expected_size_ = 0;
    current_size_  = 0;
    channel_count_ = 0;
}
