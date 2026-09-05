#include "Convolver.h"
#include "../utils/Crc32.h"
#include "../utils/WavReader.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

// Scales the IR in-place so its peak absolute value == 1.0f.
// Returns false for silent IRs (peak < 1e-6f) — caller must reject them.
// Applied on every user-supplied IR path to guard against explosive feedback
// from unnormalised WAVs or DC-biased recordings.
[[nodiscard]] static bool NormalizePeak(float* const data, const uint32_t size) noexcept {
    float peak = 0.0f;
    for (uint32_t i = 0; i < size; ++i) {
        const float v = std::fabs(data[i]);
        if (v > peak) peak = v;
    }
    if (peak < 1e-6f) return false;
    if (peak > 1.0f) {
        const float inv = 1.0f / peak;
        for (uint32_t i = 0; i < size; ++i) data[i] *= inv;
    }
    return true;
}

static void ApplyCrossChannel(float* const buf, const uint32_t n, const float cc) noexcept {
    for (uint32_t i = 0; i < n; ++i) {
        const float L = buf[i * 2];
        const float R = buf[i * 2 + 1];
        buf[i * 2]     = std::lerp(L, R, cc);
        buf[i * 2 + 1] = std::lerp(R, L, cc);
    }
}

Convolver::Convolver() = default;

// ---------------------------------------------------------------------------
// Lock-free kernel swap (audio thread)
// ---------------------------------------------------------------------------

void Convolver::ConsumeKernelSwap() noexcept {
    // acquire load pairs with the release store in CommitToStaging().
    // Intentionally NOT seq_cst — a full fence on every audio callback
    // would be unnecessary overhead with no correctness benefit.
    if (!kernel_swap_pending_.load(std::memory_order_acquire)) return;

    std::swap(kernel_ch1_, staging_ch1_);
    std::swap(kernel_ch2_, staging_ch2_);

    // release store: retiring kernels in staging_ are now visible to the
    // binder thread, which will free them on its next CommitToStaging() call.
    kernel_swap_pending_.store(false, std::memory_order_release);
}

// Binder thread: serialize callers, yield until any in-flight swap is consumed
// (≤ 1 audio callback ≈ 1–10 ms), then publish the new kernel pair atomically.
void Convolver::CommitToStaging(std::unique_ptr<PConvNUPC> ch1,
                                 std::unique_ptr<PConvNUPC> ch2) {
    const std::lock_guard lock(kernel_stage_mutex_);

    // acquire spin-wait pairs with the release store in ConsumeKernelSwap().
    while (kernel_swap_pending_.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    staging_ch1_ = std::move(ch1);
    staging_ch2_ = std::move(ch2);

    // release store: staging_ writes are now visible to the audio thread's
    // acquire load in ConsumeKernelSwap().
    kernel_swap_pending_.store(true, std::memory_order_release);
}

// ---------------------------------------------------------------------------
// Audio-thread DSP path
// ---------------------------------------------------------------------------

uint32_t Convolver::Process(
    const float* const source, float* const dest, const uint32_t frame_size
) {
    // Only place active kernels are mutated on the audio thread.
    ConsumeKernelSwap();

    if (!enable_ || !kernel_ch1_->InstanceUsable() || !kernel_ch2_->InstanceUsable()
        || frame_size == 0) {
        if (source != dest && frame_size > 0) {
            std::copy_n(source, frame_size * 2u, dest);
        }
        return frame_size;
    }

    if (source != dest) {
        std::copy_n(source, frame_size * 2u, dest);
    }

    kernel_ch1_->ProcessInterleaved(dest, dest, 0u, 2u, frame_size);
    kernel_ch2_->ProcessInterleaved(dest, dest, 1u, 2u, frame_size);

    if (is_valid_cross_channel_) {
        ApplyCrossChannel(dest, frame_size, cross_channel_);
    }

    return frame_size;
}

void Convolver::ProcessPlanar(std::span<float> L, std::span<float> R) noexcept {
    // Kernel swap check is handled in the interleaved Process(); mirror it here.
    ConsumeKernelSwap();

    if (!enable_ || !kernel_ch1_->InstanceUsable() || !kernel_ch2_->InstanceUsable()
        || L.empty()) {
        return;  // pass-through: planar buffers already contain the input
    }
    const auto frame_size = static_cast<uint32_t>(L.size());

    // Direct contiguous convolution — no interleave/deinterleave overhead.
    kernel_ch1_->Process(L.data(), L.data(), frame_size);
    kernel_ch2_->Process(R.data(), R.data(), frame_size);

    if (is_valid_cross_channel_) {
        // Planar cross-channel blend: blend = lerp(L, R, cc) and vice-versa.
        const float cc = cross_channel_;
        for (uint32_t i = 0; i < frame_size; ++i) {
            const float l = L[i];
            const float r = R[i];
            L[i] = l + (r - l) * cc;
            R[i] = r + (l - r) * cc;
        }
    }
}

void Convolver::Reset() {
    if (kernel_ch1_) kernel_ch1_->Reset();
    if (kernel_ch2_) kernel_ch2_->Reset();
}

bool     Convolver::GetEnable()   const noexcept { return enable_;    }
uint32_t Convolver::GetKernelID() const noexcept { return kernel_id_; }

void Convolver::SetConfig(const Config& config) noexcept {
    config_ = config;
    SetEnable(config.enable);
    SetCrossChannel(config.cross_channel);
}

void Convolver::SetEnable(const bool enable) {
    config_.enable = enable;
    if (enable_ != enable) {
        if (enable) Reset();
        enable_ = enable;
    }
}

// ---------------------------------------------------------------------------
// Kernel loading (binder thread)
// ---------------------------------------------------------------------------

void Convolver::SetKernel(const char* const path) {
    if (!path || path[0] == '\0') return;
    if (kernel_file_path_ == path) return;

    current_kernel_buffer_crc_ = 0;

    WavData wav;
    if (!ReadWavFile(path, wav) || wav.frame_count < 16
        || wav.channels == 0 || wav.channels > 2) {
        kernel_file_path_.clear();
        kernel_id_ = 0;
        CommitToStaging(std::make_unique<PConvNUPC>(), std::make_unique<PConvNUPC>());
        return;
    }

    auto ch1 = std::make_unique<PConvNUPC>();
    auto ch2 = std::make_unique<PConvNUPC>();
    bool success = false;

    if (wav.channels == 1) {
        if (NormalizePeak(wav.samples.get(), wav.frame_count)) {
            success = ch1->LoadKernel(wav.samples.get(), wav.frame_count)
                   && ch2->LoadKernel(wav.samples.get(), wav.frame_count);
        }
    } else {
        auto buf1 = std::make_unique<float[]>(wav.frame_count);
        auto buf2 = std::make_unique<float[]>(wav.frame_count);
        for (uint32_t i = 0; i < wav.frame_count; ++i) {
            buf1[i] = wav.samples[i * 2];
            buf2[i] = wav.samples[i * 2 + 1];
        }
        // Normalise channels independently to preserve stereo balance.
        const bool n1 = NormalizePeak(buf1.get(), wav.frame_count);
        const bool n2 = NormalizePeak(buf2.get(), wav.frame_count);
        if (n1 && n2) {
            success = ch1->LoadKernel(buf1.get(), wav.frame_count)
                   && ch2->LoadKernel(buf2.get(), wav.frame_count);
        }
    }

    if (success) {
        kernel_file_path_          = path;
        kernel_id_                 = 0;
        current_kernel_buffer_crc_ = 0;
    } else {
        kernel_file_path_.clear();
        kernel_id_ = 0;
        ch1 = std::make_unique<PConvNUPC>();
        ch2 = std::make_unique<PConvNUPC>();
    }

    CommitToStaging(std::move(ch1), std::move(ch2));
}

void Convolver::SetKernel(const float* const buf, const uint32_t size) {
    if (size < 16) return;

    // Work on a local copy so NormalizePeak() does not mutate the caller's buffer.
    auto tmp = std::make_unique<float[]>(size);
    std::copy_n(buf, size, tmp.get());

    auto ch1 = std::make_unique<PConvNUPC>();
    auto ch2 = std::make_unique<PConvNUPC>();

    if (NormalizePeak(tmp.get(), size)) {
        if (!ch1->LoadKernel(tmp.get(), size) || !ch2->LoadKernel(tmp.get(), size)) {
            ch1 = std::make_unique<PConvNUPC>();
            ch2 = std::make_unique<PConvNUPC>();
        }
    }

    kernel_id_                 = 0;
    current_kernel_buffer_crc_ = 0;
    CommitToStaging(std::move(ch1), std::move(ch2));
}

void Convolver::SetKernelStereo(
    const float* const ch_l, const float* const ch_r, const uint32_t frame_count
) {
    if (frame_count < 16) return;

    auto buf1 = std::make_unique<float[]>(frame_count);
    auto buf2 = std::make_unique<float[]>(frame_count);
    std::copy_n(ch_l, frame_count, buf1.get());
    std::copy_n(ch_r, frame_count, buf2.get());

    auto ch1 = std::make_unique<PConvNUPC>();
    auto ch2 = std::make_unique<PConvNUPC>();

    // Normalise channels independently to preserve stereo balance.
    const bool n1 = NormalizePeak(buf1.get(), frame_count);
    if (const bool n2 = NormalizePeak(buf2.get(), frame_count); n1 && n2) {
        if (!ch1->LoadKernel(buf1.get(), frame_count)
            || !ch2->LoadKernel(buf2.get(), frame_count)) {
            ch1 = std::make_unique<PConvNUPC>();
            ch2 = std::make_unique<PConvNUPC>();
        }
    }

    kernel_id_                 = 0;
    current_kernel_buffer_crc_ = 0;
    CommitToStaging(std::move(ch1), std::move(ch2));
}

void Convolver::SetCrossChannel(float value) {
    config_.cross_channel = value;
    if (value <= 0.0f) {
        is_valid_cross_channel_ = false;
        return;
    }
    cross_channel_          = std::min(value, 1.0f);
    is_valid_cross_channel_ = true;
}

void Convolver::SetSamplingRate(const uint32_t sampling_rate) {
    sampling_rate_ = sampling_rate;
    // A sample-rate change invalidates the loaded filter stages (they were
    // designed for the old rate).  Reset the CRC guard so the next commit of
    // the same IR bytes is NOT skipped as a duplicate — it must rebuild the
    // stages at the new rate.
    current_kernel_buffer_crc_ = 0;
}

// ---------------------------------------------------------------------------
// Chunked IR receive buffer
// ---------------------------------------------------------------------------

void Convolver::PrepareKernelBuffer(
    const uint32_t buf_size, const uint32_t ch_count, const bool reset
) {
    if (!reset) {
        if (ch_count - 1 < 2 && buf_size > 0) {
            kernel_buffer_.assign(buf_size, 0.0f);
            expected_size_ = buf_size;
            current_size_  = 0;
            channel_count_ = ch_count;
        }
    } else {
        kernel_buffer_.clear();
        kernel_buffer_.shrink_to_fit();
        expected_size_             = 0;
        current_size_              = 0;
        channel_count_             = 0;
        current_kernel_buffer_crc_ = 0;
        kernel_file_path_.clear();
        kernel_id_ = 0;
        CommitToStaging(std::make_unique<PConvNUPC>(), std::make_unique<PConvNUPC>());
    }
}

void Convolver::SetKernelBuffer(const float* const buf, uint32_t size) {
    if (!buf || size == 0 || kernel_buffer_.empty() || expected_size_ == 0) return;
    if (current_size_ >= expected_size_) return;
    size = std::min(size, expected_size_ - current_size_);
    std::copy_n(buf, size, kernel_buffer_.data() + current_size_);
    current_size_ += size;
}

void Convolver::CommitKernelBuffer(
    const uint32_t expected_size, const uint32_t expected_crc, const uint32_t kernel_id
) {
    if (kernel_buffer_.empty() || expected_size_ != expected_size || current_size_ != expected_size_) {
        ClearKernelBuffer();
        return;
    }

    const auto* const raw = reinterpret_cast<const std::byte*>(kernel_buffer_.data());
    const uint32_t calculated_crc = Crc32(reinterpret_cast<const uint8_t*>(raw), current_size_ * 4);
    if (channel_count_ - 1 > 1 || (expected_crc != 0 && calculated_crc != expected_crc)) {
        ClearKernelBuffer();
        return;
    }

    // Same CRC as the currently-loaded kernel: the caller is re-committing the
    // same IR (e.g. after a sample-rate reset or preset re-apply).  The current
    // kernel is already staged and valid — keep it, but still adopt the new
    // kernel_id so LoadConvolverKernel()'s GetKernelID() check succeeds.
    if (calculated_crc == current_kernel_buffer_crc_) {
        kernel_id_ = kernel_id;
        ClearKernelBuffer();
        return;
    }

    current_kernel_buffer_crc_ = calculated_crc;

    const uint32_t frames_per_channel = current_size_ / channel_count_;

    auto ch1 = std::make_unique<PConvNUPC>();
    auto ch2 = std::make_unique<PConvNUPC>();
    bool loaded = false;

    if (channel_count_ == 1) {
        auto tmp = std::make_unique<float[]>(frames_per_channel);
        std::copy_n(kernel_buffer_.data(), frames_per_channel, tmp.get());
        if (NormalizePeak(tmp.get(), frames_per_channel)) {
            loaded = ch1->LoadKernel(tmp.get(), frames_per_channel)
                  && ch2->LoadKernel(tmp.get(), frames_per_channel);
        }
    } else {
        auto buf1 = std::make_unique<float[]>(frames_per_channel);
        auto buf2 = std::make_unique<float[]>(frames_per_channel);
        for (uint32_t i = 0; i < frames_per_channel; ++i) {
            buf1[i] = kernel_buffer_[i * 2];
            buf2[i] = kernel_buffer_[i * 2 + 1];
        }
        const bool n1 = NormalizePeak(buf1.get(), frames_per_channel);
        const bool n2 = NormalizePeak(buf2.get(), frames_per_channel);
        if (n1 && n2) {
            loaded = ch1->LoadKernel(buf1.get(), frames_per_channel)
                  && ch2->LoadKernel(buf2.get(), frames_per_channel);
        }
    }

    if (!loaded) {
        current_kernel_buffer_crc_ = 0;
        kernel_id_ = 0;
        ch1 = std::make_unique<PConvNUPC>();
        ch2 = std::make_unique<PConvNUPC>();
    } else {
        kernel_file_path_.clear();
        kernel_id_ = kernel_id;
    }

    ClearKernelBuffer();
    CommitToStaging(std::move(ch1), std::move(ch2));
}

void Convolver::ClearKernelBuffer() noexcept {
    kernel_buffer_.clear();
    expected_size_ = 0;
    current_size_  = 0;
    channel_count_ = 0;
}
