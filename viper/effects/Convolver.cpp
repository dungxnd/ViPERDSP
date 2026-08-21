#include "Convolver.h"
#include "../utils/Crc32.h"
#include "../utils/WavReader.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

// kConvSegmentSize removed: PConvNUPC auto-sizes its NUPC hierarchy.

// ---------------------------------------------------------------------------
// IR peak normalisation
// ---------------------------------------------------------------------------
// Scales data in-place so peak absolute value == 1.0f.
// Returns false if the IR is silent (peak < 1e-6f) — caller should reject it.
// Applied before LoadKernel on every user-supplied IR path to guard against
// explosive feedback from unnormalised float WAVs or DC-biased recordings.
// ---------------------------------------------------------------------------

[[nodiscard]] static bool NormalizePeak(float* const data, const uint32_t size) noexcept {
    float peak = 0.0f;
    for (uint32_t i = 0; i < size; ++i) {
        const float v = std::fabs(data[i]);
        if (v > peak) peak = v;
    }
    if (peak < 1e-6f) return false;  // silent / near-zero IR — reject
    if (peak > 1.0f) {
        const float inv = 1.0f / peak;
        for (uint32_t i = 0; i < size; ++i) {
            data[i] *= inv;
        }
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

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

Convolver::Convolver()
    : kernel_ch1_(std::make_unique<PConvNUPC>())
    , kernel_ch2_(std::make_unique<PConvNUPC>()) {
}

// ---------------------------------------------------------------------------
// Thread-safe kernel staging
// ---------------------------------------------------------------------------

// ConsumeKernelSwap — called at start of every Process().
// Audio-thread only.  No allocation, no lock, O(1).
void Convolver::ConsumeKernelSwap() noexcept {
    if (!kernel_swap_pending_.load(std::memory_order_acquire)) return;

    // Swap active ↔ staging.  3 pointer moves — no heap activity.
    std::swap(kernel_ch1_, staging_ch1_);
    std::swap(kernel_ch2_, staging_ch2_);

    // Signal binder thread: staging_ slots are now safe to overwrite.
    // The old active kernels sit in staging_ch1_/ch2_ until the next
    // CommitToStaging call (binder thread) overwrites and frees them.
    kernel_swap_pending_.store(false, std::memory_order_release);
}

// CommitToStaging — called by all binder-thread kernel mutation paths.
// Serialises concurrent binder callers via kernel_stage_mutex_.
// Spinwaits for any previous pending swap to be consumed (≤ 1 callback ≈ 10 ms).
void Convolver::CommitToStaging(std::unique_ptr<PConvNUPC> ch1,
                                 std::unique_ptr<PConvNUPC> ch2) {
    const std::lock_guard lock(kernel_stage_mutex_);

    // Yield until the audio thread has consumed the previous staging swap.
    // In practice this completes in < 1 audio callback (1–10 ms).
    while (kernel_swap_pending_.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    // Write staging — visible to audio thread after the release store below.
    staging_ch1_ = std::move(ch1);
    staging_ch2_ = std::move(ch2);

    // Release: audio thread will see the full staging_ writes on acquire load.
    kernel_swap_pending_.store(true, std::memory_order_release);
}

// ---------------------------------------------------------------------------
// Process
// ---------------------------------------------------------------------------

uint32_t Convolver::Process(
    const float* const source, float* const dest, const uint32_t frame_size
) {
    // Consume any pending kernel swap at the quiescent start of the callback.
    // This is the ONLY place active kernels are mutated on the audio thread.
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

// ---------------------------------------------------------------------------
// Reset — resets active kernel state (called on audio thread via pending flag)
// ---------------------------------------------------------------------------

void Convolver::Reset() {
    if (kernel_ch1_) kernel_ch1_->Reset();
    if (kernel_ch2_) kernel_ch2_->Reset();
}

bool     Convolver::GetEnable()   const noexcept { return enable_;    }
uint32_t Convolver::GetKernelID() const noexcept { return kernel_id_; }

void Convolver::SetEnable(const bool enable) {
    if (enable_ != enable) {
        if (enable) Reset();
        enable_ = enable;
    }
}

// ---------------------------------------------------------------------------
// Kernel loading — binder-thread paths
// All these methods build PConvNUPC instances OFF the audio thread, then
// commit them atomically via CommitToStaging().
// ---------------------------------------------------------------------------

void Convolver::SetKernel(const char* const path) {
    if (!path || path[0] == '\0') return;
    if (kernel_file_path_ == path) return;

    current_kernel_buffer_crc_ = 0;

    WavData wav;
    if (!ReadWavFile(path, wav)) {
        kernel_file_path_.clear();
        kernel_id_ = 0;
        // Push empty (unloaded) kernels so the audio thread sees a clean state.
        CommitToStaging(std::make_unique<PConvNUPC>(), std::make_unique<PConvNUPC>());
        return;
    }

    if (wav.frame_count < 16 || wav.channels == 0 || wav.channels > 2) {
        kernel_file_path_.clear();
        kernel_id_ = 0;
        CommitToStaging(std::make_unique<PConvNUPC>(), std::make_unique<PConvNUPC>());
        return;
    }

    auto ch1 = std::make_unique<PConvNUPC>();
    auto ch2 = std::make_unique<PConvNUPC>();
    bool success = false;

    if (wav.channels == 1) {
        // Normalise in-place on the decoded float buffer.
        if (NormalizePeak(wav.samples.get(), wav.frame_count)) {
            const bool r1 = ch1->LoadKernel(wav.samples.get(), wav.frame_count);
            const bool r2 = ch2->LoadKernel(wav.samples.get(), wav.frame_count);
            success = r1 && r2;
        }
    } else {
        auto buf1 = std::make_unique<float[]>(wav.frame_count);
        auto buf2 = std::make_unique<float[]>(wav.frame_count);
        for (uint32_t i = 0; i < wav.frame_count; ++i) {
            buf1[i] = wav.samples[i * 2];
            buf2[i] = wav.samples[i * 2 + 1];
        }
        // Normalise each channel independently (preserves stereo balance).
        const bool n1 = NormalizePeak(buf1.get(), wav.frame_count);
        const bool n2 = NormalizePeak(buf2.get(), wav.frame_count);
        if (n1 && n2) {
            const bool r1 = ch1->LoadKernel(buf1.get(), wav.frame_count);
            const bool r2 = ch2->LoadKernel(buf2.get(), wav.frame_count);
            success = r1 && r2;
        }
    }

    if (success) {
        kernel_file_path_          = path;
        kernel_id_                 = 0;
        current_kernel_buffer_crc_ = 0;
    } else {
        kernel_file_path_.clear();
        kernel_id_ = 0;
        // Replace with empty instances (InstanceUsable() == false).
        ch1 = std::make_unique<PConvNUPC>();
        ch2 = std::make_unique<PConvNUPC>();
    }

    CommitToStaging(std::move(ch1), std::move(ch2));
}

void Convolver::SetKernel(const float* const buf, const uint32_t size) {
    if (size < 16) return;

    // Work on a local copy so normalisation does not mutate the caller's buffer.
    auto tmp = std::make_unique<float[]>(size);
    std::copy_n(buf, size, tmp.get());

    auto ch1 = std::make_unique<PConvNUPC>();
    auto ch2 = std::make_unique<PConvNUPC>();

    if (NormalizePeak(tmp.get(), size)) {
        const bool r1 = ch1->LoadKernel(tmp.get(), size);
        const bool r2 = ch2->LoadKernel(tmp.get(), size);
        if (!r1 || !r2) {
            ch1 = std::make_unique<PConvNUPC>();
            ch2 = std::make_unique<PConvNUPC>();
        }
    }

    kernel_id_                 = 0;
    current_kernel_buffer_crc_ = 0;
    CommitToStaging(std::move(ch1), std::move(ch2));
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

    auto buf1 = std::make_unique<float[]>(frame_count);
    auto buf2 = std::make_unique<float[]>(frame_count);
    std::copy_n(ch_l, frame_count, buf1.get());
    std::copy_n(ch_r, frame_count, buf2.get());

    auto ch1 = std::make_unique<PConvNUPC>();
    auto ch2 = std::make_unique<PConvNUPC>();

    const bool n1 = NormalizePeak(buf1.get(), frame_count);
    const bool n2 = NormalizePeak(buf2.get(), frame_count);
    if (n1 && n2) {
        const bool r1 = ch1->LoadKernel(buf1.get(), frame_count);
        const bool r2 = ch2->LoadKernel(buf2.get(), frame_count);
        if (!r1 || !r2) {
            ch1 = std::make_unique<PConvNUPC>();
            ch2 = std::make_unique<PConvNUPC>();
        }
    }

    kernel_id_                 = 0;
    current_kernel_buffer_crc_ = 0;
    CommitToStaging(std::move(ch1), std::move(ch2));
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
        kernel_file_path_.clear();
        kernel_id_ = 0;
        // Commit empty (unloaded) kernels so the audio thread sees cleared state.
        CommitToStaging(std::make_unique<PConvNUPC>(), std::make_unique<PConvNUPC>());
    }
}

void Convolver::CommitKernelBuffer(
    const uint32_t expected_size, const uint32_t expected_crc, const uint32_t kernel_id
) {
    if (kernel_buffer_.empty() || expected_size_ != expected_size || current_size_ == 0) {
        ClearKernelBuffer();
        return;
    }

    const auto* const raw = reinterpret_cast<const std::byte*>(kernel_buffer_.data());
    const uint32_t calculated_crc = Crc32(reinterpret_cast<const uint8_t*>(raw), current_size_ * 4);
    if (channel_count_ - 1 > 1 || calculated_crc != expected_crc
        || calculated_crc == current_kernel_buffer_crc_) {
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
            const bool r1 = ch1->LoadKernel(tmp.get(), frames_per_channel);
            const bool r2 = ch2->LoadKernel(tmp.get(), frames_per_channel);
            loaded = r1 && r2;
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
            const bool r1 = ch1->LoadKernel(buf1.get(), frames_per_channel);
            const bool r2 = ch2->LoadKernel(buf2.get(), frames_per_channel);
            loaded = r1 && r2;
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
