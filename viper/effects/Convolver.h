#pragma once

#include "../utils/PConvNUPC.h"
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class Convolver {
public:
    Convolver();
    ~Convolver() = default;

    // Non-copyable / non-movable: owns unique kernel resources
    Convolver(const Convolver&)            = delete;
    Convolver& operator=(const Convolver&) = delete;
    Convolver(Convolver&&)                 = delete;
    Convolver& operator=(Convolver&&)      = delete;

    uint32_t Process(const float* source, float* dest, uint32_t frame_size);
    void Reset();

    [[nodiscard]] bool     GetEnable()   const noexcept;
    [[nodiscard]] uint32_t GetKernelID() const noexcept;

    void SetEnable(bool enable);
    void SetKernel(const char* path);
    void SetKernel(const float* buf, uint32_t size);
    void SetKernelBuffer(const float* buf, uint32_t size);
    void SetKernelStereo(const float* ch_l, const float* ch_r, uint32_t frame_count);
    void SetCrossChannel(float value);
    void SetSamplingRate(uint32_t sampling_rate);

    void PrepareKernelBuffer(uint32_t buf_size, uint32_t ch_count, bool reset);
    void CommitKernelBuffer(uint32_t expected_size, uint32_t expected_crc, uint32_t kernel_id);

private:
    bool     enable_                    = false;
    bool     is_valid_cross_channel_    = false;
    uint32_t sampling_rate_             = 44100;
    uint32_t kernel_id_                 = 0;
    uint32_t expected_size_             = 0;
    uint32_t current_size_              = 0;
    uint32_t channel_count_             = 0;
    uint32_t current_kernel_buffer_crc_ = 0;
    float    cross_channel_             = 0.0f;

    std::string        kernel_file_path_;
    std::vector<float> kernel_buffer_;

    // -------------------------------------------------------------------------
    // Thread-safe kernel lifecycle
    // -------------------------------------------------------------------------
    // Active kernels (kernel_ch1_/ch2_) are owned exclusively by the audio
    // thread.  They are written ONLY in ConsumeKernelSwap(), which is called
    // at the very start of Process() before any DSP work.
    //
    // Staging kernels (staging_ch1_/ch2_) are written exclusively by the
    // binder thread via CommitToStaging() under kernel_stage_mutex_.
    //
    // Protocol:
    //   Binder: spinwait for !kernel_swap_pending_ (acquire), write staging_,
    //           store kernel_swap_pending_ = true (release).
    //   Audio:  load kernel_swap_pending_ (acquire); if true, swap unique_ptr
    //           pairs (3 pointer moves — no allocation), store false (release).
    //           Retired kernels land in staging_; freed next time binder
    //           overwrites them.
    //
    // The audio thread never allocates, never spins, never holds any lock.
    // -------------------------------------------------------------------------
    std::unique_ptr<PConvNUPC> kernel_ch1_;   // audio-thread owned
    std::unique_ptr<PConvNUPC> kernel_ch2_;   // audio-thread owned
    std::unique_ptr<PConvNUPC> staging_ch1_;  // binder-thread owned until swap
    std::unique_ptr<PConvNUPC> staging_ch2_;  // binder-thread owned until swap

    // Written by binder (release) after staging_ is ready.
    // Cleared by audio (release) after swap completes.
    std::atomic<bool> kernel_swap_pending_{false};

    // Serialises concurrent binder-thread kernel mutations (rare but possible
    // when multiple IPC calls arrive close together).
    std::mutex kernel_stage_mutex_;

    /// Called once at the start of every Process() call.
    /// Swaps staging↔active in O(1) with no allocation.
    void ConsumeKernelSwap() noexcept;

    /// Called by all SetKernel*/CommitKernelBuffer paths on the binder thread.
    /// Blocks (yields) until any previous pending swap has been consumed by
    /// the audio thread (at most one audio-callback cycle ≈ 1–10 ms).
    void CommitToStaging(std::unique_ptr<PConvNUPC> ch1,
                         std::unique_ptr<PConvNUPC> ch2);

    void ClearKernelBuffer() noexcept;
};
