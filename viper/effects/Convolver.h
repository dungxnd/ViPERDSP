#pragma once

#include <span>


#include "../include/ViPERParams.h"
#include "../utils/PConvNUPC.h"
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Stereo convolution reverb using PConvNUPC (Non-Uniform Partitioned Convolution).
// Supports mono and stereo IRs loaded from WAV files, float buffers, or a
// chunked buffer protocol.  Optional cross-channel blend is applied post-convolution.
//
// Thread-safety model (lock-free hot path):
//   Active kernels (kernel_ch1_/ch2_) are owned exclusively by the audio thread
//   and mutated only inside ConsumeKernelSwap(), called at the start of Process().
//   Binder-thread kernel changes build new PConvNUPC instances off-thread, then
//   publish them through a one-slot staging area protected by a single atomic flag:
//
//     Binder: acquire-spin until !pending_, write staging_, release-store pending_=true.
//     Audio:  acquire-load pending_; if set, swap ptrs (3 moves, zero alloc),
//             release-store pending_=false.  Old kernels sit in staging_ until the
//             next CommitToStaging() call overwrites and frees them.
//
//   The audio thread never allocates, spins, or holds a lock.
class Convolver {
public:
    using Config = viper::ConvolverParams;

    Convolver();
    ~Convolver() = default;

    Convolver(const Convolver&)            = delete;
    Convolver& operator=(const Convolver&) = delete;
    Convolver(Convolver&&)                 = delete;
    Convolver& operator=(Convolver&&)      = delete;

    uint32_t Process(const float* source, float* dest, uint32_t frame_size);

    // Planar processing: eliminates ProcessInterleaved stride overhead.
    // L and R are separate contiguous float arrays of `frame_size` samples.
    // Called from ViPER::Process() after deinterleave.
    void ProcessPlanar(std::span<float> L, std::span<float> R) noexcept;

    // Resets runtime state of the active kernels (clears delay lines).
    void Reset();

    [[nodiscard]] bool     GetEnable()   const noexcept;
    [[nodiscard]] bool     IsEnabled()   const noexcept { return GetEnable(); }
    [[nodiscard]] uint32_t GetKernelID() const noexcept;
    [[nodiscard]] uint32_t GetExpectedSize() const noexcept { return expected_size_; }

    void SetConfig(const Config& config) noexcept;
    [[nodiscard]] const Config& GetConfig() const noexcept { return config_; }

    void SetEnable(bool enable);
    void SetCrossChannel(float value);
    void SetSamplingRate(uint32_t sampling_rate);

    // Kernel loading — all paths execute on the binder thread and publish via
    // CommitToStaging().  Each normalises the IR with NormalizePeak() before
    // calling LoadKernel() to guard against unnormalised or DC-biased WAVs.
    void SetKernel(const char* path);
    void SetKernel(const float* buf, uint32_t size);
    void SetKernelStereo(const float* ch_l, const float* ch_r, uint32_t frame_count);

    // Chunked streaming protocol for large in-band IR transfers:
    //   PrepareKernelBuffer(size, ch_count, reset=false) — allocates receive buffer.
    //   SetKernelBuffer(buf, size)                       — appends a chunk.
    //   CommitKernelBuffer(expected_size, crc, id)       — validates CRC and loads.
    void PrepareKernelBuffer(uint32_t buf_size, uint32_t ch_count, bool reset);
    void SetKernelBuffer(const float* buf, uint32_t size);
    void CommitKernelBuffer(uint32_t expected_size, uint32_t expected_crc, uint32_t kernel_id);

private:
    Config   config_{};
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

    // Active kernels: audio-thread only, never touched by the binder thread.
    std::unique_ptr<PConvNUPC> kernel_ch1_{std::make_unique<PConvNUPC>()};
    std::unique_ptr<PConvNUPC> kernel_ch2_{std::make_unique<PConvNUPC>()};

    // Staging kernels: written by the binder under kernel_stage_mutex_, then
    // transferred to kernel_ch1_/ch2_ by the audio thread in ConsumeKernelSwap().
    std::unique_ptr<PConvNUPC> staging_ch1_;
    std::unique_ptr<PConvNUPC> staging_ch2_;

    // Acquire/release flag — intentionally NOT seq_cst (see ConsumeKernelSwap).
    std::atomic<bool> kernel_swap_pending_{false};

    // Serialises concurrent binder-thread callers (rare; multiple IPC calls can
    // arrive close together).
    std::mutex kernel_stage_mutex_;

    void ConsumeKernelSwap() noexcept;
    void CommitToStaging(std::unique_ptr<PConvNUPC> ch1,
                         std::unique_ptr<PConvNUPC> ch2);
    void ClearKernelBuffer() noexcept;
};
