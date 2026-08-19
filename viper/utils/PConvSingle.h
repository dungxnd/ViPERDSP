#pragma once

#include <cstdint>
#include <memory>

using PFFFT_Setup = struct PFFFT_Setup;

class PConvSingle {
public:
    PConvSingle() noexcept = default;
    ~PConvSingle();

    // Non-copyable: owns raw FFT-aligned memory
    PConvSingle(const PConvSingle&)            = delete;
    PConvSingle& operator=(const PConvSingle&) = delete;
    PConvSingle(PConvSingle&&)                 = delete;
    PConvSingle& operator=(PConvSingle&&)      = delete;

    void Reset();

    [[nodiscard]] uint32_t GetFFTSize()      const noexcept;
    [[nodiscard]] uint32_t GetSegmentCount() const noexcept;
    [[nodiscard]] uint32_t GetSegmentSize()  const noexcept;
    [[nodiscard]] bool     InstanceUsable()  const noexcept;

    void ConvolveInterleaved(float* buffer, int channel, uint32_t n);
    void ConvSegment(float* buffer, bool interleaved, int channel, uint32_t n);
    void Convolve(float* buffer, uint32_t n);

    uint32_t LoadKernel(const float* kernel, uint32_t kernel_size, uint32_t segment_size);
    uint32_t LoadKernel(const float* kernel, float gain, uint32_t kernel_size, uint32_t segment_size);
    uint32_t ProcessKernel(const float* kernel, uint32_t kernel_size);
    uint32_t ProcessKernel(const float* kernel, float gain, uint32_t kernel_size);

    void ReleaseResources();
    void UnloadKernel();

private:
    bool instance_usable_    = false;
    uint32_t segment_count_  = 0;
    uint32_t segment_size_   = 0;
    uint32_t fft_size_        = 0;
    uint32_t delay_line_index_ = 0;
    uint32_t input_fill_     = 0;

    // Scalar PFFFT buffers — allocated via pffft_aligned_malloc / pffft_aligned_free
    PFFFT_Setup* fft_setup_        = nullptr;
    float*       fft_work_         = nullptr;
    float*       overlap_buffer_   = nullptr;
    float*       fft_buffer_       = nullptr;
    float*       accum_buffer_     = nullptr;
    float*       mono_buffer_      = nullptr;

    // Arrays of per-segment PFFFT-aligned float* — owned via unique_ptr<float*[]>
    std::unique_ptr<float*[]> filter_segments_;
    std::unique_ptr<float*[]> input_history_;

    void ConvChunk(float* buffer, bool interleaved, int channel, uint32_t n);
};
