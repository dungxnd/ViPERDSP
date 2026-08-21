#pragma once

#include "../utils/PConvNUPC.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class Convolver {
public:
    Convolver() noexcept = default;
    ~Convolver() = default;

    // Non-copyable: owns unique kernel resources
    Convolver(const Convolver&)            = delete;
    Convolver& operator=(const Convolver&) = delete;

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
    bool     enable_                   = false;
    bool     is_valid_cross_channel_   = false;
    uint32_t sampling_rate_            = 44100;
    uint32_t kernel_id_                = 0;
    uint32_t expected_size_            = 0;
    uint32_t current_size_             = 0;
    uint32_t channel_count_            = 0;
    uint32_t current_kernel_buffer_crc_ = 0;
    float    cross_channel_            = 0.0f;

    std::string        kernel_file_path_;
    std::vector<float> kernel_buffer_;

    PConvNUPC kernel_ch1_;
    PConvNUPC kernel_ch2_;

    void ClearKernelBuffer() noexcept;
};
