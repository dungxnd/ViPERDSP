#pragma once

#include "../utils/Harmonic.h"
#include "../utils/MultiBiquad.h"
#include <array>
#include <cstdint>

class AnalogX {
public:
    enum class ProcessingModel { Soft = 0, Medium = 1, Hard = 2 };

    AnalogX();

    void Process(float* samples, uint32_t size);
    void ProcessPlanar(float* __restrict L, float* __restrict R, size_t frames) noexcept;
    void Reset();

    [[nodiscard]] bool IsEnabled() const noexcept { return enable_; }
    void SetEnable(bool enable);
    void SetProcessingModel(ProcessingModel model);
    void SetProcessingModel(int model);  // accepts raw int from parameter dispatch
    void SetSamplingRate(uint32_t sampling_rate);

private:
    bool enable_ = false;

    ProcessingModel processing_model_ = ProcessingModel::Soft;
    uint32_t        sampling_rate_    = 44100;
    uint32_t        freq_range_       = 0;

    float gain_ = 0.0f;

    std::array<MultiBiquad, 2> high_pass_;
    std::array<Harmonic, 2>    harmonic_;
    std::array<MultiBiquad, 2> low_pass_;
    std::array<MultiBiquad, 2> peak_;
    alignas(64) std::array<float, 4096u * 2u> pp_scratch_{};
};
