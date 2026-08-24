#pragma once

#include <span>


#include "../utils/Harmonic.h"
#include "../utils/MultiBiquad.h"
#include <array>
#include <cstdint>

class AnalogX {
public:
    enum class ProcessingModel { Soft = 0, Medium = 1, Hard = 2 };

    AnalogX();

    // True planar in-place — no interleave/deinterleave, no scratch buffer.
    void ProcessPlanar(std::span<float> L, std::span<float> R) noexcept;
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
    float gain_ = 0.0f;

    std::array<MultiBiquad, 2> high_pass_;
    std::array<Harmonic, 2>    harmonic_;
    std::array<MultiBiquad, 2> low_pass_;
    std::array<MultiBiquad, 2> peak_;
};
