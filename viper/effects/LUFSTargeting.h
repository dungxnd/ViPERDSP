#pragma once

#include "../utils/Biquad.h"
#include <array>
#include <cstdint>

class LUFSTargeting {
public:
    LUFSTargeting();

    void Process(float* samples, uint32_t size) noexcept;
    void Reset() noexcept;

    void SetEnable(bool enable) noexcept;
    void SetTargetLUFS(float value) noexcept;
    void SetMaxGain(float value) noexcept;
    void SetSpeed(int value) noexcept;
    void SetSamplingRate(uint32_t sampling_rate);

private:
    static constexpr uint32_t kMaxWindows        = 40;
    static constexpr double   kAbsoluteGateLufs  = -70.0;

    bool     enable_{false};
    int      speed_{1};
    uint32_t sampling_rate_{44100};
    uint32_t window_size_{0};
    uint32_t step_size_{0};
    uint32_t sample_counter_{0};
    uint32_t window_sample_count_{0};
    uint32_t window_write_idx_{0};
    uint32_t window_count_{0};

    float  target_lufs_{-14.0f};
    float  max_gain_db_{6.0f};

    double smoothed_gain_db_{0.0};
    double window_accumulator_{0.0};
    double attack_coeff_{0.0};
    double release_coeff_{0.0};

    std::array<double, kMaxWindows> window_power_{};

    Biquad k_weight_stage1_l_;
    Biquad k_weight_stage1_r_;
    Biquad k_weight_stage2_l_;
    Biquad k_weight_stage2_r_;

    void ConfigureFilters() noexcept;
    void UpdateSmoothingCoeffs() noexcept;
};
