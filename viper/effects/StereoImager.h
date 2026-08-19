#pragma once

#include "../utils/MultiBiquad.h"
#include <array>
#include <cstdint>
#include <vector>

class StereoImager {
public:
    static constexpr uint32_t kNumBands      = 3;
    static constexpr uint32_t kNumCrossovers = 2;

    StereoImager();

    void Process(float *samples, uint32_t size) noexcept;
    void Reset() noexcept;

    void SetEnable(bool enable) noexcept;
    void SetLowWidth(float value) noexcept;
    void SetMidWidth(float value) noexcept;
    void SetHighWidth(float value) noexcept;
    void SetLowCrossover(float value) noexcept;
    void SetHighCrossover(float value) noexcept;
    void SetSamplingRate(uint32_t sampling_rate) noexcept;

private:
    bool enable_{false};

    uint32_t sampling_rate_{44100u};

    std::array<float, kNumBands>      band_widths_{1.0f, 1.0f, 1.0f};
    std::array<float, kNumCrossovers> crossover_freqs_{200.0f, 4000.0f};

    std::array<MultiBiquad, kNumCrossovers> lowpass_la_{};
    std::array<MultiBiquad, kNumCrossovers> lowpass_lb_{};
    std::array<MultiBiquad, kNumCrossovers> lowpass_ra_{};
    std::array<MultiBiquad, kNumCrossovers> lowpass_rb_{};
    std::array<MultiBiquad, kNumCrossovers> highpass_la_{};
    std::array<MultiBiquad, kNumCrossovers> highpass_lb_{};
    std::array<MultiBiquad, kNumCrossovers> highpass_ra_{};
    std::array<MultiBiquad, kNumCrossovers> highpass_rb_{};

    std::array<std::vector<float>, kNumBands> band_buffers_{};

    void ConfigureCrossovers() noexcept;
};
