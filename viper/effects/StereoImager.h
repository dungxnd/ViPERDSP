#pragma once

#include "../utils/MultiBiquad.h"
#include <array>
#include <cstdint>
class StereoImager {
public:
    static constexpr uint32_t kNumBands      = 3;
    static constexpr uint32_t kNumCrossovers = 2;

    StereoImager();

    void Process(float *samples, uint32_t size) noexcept;
    void ProcessPlanar(float* __restrict L, float* __restrict R, size_t frames) noexcept;
    void Reset() noexcept;

    [[nodiscard]] bool IsEnabled() const noexcept { return enable_; }
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

    // Pre-allocated: eliminates RT-unsafe resize() in Process().
    static constexpr uint32_t kMaxFrames = 4096u;
    alignas(64) std::array<std::array<float, kMaxFrames * 2u>, kNumBands> band_buffers_{};
    alignas(64) std::array<float, 4096u * 2u> pp_scratch_{};

    void ConfigureCrossovers() noexcept;
};
