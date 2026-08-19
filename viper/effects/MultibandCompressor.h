#pragma once

#include "../utils/MultiBiquad.h"
#include "FETCompressor.h"
#include <array>
#include <cstdint>
#include <vector>

class MultibandCompressor {
public:
    static constexpr uint32_t kMaxBands      = 5;
    static constexpr uint32_t kMaxCrossovers = 4;

    MultibandCompressor();

    void Process(float* samples, uint32_t size) noexcept;
    void Reset() noexcept;

    void SetEnable(bool enable) noexcept;
    void SetBandCount(uint32_t count);
    void SetCrossoverFrequency(uint32_t index, float frequency) noexcept;
    void SetSamplingRate(uint32_t sampling_rate) noexcept;

    void SetBandEnable(uint32_t band, bool enable) noexcept;
    void SetBandThreshold(uint32_t band, float value) noexcept;
    void SetBandRatio(uint32_t band, float value) noexcept;
    void SetBandKnee(uint32_t band, float value) noexcept;
    void SetBandKneeAuto(uint32_t band, bool enable) noexcept;
    void SetBandGain(uint32_t band, float value) noexcept;
    void SetBandGainAuto(uint32_t band, bool enable) noexcept;
    void SetBandAttack(uint32_t band, float value) noexcept;
    void SetBandAttackAuto(uint32_t band, bool enable) noexcept;
    void SetBandRelease(uint32_t band, float value) noexcept;
    void SetBandReleaseAuto(uint32_t band, bool enable) noexcept;
    void SetBandKneeMulti(uint32_t band, float value) noexcept;
    void SetBandMaxAttack(uint32_t band, float value) noexcept;
    void SetBandMaxRelease(uint32_t band, float value) noexcept;
    void SetBandCrest(uint32_t band, float value) noexcept;
    void SetBandAdapt(uint32_t band, float value) noexcept;
    void SetBandNoClip(uint32_t band, bool enable) noexcept;

private:
    bool     enable_{false};
    uint32_t sampling_rate_{44100};
    uint32_t band_count_{3};

    std::array<float, kMaxCrossovers>       crossover_freqs_{200.0f, 4000.0f, 0.0f, 0.0f};

    std::array<MultiBiquad, kMaxCrossovers> lowpass_la_;
    std::array<MultiBiquad, kMaxCrossovers> lowpass_lb_;
    std::array<MultiBiquad, kMaxCrossovers> lowpass_ra_;
    std::array<MultiBiquad, kMaxCrossovers> lowpass_rb_;
    std::array<MultiBiquad, kMaxCrossovers> highpass_la_;
    std::array<MultiBiquad, kMaxCrossovers> highpass_lb_;
    std::array<MultiBiquad, kMaxCrossovers> highpass_ra_;
    std::array<MultiBiquad, kMaxCrossovers> highpass_rb_;

    std::array<FETCompressor, kMaxBands>          compressors_;
    std::array<std::vector<float>, kMaxBands>     band_buffers_;

    void ConfigureCrossovers() noexcept;
};
