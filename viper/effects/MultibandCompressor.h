#pragma once

#include <span>


#include "../include/ViPERParams.h"
#include "../dsp/LinkwitzRileyCrossover.h"
#include "FETCompressor.h"
#include <array>
#include <cstdint>
class MultibandCompressor {
public:
    using Config = viper::MultibandCompressorParams;

    static constexpr uint32_t kMaxBands      = 5;
    static constexpr uint32_t kMaxCrossovers = 4;

    MultibandCompressor();

    void ProcessPlanar(std::span<float> L, std::span<float> R) noexcept;
    void Reset() noexcept;

    [[nodiscard]] bool IsEnabled() const noexcept { return config_.enable; }
    void SetConfig(const Config& config) noexcept;
    [[nodiscard]] const Config& GetConfig() const noexcept { return config_; }

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
    Config   config_{};
    bool     enable_{false};
    uint32_t sampling_rate_{44100};
    uint32_t band_count_{3};

    std::array<float, kMaxCrossovers>       crossover_freqs_{200.0f, 4000.0f, 0.0f, 0.0f};

    viper::dsp::LinkwitzRileyCrossover<kMaxBands> crossover_;

    std::array<FETCompressor, kMaxBands>          compressors_;
    static constexpr uint32_t kMaxFrames = 4096u;
    // Per-band scratch + band-sum accumulator for ProcessPlanar().
    // Each band copies the ORIGINAL input from L/R into its scratch, filters,
    // compresses, and accumulates into accum_l_/accum_r_ (zero-filled each
    // call), which is finally copied to L/R — output is exactly Σbands.  The
    // accumulator must not be seeded with the input (that would double the
    // signal) nor used as the band-input source (band 0's output would corrupt
    // bands 1..N).
    alignas(64) std::array<float, kMaxFrames> band_scratch_l_{};
    alignas(64) std::array<float, kMaxFrames> band_scratch_r_{};
    alignas(64) std::array<float, kMaxFrames> accum_l_{};
    alignas(64) std::array<float, kMaxFrames> accum_r_{};

    void ConfigureCrossovers() noexcept;
};
