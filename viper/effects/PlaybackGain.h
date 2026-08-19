#pragma once

#include "../utils/Biquad.h"
#include <cstdint>

class PlaybackGain {
public:
    PlaybackGain();

    void Process(float* samples, uint32_t size) noexcept;
    void Reset() noexcept;

    void SetEnable(bool enable) noexcept;
    void SetMaxGainFactor(float max_gain_factor) noexcept;
    void SetRatio(float ratio) noexcept;
    void SetVolume(float volume) noexcept;
    void SetSamplingRate(uint32_t sampling_rate) noexcept;

private:
    static constexpr float  kWarmupSeconds = 0.4f;
    static constexpr float  kBandpassFreq  = 2200.0f;
    static constexpr float  kBandpassQ     = 0.33f;
    static constexpr float  kLogCoeff      = 0.4342945f;  // log10(e)

    bool     enable_{false};
    uint32_t sampling_rate_{44100};
    uint32_t ramp_progress_{0};
    uint32_t ramp_frames_{static_cast<uint32_t>(44100 * kWarmupSeconds)};

    float log_coeff_{kLogCoeff};
    float ratio1_{2.0f};
    float ratio2_{0.5f};
    float volume_{1.0f};
    float max_gain_factor_{1.0f};
    float current_gain_l_{1.0f};
    float current_gain_r_{1.0f};

    Biquad biquad1_;
    Biquad biquad2_;

    [[nodiscard]] double AnalyseWave(const float* samples, uint32_t size) noexcept;
};
