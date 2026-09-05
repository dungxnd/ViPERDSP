#pragma once

#include <span>


#include "../include/ViPERParams.h"
#include "../utils/Biquad.h"
#include <array>
#include <cstdint>
#include <numbers>

class PlaybackGain {
public:
    using Config = viper::PlaybackGainControlParams;

    PlaybackGain();

    void ProcessPlanar(std::span<float> L, std::span<float> R) noexcept;
    void Process(float* samples, uint32_t size) noexcept;
    void Reset() noexcept;

    [[nodiscard]] bool IsEnabled() const noexcept { return config_.enable; }
    void SetConfig(const Config& config) noexcept;
    [[nodiscard]] const Config& GetConfig() const noexcept { return config_; }

    void SetEnable(bool enable) noexcept;
    void SetMaxGainFactor(float max_gain_factor) noexcept;
    void SetRatio(float ratio) noexcept;
    void SetVolume(float volume) noexcept;
    void SetSamplingRate(uint32_t sampling_rate) noexcept;

private:
    Config config_{};
    static constexpr float  kWarmupSeconds = 0.4f;
    static constexpr float  kBandpassFreq  = 2200.0f;
    static constexpr float  kBandpassQ     = 0.33f;
    static constexpr float  kLogCoeff      = std::numbers::log10e_v<float>;

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

};
