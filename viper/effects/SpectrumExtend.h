#pragma once

#include <span>


#include "../include/ViPERParams.h"
#include "../utils/Harmonic.h"
#include "../utils/MultiBiquad.h"
#include <array>
#include <cstdint>

class SpectrumExtend {
public:
    using Config = viper::SpectrumExtensionParams;

    SpectrumExtend();

    void ProcessPlanar(std::span<float> L, std::span<float> R) noexcept;
    void Reset() noexcept;

    [[nodiscard]] bool IsEnabled() const noexcept { return config_.enable; }
    void SetConfig(const Config& config) noexcept;
    [[nodiscard]] const Config& GetConfig() const noexcept { return config_; }

    void SetEnable(bool enable) noexcept;
    void SetExciter(float value) noexcept;
    void SetReferenceFrequency(uint32_t value) noexcept;
    void SetSamplingRate(uint32_t sampling_rate) noexcept;

private:
    void Process(float *samples, uint32_t size) noexcept;

    Config   config_{};
    bool enable_{false};

    uint32_t sampling_rate_{44100u};
    uint32_t reference_freq_{7600u};

    float exciter_{0.0f};

    std::array<MultiBiquad, 2> highpass_{};
    std::array<MultiBiquad, 2> lowpass_{};
    std::array<Harmonic, 2>    harmonics_{};
};
