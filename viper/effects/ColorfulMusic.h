#pragma once

#include <cstdint>
#include <span>

#include "../include/ViPERParams.h"
#include "../utils/DepthSurround.h"
#include "../utils/Stereo3DSurround.h"

class ColorfulMusic {
public:
    using Config = viper::FieldSurroundParams;

    ColorfulMusic();

    void ProcessPlanar(std::span<float> L, std::span<float> R) noexcept;
    void Reset();

    [[nodiscard]] bool IsEnabled() const noexcept { return config_.enable; }
    void SetConfig(const Config& config) noexcept;
    [[nodiscard]] const Config& GetConfig() const noexcept { return config_; }

    void SetEnable(bool enable);
    void SetDepthValue(uint32_t value);
    void SetMidImageValue(float value);
    void SetWidenValue(float value);
    void SetSamplingRate(uint32_t sampling_rate);

private:
    void Process(float* samples, uint32_t size);

    Config   config_{};
    bool     enabled_      = false;
    uint32_t sampling_rate_ = 44100;

    Stereo3DSurround stereo_3d_surround_;
    DepthSurround    depth_surround_;
};
