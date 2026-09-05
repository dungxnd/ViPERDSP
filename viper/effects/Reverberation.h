#pragma once

#include <span>


#include "../include/ViPERParams.h"
#include "../utils/CRevModel.h"
#include <array>
#include <cstdint>

class Reverberation {
public:
    using Config = viper::ReverbParams;

    Reverberation();

    void ProcessPlanar(std::span<float> L, std::span<float> R) noexcept;
    void Reset() noexcept;

    [[nodiscard]] bool IsEnabled() const noexcept { return config_.enable; }
    void SetConfig(const Config& config) noexcept;
    [[nodiscard]] const Config& GetConfig() const noexcept { return config_; }

    void SetEnable(bool enable) noexcept;
    void SetSamplingRate(uint32_t sampling_rate) noexcept;
    void SetDamp(float value)     noexcept;
    void SetDry(float value)      noexcept;
    void SetRoomSize(float value) noexcept;
    void SetWet(float value)      noexcept;
    void SetWidth(float value)    noexcept;

private:
    void Process(float* buffer, uint32_t size) noexcept;

    Config    config_{};
    bool      enable_{false};
    CRevModel model_;
};
