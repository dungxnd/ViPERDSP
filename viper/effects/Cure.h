#pragma once

#include <span>


#include "../include/ViPERParams.h"
#include "../utils/Crossfeed.h"
#include "../utils/PassFilter.h"
#include <array>

class Cure {
public:
    using Config = viper::CureParams;

    Cure() = default;

    void ProcessPlanar(std::span<float> L, std::span<float> R) noexcept;
    void Reset() noexcept;

    [[nodiscard]] bool IsEnabled() const noexcept { return config_.enable; }
    void SetConfig(const Config& config) noexcept;
    [[nodiscard]] const Config& GetConfig() const noexcept { return config_; }

    [[nodiscard]] uint32_t GetCutoff() const noexcept;
    [[nodiscard]] float GetFeedback() const noexcept;
    [[nodiscard]] float GetLevelDelay() const noexcept;
    [[nodiscard]] Crossfeed::Preset GetPreset() const noexcept;

    void SetEnable(bool enable) noexcept;
    void SetCutoff(uint32_t value) noexcept;
    void SetFeedback(float value) noexcept;
    void SetPreset(uint32_t value) noexcept;
    void SetSamplingRate(uint32_t sampling_rate) noexcept;

private:
    Config config_{};
    bool enabled_{false};

    Crossfeed crossfeed_{};
    PassFilter pass_filter_{};
};
