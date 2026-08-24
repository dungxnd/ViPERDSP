#pragma once

#include <span>


#include "../utils/Crossfeed.h"
#include "../utils/PassFilter.h"
#include <array>

class Cure {
public:
    Cure() = default;

    void ProcessPlanar(std::span<float> L, std::span<float> R) noexcept;
    void Reset() noexcept;

    [[nodiscard]] bool IsEnabled() const noexcept { return enabled_; }

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
    bool enabled_{false};

    Crossfeed crossfeed_{};
    PassFilter pass_filter_{};
};
