#pragma once

#include "../utils/Crossfeed.h"
#include "../utils/PassFilter.h"

class Cure {
public:
    Cure() = default;

    void Process(float *buffer, uint32_t size) noexcept;
    void Reset() noexcept;

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
