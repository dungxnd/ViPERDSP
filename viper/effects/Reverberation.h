#pragma once

#include <span>


#include "../utils/CRevModel.h"
#include <array>
#include <cstdint>

class Reverberation {
public:
    Reverberation();

    void ProcessPlanar(std::span<float> L, std::span<float> R) noexcept;
    void Reset() noexcept;

    [[nodiscard]] bool IsEnabled() const noexcept { return enable_; }
    void SetEnable(bool enable) noexcept;
    void SetSamplingRate(uint32_t sampling_rate) noexcept;
    void SetDamp(float value)     noexcept;
    void SetDry(float value)      noexcept;
    void SetRoomSize(float value) noexcept;
    void SetWet(float value)      noexcept;
    void SetWidth(float value)    noexcept;

private:
    void Process(float* buffer, uint32_t size) noexcept;

    bool      enable_{false};
    CRevModel model_;
};
