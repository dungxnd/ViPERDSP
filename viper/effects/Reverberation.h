#pragma once

#include "../utils/CRevModel.h"
#include <array>
#include <cstdint>

class Reverberation {
public:
    Reverberation();

    void ProcessPlanar(float* __restrict L, float* __restrict R, size_t frames) noexcept;
    void Reset() noexcept;

    [[nodiscard]] bool IsEnabled() const noexcept { return enable_; }
    void SetEnable(bool enable) noexcept;
    void SetDamp(float value)     noexcept;
    void SetDry(float value)      noexcept;
    void SetRoomSize(float value) noexcept;
    void SetWet(float value)      noexcept;
    void SetWidth(float value)    noexcept;

private:
    void Process(float* buffer, uint32_t size) noexcept;

    bool      enable_{false};
    CRevModel model_;
    alignas(64) std::array<float, 4096u * 2u> scratch_{};
};
