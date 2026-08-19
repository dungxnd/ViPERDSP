#pragma once

#include "../utils/CRevModel.h"
#include <cstdint>

class Reverberation {
public:
    Reverberation();

    void Process(float* buffer, uint32_t size) noexcept;
    void Reset() noexcept;

    void SetEnable(bool enable) noexcept;
    void SetDamp(float value)     noexcept;
    void SetDry(float value)      noexcept;
    void SetRoomSize(float value) noexcept;
    void SetWet(float value)      noexcept;
    void SetWidth(float value)    noexcept;

private:
    bool      enable_{false};
    CRevModel model_;
};
