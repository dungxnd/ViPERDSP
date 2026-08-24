#pragma once

#include <cstdint>
#include <span>

#include "../utils/DepthSurround.h"
#include "../utils/Stereo3DSurround.h"

class ColorfulMusic {
public:
    ColorfulMusic();

    void ProcessPlanar(std::span<float> L, std::span<float> R) noexcept;
    void Reset();

    [[nodiscard]] bool IsEnabled() const noexcept { return enabled_; }
    void SetEnable(bool enable);
    void SetDepthValue(uint32_t value);
    void SetMidImageValue(float value);
    void SetWidenValue(float value);
    void SetSamplingRate(uint32_t sampling_rate);

private:
    void Process(float* samples, uint32_t size);

    bool     enabled_      = false;
    uint32_t sampling_rate_ = 44100;

    Stereo3DSurround stereo_3d_surround_;
    DepthSurround    depth_surround_;
};
