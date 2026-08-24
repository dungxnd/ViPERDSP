#pragma once

#include "../utils/DepthSurround.h"
#include "../utils/Stereo3DSurround.h"
#include <array>
#include <cstdint>

class ColorfulMusic {
public:
    ColorfulMusic();

    void ProcessPlanar(float* __restrict L, float* __restrict R, size_t frames) noexcept;
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
    alignas(64) std::array<float, 4096u * 2u> scratch_{};
};
