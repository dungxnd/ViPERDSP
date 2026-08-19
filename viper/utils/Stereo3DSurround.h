#pragma once

#include <cstdint>

class Stereo3DSurround {
public:
    Stereo3DSurround() noexcept;

    void Process(float* samples, uint32_t size) const noexcept;

    void SetMiddleImage(float value) noexcept;
    void SetStereoWiden(float value) noexcept;

private:
    float stereo_widen_  = 0.0f;
    float middle_image_  = 1.0f;
    float coeff_left_    = 0.5f;
    float coeff_right_   = 0.5f;

    void ConfigureVariables() noexcept;
};
