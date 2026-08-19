#include "Stereo3DSurround.h"

Stereo3DSurround::Stereo3DSurround() noexcept = default;

void Stereo3DSurround::Process(float* const samples, const uint32_t size) const noexcept {
    if (size == 0) return;

    const uint32_t pairs     = size / 2;
    const uint32_t remainder = size % 2;

    for (uint32_t i = 0; i < pairs; ++i) {
        const float a = coeff_left_  * (samples[4 * i]     + samples[4 * i + 1]);
        const float b = coeff_right_ * (samples[4 * i + 1] - samples[4 * i]);
        const float c = coeff_left_  * (samples[4 * i + 2] + samples[4 * i + 3]);
        const float d = coeff_right_ * (samples[4 * i + 3] - samples[4 * i + 2]);

        samples[4 * i]     = a - b;
        samples[4 * i + 1] = a + b;
        samples[4 * i + 2] = c - d;
        samples[4 * i + 3] = c + d;
    }

    for (uint32_t i = 4 * pairs; remainder > 0 && i < 2 * size; i += 2) {
        const float a = samples[i];
        const float b = samples[i + 1];
        samples[i]     = coeff_left_  * (a + b) - coeff_right_ * (b - a);
        samples[i + 1] = coeff_left_  * (a + b) + coeff_right_ * (b - a);
    }
}

void Stereo3DSurround::SetMiddleImage(const float value) noexcept {
    middle_image_ = value;
    ConfigureVariables();
}

void Stereo3DSurround::SetStereoWiden(const float value) noexcept {
    stereo_widen_ = value;
    ConfigureVariables();
}

void Stereo3DSurround::ConfigureVariables() noexcept {
    const float tmp = stereo_widen_ + 1.0f;
    const float x   = tmp + 1.0f;
    const float y   = (x < 2.0f) ? 0.5f : (1.0f / x);
    coeff_left_  = middle_image_ * y;
    coeff_right_ = tmp * y;
}
