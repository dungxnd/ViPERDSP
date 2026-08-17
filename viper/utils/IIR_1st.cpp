#include "IIR_1st.h"
#include <cmath>
#include <numbers>

namespace {

[[nodiscard]] float omega2(const float frequency, const uint32_t sampling_rate) noexcept {
    return std::numbers::pi_v<float> * frequency / static_cast<float>(sampling_rate);
}

} // namespace

void IIR_1st::Mute() noexcept {
    prev_sample_ = 0.0f;
}

void IIR_1st::SetCoefficients(const float b0, const float b1, const float a1) noexcept {
    b0_ = b0;
    b1_ = b1;
    a1_ = a1;
}

void IIR_1st::SetHighPassFilterBW(const float frequency, const uint32_t sampling_rate) noexcept {
    const float t = std::tan(omega2(frequency, sampling_rate));
    b0_ = 1.0f / (1.0f + t);
    b1_ = -b0_;
    a1_ = (1.0f - t) / (1.0f + t);
}

void IIR_1st::SetLowPassFilterBW(const float frequency, const uint32_t sampling_rate) noexcept {
    const float t = std::tan(omega2(frequency, sampling_rate));
    a1_ = (1.0f - t) / (1.0f + t);
    b0_ = t / (1.0f + t);
    b1_ = b0_;
}
