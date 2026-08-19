#include "PolesFilter.h"
#include <numbers>

namespace {
constexpr float kAntiDenormal = 1e-25f;

void FilterSide(
    PolesFilter::Channel &side,
    const float sample,
    float *out1, float *out2, float *out3
) noexcept {
    const float oldest_in = side.in[2];
    side.in[2] = side.in[1];
    side.in[1] = side.in[0];
    side.in[0] = sample;

    side.x[0] += side.lower_angle * (sample   - side.x[0]) + kAntiDenormal;
    side.x[1] += side.lower_angle * (side.x[0] - side.x[1]) + kAntiDenormal;
    side.x[2] += side.lower_angle * (side.x[1] - side.x[2]) + kAntiDenormal;
    side.x[3] += side.lower_angle * (side.x[2] - side.x[3]) + kAntiDenormal;

    side.y[0] += side.upper_angle * (sample   - side.y[0]) + kAntiDenormal;
    side.y[1] += side.upper_angle * (side.y[0] - side.y[1]) + kAntiDenormal;
    side.y[2] += side.upper_angle * (side.y[1] - side.y[2]) + kAntiDenormal;
    side.y[3] += side.upper_angle * (side.y[2] - side.y[3]) + kAntiDenormal;

    *out1 = side.x[3];
    *out2 = oldest_in - side.y[3];
    *out3 = side.y[3]  - side.x[3];
}
} // namespace

PolesFilter::PolesFilter() {
    Reset();
}

void PolesFilter::FilterLeft(
    const float sample, float *out1, float *out2, float *out3
) noexcept {
    FilterSide(channels_[0], sample, out1, out2, out3);
}

void PolesFilter::FilterRight(
    const float sample, float *out1, float *out2, float *out3
) noexcept {
    FilterSide(channels_[1], sample, out1, out2, out3);
}

void PolesFilter::Reset() noexcept {
    channels_[0] = Channel{};
    channels_[1] = Channel{};
    UpdateCoeff();
}

void PolesFilter::SetPassFilter(const uint32_t lower_freq, const uint32_t upper_freq) noexcept {
    lower_freq_ = lower_freq;
    upper_freq_ = upper_freq;
    UpdateCoeff();
}

void PolesFilter::SetSamplingRate(const uint32_t sampling_rate) noexcept {
    sampling_rate_ = sampling_rate;
    UpdateCoeff();
}

void PolesFilter::UpdateCoeff() noexcept {
    const auto sr = static_cast<float>(sampling_rate_);
    const float lower_angle = static_cast<float>(lower_freq_) * std::numbers::pi_v<float> / sr;
    const float upper_angle = static_cast<float>(upper_freq_) * std::numbers::pi_v<float> / sr;

    for (auto& ch : channels_) {
        ch.lower_angle = lower_angle;
        ch.upper_angle = upper_angle;
    }
}
