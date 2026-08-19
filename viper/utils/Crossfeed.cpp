#include "Crossfeed.h"
#include <cmath>
#include <numbers>

Crossfeed::Crossfeed() {
    Reset();
}

void Crossfeed::ProcessFrames(float *buffer, const uint32_t size) noexcept {
    for (uint32_t i = 0; i < size * 2; i += 2) {
        FilterSample(buffer + i);
    }
}

void Crossfeed::Reset() noexcept {
    const uint32_t cutoff = preset_.cutoff;
    const double level = preset_.feedback / 10.0;

    const double gb_lo = level * -5.0 / 6.0 - 3.0;
    const double gb_hi = level / 6.0 - 3.0;

    const double g_lo = std::pow(10.0, gb_lo / 20.0);
    const double g_hi = 1.0 - std::pow(10.0, gb_hi / 20.0);
    const double fc_hi = cutoff * std::pow(2.0, (gb_lo - 20.0 * std::log10(g_hi)) / 12.0);

    const double pi2 = 2.0 * std::numbers::pi;
    double x = std::exp(-pi2 * cutoff / sampling_rate_);
    b1_lo_ = static_cast<float>(x);
    a0_lo_ = static_cast<float>(g_lo * (1.0 - x));

    x = std::exp(-pi2 * fc_hi / sampling_rate_);
    b1_hi_ = static_cast<float>(x);
    a0_hi_ = static_cast<float>(1.0 - g_hi * (1.0 - x));
    a1_hi_ = static_cast<float>(-x);

    gain_ = static_cast<float>(1.0 / (1.0 - g_hi + g_lo));
    lfs_ = {};
}

uint32_t Crossfeed::GetCutoff() const noexcept {
    return preset_.cutoff;
}

float Crossfeed::GetFeedback() const noexcept {
    return static_cast<float>(preset_.feedback) / 10.0f;
}

float Crossfeed::GetLevelDelay() const noexcept {
    if (preset_.cutoff <= 1800) {
        return 18700.0f / static_cast<float>(preset_.cutoff) * 10.0f;
    }
    return 0.0f;
}

Crossfeed::Preset Crossfeed::GetPreset() const noexcept {
    return preset_;
}

void Crossfeed::SetCutoff(const uint32_t value) noexcept {
    preset_.cutoff = value;
    Reset();
}

void Crossfeed::SetFeedback(const float value) noexcept {
    preset_.feedback = static_cast<uint32_t>(value * 10.0f);
    Reset();
}

void Crossfeed::SetPreset(const Preset preset) noexcept {
    preset_ = preset;
    Reset();
}

void Crossfeed::SetSamplingRate(const uint32_t sampling_rate) noexcept {
    if (sampling_rate_ != sampling_rate) {
        sampling_rate_ = sampling_rate;
        Reset();
    }
}

void Crossfeed::FilterSample(float *sample) noexcept {
    lfs_.lo[0] = ApplyLoFilter(sample[0], lfs_.lo[0]);
    lfs_.lo[1] = ApplyLoFilter(sample[1], lfs_.lo[1]);

    lfs_.hi[0] = ApplyHiFilter(sample[0], lfs_.asis[0], lfs_.hi[0]);
    lfs_.hi[1] = ApplyHiFilter(sample[1], lfs_.asis[1], lfs_.hi[1]);
    lfs_.asis[0] = sample[0];
    lfs_.asis[1] = sample[1];

    sample[0] = (lfs_.hi[0] + lfs_.lo[1]) * gain_;
    sample[1] = (lfs_.hi[1] + lfs_.lo[0]) * gain_;
}
