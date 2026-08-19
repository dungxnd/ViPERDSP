#include "Subwoofer.h"
#include <cmath>

Subwoofer::Subwoofer() noexcept {
    for (auto& p : peak_) {
        p.RefreshFilter(MultiBiquad::FilterType::PEAK, gain_, 37.0f, sampling_rate_, 1.0f, false);
    }
    for (auto& p : peak_low_) {
        p.RefreshFilter(MultiBiquad::FilterType::PEAK, gain_lower_, 75.0f, sampling_rate_, 1.0f, false);
    }
    for (auto& lp : lowpass_) {
        lp.RefreshFilter(MultiBiquad::FilterType::LOW_PASS, 0.0f, 200.0f, sampling_rate_, 1.0f, false);
    }
}

void Subwoofer::Process(float* samples, const uint32_t size) noexcept {
    for (uint32_t i = 0; i < size * 2; i += 2) {
        double tmp = peak_[0].ProcessSample(samples[i]);
        tmp = peak_low_[0].ProcessSample(tmp);
        tmp = lowpass_[0].ProcessSample(tmp - samples[i]);
        samples[i] = samples[i] * 0.5f + static_cast<float>(tmp) * 0.6f;

        tmp = peak_[1].ProcessSample(samples[i + 1]);
        tmp = peak_low_[1].ProcessSample(tmp);
        tmp = lowpass_[1].ProcessSample(tmp - samples[i + 1]);
        samples[i + 1] = samples[i + 1] * 0.5f + static_cast<float>(tmp) * 0.6f;
    }
}

void Subwoofer::SetBassGain(const uint32_t sampling_rate, const float gain_db) noexcept {
    gain_       = 20.0f * std::log10(gain_db);
    gain_lower_ = 20.0f * std::log10(gain_db / 8.0f);

    for (auto& p : peak_) {
        p.RefreshFilter(MultiBiquad::FilterType::PEAK, gain_, 44.0f, sampling_rate, 0.75f, true);
    }
    for (auto& p : peak_low_) {
        p.RefreshFilter(MultiBiquad::FilterType::PEAK, gain_lower_, 80.0f, sampling_rate, 0.2f, true);
    }
    for (auto& lp : lowpass_) {
        lp.RefreshFilter(MultiBiquad::FilterType::LOW_PASS, 0.0f, 380.0f, sampling_rate, 0.6f, false);
    }
}
