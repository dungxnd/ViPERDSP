#include "Subwoofer.h"
#include <algorithm>
#include <cmath>

Subwoofer::Subwoofer() noexcept {
    for (auto& p : peak_) {
        p.RefreshFilter(MultiBiquad::FilterType::Peak, gain_, 37.0f, sampling_rate_, 1.0f, false);
    }
    for (auto& p : peak_low_) {
        p.RefreshFilter(MultiBiquad::FilterType::Peak, gain_lower_, 75.0f, sampling_rate_, 1.0f, false);
    }
    for (auto& lp : lowpass_) {
        lp.RefreshFilter(MultiBiquad::FilterType::LowPass, 0.0f, 200.0f, sampling_rate_, 1.0f, false);
    }
}

void Subwoofer::Process(float* samples, const uint32_t size) noexcept {
    // at zero bass gain the transfer function is ~0.5*dry - 0.6*LPF(dry),
    // which introduces a -6 dB shelf and a phase-cancellation notch at ~380 Hz.
    // Bypass entirely so the output is unity-gain dry when gain is disabled.
    if (bypassed_) return;
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

void Subwoofer::SetBassGain(const uint32_t sampling_rate, const float linear_gain) noexcept {
    // Bypass flag: when the caller passes 0.0f the subwoofer is effectively
    // disabled — Process() returns immediately, preserving unity-gain dry.
    bypassed_ = (linear_gain <= 0.0f);

    // linear_gain is a linear amplitude multiplier (not dB).
    // Clamp to a small positive floor to prevent log10(0) = -inf or log10(neg) = NaN,
    // which would poison all biquad coefficients permanently.
    // 1e-4f ≈ -80 dB: filters are effectively silent at this floor.
    const float safe_gain = std::max(linear_gain, 1e-4f);
    gain_       = 20.0f * std::log10(safe_gain);
    gain_lower_ = 20.0f * std::log10(safe_gain / 8.0f);

    for (auto& p : peak_) {
        p.RefreshFilter(MultiBiquad::FilterType::Peak, gain_, 44.0f, sampling_rate, 0.75f, true);
    }
    for (auto& p : peak_low_) {
        p.RefreshFilter(MultiBiquad::FilterType::Peak, gain_lower_, 80.0f, sampling_rate, 0.2f, true);
    }
    for (auto& lp : lowpass_) {
        lp.RefreshFilter(MultiBiquad::FilterType::LowPass, 0.0f, 380.0f, sampling_rate, 0.6f, false);
    }
}
