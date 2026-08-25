#include "DiffSurround.h"
#include <algorithm>

DiffSurround::DiffSurround() {
    Reset();
}

void DiffSurround::Reset() {
    ring_[0].Reset();
    ring_[1].Reset();

    const auto delay_samples =
        static_cast<uint32_t>(delay_time_ / 1000.0f * static_cast<float>(sampling_rate_));

    // Pre-fill the delayed channel with silence equal to the requested delay.
    const uint32_t delayed_ch = reverse_ ? 0u : 1u;
    ring_[delayed_ch].SetDelay(delay_samples);

    lp_filter_.Reset();
    if (lp_cutoff_ > 0.0f) {
        lp_filter_.RefreshFilter(
            MultiBiquad::FilterType::LowPass,
            0.0f, lp_cutoff_, sampling_rate_, 0.7071f, false
        );
    }
}

void DiffSurround::SetEnable(const bool enable) {
    if (enable_ != enable) {
        if (!enable_) Reset();
        enable_ = enable;
    }
}

void DiffSurround::SetDelayTime(const float value) {
    if (delay_time_ != value) {
        delay_time_ = value;
        Reset();
    }
}

void DiffSurround::SetReverse(const bool value) {
    if (reverse_ != value) {
        reverse_ = value;
        Reset();
    }
}

void DiffSurround::SetWetDryMix(float value) {
    wet_dry_mix_ = std::clamp(value, 0.0f, 1.0f);
}

void DiffSurround::SetLPCutoff(float value) {
    value = std::clamp(value, 0.0f, 20000.0f);
    if (lp_cutoff_ != value) {
        lp_cutoff_ = value;
        if (value > 0.0f) {
            lp_filter_.RefreshFilter(
                MultiBiquad::FilterType::LowPass,
                0.0f, value, sampling_rate_, 0.7071f, false
            );
        }
    }
}

void DiffSurround::SetSamplingRate(const uint32_t sampling_rate) {
    if (sampling_rate_ != sampling_rate) {
        sampling_rate_ = sampling_rate;
        Reset();
    }
}

void DiffSurround::ProcessPlanar(std::span<float> L, std::span<float> R) noexcept {
    if (!IsEnabled() || L.empty()) return;

    const uint32_t delayed_ch = reverse_ ? 0u : 1u;
    const uint32_t direct_ch  = 1u - delayed_ch;
    const float wet = wet_dry_mix_;
    const float dry = 1.0f - wet;

    // Use scratch for aliasing safety when L/R point into the same buffer region,
    // but process directly in the planar domain — no interleave/deinterleave.
    float* const ch[2] = {L.data(), R.data()};

    for (size_t i = 0; i < L.size(); ++i) {
        const float direct_in  = ch[direct_ch][i];
        const float delayed_in = ch[delayed_ch][i];

        // Both channels clock through their ring delays every sample.
        const float direct_out  = ring_[direct_ch].Process(direct_in);
        float       delayed_out = ring_[delayed_ch].Process(delayed_in);

        if (lp_cutoff_ > 0.0f) {
            delayed_out = static_cast<float>(lp_filter_.ProcessSample(delayed_out));
        }

        ch[direct_ch][i]  = direct_out;
        // Dry term = the delayed channel's OWN undelayed input.  The previous
        // `dry * direct_out` cross-fed the direct channel into the delayed
        // channel, collapsing stereo to mono at wet_dry_mix_ = 0 (100% dry).
        ch[delayed_ch][i] = dry * delayed_in + wet * delayed_out;
    }
}
