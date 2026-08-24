#include "PlaybackGain.h"
#include <algorithm>
#include <cmath>

PlaybackGain::PlaybackGain() {
    // All members already initialized by in-class defaults.
    biquad1_.SetBandPassParameter(kBandpassFreq, sampling_rate_, kBandpassQ);
    biquad2_.SetBandPassParameter(kBandpassFreq, sampling_rate_, kBandpassQ);
}

void PlaybackGain::Reset() noexcept {
    biquad1_.SetBandPassParameter(kBandpassFreq, sampling_rate_, kBandpassQ);
    biquad2_.SetBandPassParameter(kBandpassFreq, sampling_rate_, kBandpassQ);
    current_gain_l_ = 1.0f;
    current_gain_r_ = 1.0f;
    ramp_progress_  = 0;
}

void PlaybackGain::SetEnable(const bool enable) noexcept {
    if (enable_ != enable) {
        if (enable) Reset();
        enable_ = enable;
    }
}

void PlaybackGain::SetMaxGainFactor(const float max_gain_factor) noexcept {
    max_gain_factor_ = max_gain_factor;
}

void PlaybackGain::SetRatio(const float ratio) noexcept {
    ratio1_ = ratio + 1.0f;
    ratio2_ = 1.0f / ratio1_;
}

void PlaybackGain::SetVolume(const float volume) noexcept {
    volume_ = volume;
}

void PlaybackGain::SetSamplingRate(const uint32_t sampling_rate) noexcept {
    if (sampling_rate_ == sampling_rate) return;
    sampling_rate_ = sampling_rate;
    ramp_frames_   = static_cast<uint32_t>(sampling_rate * kWarmupSeconds);
    Reset();
}

void PlaybackGain::ProcessPlanar(float* __restrict L, float* __restrict R, const size_t frames) noexcept {
    if (!IsEnabled() || frames == 0) return;

    double tmp_l = 0.0, tmp_r = 0.0;
    for (size_t i = 0u; i < frames; ++i) {
        const double sl = biquad1_.ProcessSample(L[i]);
        const double sr = biquad2_.ProcessSample(R[i]);
        tmp_l += sl * sl;
        tmp_r += sr * sr;
    }
    const double analyzed = std::max(std::max(tmp_l, tmp_r) / static_cast<double>(frames), 1e-10);
    const double a = std::log(analyzed);

    const auto n = static_cast<uint32_t>(frames);
    if (ramp_progress_ < ramp_frames_) {
        ramp_progress_ = std::min(ramp_progress_ + n, ramp_frames_);
    }
    const double ramp = ramp_frames_ == 0 ? 1.0
        : static_cast<double>(ramp_progress_) / static_cast<double>(ramp_frames_);

    const double b      = a * log_coeff_ * 10.0 + 23.0;
    const double c      = ramp * (b * ratio2_ - b);
    const double d      = c / 100.0;
    const double target = std::pow(10.0, (c - d * d * 50.0) / 20.0) * volume_;

    const uint32_t ramp_len = std::max(n, sampling_rate_ / 40u);

    // Left channel
    {
        double g = (target - current_gain_l_) / ramp_len;
        if (g >= 0.0) g *= 0.0625;
        for (size_t i = 0u; i < frames; ++i) {
            L[i] *= current_gain_l_;
            current_gain_l_ = std::clamp(
                current_gain_l_ + static_cast<float>(g),
                -max_gain_factor_, max_gain_factor_
            );
        }
    }
    // Right channel
    {
        double g = (target - current_gain_r_) / ramp_len;
        if (g >= 0.0) g *= 0.0625;
        for (size_t i = 0u; i < frames; ++i) {
            R[i] *= current_gain_r_;
            current_gain_r_ = std::clamp(
                current_gain_r_ + static_cast<float>(g),
                -max_gain_factor_, max_gain_factor_
            );
        }
    }
}
