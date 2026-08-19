#include "PlaybackGain.h"
#include <algorithm>
#include <cmath>

PlaybackGain::PlaybackGain() {
    // All members already initialized by in-class defaults.
    biquad1_.SetBandPassParameter(kBandpassFreq, sampling_rate_, kBandpassQ);
    biquad2_.SetBandPassParameter(kBandpassFreq, sampling_rate_, kBandpassQ);
}

void PlaybackGain::Process(float* const samples, const uint32_t size) noexcept {
    if (!enable_ || size == 0) return;

    const double analyzed = std::max(AnalyseWave(samples, size), 1e-10);
    const double a        = std::log(analyzed);

    if (ramp_progress_ < ramp_frames_) {
        ramp_progress_ = std::min(ramp_progress_ + size, ramp_frames_);
    }
    const double ramp = ramp_frames_ == 0
                        ? 1.0
                        : static_cast<double>(ramp_progress_)
                          / static_cast<double>(ramp_frames_);

    const double b      = a * log_coeff_ * 10.0 + 23.0;
    const double c      = ramp * (b * ratio2_ - b);
    const double d      = c / 100.0;
    const double target = std::pow(10.0, (c - d * d * 50.0) / 20.0) * volume_;

    const uint32_t ramp_len = std::max(size, sampling_rate_ / 40u);

    for (uint32_t ch = 0; ch < 2; ++ch) {
        float& gain = (ch == 0) ? current_gain_l_ : current_gain_r_;
        double g    = (target - gain) / ramp_len;
        if (g >= 0.0) g *= 0.0625;

        for (uint32_t i = 0; i < size; ++i) {
            samples[i * 2 + ch] *= gain;
            gain = std::clamp(gain + static_cast<float>(g),
                              -max_gain_factor_, max_gain_factor_);
        }
    }
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

double PlaybackGain::AnalyseWave(const float* const samples, const uint32_t size) noexcept {
    double tmp_l = 0.0;
    double tmp_r = 0.0;

    for (uint32_t i = 0; i < size * 2; i += 2) {
        const double s_l = biquad1_.ProcessSample(samples[i]);
        const double s_r = biquad2_.ProcessSample(samples[i + 1]);
        tmp_l += s_l * s_l;
        tmp_r += s_r * s_r;
    }

    return std::max(tmp_l, tmp_r) / static_cast<double>(size);
}
