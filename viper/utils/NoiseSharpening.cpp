#include "NoiseSharpening.h"
#include <algorithm>

NoiseSharpening::NoiseSharpening() noexcept {
    Reset();
}

void NoiseSharpening::Process(float* buffer, const uint32_t size) noexcept {
    for (uint32_t i = 0; i < size; ++i) {
        const float sample_l = buffer[i * 2];
        const float sample_r = buffer[i * 2 + 1];
        const float prev_l   = in_[0];
        const float prev_r   = in_[1];
        in_[0] = sample_l;
        in_[1] = sample_r;
        const float diff_l = (sample_l - prev_l) * gain_;
        const float diff_r = (sample_r - prev_r) * gain_;

        const float sample_l_in = sample_l + diff_l;
        const float sample_r_in = sample_r + diff_r;

        float hist = sample_l_in * filters_[0].b1_;
        const float left = filters_[0].prev_sample_ + sample_l_in * filters_[0].b0_;
        filters_[0].prev_sample_ = sample_l_in * filters_[0].a1_ + hist;

        hist = sample_r_in * filters_[1].b1_;
        const float right = filters_[1].prev_sample_ + sample_r_in * filters_[1].b0_;
        filters_[1].prev_sample_ = sample_r_in * filters_[1].a1_ + hist;

        buffer[i * 2]     = left;
        buffer[i * 2 + 1] = right;
    }
}

void NoiseSharpening::Reset() noexcept {
    const float max_cutoff = static_cast<float>(sampling_rate_) * 0.45f;
    const float cutoff = std::clamp(
        static_cast<float>(sampling_rate_ / 2.0 - 1000.0), 20.0f, max_cutoff
    );
    for (auto& f : filters_) {
        f.SetLowPassFilterBW(cutoff, sampling_rate_);
        f.Mute();
    }
    in_.fill(0.0f);
}

void NoiseSharpening::SetGain(const float gain) noexcept {
    gain_ = gain;
}

void NoiseSharpening::SetSamplingRate(const uint32_t sampling_rate) noexcept {
    sampling_rate_ = sampling_rate;
    Reset();
}
