#include "DynamicBass.h"
#include <algorithm>

DynamicBass::DynamicBass() {
    SetSamplingRate(sampling_rate_);
    high_freq_y_ = sampling_rate_ / 4;
    filter_x_.SetPassFilter(low_freq_x_, high_freq_x_);
    filter_y_.SetPassFilter(low_freq_y_, high_freq_y_);
    RecalcLowPass();
}

void DynamicBass::FilterSamples(float *samples, const uint32_t size) noexcept {
    if (low_freq_x_ <= 120) {
        for (uint32_t i = 0; i < size; ++i) {
            const float l = samples[2 * i];
            const float r = samples[2 * i + 1];
            const float avg = static_cast<float>(low_pass_.ProcessSample(l + r));
            samples[2 * i]     = l + avg;
            samples[2 * i + 1] = r + avg;
        }
    } else {
        for (uint32_t i = 0; i < size; ++i) {
            float x1, x2, x3, x4, x5, x6, y1, y2, y3, y4, y5, y6;

            filter_x_.FilterLeft (samples[2 * i],     &x1, &x2, &x3);
            filter_x_.FilterRight(samples[2 * i + 1], &x4, &x5, &x6);
            filter_y_.FilterLeft (bass_gain_ * x1,    &y1, &y2, &y3);
            filter_y_.FilterRight(bass_gain_ * x4,    &y4, &y5, &y6);

            samples[2 * i]     = x2 + y3 + side_gain_x_ * y2 + side_gain_y_ * y1 + x3;
            samples[2 * i + 1] = x5 + y6 + side_gain_x_ * y5 + side_gain_y_ * y4 + x6;
        }
    }
}

void DynamicBass::Reset() noexcept {
    filter_x_.Reset();
    filter_y_.Reset();
    RecalcLowPass();
}

void DynamicBass::SetBassGain(const float value) noexcept {
    bass_gain_ = value;
    const double x = std::min((value - 1.0) / 20.0 * 1600.0, 1600.0);
    q_peak_ = static_cast<float>(x);
    RecalcLowPass();
}

void DynamicBass::SetFilterXPassFrequency(const uint32_t low, const uint32_t high) noexcept {
    low_freq_x_  = low;
    high_freq_x_ = high;
    filter_x_.SetPassFilter(low, high);
    filter_x_.SetSamplingRate(sampling_rate_);
    RecalcLowPass();
}

void DynamicBass::SetFilterYPassFrequency(const uint32_t low, const uint32_t high) noexcept {
    low_freq_y_  = low;
    high_freq_y_ = high;
    filter_y_.SetPassFilter(low, high);
    filter_y_.SetSamplingRate(sampling_rate_);
    RecalcLowPass();
}

void DynamicBass::SetSideGain(const float gain_x, const float gain_y) noexcept {
    side_gain_x_ = gain_x;
    side_gain_y_ = gain_y;
}

void DynamicBass::SetSamplingRate(const uint32_t sampling_rate) noexcept {
    sampling_rate_ = sampling_rate;
    filter_x_.SetSamplingRate(sampling_rate);
    filter_y_.SetSamplingRate(sampling_rate);
    RecalcLowPass();
}

void DynamicBass::RecalcLowPass() noexcept {
    low_pass_.SetLowPassParameter(55.0f, sampling_rate_, q_peak_ / 666.0f + 0.5f);
}
