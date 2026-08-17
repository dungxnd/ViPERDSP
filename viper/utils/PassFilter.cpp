#include "PassFilter.h"
#include "../constants.h"

PassFilter::PassFilter()
    : sampling_rate_(VIPER_DEFAULT_SAMPLING_RATE),
      filters_({IIR_NOrder_BW_LH(3), IIR_NOrder_BW_LH(3),
                IIR_NOrder_BW_LH(1), IIR_NOrder_BW_LH(1)}) {
    Reset();
}

void PassFilter::ProcessFrames(float *buffer, const uint32_t size) noexcept {
    for (uint32_t x = 0; x < size; ++x) {
        float left  = buffer[2 * x];
        float right = buffer[2 * x + 1];

        left  = FilterLH(filters_[2], left);
        left  = FilterLH(filters_[0], left);
        right = FilterLH(filters_[3], right);
        right = FilterLH(filters_[1], right);

        buffer[2 * x]     = left;
        buffer[2 * x + 1] = right;
    }
}

void PassFilter::Reset() noexcept {
    const float cutoff = (sampling_rate_ < 44100)
        ? static_cast<float>(sampling_rate_) - 100.0f
        : 18000.0f;

    filters_[0].SetLPF(cutoff, sampling_rate_);
    filters_[1].SetLPF(cutoff, sampling_rate_);
    filters_[2].SetHPF(10.0f, sampling_rate_);
    filters_[3].SetHPF(10.0f, sampling_rate_);

    for (auto& f : filters_) {
        f.Mute();
    }
}

void PassFilter::SetSamplingRate(const uint32_t sampling_rate) noexcept {
    if (sampling_rate_ != sampling_rate) {
        sampling_rate_ = sampling_rate;
        Reset();
    }
}
