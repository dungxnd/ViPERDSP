#include "PassFilter.h"
#include <algorithm>

PassFilter::PassFilter() {
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

void PassFilter::ProcessPlanar(float* __restrict L, float* __restrict R, const size_t frames) noexcept {
    for (size_t x = 0; x < frames; ++x) {
        L[x] = FilterLH(filters_[2], L[x]);
        L[x] = FilterLH(filters_[0], L[x]);
        R[x] = FilterLH(filters_[3], R[x]);
        R[x] = FilterLH(filters_[1], R[x]);
    }
}

void PassFilter::Reset() noexcept {
    // Clamp the low-pass cutoff to a safe fraction of Nyquist.  The previous
    // `fs - 100` fallback exceeded Nyquist at every rate below 44.1 kHz, which
    // made tan(pi*f/fs) negative and pushed the IIR pole outside the unit
    // circle (|a1| > 1) — instant ±inf/NaN divergence.
    const float cutoff = std::min(18000.0f, static_cast<float>(sampling_rate_) * 0.45f);

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
