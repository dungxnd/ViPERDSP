#include "IIR_NOrder_BW_LH.h"

IIR_NOrder_BW_LH::IIR_NOrder_BW_LH(const uint32_t order)
    : order_(order), filters_(order) {
    for (auto& f : filters_) {
        f.Mute();
    }
}

void IIR_NOrder_BW_LH::Mute() noexcept {
    for (auto& f : filters_) {
        f.Mute();
    }
}

void IIR_NOrder_BW_LH::SetLPF(const float frequency, const uint32_t sampling_rate) noexcept {
    for (auto& f : filters_) {
        f.SetLowPassFilterBW(frequency, sampling_rate);
    }
}

void IIR_NOrder_BW_LH::SetHPF(const float frequency, const uint32_t sampling_rate) noexcept {
    for (auto& f : filters_) {
        f.SetHighPassFilterBW(frequency, sampling_rate);
    }
}
