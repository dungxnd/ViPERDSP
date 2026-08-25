#include "IIR_NOrder_BW_BP.h"

IIR_NOrder_BW_BP::IIR_NOrder_BW_BP(const uint32_t order)
    : order_(order), lowpass_(order), highpass_(order) {
    for (auto& f : lowpass_)  { f.Mute(); }
    for (auto& f : highpass_) { f.Mute(); }
}

void IIR_NOrder_BW_BP::Mute() noexcept {
    for (auto& f : lowpass_)  { f.Mute(); }
    for (auto& f : highpass_) { f.Mute(); }
}

void IIR_NOrder_BW_BP::SetBPF(
    const float low_edge, const float high_edge, const uint32_t sampling_rate
) noexcept {
    for (auto& f : lowpass_)  { f.SetLowPassFilterBW(low_edge,   sampling_rate); }
    for (auto& f : highpass_) { f.SetHighPassFilterBW(high_edge, sampling_rate); }
}
