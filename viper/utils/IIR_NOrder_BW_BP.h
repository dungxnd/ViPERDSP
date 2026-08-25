#pragma once

#include "IIR_1st.h"
#include <cstdint>
#include <vector>

class IIR_NOrder_BW_BP {
public:
    explicit IIR_NOrder_BW_BP(uint32_t order);

    void Mute() noexcept;
    // Band-pass as LP(low_edge) · HP(high_edge): the low-pass keeps everything
    // below the UPPER edge and the high-pass keeps everything above the LOWER
    // edge, passing [high_edge, low_edge].  The parameter names are the band
    // edges, NOT the filter they feed — `low_edge` goes to the low-pass and
    // `high_edge` to the high-pass.
    void SetBPF(float low_edge, float high_edge, uint32_t sampling_rate) noexcept;

    uint32_t order_;

    std::vector<IIR_1st> lowpass_;
    std::vector<IIR_1st> highpass_;
};

inline float FilterBPLowPass(IIR_NOrder_BW_BP& filter, float sample) noexcept {
    for (uint32_t idx = 0; idx < filter.order_; ++idx) {
        sample = Filter(&filter.lowpass_[idx], sample);
    }
    return sample;
}

inline float FilterBPHighPass(IIR_NOrder_BW_BP& filter, float sample) noexcept {
    for (uint32_t idx = 0; idx < filter.order_; ++idx) {
        sample = Filter(&filter.highpass_[idx], sample);
    }
    return sample;
}

inline float FilterBP(IIR_NOrder_BW_BP& filter, const float sample) noexcept {
    return FilterBPHighPass(filter, FilterBPLowPass(filter, sample));
}
