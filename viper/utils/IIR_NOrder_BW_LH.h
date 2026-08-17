#pragma once

#include "IIR_1st.h"
#include <vector>
#include <cstdint>

class IIR_NOrder_BW_LH {
public:
    explicit IIR_NOrder_BW_LH(uint32_t order);

    void Mute() noexcept;

    void SetLPF(float frequency, uint32_t sampling_rate) noexcept;
    void SetHPF(float frequency, uint32_t sampling_rate) noexcept;

    uint32_t order_;
    std::vector<IIR_1st> filters_;
};

[[nodiscard]] inline float FilterLH(IIR_NOrder_BW_LH *filter, float sample) noexcept {
    for (uint32_t idx = 0; idx < filter->order_; ++idx) {
        sample = Filter(&filter->filters_[idx], sample);
    }
    return sample;
}

[[nodiscard]] inline float FilterLH(IIR_NOrder_BW_LH &filter, float sample) noexcept {
    for (auto& f : filter.filters_) {
        sample = Filter(&f, sample);
    }
    return sample;
}
