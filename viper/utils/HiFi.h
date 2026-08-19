#pragma once

#include "IIR_NOrder_BW_BP.h"
#include "IIR_NOrder_BW_LH.h"
#include "WaveBuffer.h"
#include <array>
#include <cstdint>
#include <memory>

class HiFi {
public:
    HiFi();
    ~HiFi() = default;

    // Non-copyable (owns unique_ptrs)
    HiFi(const HiFi&)            = delete;
    HiFi& operator=(const HiFi&) = delete;
    HiFi(HiFi&&)                 = default;
    HiFi& operator=(HiFi&&)      = default;

    void Process(float* samples, uint32_t size);
    void Reset();

    void SetClarity(float value) noexcept;
    void SetSamplingRate(uint32_t sampling_rate);

private:
    uint32_t sampling_rate_{44100u};
    float    gain_{1.0f};

    struct FilterSet {
        std::unique_ptr<IIR_NOrder_BW_LH> lowpass;
        std::unique_ptr<IIR_NOrder_BW_LH> highpass;
        std::unique_ptr<IIR_NOrder_BW_BP> bandpass;
    };

    std::array<std::unique_ptr<WaveBuffer>, 2> buffers_;
    std::array<FilterSet, 2>                   filters_;
};
