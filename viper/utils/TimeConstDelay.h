#pragma once

#include <cstdint>
#include <vector>

class TimeConstDelay {
public:
    TimeConstDelay() noexcept = default;

    float ProcessSample(float sample) noexcept;
    void  SetParameters(uint32_t sampling_rate, float delay);

private:
    uint32_t           offset_       = 0;
    uint32_t           sample_count_ = 0;
    std::vector<float> samples_;
};
