#pragma once

#include "../utils/QuadricTube.h"
#include "../utils/MultiBiquad.h"
#include <array>

class TubeSimulator {
public:
    // Model 0 = 12AX7 (default, high-gain)
    // Model 1 = 6N1J  (Soviet medium-gain, warmer character)
    enum class TubeType : int {
        k12AX7 = 0,
        k6N1J  = 1,
    };

    TubeSimulator();

    void Process(float *buffer, uint32_t size);
    void Reset();

    void SetEnable(bool enable);
    void SetTubeType(int model);
    void SetTubeMix(float mix);
    void SetTubeDrive(float drive);
    void SetSamplingRate(uint32_t sampling_rate);

private:
    bool enable_;
    TubeType tube_type_;
    float mix_amount_ = 0.3f;  // Wet/dry ratio [0.0 - 1.0]; default 30%
    uint32_t sampling_rate_;

    std::array<MultiBiquad, 2> high_pass_;
    std::array<QuadricTube, 2> tube_;
    std::array<MultiBiquad, 2> low_pass_;
};
