#pragma once

#include "../utils/QuadricTube.h"
#include "../utils/MultiBiquad.h"
#include <array>

class TubeSimulator {
public:
    // Model 0 = 12AX7 (default, high-gain)
    // Model 1 = 6N1P  (Soviet medium-gain, warm H2 character)
    // Model 2 = 12AU7 (clean, low-distortion, high headroom)
    // Model 3 = 12AT7 (high-transconductance, medium-high gain)
    // Model 4 = 6DJ8  (high-gm frame-grid triode, open/airy character)
    enum class TubeType : int {
        k12AX7 = 0,
        k6N1J  = 1,
        k12AU7 = 2,
        k12AT7 = 3,
        k6DJ8  = 4,
    };

    TubeSimulator();

    void Process(float *buffer, uint32_t size);
    void Reset();

    void SetEnable(bool enable);
    void SetTubeType(int model);
    void SetTubeMix(float mix);
    void SetTubeDrive(float drive);
    void SetTubeHpfCutoff(float cutoff_hz);  // [20 Hz – 250 Hz]; default 120 Hz
    void SetSamplingRate(uint32_t sampling_rate);

private:
    bool enable_;
    TubeType tube_type_;
    float mix_amount_    = 0.3f;    // Wet/dry ratio [0.0 - 1.0]; default 30%
    float hpf_cutoff_hz_ = 120.0f;  // Tunable HPF cutoff [20 – 250 Hz]; default 120 Hz
    uint32_t sampling_rate_;

    std::array<MultiBiquad, 2> high_pass_;
    std::array<QuadricTube, 2> tube_;
    std::array<MultiBiquad, 2> low_pass_;

    // Matched allpass filters for the dry path — same poles as HPF/LPF above,
    // so dry and wet signal accumulate identical phase rotation before blending.
    std::array<MultiBiquad, 2> dry_apf_hpf_;
    std::array<MultiBiquad, 2> dry_apf_lpf_;
};
