#pragma once

#include <span>


#include "../utils/QuadricTube.h"
#include "../utils/QuadricTubeWDF.h"
#include "../utils/MultiBiquad.h"
#include <array>
#include <cstdint>

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

    enum class TubeMode : int {
        kStatic = 0,
        kWDF    = 1,
    };

    TubeSimulator();

    void ProcessPlanar(std::span<float> L, std::span<float> R) noexcept;
    void Reset() noexcept;

    [[nodiscard]] bool IsEnabled() const noexcept { return enable_; }
    void SetEnable(bool enable) noexcept;
    void SetTubeType(int model) noexcept;
    void SetTubeMode(int mode) noexcept;
    void SetTubeMix(float mix) noexcept;
    void SetTubeDrive(float drive) noexcept;
    void SetTubeHpfCutoff(float cutoff_hz) noexcept;  // [20 Hz – 250 Hz]; default 120 Hz
    void SetSamplingRate(uint32_t sampling_rate) noexcept;

private:
    void Process(float *buffer, uint32_t size) noexcept;

    bool     enable_{false};
    TubeType tube_type_{TubeType::k12AX7};
    TubeMode tube_mode_{TubeMode::kStatic};
    float    mix_amount_{0.3f};     // Wet/dry ratio [0.0 - 1.0]; default 30%
    float    hpf_cutoff_hz_{120.0f}; // Tunable HPF cutoff [20 – 250 Hz]; default 120 Hz
    uint32_t sampling_rate_{44100u};

    std::array<MultiBiquad, 2>    high_pass_{};
    std::array<QuadricTube, 2>    tube_{};
    std::array<QuadricTubeWDF, 2> tube_wdf_{};
    std::array<MultiBiquad, 2>    low_pass_{};

    // Matched allpass filters for the dry path — same poles as HPF/LPF above,
    // so dry and wet signal accumulate identical phase rotation before blending.
    std::array<MultiBiquad, 2> dry_apf_hpf_{};
    std::array<MultiBiquad, 2> dry_apf_lpf_{};
};
