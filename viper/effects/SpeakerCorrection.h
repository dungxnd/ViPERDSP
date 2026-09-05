#pragma once

#include <span>


#include "../include/ViPERParams.h"
#include "../utils/Biquad.h"
#include "../utils/MultiBiquad.h"
#include <array>
#include <cstdint>

class SpeakerCorrection {
public:
    using Config = viper::SpeakerCorrectionParams;

    SpeakerCorrection();

    // True planar in-place — no interleave/deinterleave, no scratch buffer.
    void ProcessPlanar(std::span<float> L, std::span<float> R) noexcept;
    void Reset() noexcept;

    [[nodiscard]] bool IsEnabled() const noexcept { return config_.enable; }
    void SetConfig(const Config& config) noexcept;
    [[nodiscard]] const Config& GetConfig() const noexcept { return config_; }

    void SetEnable(bool enable) noexcept;
    void SetHighPassCutoff(uint32_t value) noexcept;
    void SetLowPassCutoff(uint32_t value) noexcept;
    void SetBandPassCenter(uint32_t value) noexcept;
    void SetBandPassQ(float value) noexcept;
    void SetSamplingRate(uint32_t sampling_rate) noexcept;

private:
    Config config_{};
    bool enable_{false};

    uint32_t sampling_rate_{44100u};
    uint32_t hp_cutoff_{80u};
    uint32_t lp_cutoff_{13500u};
    uint32_t bp_center_{420u};
    float    bp_q_{3.88f};

    std::array<MultiBiquad, 2> high_pass_{};
    std::array<Biquad, 2>      low_pass_{};
    std::array<Biquad, 2>      band_pass_{};

    void RefreshHighPass() noexcept;
    void RefreshLowPass() noexcept;
    void RefreshBandPass() noexcept;
};
