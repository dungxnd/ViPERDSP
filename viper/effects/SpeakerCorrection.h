#pragma once

#include "../utils/Biquad.h"
#include "../utils/MultiBiquad.h"
#include <array>
#include <cstdint>

class SpeakerCorrection {
public:
    SpeakerCorrection();

    void Process(float *samples, uint32_t size) noexcept;
    void ProcessPlanar(float* __restrict L, float* __restrict R, size_t frames) noexcept;
    void Reset() noexcept;

    [[nodiscard]] bool IsEnabled() const noexcept { return enable_; }
    void SetEnable(bool enable) noexcept;
    void SetHighPassCutoff(uint32_t value) noexcept;
    void SetLowPassCutoff(uint32_t value) noexcept;
    void SetBandPassCenter(uint32_t value) noexcept;
    void SetBandPassQ(float value) noexcept;
    void SetSamplingRate(uint32_t sampling_rate) noexcept;

private:
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
    alignas(64) std::array<float, 4096u * 2u> pp_scratch_{};
};
