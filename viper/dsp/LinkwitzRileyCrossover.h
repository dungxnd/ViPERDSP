#pragma once
#include "../utils/StereoBiquadSIMD.h"
#include <array>
#include <cstddef>
#include <cstdint>

namespace viper::dsp {

template <uint32_t MaxBands>
class LinkwitzRileyCrossover {
public:
    static constexpr uint32_t kMaxCrossovers = MaxBands - 1u;
    static constexpr float    kButterworthQ  = 0.70710678f;

    void Configure(const float* freqs, uint32_t num_crossovers, uint32_t sr) noexcept {
        const float fs = static_cast<float>(sr);
        for (uint32_t i = 0; i < num_crossovers && i < kMaxCrossovers; ++i) {
            lp_coeffs_[i] = StereoBiquadCoeffs::MakeLowPass (freqs[i], kButterworthQ, fs);
            hp_coeffs_[i] = StereoBiquadCoeffs::MakeHighPass(freqs[i], kButterworthQ, fs);
        }
    }

    void Reset() noexcept {
        for (auto& s : lp_state1_) s.Reset();
        for (auto& s : lp_state2_) s.Reset();
        for (auto& s : hp_state1_) s.Reset();
        for (auto& s : hp_state2_) s.Reset();
    }

    // Filter band `b` of `num_bands` in-place on planar out_l/out_r.
    // Caller must pre-fill out_l/out_r with a copy of the input before calling.
    void ProcessBand(uint32_t b, uint32_t num_bands,
                     float* __restrict out_l, float* __restrict out_r,
                     size_t frames) noexcept {
        const uint32_t nc = num_bands - 1u;
        if (b == 0u) {
            StereoBiquadBank::Process(lp_coeffs_[0], lp_state1_[0], out_l, out_r, frames);
            StereoBiquadBank::Process(lp_coeffs_[0], lp_state2_[0], out_l, out_r, frames);
        } else if (b == nc) {
            StereoBiquadBank::Process(hp_coeffs_[nc-1u], hp_state1_[nc-1u], out_l, out_r, frames);
            StereoBiquadBank::Process(hp_coeffs_[nc-1u], hp_state2_[nc-1u], out_l, out_r, frames);
        } else {
            StereoBiquadBank::Process(hp_coeffs_[b-1u], hp_state1_[b-1u], out_l, out_r, frames);
            StereoBiquadBank::Process(hp_coeffs_[b-1u], hp_state2_[b-1u], out_l, out_r, frames);
            StereoBiquadBank::Process(lp_coeffs_[b],    lp_state1_[b],    out_l, out_r, frames);
            StereoBiquadBank::Process(lp_coeffs_[b],    lp_state2_[b],    out_l, out_r, frames);
        }
    }

private:
    std::array<StereoBiquadCoeffs, kMaxCrossovers> lp_coeffs_{};
    std::array<StereoBiquadCoeffs, kMaxCrossovers> hp_coeffs_{};
    std::array<StereoBiquadState,  kMaxCrossovers> lp_state1_{};
    std::array<StereoBiquadState,  kMaxCrossovers> lp_state2_{};
    std::array<StereoBiquadState,  kMaxCrossovers> hp_state1_{};
    std::array<StereoBiquadState,  kMaxCrossovers> hp_state2_{};
};

} // namespace viper::dsp
