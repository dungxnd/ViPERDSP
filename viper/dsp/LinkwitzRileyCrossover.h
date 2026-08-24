#pragma once
#include "../utils/StereoBiquadSIMD.h"
#include <array>
#include <cstddef>
#include <cstdint>

namespace viper::dsp {

// Phase-corrected Linkwitz-Riley 4th-order (LR-4) crossover.
//
// For an N-band split (nc = N-1 crossovers at f_0 < f_1 < ... < f_{nc-1}),
// the standard sequential-tree topology assigns each band b:
//
//   H_b(z) = [∏_{i<b}  HP_LR4(f_i)]
//           · [LP_LR4(f_b)          ]  (if b < nc)
//           · [∏_{j>b}  AP_LR4(f_j)]  (phase compensation)
//
// where AP_LR4(f_j) = LP_LR4(f_j) + HP_LR4(f_j) is the 4th-order allpass
// that arises from the Linkwitz-Riley identity.  The AP terms compensate the
// phase difference between sub-trees in the sequential cascade so that:
//
//   ∑_b H_b(z) = ∏_{k=0}^{nc-1} AP_LR4(f_k)   (unit magnitude at all ω)
//
// Without this compensation, omitting the AP on the edge bands creates deep
// destructive interference notches in the transition regions whenever the bands
// are recombined (e.g. MultibandCompressor at 1:1 ratio, StereoImager at 100%).
//
// State layout: each band has its own independent state arrays so that
// concurrent ProcessBand() calls on distinct scratch buffers do not alias.
//
// Reference: Cecchi & Välimäki et al., "Crossover Networks: A Review",
//            JAES 2023, Section 2.2, Eqs. (7)–(9), Fig. 4(b).

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
            ap_coeffs_[i] = StereoBiquadCoeffs::MakeAllPass  (freqs[i], kButterworthQ, fs);
        }
    }

    void Reset() noexcept {
        for (auto& row : lp_state1_) for (auto& s : row) s.Reset();
        for (auto& row : lp_state2_) for (auto& s : row) s.Reset();
        for (auto& row : hp_state1_) for (auto& s : row) s.Reset();
        for (auto& row : hp_state2_) for (auto& s : row) s.Reset();
        for (auto& row : ap_state1_) for (auto& s : row) s.Reset();
        for (auto& row : ap_state2_) for (auto& s : row) s.Reset();
    }

    // Filter band `b` of `num_bands` in-place on planar out_l/out_r.
    // Caller must pre-fill out_l/out_r with a fresh copy of the input before
    // each call; bands must be processed in order b = 0 .. num_bands-1.
    //
    // Phase-corrected tree topology (3-band example, nc=2):
    //   Band 0 (Low):  LP_LR4(f0)^2                · AP_LR4(f1)^2
    //   Band 1 (Mid):  HP_LR4(f0)^2 · LP_LR4(f1)^2  (no AP — already in the middle)
    //   Band 2 (High): HP_LR4(f0)^2 · HP_LR4(f1)^2  (no AP — top band)
    void ProcessBand(const uint32_t b, const uint32_t num_bands,
                     float* __restrict out_l, float* __restrict out_r,
                     const size_t frames) noexcept {
        const uint32_t nc = num_bands - 1u;

        // Step 1: HP_LR4 for every crossover strictly below this band.
        for (uint32_t i = 0u; i < b; ++i) {
            StereoBiquadBank::Process(hp_coeffs_[i], hp_state1_[b][i], out_l, out_r, frames);
            StereoBiquadBank::Process(hp_coeffs_[i], hp_state2_[b][i], out_l, out_r, frames);
        }

        // Step 2: LP_LR4 at this band's upper crossover (skip for the highest band).
        if (b < nc) {
            StereoBiquadBank::Process(lp_coeffs_[b], lp_state1_[b][b], out_l, out_r, frames);
            StereoBiquadBank::Process(lp_coeffs_[b], lp_state2_[b][b], out_l, out_r, frames);
        }

        // Step 3: AP_LR4 for every crossover strictly above this band's top.
        // For 2-band (nc==1): this loop is always empty — no regression vs old code.
        for (uint32_t j = b + 1u; j < nc; ++j) {
            StereoBiquadBank::Process(ap_coeffs_[j], ap_state1_[b][j], out_l, out_r, frames);
            StereoBiquadBank::Process(ap_coeffs_[j], ap_state2_[b][j], out_l, out_r, frames);
        }
    }

private:
    // Coefficients (shared across bands — read-only during ProcessBand).
    std::array<StereoBiquadCoeffs, kMaxCrossovers> lp_coeffs_{};
    std::array<StereoBiquadCoeffs, kMaxCrossovers> hp_coeffs_{};
    std::array<StereoBiquadCoeffs, kMaxCrossovers> ap_coeffs_{};

    // Per-band, per-crossover independent states.
    // lp/hp/ap_stateN_[b][i]: state for band b's filter at crossover index i.
    // Only the entries (b, i) that are actually used in ProcessBand are accessed.
    using StateMatrix = std::array<std::array<StereoBiquadState, kMaxCrossovers>, MaxBands>;
    StateMatrix lp_state1_{};
    StateMatrix lp_state2_{};
    StateMatrix hp_state1_{};
    StateMatrix hp_state2_{};
    StateMatrix ap_state1_{};
    StateMatrix ap_state2_{};
};

} // namespace viper::dsp
