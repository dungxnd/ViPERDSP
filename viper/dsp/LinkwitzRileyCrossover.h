#pragma once
// LinkwitzRileyCrossover.h — Reusable LR4 crossover network.
//
// Implements a 4th-order Linkwitz-Riley (2 cascaded 2nd-order Butterworth)
// crossover for up to MaxBands bands.  Used by both MultibandCompressor
// (≤5 bands) and StereoImager (3 bands) to eliminate the duplicated 8-array
// per-crossover boilerplate.
//
// Each crossover frequency requires 8 MultiBiquad instances (LP×2 + HP×2 per
// channel pair).  This class owns all of them and provides a unified
// Configure() + Reset() + ProcessSampleStereo() interface.

#include "../utils/MultiBiquad.h"
#include <array>
#include <cstdint>

namespace viper::dsp {

template <uint32_t MaxBands>
class LinkwitzRileyCrossover {
public:
    static constexpr uint32_t kMaxCrossovers = MaxBands - 1u;
    static constexpr float    kButterworthQ  = 0.7071f;

    // Configure all crossover filters.
    // `freqs`         — array of at least `num_crossovers` frequencies (Hz)
    // `num_crossovers`— number of split points (= band_count - 1)
    // `sampling_rate` — current sample rate
    void Configure(const float* freqs, uint32_t num_crossovers,
                   uint32_t sampling_rate) noexcept {
        for (uint32_t i = 0; i < num_crossovers && i < kMaxCrossovers; ++i) {
            const float f = freqs[i];
            lowpass_la_[i].RefreshFilter(MultiBiquad::LOW_PASS,  0.0f, f, sampling_rate, kButterworthQ, false);
            lowpass_lb_[i].RefreshFilter(MultiBiquad::LOW_PASS,  0.0f, f, sampling_rate, kButterworthQ, false);
            lowpass_ra_[i].RefreshFilter(MultiBiquad::LOW_PASS,  0.0f, f, sampling_rate, kButterworthQ, false);
            lowpass_rb_[i].RefreshFilter(MultiBiquad::LOW_PASS,  0.0f, f, sampling_rate, kButterworthQ, false);
            highpass_la_[i].RefreshFilter(MultiBiquad::HIGH_PASS, 0.0f, f, sampling_rate, kButterworthQ, false);
            highpass_lb_[i].RefreshFilter(MultiBiquad::HIGH_PASS, 0.0f, f, sampling_rate, kButterworthQ, false);
            highpass_ra_[i].RefreshFilter(MultiBiquad::HIGH_PASS, 0.0f, f, sampling_rate, kButterworthQ, false);
            highpass_rb_[i].RefreshFilter(MultiBiquad::HIGH_PASS, 0.0f, f, sampling_rate, kButterworthQ, false);
        }
    }

    // Reset all filter states (does NOT reconfigure coefficients).
    void Reset() noexcept {
        for (uint32_t i = 0; i < kMaxCrossovers; ++i) {
            lowpass_la_[i].Reset();  lowpass_lb_[i].Reset();
            lowpass_ra_[i].Reset();  lowpass_rb_[i].Reset();
            highpass_la_[i].Reset(); highpass_lb_[i].Reset();
            highpass_ra_[i].Reset(); highpass_rb_[i].Reset();
        }
    }

    // Route one stereo sample through crossover band `b`.
    // `num_bands` = total band count (crossover count = num_bands - 1).
    void ProcessSampleStereo(uint32_t b, uint32_t num_bands,
                              double& l, double& r) noexcept {
        const uint32_t nc = num_bands - 1u;
        if (b == 0u) {
            l = lowpass_la_[0].ProcessSample(l);
            l = lowpass_lb_[0].ProcessSample(l);
            r = lowpass_ra_[0].ProcessSample(r);
            r = lowpass_rb_[0].ProcessSample(r);
        } else if (b == nc) {
            l = highpass_la_[nc - 1u].ProcessSample(l);
            l = highpass_lb_[nc - 1u].ProcessSample(l);
            r = highpass_ra_[nc - 1u].ProcessSample(r);
            r = highpass_rb_[nc - 1u].ProcessSample(r);
        } else {
            l = highpass_la_[b - 1u].ProcessSample(l);
            l = highpass_lb_[b - 1u].ProcessSample(l);
            l = lowpass_la_[b].ProcessSample(l);
            l = lowpass_lb_[b].ProcessSample(l);
            r = highpass_ra_[b - 1u].ProcessSample(r);
            r = highpass_rb_[b - 1u].ProcessSample(r);
            r = lowpass_ra_[b].ProcessSample(r);
            r = lowpass_rb_[b].ProcessSample(r);
        }
    }

private:
    std::array<MultiBiquad, kMaxCrossovers> lowpass_la_{};
    std::array<MultiBiquad, kMaxCrossovers> lowpass_lb_{};
    std::array<MultiBiquad, kMaxCrossovers> lowpass_ra_{};
    std::array<MultiBiquad, kMaxCrossovers> lowpass_rb_{};
    std::array<MultiBiquad, kMaxCrossovers> highpass_la_{};
    std::array<MultiBiquad, kMaxCrossovers> highpass_lb_{};
    std::array<MultiBiquad, kMaxCrossovers> highpass_ra_{};
    std::array<MultiBiquad, kMaxCrossovers> highpass_rb_{};
};

} // namespace viper::dsp
