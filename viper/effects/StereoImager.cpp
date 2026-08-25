#include "StereoImager.h"
#include <algorithm>
#include <array>

StereoImager::StereoImager() {
    ConfigureCrossovers();
}

void StereoImager::Process(float *samples, const uint32_t size) noexcept {
    if (!enable_ || size == 0) return;
    if (size > kMaxFrames) return;

    // Band-sum accumulator — zero-filled.  Each band copies the ORIGINAL input
    // (L/R, untouched until the final copy-out) into scratch, filters it, and
    // accumulates into accum_l_/accum_r_.  The input must NOT seed the
    // accumulator: that would double the signal (Input + Σbands = 2·Input) and
    // corrupt band 1..N with band 0's output.
    std::fill_n(accum_l_.data(), size, 0.0f);
    std::fill_n(accum_r_.data(), size, 0.0f);

    for (uint32_t b = 0; b < kNumBands; ++b) {
        // Copy the original interleaved input into planar scratch
        for (uint32_t f = 0; f < size; ++f) {
            scratch_l_[f] = samples[f * 2u];
            scratch_r_[f] = samples[f * 2u + 1u];
        }
        // Block crossover filter for band b (in-place on scratch)
        crossover_.ProcessBand(b, kNumBands, scratch_l_.data(), scratch_r_.data(), size);
        // Apply stereo width; accumulate into the band-sum accumulator
        const float w = band_widths_[b];
        for (uint32_t f = 0; f < size; ++f) {
            const float mid  = (scratch_l_[f] + scratch_r_[f]) * 0.5f;
            const float side = (scratch_l_[f] - scratch_r_[f]) * 0.5f * w;
            accum_l_[f] += mid + side;
            accum_r_[f] += mid - side;
        }
    }

    // Copy the band-summed result to the interleaved output.
    for (uint32_t f = 0; f < size; ++f) {
        samples[f * 2u]      = accum_l_[f];
        samples[f * 2u + 1u] = accum_r_[f];
    }
}

void StereoImager::Reset() noexcept {
    crossover_.Reset();
    ConfigureCrossovers();
}

void StereoImager::SetEnable(const bool enable) noexcept {
    if (enable_ != enable) {
        if (enable) Reset();
        enable_ = enable;
    }
}

void StereoImager::SetLowWidth(const float value) noexcept {
    band_widths_[0] = value / 100.0f;
}

void StereoImager::SetMidWidth(const float value) noexcept {
    band_widths_[1] = value / 100.0f;
}

void StereoImager::SetHighWidth(const float value) noexcept {
    band_widths_[2] = value / 100.0f;
}

void StereoImager::SetLowCrossover(const float value) noexcept {
    const float clamped = ClampCrossover(value);
    if (crossover_freqs_[0] != clamped) {
        crossover_freqs_[0] = clamped;
        ConfigureCrossovers();
    }
}

void StereoImager::SetHighCrossover(const float value) noexcept {
    const float clamped = ClampCrossover(value);
    if (crossover_freqs_[1] != clamped) {
        crossover_freqs_[1] = clamped;
        ConfigureCrossovers();
    }
}

void StereoImager::SetSamplingRate(const uint32_t sampling_rate) noexcept {
    if (sampling_rate_ != sampling_rate) {
        sampling_rate_ = sampling_rate;
        ConfigureCrossovers();
    }
}

void StereoImager::ConfigureCrossovers() noexcept {
    crossover_.Configure(crossover_freqs_.data(), kNumCrossovers, sampling_rate_);
}

float StereoImager::ClampCrossover(const float value) const noexcept {
    // A 0 Hz crossover would put a pole exactly on the unit circle (y[n] =
    // 2·y[n-1] - y[n-2]) and explode on any non-zero signal; anything at or
    // above Nyquist degenerates into a divide-by-zero / useless filter.
    // Fall back to the neutral defaults, then clamp into [20 Hz, 0.45·fs].
    const float valid = value > 0.0f ? value : 200.0f;
    const float max_freq = static_cast<float>(sampling_rate_) * 0.45f;
    return std::clamp(valid, 20.0f, max_freq);
}

void StereoImager::ProcessPlanar(std::span<float> L, std::span<float> R) noexcept {
    if (!IsEnabled() || L.empty()) return;
    if (L.size() > kMaxFrames) return;

    const auto size = static_cast<uint32_t>(L.size());

    // Band-sum accumulator — zero-filled.  Each band copies the ORIGINAL input
    // from L/R (untouched until the final copy-out) into scratch, so band 0's
    // output never seeds band 1..N and the final sum is Σbands (no +6 dB).
    std::fill_n(accum_l_.data(), size, 0.0f);
    std::fill_n(accum_r_.data(), size, 0.0f);

    for (uint32_t b = 0u; b < kNumBands; ++b) {
        // Copy the ORIGINAL input into planar scratch — the crossover filters
        // the scratch in-place.
        std::copy_n(L.data(), size, scratch_l_.data());
        std::copy_n(R.data(), size, scratch_r_.data());
        crossover_.ProcessBand(b, kNumBands, scratch_l_.data(), scratch_r_.data(), size);
        // Apply stereo width; accumulate into the band-sum accumulator
        const float w = band_widths_[b];
        #pragma clang loop vectorize(enable)
        for (uint32_t f = 0u; f < size; ++f) {
            const float mid  = (scratch_l_[f] + scratch_r_[f]) * 0.5f;
            const float side = (scratch_l_[f] - scratch_r_[f]) * 0.5f * w;
            accum_l_[f] += mid + side;
            accum_r_[f] += mid - side;
        }
    }

    // Copy the band-summed result to the output.
    std::copy_n(accum_l_.data(), size, L.data());
    std::copy_n(accum_r_.data(), size, R.data());
}
