#include "MultibandCompressor.h"
#include <algorithm>
#include <array>

namespace {

constexpr std::array<float, 2> kDefault3BandFreqs = { 200.0f, 4000.0f };
constexpr std::array<float, 4> kDefault5BandFreqs = { 120.0f, 500.0f, 4000.0f, 8000.0f };

} // anonymous namespace

MultibandCompressor::MultibandCompressor() {
    // crossover_freqs_ already initialized by in-class default {200, 4000, 0, 0}.
    for (auto& comp : compressors_) {
        comp.SetSamplingRate(sampling_rate_);
        comp.Reset();
    }
    ConfigureCrossovers();
}

void MultibandCompressor::Reset() noexcept {
    for (auto& comp : compressors_) comp.Reset();
    crossover_.Reset();
    ConfigureCrossovers();
}

void MultibandCompressor::SetEnable(const bool enable) noexcept {
    if (enable_ != enable) {
        if (enable) Reset();
        enable_ = enable;
    }
}

void MultibandCompressor::SetBandCount(const uint32_t count) {
    if (count != 3 && count != 5) return;
    if (band_count_ == count) return;
    band_count_ = count;
    if (count == 3) {
        crossover_freqs_[0] = kDefault3BandFreqs[0];
        crossover_freqs_[1] = kDefault3BandFreqs[1];
        crossover_freqs_[2] = 0.0f;
        crossover_freqs_[3] = 0.0f;
    } else {
        std::ranges::copy(kDefault5BandFreqs, crossover_freqs_.begin());
    }
    Reset();
}

void MultibandCompressor::SetCrossoverFrequency(
    const uint32_t index, const float frequency
) noexcept {
    if (index >= band_count_ - 1) return;
    // A 0 Hz crossover puts a pole exactly on the unit circle (y[n] =
    // 2·y[n-1] - y[n-2]) and explodes on any non-zero signal; anything at or
    // above Nyquist degenerates into a useless/unstable filter.  Same clamp as
    // StereoImager::ClampCrossover: fall back to the neutral default, then
    // clamp into [20 Hz, 0.45·fs].
    const float max_freq = static_cast<float>(sampling_rate_) * 0.45f;
    const float clamped  = std::clamp(
        frequency > 0.0f ? frequency : 200.0f, 20.0f, max_freq);
    if (crossover_freqs_[index] == clamped) return;
    crossover_freqs_[index] = clamped;
    ConfigureCrossovers();
}

void MultibandCompressor::SetSamplingRate(const uint32_t sampling_rate) noexcept {
    if (sampling_rate_ == sampling_rate) return;
    sampling_rate_ = sampling_rate;
    for (auto& comp : compressors_) comp.SetSamplingRate(sampling_rate_);
    ConfigureCrossovers();
}

void MultibandCompressor::SetBandEnable(const uint32_t band, const bool enable) noexcept {
    if (band >= band_count_) return;
    compressors_[band].SetEnable(enable);
}

void MultibandCompressor::SetBandThreshold(const uint32_t band, const float value) noexcept {
    if (band >= band_count_) return;
    compressors_[band].SetThreshold(value);
}

void MultibandCompressor::SetBandRatio(const uint32_t band, const float value) noexcept {
    if (band >= band_count_) return;
    compressors_[band].SetRatio(value);
}

void MultibandCompressor::SetBandKnee(const uint32_t band, const float value) noexcept {
    if (band >= band_count_) return;
    compressors_[band].SetKnee(value);
}

void MultibandCompressor::SetBandKneeAuto(const uint32_t band, const bool enable) noexcept {
    if (band >= band_count_) return;
    compressors_[band].SetKneeAuto(enable);
}

void MultibandCompressor::SetBandGain(const uint32_t band, const float value) noexcept {
    if (band >= band_count_) return;
    compressors_[band].SetGain(value);
}

void MultibandCompressor::SetBandGainAuto(const uint32_t band, const bool enable) noexcept {
    if (band >= band_count_) return;
    compressors_[band].SetGainAuto(enable);
}

void MultibandCompressor::SetBandAttack(const uint32_t band, const float value) noexcept {
    if (band >= band_count_) return;
    compressors_[band].SetAttack(value);
}

void MultibandCompressor::SetBandAttackAuto(const uint32_t band, const bool enable) noexcept {
    if (band >= band_count_) return;
    compressors_[band].SetAttackAuto(enable);
}

void MultibandCompressor::SetBandRelease(const uint32_t band, const float value) noexcept {
    if (band >= band_count_) return;
    compressors_[band].SetRelease(value);
}

void MultibandCompressor::SetBandReleaseAuto(const uint32_t band, const bool enable) noexcept {
    if (band >= band_count_) return;
    compressors_[band].SetReleaseAuto(enable);
}

void MultibandCompressor::SetBandKneeMulti(const uint32_t band, const float value) noexcept {
    if (band >= band_count_) return;
    compressors_[band].SetKneeMulti(value);
}

void MultibandCompressor::SetBandMaxAttack(const uint32_t band, const float value) noexcept {
    if (band >= band_count_) return;
    compressors_[band].SetMaxAttack(value);
}

void MultibandCompressor::SetBandMaxRelease(const uint32_t band, const float value) noexcept {
    if (band >= band_count_) return;
    compressors_[band].SetMaxRelease(value);
}

void MultibandCompressor::SetBandCrest(const uint32_t band, const float value) noexcept {
    if (band >= band_count_) return;
    compressors_[band].SetCrest(value);
}

void MultibandCompressor::SetBandAdapt(const uint32_t band, const float value) noexcept {
    if (band >= band_count_) return;
    compressors_[band].SetAdapt(value);
}

void MultibandCompressor::SetBandNoClip(const uint32_t band, const bool enable) noexcept {
    if (band >= band_count_) return;
    compressors_[band].SetNoClip(enable);
}

void MultibandCompressor::ConfigureCrossovers() noexcept {
    crossover_.Configure(crossover_freqs_.data(), band_count_ - 1u, sampling_rate_);
}

void MultibandCompressor::ProcessPlanar(std::span<float> L, std::span<float> R) noexcept {
    if (!IsEnabled() || L.empty() || L.size() > kMaxFrames) return;
    const size_t frames = L.size();

    // Band-sum accumulator — zero-filled.  Each band copies the ORIGINAL input
    // from L/R (untouched until the final copy-out) into its scratch, filters,
    // compresses, and accumulates.  The input must NOT seed the accumulator:
    // that would double the signal (Input + Σbands = 2·Input) and corrupt
    // band 1..N with band 0's output.
    std::fill_n(accum_l_.data(), frames, 0.0f);
    std::fill_n(accum_r_.data(), frames, 0.0f);

    for (uint32_t b = 0u; b < band_count_; ++b) {
        // Copy the ORIGINAL input into band scratch.
        std::copy_n(L.data(), frames, band_scratch_l_.data());
        std::copy_n(R.data(), frames, band_scratch_r_.data());

        // Planar Linkwitz-Riley crossover for band b
        crossover_.ProcessBand(b, band_count_, band_scratch_l_.data(), band_scratch_r_.data(), static_cast<uint32_t>(frames));

        // Planar FET compression
        compressors_[b].ProcessPlanar(
            std::span<float>{band_scratch_l_.data(), frames},
            std::span<float>{band_scratch_r_.data(), frames}
        );

        // Accumulate band output into the band-sum accumulator.
#pragma clang loop vectorize(enable)
        for (size_t f = 0u; f < frames; ++f) {
            accum_l_[f] += band_scratch_l_[f];
            accum_r_[f] += band_scratch_r_[f];
        }
    }

    // Copy the band-summed result to the output.
    std::copy_n(accum_l_.data(), frames, L.data());
    std::copy_n(accum_r_.data(), frames, R.data());
}
