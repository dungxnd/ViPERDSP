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

void MultibandCompressor::Process(float* const samples, const uint32_t size) noexcept {
    if (!enable_ || size == 0) return;
    if (size > kMaxFrames) return;

    const uint32_t num_crossovers = band_count_ - 1;
    const uint32_t frame_count    = size * 2;

    for (uint32_t b = 0; b < band_count_; ++b) {
        for (uint32_t i = 0; i < frame_count; i += 2) {
            double sample_l = static_cast<double>(samples[i]);
            double sample_r = static_cast<double>(samples[i + 1]);

            crossover_.ProcessSampleStereo(b, band_count_, sample_l, sample_r);

            band_buffers_[b][i]     = static_cast<float>(sample_l);
            band_buffers_[b][i + 1] = static_cast<float>(sample_r);
        }

        compressors_[b].Process(band_buffers_[b].data(), size);
    }

    for (uint32_t i = 0; i < frame_count; i += 2) {
        float sum_l = 0.0f;
        float sum_r = 0.0f;
        for (uint32_t b = 0; b < band_count_; ++b) {
            sum_l += band_buffers_[b][i];
            sum_r += band_buffers_[b][i + 1];
        }
        samples[i]     = sum_l;
        samples[i + 1] = sum_r;
    }
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
    if (crossover_freqs_[index] == frequency) return;
    crossover_freqs_[index] = frequency;
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

void MultibandCompressor::ProcessPlanar(float* __restrict L, float* __restrict R, const size_t frames) noexcept {
    if (!IsEnabled() || frames == 0) return;
    const auto n = static_cast<uint32_t>(frames);
    // Process() already owns band_buffers_ — reuse band_buffers_[0] as interleave
    // scratch; it is unconditionally overwritten on entry to Process().
    float* const sc = band_buffers_[0].data();
    for (size_t i = 0u; i < frames; ++i) {
        sc[2u * i]      = L[i];
        sc[2u * i + 1u] = R[i];
    }
    Process(sc, n);
    for (size_t i = 0u; i < frames; ++i) {
        L[i] = sc[2u * i];
        R[i] = sc[2u * i + 1u];
    }
}
