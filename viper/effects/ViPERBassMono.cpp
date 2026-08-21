#include "ViPERBassMono.h"
#include <algorithm>
#include <cmath>
#include <mdspan>
#include <numbers>
#include <ranges>
#include <utility>

ViPERBassMono::ViPERBassMono() {
    // scratch_buffer_ must be valid before any Process() call, including at the
    // default 44100 Hz rate where SetSamplingRate() is never invoked.
    // PureBassPlus needs 3*frames floats (2*frames FIR + 1*frames delayed bass).
    // Size to 4096*4 for headroom consistency with ViPERBass and future safety.
    scratch_buffer_.resize(4096u * 4u, 0.0f);
    biquad_.Reset();
    biquad_.SetLowPassParameter(static_cast<float>(frequency_), sampling_rate_, 0.53f);
    subwoofer_.SetBassGain(sampling_rate_, 0.0f);
    Reset();
}

void ViPERBassMono::ShapeMix(const float bass, float& left, float& right) noexcept {
    const float y = dc_block_coeff_ * (dc_y1_ + bass - dc_x1_);
    dc_x1_ = bass;
    dc_y1_ = y;
    const float clipped = BassSoftClip(y, 0.8f);
    left  = BassSoftClip(left  + clipped, 0.95f);
    right = BassSoftClip(right + clipped, 0.95f);
}

void ViPERBassMono::ProcessNaturalBass(StereoView audio) noexcept {
    for (size_t f = 0; f < audio.extent(0); ++f) {
        bass_factor_smoothed_ +=
            (bass_factor_ - bass_factor_smoothed_) * smoothing_coeff_;
        const double sample = (static_cast<double>(audio[f, 0])
                               + static_cast<double>(audio[f, 1])) * 0.5;
        float bass = static_cast<float>(biquad_.ProcessSample(sample))
                     * bass_factor_smoothed_;
        if (anti_pop_ < 1.0f) {
            bass      *= anti_pop_;
            anti_pop_  = std::min(anti_pop_ + anti_pop_step_, 1.0f);
        }
        ShapeMix(bass, audio[f, 0], audio[f, 1]);
    }
}

// PureBass+ mono design:
//
//   mid(x[n]) ─► [ Biquad LPF ] ─► b[n] ─► [ ring delay D=31 ] ─► b_del[n] ─┐
//   x[n]      ─► [ copy to scratch ]                                          │
//               [ Polyphase FIR (D=31 output lag) ] ─► fir[n]                │
//                                                                             ▼
//                                                              [ ShapeMix ] ─► out[n]
void ViPERBassMono::ProcessPureBassPlus(std::span<float> samples, StereoView audio) noexcept {
    const size_t frames = audio.extent(0);

    // scratch_buffer_ layout:
    //   [0 .. 2f-1]  : FIR input (interleaved stereo copy of raw input)
    //   [2f .. 3f-1] : delayed mono bass per frame
    float* const fir_buf  = scratch_buffer_.data();
    float* const bass_buf = scratch_buffer_.data() + frames * 2u;

    for (size_t f = 0; f < frames; ++f) {
        // Mono-mix and biquad low-pass.
        const double sample = (static_cast<double>(audio[f, 0])
                               + static_cast<double>(audio[f, 1])) * 0.5;
        const float b = static_cast<float>(biquad_.ProcessSample(sample));

        // Push into mono ring delay.
        bass_delay_[delay_write_idx_] = b;
        const size_t read_idx = (delay_write_idx_ - Polyphase::kLatency) & kDelayMask;
        delay_write_idx_ = (delay_write_idx_ + 1u) & kDelayMask;

        bass_buf[f] = bass_delay_[read_idx];

        // Copy raw stereo input for FIR.
        fir_buf[f * 2u]      = audio[f, 0];
        fir_buf[f * 2u + 1u] = audio[f, 1];
    }

    // FIR over stereo copy — introduces kLatency=31 group delay.
    polyphase_.Process(fir_buf, static_cast<uint32_t>(frames));

    for (size_t f = 0; f < frames; ++f) {
        bass_factor_smoothed_ +=
            (bass_factor_ - bass_factor_smoothed_) * smoothing_coeff_;

        float bass = bass_buf[f] * bass_factor_smoothed_;
        if (anti_pop_ < 1.0f) {
            bass      *= anti_pop_;
            anti_pop_  = std::min(anti_pop_ + anti_pop_step_, 1.0f);
        }

        float fir_l = fir_buf[f * 2u];
        float fir_r = fir_buf[f * 2u + 1u];
        ShapeMix(bass, fir_l, fir_r);

        audio[f, 0] = fir_l;
        audio[f, 1] = fir_r;
    }
}

void ViPERBassMono::ProcessSubwoofer(std::span<float> samples, StereoView audio) noexcept {
    if (anti_pop_ >= 1.0f) [[likely]] {
        subwoofer_.Process(samples);
        return;
    }

    // Crossfade dry→wet over anti_pop_ ramp to eliminate the onset transient.
    const uint32_t size         = static_cast<uint32_t>(audio.extent(0));
    const uint32_t sample_count = size * 2u;
    std::copy_n(samples.data(), sample_count, scratch_buffer_.data());
    subwoofer_.Process(scratch_buffer_.data(), size);

    const StereoView wet(scratch_buffer_.data(), size, 2u);
    for (size_t f = 0; f < size; ++f) {
        audio[f, 0] += anti_pop_ * (wet[f, 0] - audio[f, 0]);
        audio[f, 1] += anti_pop_ * (wet[f, 1] - audio[f, 1]);
        anti_pop_ = std::min(anti_pop_ + anti_pop_step_, 1.0f);
    }
}

void ViPERBassMono::Process(std::span<float> samples) noexcept {
    if (!enable_ || samples.empty()) return;
    [[assume(samples.size() % 2 == 0)]];

    StereoView audio(samples.data(), samples.size() / 2, 2u);

    using enum ProcessMode;
    switch (process_mode_) {
        case NaturalBass:  ProcessNaturalBass (audio);          break;
        case PureBassPlus: ProcessPureBassPlus(samples, audio); break;
        case Subwoofer:    ProcessSubwoofer   (samples, audio); break;
        default:           return; // guard against corrupt IPC value
    }
}

void ViPERBassMono::Reset() noexcept {
    polyphase_.SetSamplingRate(sampling_rate_);
    polyphase_.Reset();
    subwoofer_.Reset(); // clear biquad delay-state before reconfiguring coefficients
    subwoofer_.SetBassGain(sampling_rate_, bass_factor_ * 2.5f);
    biquad_.SetLowPassParameter(static_cast<float>(frequency_), sampling_rate_, 0.53f);

    std::ranges::fill(bass_delay_, 0.0f);
    delay_write_idx_ = 0u;

    const auto sr_f       = static_cast<float>(sampling_rate_);
    anti_pop_step_        = 1.0f / (0.020f * sr_f);
    anti_pop_             = 0.0f;
    smoothing_coeff_      = 1.0f - std::exp(-1.0f / (0.030f * sr_f));
    bass_factor_smoothed_ = bass_factor_;
    dc_block_coeff_       = std::exp(-2.0f * std::numbers::pi_v<float> * 18.0f / sr_f);
    dc_x1_ = 0.0f;
    dc_y1_ = 0.0f;
}

void ViPERBassMono::SetEnable(const bool enable) noexcept {
    if (enable_ != enable) {
        if (enable) Reset();
        enable_ = enable;
    }
}

void ViPERBassMono::SetProcessMode(const ProcessMode mode) noexcept {
    const auto safe_mode = static_cast<ProcessMode>(
        std::clamp(std::to_underlying(mode), uint8_t{0}, uint8_t{2}));
    if (process_mode_ != safe_mode) {
        process_mode_ = safe_mode;
        Reset();
    }
}

void ViPERBassMono::SetBassFactor(const float value) noexcept {
    if (bass_factor_ != value) {
        bass_factor_ = value;
        subwoofer_.SetBassGain(sampling_rate_, bass_factor_ * 2.5f);
    }
}

void ViPERBassMono::SetFrequency(const uint32_t value) noexcept {
    if (frequency_ != value) {
        frequency_ = value;
        biquad_.SetLowPassParameter(
            static_cast<float>(frequency_), sampling_rate_, 0.53f);
    }
}

void ViPERBassMono::SetAntiPop(const bool enable) noexcept {
    anti_pop_ = enable ? 0.0f : 1.0f;
}

void ViPERBassMono::SetSamplingRate(const uint32_t sampling_rate) noexcept {
    if (sampling_rate_ != sampling_rate) {
        sampling_rate_ = sampling_rate;
        scratch_buffer_.resize(4096u * 4u);
        Reset();
    }
}
