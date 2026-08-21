#include "ViPERBass.h"
#include <algorithm>
#include <cmath>
#include <mdspan>
#include <numbers>
#include <ranges>
#include <utility>

ViPERBass::ViPERBass() {
    // scratch_buffer_ must be valid before any Process() call, including at the
    // default 44100 Hz rate where SetSamplingRate() is never invoked.
    // PureBassPlus needs 4*frames floats (2*frames FIR + 2*frames delayed bass).
    // Size to 4096*4 so any block size up to 4096 frames is safe without RT realloc.
    scratch_buffer_.resize(4096u * 4u, 0.0f);
    for (auto& bq : biquad_) {
        bq.Reset();
        bq.SetLowPassParameter(static_cast<float>(frequency_), sampling_rate_, 0.53f);
    }
    subwoofer_.SetBassGain(sampling_rate_, 0.0f);
    Reset();
}

void ViPERBass::ShapeMix(float& left, float& right, float bass_l, float bass_r) noexcept {
    const float yl = dc_block_coeff_ * (dc_y1_[0] + bass_l - dc_x1_[0]);
    dc_x1_[0] = bass_l;  dc_y1_[0] = yl;
    const float yr = dc_block_coeff_ * (dc_y1_[1] + bass_r - dc_x1_[1]);
    dc_x1_[1] = bass_r;  dc_y1_[1] = yr;

    left  = BassSoftClip(left  + BassSoftClip(yl, 0.8f), 0.95f);
    right = BassSoftClip(right + BassSoftClip(yr, 0.8f), 0.95f);
}

void ViPERBass::ApplyAntiPop(float& bass_l, float& bass_r) noexcept {
    if (anti_pop_ >= 1.0f) [[likely]] return;
    bass_l   *= anti_pop_;
    bass_r   *= anti_pop_;
    anti_pop_ = std::min(anti_pop_ + anti_pop_step_, 1.0f);
}

void ViPERBass::ProcessNaturalBass(StereoView audio) noexcept {
    for (size_t f = 0; f < audio.extent(0); ++f) {
        bass_factor_smoothed_ +=
            (bass_factor_ - bass_factor_smoothed_) * smoothing_coeff_;
        float bass_l = static_cast<float>(biquad_[0].ProcessSample(audio[f, 0]))
                       * bass_factor_smoothed_;
        float bass_r = static_cast<float>(biquad_[1].ProcessSample(audio[f, 1]))
                       * bass_factor_smoothed_;
        ApplyAntiPop(bass_l, bass_r);
        ShapeMix(audio[f, 0], audio[f, 1], bass_l, bass_r);
    }
}

// PureBass+ design:
//
//   x[n] ─► [ Biquad LPF ] ─► b[n] ─► [ ring delay D=31 ] ─► b_del[n]  ─┐
//   x[n] ─► [ copy to scratch ]                                           │
//            [ Polyphase FIR (D=31 output lag) ] ─► fir[n]               │
//                                                                         ▼
//                                                            [ ShapeMix ] ─► out[n]
//
// Both paths converge with zero net phase offset.
void ViPERBass::ProcessPureBassPlus(std::span<float> samples, StereoView audio) noexcept {
    const size_t frames = audio.extent(0);

    // ── Step 1: biquad + ring delay ──────────────────────────────────────────
    // Re-use the scratch buffer front half to stash per-frame delayed bass,
    // then use the same buffer for the Polyphase FIR pass below.
    // scratch_buffer_ layout: [0 .. 2*frames-1] = biquad-delayed bass (L,R,L,R,...)
    //                         (overwritten by FIR in step 3)
    // We need both results simultaneously, so keep delayed bass in a stack VLA-
    // equivalent via audio[f,0/1] temporary storage trick — instead write to a
    // local span pointing to the *back* half of scratch_buffer_.

    // ── Use first 2*frames for scratch FIR input, last 2*frames for delayed bass.
    // Total needed: 4 * frames.  scratch_buffer_ is pre-sized to 4096*2 = 8192
    // floats which covers any realistic block size (≤ 4096 frames).
    float* const fir_buf  = scratch_buffer_.data();                 // [0 .. 2f-1]
    float* const bass_buf = scratch_buffer_.data() + frames * 2u;  // [2f .. 4f-1]

    for (size_t f = 0; f < frames; ++f) {
        // Biquad low-pass on raw input.
        const float b_l = static_cast<float>(biquad_[0].ProcessSample(audio[f, 0]));
        const float b_r = static_cast<float>(biquad_[1].ProcessSample(audio[f, 1]));

        // Push into ring delay.
        bass_delay_[0][delay_write_idx_] = b_l;
        bass_delay_[1][delay_write_idx_] = b_r;

        // Read out sample delayed by exactly kLatency (31) frames.
        const size_t read_idx = (delay_write_idx_ - Polyphase::kLatency) & kDelayMask;
        delay_write_idx_ = (delay_write_idx_ + 1u) & kDelayMask;

        bass_buf[f * 2u]      = bass_delay_[0][read_idx];
        bass_buf[f * 2u + 1u] = bass_delay_[1][read_idx];

        // Copy raw input for FIR.
        fir_buf[f * 2u]      = audio[f, 0];
        fir_buf[f * 2u + 1u] = audio[f, 1];
    }

    // ── Step 2: mid/high FIR (introduces kLatency=31 frames group delay) ─────
    polyphase_.Process(fir_buf, static_cast<uint32_t>(frames));

    // ── Step 3: coherent mix ─────────────────────────────────────────────────
    for (size_t f = 0; f < frames; ++f) {
        bass_factor_smoothed_ +=
            (bass_factor_ - bass_factor_smoothed_) * smoothing_coeff_;

        float bass_l = bass_buf[f * 2u]      * bass_factor_smoothed_;
        float bass_r = bass_buf[f * 2u + 1u] * bass_factor_smoothed_;
        ApplyAntiPop(bass_l, bass_r);

        float fir_l = fir_buf[f * 2u];
        float fir_r = fir_buf[f * 2u + 1u];
        ShapeMix(fir_l, fir_r, bass_l, bass_r);

        audio[f, 0] = fir_l;
        audio[f, 1] = fir_r;
    }
}

void ViPERBass::ProcessSubwoofer(std::span<float> samples, StereoView audio) noexcept {
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

void ViPERBass::Process(std::span<float> samples) noexcept {
    if (!enable_ || samples.empty()) return;
    const size_t size = samples.size();
    [[assume(size % 2 == 0)]];

    StereoView audio(samples.data(), size / 2, 2u);

    using enum ProcessMode;
    switch (process_mode_) {
        case NaturalBass:  ProcessNaturalBass (audio);          break;
        case PureBassPlus: ProcessPureBassPlus(samples, audio); break;
        case Subwoofer:    ProcessSubwoofer   (samples, audio); break;
        default:           return; // guard against corrupt IPC value
    }
}

void ViPERBass::Reset() noexcept {
    polyphase_.SetSamplingRate(sampling_rate_);
    polyphase_.Reset();
    subwoofer_.Reset(); // clear biquad delay-state before reconfiguring coefficients
    subwoofer_.SetBassGain(sampling_rate_, bass_factor_ * 2.5f);

    for (auto& ch : bass_delay_) {
        std::ranges::fill(ch, 0.0f);
    }
    delay_write_idx_ = 0u;

    const auto sr_f = static_cast<float>(sampling_rate_);
    for (auto& bq : biquad_) {
        bq.SetLowPassParameter(static_cast<float>(frequency_), sampling_rate_, 0.53f);
    }
    anti_pop_step_        = 1.0f / (0.020f * sr_f);
    anti_pop_             = 0.0f;
    smoothing_coeff_      = 1.0f - std::exp(-1.0f / (0.030f * sr_f));
    bass_factor_smoothed_ = bass_factor_;
    dc_block_coeff_       = std::exp(-2.0f * std::numbers::pi_v<float> * 18.0f / sr_f);
    dc_x1_.fill(0.0f);
    dc_y1_.fill(0.0f);
}

void ViPERBass::SetEnable(const bool enable) noexcept {
    if (enable_ != enable) {
        if (enable) Reset();
        enable_ = enable;
    }
}

void ViPERBass::SetProcessMode(const ProcessMode mode) noexcept {
    const auto safe_mode = static_cast<ProcessMode>(
        std::clamp(std::to_underlying(mode), uint8_t{0}, uint8_t{2}));
    if (process_mode_ != safe_mode) {
        process_mode_ = safe_mode;
        Reset();
    }
}

void ViPERBass::SetBassFactor(const float value) noexcept {
    if (bass_factor_ != value) {
        bass_factor_ = value;
        subwoofer_.SetBassGain(sampling_rate_, bass_factor_ * 2.5f);
    }
}

void ViPERBass::SetFrequency(const uint32_t value) noexcept {
    if (frequency_ != value) {
        frequency_ = value;
        for (auto& bq : biquad_) {
            bq.SetLowPassParameter(
                static_cast<float>(frequency_), sampling_rate_, 0.53f);
        }
    }
}

void ViPERBass::SetAntiPop(const bool enable) noexcept {
    anti_pop_ = enable ? 0.0f : 1.0f;
}

void ViPERBass::SetSamplingRate(const uint32_t sampling_rate) noexcept {
    if (sampling_rate_ != sampling_rate) {
        sampling_rate_ = sampling_rate;
        scratch_buffer_.resize(4096u * 4u);
        Reset();
    }
}
