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
    scratch_buffer_.resize(4096u * 2u, 0.0f);
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

void ViPERBass::ProcessPureBassPlus(std::span<float> samples, StereoView audio) noexcept {
    const uint32_t size = static_cast<uint32_t>(audio.extent(0));

    if (!wave_buffer_.PushSamples(samples)) return;

    float* const buffer         = wave_buffer_.GetBuffer();
    const uint32_t write_offset = (wave_buffer_.GetBufferOffset() - size) * 2u;

    StereoView delay_write(buffer + write_offset, size, 2u);
    for (size_t f = 0; f < size; ++f) {
        delay_write[f, 0] = static_cast<float>(biquad_[0].ProcessSample(audio[f, 0]));
        delay_write[f, 1] = static_cast<float>(biquad_[1].ProcessSample(audio[f, 1]));
    }

    // Polyphase accumulates 1008 frames before producing output; PopSamples is
    // called unconditionally so the delay line stays aligned regardless.
    const bool polyphase_ready = (polyphase_.Process(samples) == size);

    if (polyphase_ready) {
        StereoView delay_read(buffer, size, 2u);
        for (size_t f = 0; f < size; ++f) {
            bass_factor_smoothed_ +=
                (bass_factor_ - bass_factor_smoothed_) * smoothing_coeff_;
            float bass_l = delay_read[f, 0] * bass_factor_smoothed_;
            float bass_r = delay_read[f, 1] * bass_factor_smoothed_;
            ApplyAntiPop(bass_l, bass_r);
            ShapeMix(audio[f, 0], audio[f, 1], bass_l, bass_r);
        }
    }
    wave_buffer_.PopSamples(size, true);
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

void ViPERBass::Reset() noexcept {
    polyphase_.SetSamplingRate(sampling_rate_);
    polyphase_.Reset();
    wave_buffer_.Reset();
    wave_buffer_.PushZeros(polyphase_.GetLatency());
    subwoofer_.Reset(); // clear biquad delay-state before reconfiguring coefficients
    subwoofer_.SetBassGain(sampling_rate_, bass_factor_ * 2.5f);

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
        scratch_buffer_.resize(4096u * 2u);
        Reset();
    }
}
