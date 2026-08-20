#include "ViPERBassMono.h"
#include <algorithm>
#include <cmath>
#include <mdspan>
#include <numbers>
#include <utility>

ViPERBassMono::ViPERBassMono() {
    // scratch_buffer_ must be valid before any Process() call, including at the
    // default 44100 Hz rate where SetSamplingRate() is never invoked.
    scratch_buffer_.resize(4096u * 2u, 0.0f);
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

void ViPERBassMono::ProcessPureBassPlus(std::span<float> samples, StereoView audio) noexcept {
    const uint32_t size = static_cast<uint32_t>(audio.extent(0));

    if (!wave_buffer_.PushSamples(samples)) return;

    float* const buffer         = wave_buffer_.GetBuffer();
    const uint32_t write_offset = (wave_buffer_.GetBufferOffset() - size) * 2u;

    // Only channel-0 carries the mono-mixed biquad signal; channel-1 is unused.
    StereoView delay_write(buffer + write_offset, size, 2u);
    for (size_t f = 0; f < size; ++f) {
        const double sample = (static_cast<double>(audio[f, 0])
                               + static_cast<double>(audio[f, 1])) * 0.5;
        delay_write[f, 0] = static_cast<float>(biquad_.ProcessSample(sample));
    }

    // Polyphase accumulates 1008 frames before producing output; PopSamples is
    // called unconditionally so the delay line stays aligned regardless.
    const bool polyphase_ready = (polyphase_.Process(samples) == size);

    if (polyphase_ready) {
        StereoView delay_read(buffer, size, 2u);
        for (size_t f = 0; f < size; ++f) {
            bass_factor_smoothed_ +=
                (bass_factor_ - bass_factor_smoothed_) * smoothing_coeff_;
            float bass = delay_read[f, 0] * bass_factor_smoothed_;
            if (anti_pop_ < 1.0f) {
                bass      *= anti_pop_;
                anti_pop_  = std::min(anti_pop_ + anti_pop_step_, 1.0f);
            }
            ShapeMix(bass, audio[f, 0], audio[f, 1]);
        }
    }
    wave_buffer_.PopSamples(size, true);
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
    wave_buffer_.Reset();
    wave_buffer_.PushZeros(polyphase_.GetLatency());
    subwoofer_.Reset(); // clear biquad delay-state before reconfiguring coefficients
    subwoofer_.SetBassGain(sampling_rate_, bass_factor_ * 2.5f);
    biquad_.SetLowPassParameter(static_cast<float>(frequency_), sampling_rate_, 0.53f);

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
        scratch_buffer_.resize(4096u * 2u);
        Reset();
    }
}
