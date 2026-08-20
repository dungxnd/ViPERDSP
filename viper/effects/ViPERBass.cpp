#include "ViPERBass.h"
#include <algorithm>
#include <cmath>
#include <numbers>
#include <utility>

ViPERBass::ViPERBass() {
    // Pre-allocate scratch_buffer_ for worst-case 4096 stereo frames so that
    // ProcessSubwoofer never hits a null/empty buffer at default 44100 Hz
    // (SetSamplingRate skips allocation when the rate equals the default).
    scratch_buffer_.resize(4096u * 2u, 0.0f);
    for (auto& bq : biquad_) {
        bq.Reset();
        bq.SetLowPassParameter(static_cast<float>(frequency_), sampling_rate_, 0.53f);
    }
    subwoofer_.SetBassGain(sampling_rate_, 0.0f);
    Reset();
}

void ViPERBass::ShapeMix(
    std::span<float> samples, const uint32_t i, float bass_l, float bass_r
) noexcept {
    const float yl = dc_block_coeff_ * (dc_y1_[0] + bass_l - dc_x1_[0]);
    dc_x1_[0] = bass_l;  dc_y1_[0] = yl;
    const float yr = dc_block_coeff_ * (dc_y1_[1] + bass_r - dc_x1_[1]);
    dc_x1_[1] = bass_r;  dc_y1_[1] = yr;

    samples[i]     = BassSoftClip(samples[i]     + BassSoftClip(yl, 0.8f), 0.95f);
    samples[i + 1] = BassSoftClip(samples[i + 1] + BassSoftClip(yr, 0.8f), 0.95f);
}

void ViPERBass::ApplyAntiPop(float& bass_l, float& bass_r) noexcept {
    if (anti_pop_ >= 1.0f) [[likely]] return;
    bass_l   *= anti_pop_;
    bass_r   *= anti_pop_;
    anti_pop_ = std::min(anti_pop_ + anti_pop_step_, 1.0f);
}

void ViPERBass::ProcessNaturalBass(std::span<float> samples) noexcept {
    const uint32_t size = static_cast<uint32_t>(samples.size() / 2);
    [[assume(samples.size() % 2 == 0)]];
    for (uint32_t i = 0; i < size * 2; i += 2) {
        bass_factor_smoothed_ +=
            (bass_factor_ - bass_factor_smoothed_) * smoothing_coeff_;
        float bass_l = static_cast<float>(biquad_[0].ProcessSample(samples[i]))
                       * bass_factor_smoothed_;
        float bass_r = static_cast<float>(biquad_[1].ProcessSample(samples[i + 1]))
                       * bass_factor_smoothed_;
        ApplyAntiPop(bass_l, bass_r);
        ShapeMix(samples, i, bass_l, bass_r);
    }
}

void ViPERBass::ProcessPureBassPlus(std::span<float> samples) noexcept {
    const uint32_t size = static_cast<uint32_t>(samples.size() / 2);

    if (!wave_buffer_.PushSamples(samples.data(), size)) return;

    // Write biquad-filtered bass into the delay buffer at the current write head.
    // write_offset is the frame index of the samples just pushed.
    float* const buffer         = wave_buffer_.GetBuffer();
    const uint32_t write_offset = (wave_buffer_.GetBufferOffset() - size) * 2u;

    for (uint32_t i = 0; i < size * 2; i += 2) {
        buffer[write_offset + i]     = static_cast<float>(biquad_[0].ProcessSample(samples[i]));
        buffer[write_offset + i + 1] = static_cast<float>(biquad_[1].ProcessSample(samples[i + 1]));
    }

    // Polyphase may return 0 until it has accumulated 1008 frames.
    // We ALWAYS pop from wave_buffer_ so it stays in sync with the input stream.
    // Only mix the bass output when polyphase has actually produced data.
    const bool polyphase_ready = (polyphase_.Process(samples.data(), size) == size);

    if (polyphase_ready) {
        for (uint32_t i = 0; i < size * 2; i += 2) {
            bass_factor_smoothed_ +=
                (bass_factor_ - bass_factor_smoothed_) * smoothing_coeff_;
            float bass_l = buffer[i]     * bass_factor_smoothed_;
            float bass_r = buffer[i + 1] * bass_factor_smoothed_;
            ApplyAntiPop(bass_l, bass_r);
            ShapeMix(samples, i, bass_l, bass_r);
        }
    }
    // Always pop to keep the delay line in sync regardless of polyphase output.
    wave_buffer_.PopSamples(size, true);
}

void ViPERBass::ProcessSubwoofer(std::span<float> samples) noexcept {
    const uint32_t size = static_cast<uint32_t>(samples.size() / 2);

    if (anti_pop_ >= 1.0f) [[likely]] {
        subwoofer_.Process(samples.data(), size);
        return;
    }

    // Anti-pop blend: process a copy of the dry signal through the subwoofer,
    // then lerp between dry and wet per frame as the ramp advances.
    // scratch_buffer_ is pre-sized in Reset() to avoid RT-thread allocation.
    const uint32_t sample_count = size * 2u;
    std::copy_n(samples.data(), sample_count, scratch_buffer_.data());
    subwoofer_.Process(scratch_buffer_.data(), size);

    for (uint32_t i = 0; i < sample_count; i += 2) {
        samples[i]     += anti_pop_ * (scratch_buffer_[i]     - samples[i]);
        samples[i + 1] += anti_pop_ * (scratch_buffer_[i + 1] - samples[i + 1]);
        anti_pop_ = std::min(anti_pop_ + anti_pop_step_, 1.0f);
    }
}

void ViPERBass::Process(std::span<float> samples) noexcept {
    if (!enable_ || samples.empty()) return;
    [[assume(samples.size() % 2 == 0)]];

    using enum ProcessMode;
    switch (process_mode_) {
        case NaturalBass:  ProcessNaturalBass (samples); break;
        case PureBassPlus: ProcessPureBassPlus(samples); break;
        case Subwoofer:    ProcessSubwoofer   (samples); break;
        default:           return; // unreachable in practice; guard against corrupt IPC value
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
    // 20 ms anti-pop ramp (down from 1000 ms).
    anti_pop_step_         = 1.0f / (0.020f * sr_f);
    anti_pop_              = 0.0f;
    smoothing_coeff_       = 1.0f - std::exp(-1.0f / (0.030f * sr_f));
    bass_factor_smoothed_  = bass_factor_;
    dc_block_coeff_        = std::exp(-2.0f * std::numbers::pi_v<float> * 18.0f / sr_f);
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
    // Clamp to [0, 2] so an invalid IPC integer never reaches std::unreachable().
    const auto safe_mode = static_cast<ProcessMode>(
        std::clamp(static_cast<int>(mode), 0, 2));
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
        // Pre-size the scratch buffer for the worst-case frame count before Reset()
        // so that ProcessSubwoofer never allocates on the RT thread.
        // Maximum frame count assumed: 4096 stereo frames = 8192 samples.
        scratch_buffer_.resize(4096u * 2u);
        // Reset() recalculates all SR-dependent state.
        Reset();
    }
}
