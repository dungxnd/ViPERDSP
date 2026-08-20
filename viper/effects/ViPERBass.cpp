#include "ViPERBass.h"
#include <cmath>
#include <numbers>
#include <vector>

ViPERBass::ViPERBass() {
    for (auto& bq : biquad_) {
        bq.Reset();
        bq.SetLowPassParameter(static_cast<float>(frequency_), sampling_rate_, 0.53f);
    }
    subwoofer_.SetBassGain(sampling_rate_, 0.0f);
    Reset();
}

namespace {

[[nodiscard]] float SoftClip(const float v, const float knee) noexcept {
    const float drive = std::fabs(v);
    if (drive <= knee) return v;
    const float over   = drive - knee;
    const float shaped = knee + over / std::sqrt(1.0f + over * over);
    return v * (shaped / drive);
}

} // anonymous namespace

void ViPERBass::ShapeMix(
    float* const samples, const uint32_t i, float bass_l, float bass_r
) noexcept {
    const float yl = dc_block_coeff_ * (dc_y1_[0] + bass_l - dc_x1_[0]);
    dc_x1_[0] = bass_l;  dc_y1_[0] = yl;
    const float yr = dc_block_coeff_ * (dc_y1_[1] + bass_r - dc_x1_[1]);
    dc_x1_[1] = bass_r;  dc_y1_[1] = yr;

    samples[i]     = SoftClip(samples[i]     + SoftClip(yl, 0.8f), 0.95f);
    samples[i + 1] = SoftClip(samples[i + 1] + SoftClip(yr, 0.8f), 0.95f);
}

void ViPERBass::ApplyAntiPop(float& bass_l, float& bass_r) noexcept {
    if (anti_pop_ >= 1.0f) return;
    bass_l *= anti_pop_;
    bass_r *= anti_pop_;
    anti_pop_ = std::min(anti_pop_ + sampling_rate_period_, 1.0f);
}

void ViPERBass::ProcessNaturalBass(float* const samples, const uint32_t size) noexcept {
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

void ViPERBass::ProcessPureBassPlus(float* const samples, const uint32_t size) noexcept {
    if (!wave_buffer_.PushSamples(samples, size)) return;

    // GetBufferOffset() returns frames (index_ / channels_).
    // Convert to float offset: multiply by 2 (stereo).
    float* const buffer            = wave_buffer_.GetBuffer();
    const uint32_t write_offset    = (wave_buffer_.GetBufferOffset() - size) * 2u;

    for (uint32_t i = 0; i < size * 2; i += 2) {
        buffer[write_offset + i]     = static_cast<float>(biquad_[0].ProcessSample(samples[i]));
        buffer[write_offset + i + 1] = static_cast<float>(biquad_[1].ProcessSample(samples[i + 1]));
    }

    if (polyphase_.Process(samples, size) != size) return;

    for (uint32_t i = 0; i < size * 2; i += 2) {
        bass_factor_smoothed_ +=
            (bass_factor_ - bass_factor_smoothed_) * smoothing_coeff_;
        float bass_l = buffer[i]     * bass_factor_smoothed_;
        float bass_r = buffer[i + 1] * bass_factor_smoothed_;
        ApplyAntiPop(bass_l, bass_r);
        ShapeMix(samples, i, bass_l, bass_r);
    }
    wave_buffer_.PopSamples(size, true);
}

void ViPERBass::ProcessSubwoofer(float* const samples, const uint32_t size) noexcept {
    if (anti_pop_ >= 1.0f) {
        subwoofer_.Process(samples, size);
        return;
    }
    // Anti-pop ramp: copy dry, process full block, then blend per-frame.
    // Avoids per-sample Process() calls that prevent vectorization.
    std::vector<float> wet(samples, samples + size * 2u);
    subwoofer_.Process(wet.data(), size);
    for (uint32_t i = 0; i < size * 2; i += 2) {
        samples[i]     += anti_pop_ * (wet[i]     - samples[i]);
        samples[i + 1] += anti_pop_ * (wet[i + 1] - samples[i + 1]);
        anti_pop_ = std::min(anti_pop_ + sampling_rate_period_, 1.0f);
    }
}

void ViPERBass::Process(float* const samples, const uint32_t size) noexcept {
    if (!enable_ || size == 0) return;

    using enum ProcessMode;
    switch (process_mode_) {
        case NaturalBass: ProcessNaturalBass(samples, size); break;
        case PureBassPlus: ProcessPureBassPlus(samples, size); break;
        case Subwoofer:   ProcessSubwoofer  (samples, size); break;
    }
}

void ViPERBass::Reset() noexcept {
    polyphase_.SetSamplingRate(sampling_rate_);
    polyphase_.Reset();
    wave_buffer_.Reset();
    wave_buffer_.PushZeros(polyphase_.GetLatency());
    subwoofer_.SetBassGain(sampling_rate_, bass_factor_ * 2.5f);

    const auto sr_f = static_cast<float>(sampling_rate_);
    for (auto& bq : biquad_) {
        bq.SetLowPassParameter(static_cast<float>(frequency_), sampling_rate_, 0.53f);
    }
    sampling_rate_period_  = 1.0f / sr_f;
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
    if (process_mode_ != mode) {
        process_mode_ = mode;
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
        // Reset() recalculates all SR-dependent state: dc_block_coeff_,
        // smoothing_coeff_, sampling_rate_period_, polyphase latency,
        // wave_buffer_ pre-fill, biquad coeffs, and subwoofer gain.
        Reset();
    }
}
