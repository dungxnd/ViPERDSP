#include "ViPERBass.h"
#include <cmath>
#include <numbers>

ViPERBass::ViPERBass()
    : polyphase_(2),
      wave_buffer_(2, 4096) {
    for (auto& bq : biquad_) {
        bq.Reset();
        bq.SetLowPassParameter(static_cast<float>(frequency_), sampling_rate_, 0.53f);
    }
    subwoofer_.SetBassGain(sampling_rate_, 0.0f);
    Reset();
}

void ViPERBass::Process(float *samples, const uint32_t size) noexcept {
    if (!enable_ || size == 0) return;

    auto soft_clip = [](const float v, const float knee) noexcept {
        const float drive = std::fabs(v);
        if (drive <= knee) return v;
        const float over   = drive - knee;
        const float shaped = knee + over / std::sqrt(1.0f + over * over);
        return v * (shaped / drive);
    };

    auto dc_block = [this](const float x, const int ch) noexcept {
        const float y = dc_block_coeff_ * (dc_y1_[ch] + x - dc_x1_[ch]);
        dc_x1_[ch] = x;
        dc_y1_[ch] = y;
        return y;
    };

    auto shape_mix = [&](float bass_l, float bass_r, const uint32_t i) noexcept {
        bass_l = soft_clip(dc_block(bass_l, 0), 0.8f);
        bass_r = soft_clip(dc_block(bass_r, 1), 0.8f);
        samples[i]     = soft_clip(samples[i]     + bass_l, 0.95f);
        samples[i + 1] = soft_clip(samples[i + 1] + bass_r, 0.95f);
    };

    switch (process_mode_) {
        case ProcessMode::NaturalBass: {
            for (uint32_t i = 0; i < size * 2; i += 2) {
                bass_factor_smoothed_ +=
                    (bass_factor_ - bass_factor_smoothed_) * smoothing_coeff_;
                float bass_l = static_cast<float>(biquad_[0].ProcessSample(samples[i]))
                               * bass_factor_smoothed_;
                float bass_r = static_cast<float>(biquad_[1].ProcessSample(samples[i + 1]))
                               * bass_factor_smoothed_;
                if (anti_pop_ < 1.0f) {
                    bass_l *= anti_pop_;
                    bass_r *= anti_pop_;
                    anti_pop_ = std::min(anti_pop_ + sampling_rate_period_, 1.0f);
                }
                shape_mix(bass_l, bass_r, i);
            }
            break;
        }
        case ProcessMode::PureBasPlus: {
            if (wave_buffer_.PushSamples(samples, size)) {
                float *buffer             = wave_buffer_.GetBuffer();
                const uint32_t buf_offset = wave_buffer_.GetBufferOffset();

                for (uint32_t i = 0; i < size * 2; i += 2) {
                    buffer[buf_offset - size + i] =
                        static_cast<float>(biquad_[0].ProcessSample(samples[i]));
                    buffer[buf_offset - size + i + 1] =
                        static_cast<float>(biquad_[1].ProcessSample(samples[i + 1]));
                }

                if (polyphase_.Process(samples, size) == size) {
                    for (uint32_t i = 0; i < size * 2; i += 2) {
                        bass_factor_smoothed_ +=
                            (bass_factor_ - bass_factor_smoothed_) * smoothing_coeff_;
                        float bass_l = buffer[i]     * bass_factor_smoothed_;
                        float bass_r = buffer[i + 1] * bass_factor_smoothed_;
                        if (anti_pop_ < 1.0f) {
                            bass_l *= anti_pop_;
                            bass_r *= anti_pop_;
                            anti_pop_ = std::min(anti_pop_ + sampling_rate_period_, 1.0f);
                        }
                        shape_mix(bass_l, bass_r, i);
                    }
                    wave_buffer_.PopSamples(size, true);
                }
            }
            break;
        }
        case ProcessMode::Subwoofer: {
            if (anti_pop_ < 1.0f) {
                for (uint32_t i = 0; i < size * 2; i += 2) {
                    const float dry_l = samples[i];
                    const float dry_r = samples[i + 1];
                    float tmp[2] = {dry_l, dry_r};
                    subwoofer_.Process(tmp, 1);
                    samples[i]     = dry_l + anti_pop_ * (tmp[0] - dry_l);
                    samples[i + 1] = dry_r + anti_pop_ * (tmp[1] - dry_r);
                    anti_pop_ = std::min(anti_pop_ + sampling_rate_period_, 1.0f);
                }
            } else {
                subwoofer_.Process(samples, size);
            }
            break;
        }
    }
}

void ViPERBass::Reset() noexcept {
    polyphase_.SetSamplingRate(sampling_rate_);
    polyphase_.Reset();
    wave_buffer_.Reset();
    wave_buffer_.PushZeros(polyphase_.GetLatency());
    subwoofer_.SetBassGain(sampling_rate_, bass_factor_ * 2.5f);

    const float sr_f = static_cast<float>(sampling_rate_);
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
        sampling_rate_        = sampling_rate;
        sampling_rate_period_ = 1.0f / static_cast<float>(sampling_rate);
        polyphase_.SetSamplingRate(sampling_rate_);
        for (auto& bq : biquad_) {
            bq.SetLowPassParameter(
                static_cast<float>(frequency_), sampling_rate_, 0.53f);
        }
        subwoofer_.SetBassGain(sampling_rate_, bass_factor_ * 2.5f);
    }
}
