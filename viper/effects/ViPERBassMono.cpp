#include "ViPERBassMono.h"
#include <algorithm>
#include <cmath>
#include <numbers>

ViPERBassMono::ViPERBassMono() {
    biquad_.Reset();
    biquad_.SetLowPassParameter(static_cast<float>(frequency_), sampling_rate_, 0.53f);
    subwoofer_.SetBassGain(sampling_rate_, 0.0f);
    Reset();
}

void ViPERBassMono::Process(float *samples, const uint32_t size) noexcept {
    if (!enable_ || size == 0) return;

    if (anti_pop_ < 1.0f) {
        for (uint32_t i = 0; i < size * 2; i += 2) {
            samples[i]     *= anti_pop_;
            samples[i + 1] *= anti_pop_;
            anti_pop_ = std::min(anti_pop_ + sampling_rate_period_, 1.0f);
        }
    }

    auto soft_clip = [](const float v, const float knee) noexcept {
        const float drive = std::fabs(v);
        if (drive <= knee) return v;
        const float over   = drive - knee;
        const float shaped = knee + over / std::sqrt(1.0f + over * over);
        return v * (shaped / drive);
    };

    auto shape_mix = [&](float bass, const uint32_t i) noexcept {
        const float y = dc_block_coeff_ * (dc_y1_ + bass - dc_x1_);
        dc_x1_ = bass;
        dc_y1_ = y;
        bass = soft_clip(y, 0.8f);
        samples[i]     = soft_clip(samples[i]     + bass, 0.95f);
        samples[i + 1] = soft_clip(samples[i + 1] + bass, 0.95f);
    };

    switch (process_mode_) {
        case ProcessMode::NaturalBass: {
            for (uint32_t i = 0; i < size * 2; i += 2) {
                bass_factor_smoothed_ +=
                    (bass_factor_ - bass_factor_smoothed_) * smoothing_coeff_;
                const double sample = (static_cast<double>(samples[i])
                                       + static_cast<double>(samples[i + 1])) / 2.0;
                const float x = static_cast<float>(biquad_.ProcessSample(sample))
                                 * bass_factor_smoothed_;
                shape_mix(x, i);
            }
            break;
        }
        case ProcessMode::PureBassPlus: {
            if (wave_buffer_.PushSamples(samples, size)) {
                float *buffer             = wave_buffer_.GetBuffer();
                const uint32_t buf_offset = wave_buffer_.GetBufferOffset();

                for (uint32_t i = 0; i < size * 2; i += 2) {
                    const double sample = (static_cast<double>(samples[i])
                                           + static_cast<double>(samples[i + 1])) / 2.0;
                    buffer[buf_offset - size + i / 2] =
                        static_cast<float>(biquad_.ProcessSample(sample));
                }

                if (polyphase_.Process(samples, size) == size) {
                    for (uint32_t i = 0; i < size * 2; i += 2) {
                        bass_factor_smoothed_ +=
                            (bass_factor_ - bass_factor_smoothed_) * smoothing_coeff_;
                        shape_mix(buffer[i / 2] * bass_factor_smoothed_, i);
                    }
                    wave_buffer_.PopSamples(size, true);
                }
            }
            break;
        }
        case ProcessMode::Subwoofer: {
            subwoofer_.Process(samples, size);
            break;
        }
    }
}

void ViPERBassMono::Reset() noexcept {
    polyphase_.SetSamplingRate(sampling_rate_);
    polyphase_.Reset();
    wave_buffer_.Reset();
    wave_buffer_.PushZeros(polyphase_.GetLatency());
    subwoofer_.SetBassGain(sampling_rate_, bass_factor_ * 2.5f);
    biquad_.SetLowPassParameter(static_cast<float>(frequency_), sampling_rate_, 0.53f);

    const auto sr_f        = static_cast<float>(sampling_rate_);
    sampling_rate_period_  = 1.0f / sr_f;
    anti_pop_              = 0.0f;
    smoothing_coeff_       = 1.0f - std::exp(-1.0f / (0.030f * sr_f));
    bass_factor_smoothed_  = bass_factor_;
    dc_block_coeff_        = std::exp(-2.0f * std::numbers::pi_v<float> * 18.0f / sr_f);
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
    if (process_mode_ != mode) {
        process_mode_ = mode;
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
        sampling_rate_        = sampling_rate;
        sampling_rate_period_ = 1.0f / static_cast<float>(sampling_rate);
        polyphase_.SetSamplingRate(sampling_rate_);
        biquad_.SetLowPassParameter(
            static_cast<float>(frequency_), sampling_rate_, 0.53f);
        subwoofer_.SetBassGain(sampling_rate_, bass_factor_ * 2.5f);
    }
}
