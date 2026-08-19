#include "ViPERClarity.h"

ViPERClarity::ViPERClarity() {
    for (auto& hs : high_shelf_) {
        hs.SetFrequency(12000.0f);
        hs.SetGain(1.0f);
        hs.SetSamplingRate(sampling_rate_);
    }
    Reset();
}

void ViPERClarity::Process(float *samples, const uint32_t size) noexcept {
    if (!enable_) return;

    using enum ClarityMode;
    switch (process_mode_) {
        case Natural: {
            noise_sharpening_.Process(samples, size);
            break;
        }
        case Ozone: {
            for (uint32_t i = 0; i < size * 2; i += 2) {
                samples[i]     = static_cast<float>(high_shelf_[0].Process(samples[i]));
                samples[i + 1] = static_cast<float>(high_shelf_[1].Process(samples[i + 1]));
            }
            break;
        }
        case XHiFi: {
            hifi_.Process(samples, size);
            break;
        }
    }
}

void ViPERClarity::Reset() noexcept {
    noise_sharpening_.SetSamplingRate(sampling_rate_);
    noise_sharpening_.Reset();
    SetClarityToFilter();
    for (auto& hs : high_shelf_) {
        hs.SetFrequency(8250.0f);
        hs.SetSamplingRate(sampling_rate_);
    }
    hifi_.SetSamplingRate(sampling_rate_);
    hifi_.Reset();
}

void ViPERClarity::SetEnable(const bool enable) noexcept {
    if (enable_ != enable) {
        if (enable) Reset();
        enable_ = enable;
    }
}

void ViPERClarity::SetProcessMode(const ClarityMode mode) noexcept {
    if (process_mode_ != mode) {
        process_mode_ = mode;
        Reset();
    }
}

void ViPERClarity::SetClarityGain(const float value) noexcept {
    gain_ = value;
    if (process_mode_ == ClarityMode::Ozone) {
        Reset();
    } else {
        SetClarityToFilter();
    }
}

void ViPERClarity::SetClarityToFilter() noexcept {
    noise_sharpening_.SetGain(gain_);
    for (auto& hs : high_shelf_) hs.SetGain(gain_ + 1.0f);
    hifi_.SetClarity(gain_ + 1.0f);
}

void ViPERClarity::SetSamplingRate(const uint32_t sampling_rate) noexcept {
    if (sampling_rate_ != sampling_rate) {
        sampling_rate_ = sampling_rate;
        Reset();
    }
}
