#include "Cure.h"
#include <array>

void Cure::Process(float *buffer, const uint32_t size) noexcept {
    if (!enabled_) return;

    crossfeed_.ProcessFrames(buffer, size);
    pass_filter_.ProcessFrames(buffer, size);
}

void Cure::Reset() noexcept {
    crossfeed_.Reset();
    pass_filter_.Reset();
}

uint32_t Cure::GetCutoff() const noexcept {
    return crossfeed_.GetCutoff();
}

float Cure::GetFeedback() const noexcept {
    return crossfeed_.GetFeedback();
}

float Cure::GetLevelDelay() const noexcept {
    return crossfeed_.GetLevelDelay();
}

Crossfeed::Preset Cure::GetPreset() const noexcept {
    return crossfeed_.GetPreset();
}

void Cure::SetEnable(const bool enable) noexcept {
    if (enabled_ != enable) {
        if (enable) Reset();
        enabled_ = enable;
    }
}

void Cure::SetCutoff(const uint32_t value) noexcept {
    crossfeed_.SetCutoff(value);
}

void Cure::SetFeedback(const float value) noexcept {
    crossfeed_.SetFeedback(value);
}

void Cure::SetPreset(const uint32_t value) noexcept {
    static constexpr std::array<Crossfeed::Preset, 3> kPresets{{
        {650, 95},
        {700, 60},
        {700, 45},
    }};
    if (value < 3) {
        crossfeed_.SetPreset(kPresets[value]);
    }
}

void Cure::SetSamplingRate(const uint32_t sampling_rate) noexcept {
    crossfeed_.SetSamplingRate(sampling_rate);
    pass_filter_.SetSamplingRate(sampling_rate);
}
