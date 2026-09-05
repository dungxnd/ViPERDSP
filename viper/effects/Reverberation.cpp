#include "Reverberation.h"

Reverberation::Reverberation() {
    model_.SetRoomSize(0.0f);
    model_.SetWidth(0.0f);
    model_.SetDamp(0.0f);
    model_.SetWet(0.0f);
    model_.SetDry(0.5f);
    model_.Reset();
}

void Reverberation::Process(float* const buffer, const uint32_t size) noexcept {
    if (enable_) {
        model_.ProcessReplace(buffer, buffer + 1, size);
    }
}

// NOLINTNEXTLINE(readability-make-member-function-const) — model_.Reset() mutates state
void Reverberation::Reset() noexcept {
    model_.Reset();
}

void Reverberation::SetConfig(const Config& config) noexcept {
    config_ = config;
    SetEnable(config.enable);
    SetRoomSize(config.room_size);
    SetWidth(config.width);
    SetDamp(config.damp);
    SetWet(config.wet);
    SetDry(config.dry);
}

void Reverberation::SetEnable(const bool enable) noexcept {
    config_.enable = enable;
    if (enable_ != enable) {
        if (!enable_) Reset();  // reset when transitioning from disabled -> enabled
        enable_ = enable;
    }
}

void Reverberation::SetSamplingRate(const uint32_t sampling_rate) noexcept {
    model_.SetSamplingRate(sampling_rate);
}

void Reverberation::SetDamp(const float value)     noexcept { model_.SetDamp(value); }
void Reverberation::SetDry(const float value)      noexcept { model_.SetDry(value); }
void Reverberation::SetRoomSize(const float value) noexcept { model_.SetRoomSize(value); }
void Reverberation::SetWet(const float value)      noexcept { model_.SetWet(value); }
void Reverberation::SetWidth(const float value)    noexcept { model_.SetWidth(value); }

void Reverberation::ProcessPlanar(std::span<float> L, std::span<float> R) noexcept {
    if (!IsEnabled() || L.empty()) return;
    // CRevModel::ProcessPlanar operates stride-1 on contiguous L/R arrays —
    // zero interleave copies, full auto-vectorization of the comb/allpass banks.
    model_.ProcessPlanar(L.data(), R.data(), static_cast<uint32_t>(L.size()));
}
