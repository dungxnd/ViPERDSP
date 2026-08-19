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

void Reverberation::Reset() noexcept {
    model_.Reset();
}

void Reverberation::SetEnable(const bool enable) noexcept {
    if (enable_ != enable) {
        if (!enable_) Reset();  // reset when transitioning from disabled -> enabled
        enable_ = enable;
    }
}

void Reverberation::SetDamp(const float value)     noexcept { model_.SetDamp(value); }
void Reverberation::SetDry(const float value)      noexcept { model_.SetDry(value); }
void Reverberation::SetRoomSize(const float value) noexcept { model_.SetRoomSize(value); }
void Reverberation::SetWet(const float value)      noexcept { model_.SetWet(value); }
void Reverberation::SetWidth(const float value)    noexcept { model_.SetWidth(value); }
