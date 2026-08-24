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

void Reverberation::ProcessPlanar(float* __restrict L, float* __restrict R, const size_t frames) noexcept {
    if (!IsEnabled() || frames == 0) return;
    const auto n = static_cast<uint32_t>(frames);
    for (size_t i = 0; i < frames; ++i) {
        pp_scratch_[2u * i]      = L[i];
        pp_scratch_[2u * i + 1u] = R[i];
    }
    Process(pp_scratch_.data(), n);
    for (size_t i = 0; i < frames; ++i) {
        L[i] = pp_scratch_[2u * i];
        R[i] = pp_scratch_[2u * i + 1u];
    }
}
