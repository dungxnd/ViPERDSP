#pragma once

#include <cmath>
#include <cstdint>

// Shared definitions for ViPERBass and ViPERBassMono.

// ---------------------------------------------------------------------------
// ProcessMode — identical semantics in both stereo and mono variants.
// ---------------------------------------------------------------------------
enum class BassProcessMode : uint8_t {
    NaturalBass  = 0,
    PureBassPlus = 1,
    Subwoofer    = 2,
};

// ---------------------------------------------------------------------------
// BassSoftClip — algebraic waveshaper: x/sqrt(1+x²) knee variant.
// Produces smooth C²-continuous saturation with odd-order harmonics.
// The function value, first derivative, and second derivative are all
// continuous at the knee threshold (the linear-to-sqrt transition).
// knee: linear threshold below which signal passes unchanged.
// Note: ceiling = knee + 1.0 (not a hard 0 dBFS limiter; by design).
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr float BassSoftClip(const float v, const float knee) noexcept {
    const float drive = std::fabs(v);
    if (drive <= knee) [[likely]] return v;
    const float over   = drive - knee;
    const float shaped = knee + over / std::sqrt(1.0f + over * over);
    return v * (shaped / drive);
}
