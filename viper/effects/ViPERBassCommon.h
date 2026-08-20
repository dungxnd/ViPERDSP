#pragma once

#include <cmath>

// Shared definitions for ViPERBass and ViPERBassMono.

// ---------------------------------------------------------------------------
// ProcessMode — identical semantics in both stereo and mono variants.
// ---------------------------------------------------------------------------
enum class BassProcessMode {
    NaturalBass  = 0,
    PureBassPlus = 1,
    Subwoofer    = 2,
    // ALL_CAPS aliases for source compatibility
    NATURAL_BASS  = NaturalBass,
    PURE_BASS_PLUS = PureBassPlus,
    SUBWOOFER     = Subwoofer,
};

// ---------------------------------------------------------------------------
// BassSoftClip — algebraic waveshaper: x/sqrt(1+x²) knee variant.
// Produces smooth C¹-continuous saturation with odd-order harmonics.
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
