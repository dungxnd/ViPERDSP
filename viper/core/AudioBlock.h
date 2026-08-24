#pragma once
// Planar (Structure-of-Arrays) stereo audio block.
//
// Storing L and R channels in separate contiguous arrays lets the compiler
// auto-vectorise per-channel DSP loops without the gather/scatter overhead
// of interleaved `[L0 R0 L1 R1 …]` layout.
//
// Usage in an effect's ProcessPlanar():
//   void MyEffect::ProcessPlanar(float* __restrict L,
//                                float* __restrict R,
//                                size_t frames) noexcept { … }
//
// Usage in ViPER::Process():
//   block_.Deinterleave(buffer.data(), size);
//   myEffect.ProcessPlanar(block_.left.data(), block_.right.data(), size);
//   block_.Interleave(buffer.data());

#include <array>
#include <cstddef>

namespace viper::core {

// MaxFrames must be >= the host's maximum audio block size.
// 4096 covers all known Android AudioFlinger configurations (typically 128–512).
template <size_t MaxFrames = 4096>
struct alignas(64) PlanarAudioBlock {
    alignas(64) std::array<float, MaxFrames> left{};
    alignas(64) std::array<float, MaxFrames> right{};
    size_t num_frames{0};

    // Deinterleave: [L0 R0 L1 R1 …] → separate left[] / right[] arrays.
    // src must point to `frames * 2` floats.
    void Deinterleave(const float* __restrict src, size_t frames) noexcept {
        num_frames = frames;
#pragma clang loop vectorize(enable)
        for (size_t i = 0; i < frames; ++i) {
            left[i]  = src[2u * i];
            right[i] = src[2u * i + 1u];
        }
    }

    // Interleave: separate left[] / right[] → [L0 R0 L1 R1 …] in dst.
    // dst must point to `num_frames * 2` floats.
    void Interleave(float* __restrict dst) const noexcept {
#pragma clang loop vectorize(enable)
        for (size_t i = 0; i < num_frames; ++i) {
            dst[2u * i]       = left[i];
            dst[2u * i + 1u]  = right[i];
        }
    }

    // Apply per-channel gain scalars in-place (planar domain).
    void ApplyGainPan(float scale, float l_pan, float r_pan) noexcept {
        const float gl = scale * l_pan;
        const float gr = scale * r_pan;
#pragma clang loop vectorize(enable)
        for (size_t i = 0; i < num_frames; ++i) {
            left[i]  *= gl;
            right[i] *= gr;
        }
    }
};

} // namespace viper::core
