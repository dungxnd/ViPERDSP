#pragma once
// AudioProcessContext — cache-aligned planar (SoA) stereo block for the
// ViPER DSP pipeline.  One deinterleave at ingress; one interleave at egress.
// All effect ProcessPlanar() implementations receive raw float* pointers into
// these arrays — no copies, no indirection.
//
// PlanarAudioEffect concept enforces the RT-safe planar interface:
//   effect.ProcessPlanar(float* __restrict L, float* __restrict R, size_t frames) noexcept
//   effect.Reset()           noexcept
//   effect.SetSamplingRate(uint32_t)   (non-RT, called from control plane)
//   effect.IsEnabled()       noexcept -> bool
//
// The concept intentionally does NOT constrain return type of SetSamplingRate
// (some effects return void, others are implicitly void through noexcept).

#include <array>
#include <concepts>
#include <cstddef>
#include <span>

namespace viper::core {

template <size_t MaxFrames = 4096>
struct alignas(64) AudioProcessContext {
    alignas(64) std::array<float, MaxFrames> left{};
    alignas(64) std::array<float, MaxFrames> right{};
    size_t num_frames{0};

    [[nodiscard]] std::span<float> Left() noexcept {
        return {left.data(), num_frames};
    }
    [[nodiscard]] std::span<float> Right() noexcept {
        return {right.data(), num_frames};
    }
    [[nodiscard]] std::span<const float> Left() const noexcept {
        return {left.data(), num_frames};
    }
    [[nodiscard]] std::span<const float> Right() const noexcept {
        return {right.data(), num_frames};
    }

    // One SIMD deinterleave pass at pipeline ingress.
    // src must point to frames*2 floats: [L0 R0 L1 R1 ...].
    void Deinterleave(const float* __restrict src, size_t frames) noexcept {
        num_frames = frames;
        float* __restrict l = left.data();
        float* __restrict r = right.data();
#pragma clang loop vectorize(enable)
        for (size_t i = 0; i < frames; ++i) {
            l[i] = src[2u * i];
            r[i] = src[2u * i + 1u];
        }
    }

    // One SIMD interleave pass at pipeline egress.
    // dst must point to num_frames*2 floats.
    void Interleave(float* __restrict dst) const noexcept {
        const float* __restrict l = left.data();
        const float* __restrict r = right.data();
#pragma clang loop vectorize(enable)
        for (size_t i = 0; i < num_frames; ++i) {
            dst[2u * i]       = l[i];
            dst[2u * i + 1u]  = r[i];
        }
    }

    // Inline gain+pan in planar domain (vectorizer-friendly: no stride).
    void ApplyGainPan(float scale, float l_pan, float r_pan) noexcept {
        const float gl = scale * l_pan;
        const float gr = scale * r_pan;
        float* __restrict l = left.data();
        float* __restrict r = right.data();
#pragma clang loop vectorize(enable)
        for (size_t i = 0; i < num_frames; ++i) {
            l[i] *= gl;
            r[i] *= gr;
        }
    }
};

// C++23 concept: constrains the RT-safe planar DSP interface.
// Effects satisfying this concept can participate in the planar pipeline.
// All ProcessPlanar implementations receive non-owning std::span<float> views
// into the pipeline's AudioProcessContext buffers — zero copies, no indirection.
template <typename T>
concept PlanarAudioEffect = requires(T effect, std::span<float> l, std::span<float> r, uint32_t sr) {
    { effect.ProcessPlanar(l, r) } noexcept -> std::same_as<void>;
    { effect.Reset()             } noexcept;
    { effect.SetSamplingRate(sr) };
    { effect.IsEnabled()         } noexcept -> std::same_as<bool>;
};

} // namespace viper::core
