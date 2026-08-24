#pragma once

#include "../utils/MinPhaseIIRCoeffs.h"
#include <array>
#include <cstdint>
#include <mdspan>
#include <span>

class IIRFilter {
public:
    static constexpr uint32_t kMaxBands = 31u;

    explicit IIRFilter(uint32_t bands = 10);

    // Primary span-based entry point — size is number of interleaved stereo samples
    // (frame_count * 2).
    void Process(std::span<float> samples) noexcept;

    // Legacy entry point kept for ViPER::Process() call site compatibility.
    void Process(float* samples, uint32_t size) noexcept {
        if (samples) Process(std::span<float>(samples, size * 2u));
    }

    void ProcessPlanar(float* __restrict L, float* __restrict R, size_t frames) noexcept;
    void Reset() noexcept;

    [[nodiscard]] bool IsEnabled() const noexcept { return enable_; }

    void SetEnable(bool enable) noexcept;
    void SetBandCount(uint32_t bands);
    void SetBandLevel(uint32_t band, float level) noexcept;

    // Span overload — preferred; caps at the number of active bands.
    void SetBandLevels(std::span<const float> levels) noexcept;

    // Pointer overload for ViPER::DispatchRawParam compatibility.
    void SetBandLevels(const float* levels, uint32_t count) noexcept {
        if (levels) SetBandLevels(std::span<const float>(levels, count));
    }

    void SetSamplingRate(uint32_t sampling_rate);

private:
    using StereoView = std::mdspan<float, std::dextents<size_t, 2>, std::layout_right>;

    // Per-channel, per-band Transposed Direct Form II state registers.
    // Only 2 floats per band per channel — 8× smaller than the old 496-byte buf_.
    struct alignas(8) BiquadState {
        float s1{0.0f};
        float s2{0.0f};
    };

    static constexpr float kDefaultLevelQ = 0.636f;

    bool     enable_{false};
    bool     gains_dirty_{false};  // true while target != current; clears on convergence
    uint32_t bands_{0};
    uint32_t sampling_rate_{44100u};

    // Precomputed in ctor / SetSamplingRate so Process() pays no exp() cost.
    float gain_smooth_coeff_{0.0f};
    float fade_in_gain_{1.0f};   // [0,1] ramp applied on first enable; 1 = fully on
    float fade_in_step_{0.0f};   // increment per frame = 1 / (0.010 · Fs), i.e. 10 ms

    // 2 channels × kMaxBands TDF-II delay registers.
    std::array<std::array<BiquadState, kMaxBands>, 2> state_{};

    // target_gains_: what the user set (changes instantly on SetBandLevel).
    // current_gains_: smoothed toward target at ~20 ms to prevent zipper noise.
    std::array<float, kMaxBands> target_gains_{};
    std::array<float, kMaxBands> current_gains_{};

    MinPhaseIIRCoeffs min_phase_iir_coeffs_;

    alignas(64) std::array<float, 4096u * 2u> scratch_{};

    void UpdateCoeffConstants() noexcept;  // updates gain_smooth_coeff_ + fade_in_step_
};
