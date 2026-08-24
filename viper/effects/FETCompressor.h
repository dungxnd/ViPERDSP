#pragma once

#include <array>
#include <cstdint>

class FETCompressor {
public:
    FETCompressor();

    // Main stereo interleaved processing: [L0, R0, L1, R1, ...]
    void Process(float *samples, uint32_t size) noexcept;
    void ProcessPlanar(float* __restrict L, float* __restrict R, size_t frames) noexcept;
    void Reset() noexcept;

    // Enable / Disable
    void SetEnable(bool enable) noexcept;
    [[nodiscard]] bool IsEnabled() const noexcept { return enable_; }

    // Threshold: normalized [0.0 (0 dB) to 1.0 (-60 dB)] or explicit dB [-60.0 to 0.0]
    void SetThreshold(float value) noexcept;
    void SetThresholdDb(float db) noexcept;

    // Ratio: standard ratio >= 1.0 (e.g., 1.0 for 1:1, 4.0 for 4:1) or normalized slope in [0.0, 1.0)
    void SetRatio(float value) noexcept;
    // Explicit normalized slope [0.0 (1:1 / no compression) to 1.0 (inf:1 / limiter)]
    void SetRatioSlope(float slope) noexcept;

    // Knee Width: normalized [0.0 to 1.0] (maps to 0.0 to 60.0 dB) or explicit dB [0.0 to 60.0 dB]
    void SetKnee(float value) noexcept;
    void SetKneeWidthDb(float db) noexcept;
    void SetKneeAuto(bool enable) noexcept;

    // Manual Make-Up Gain: normalized [0.0 (0 dB) to 1.0 (+60 dB)] or explicit dB [0.0 to 60.0]
    void SetGain(float value) noexcept;
    void SetGainDb(float db) noexcept;
    void SetGainAuto(bool enable) noexcept;

    // Attack & Release (Normalized 0.0 to 1.0)
    void SetAttack(float value) noexcept;
    void SetAttackAuto(bool enable) noexcept;
    void SetRelease(float value) noexcept;
    void SetReleaseAuto(bool enable) noexcept;

    // Dynamic Ballistics & Knee Tuning
    void SetKneeMulti(float value) noexcept;
    void SetMaxAttack(float value) noexcept;
    void SetMaxRelease(float value) noexcept;
    void SetCrest(float value) noexcept;
    void SetAdapt(float value) noexcept;
    void SetNoClip(bool enable) noexcept;

    // Sampling Rate
    void SetSamplingRate(uint32_t sampling_rate) noexcept;

private:
    // Core sidechain computation returning the linear gain multiplier
    double ProcessSidechain(double in) noexcept;

    // Helper to calculate one-pole filter coefficient (1 - e^(-1 / (tau * fs)))
    [[nodiscard]] static float CalculateAlpha(uint32_t sampling_rate, float time_seconds) noexcept;

    // State / Config flags
    bool enable_{true};
    bool auto_knee_{true};
    bool auto_gain_{true};
    bool auto_attack_{true};
    bool auto_release_{true};
    bool no_clip_{true};

    uint32_t sampling_rate_{44100u};

    // Raw normalized control values
    float attack_raw_{0.514679f};
    float release_raw_{0.384311f};
    float crest_raw_{0.615689f};
    float adapt_raw_{0.660964f};

    // Target parameters in Natural Log (ln) domain
    float target_threshold_ln_{0.0f};
    float target_gain_ln_{0.0f};
    float target_knee_ln_{0.0f};
    float ratio_slope_{0.0f}; // CS = (1 - 1/R) in [0.0, 1.0]

    // Smoothed parameters (zipper-noise prevention)
    float smoothed_threshold_ln_{0.0f};
    float smoothed_gain_ln_{0.0f};
    float smoothed_knee_ln_{0.0f};
    float param_smoothing_coeff_{0.0f};

    // Giannoulis Smooth Decoupled Detector states (ln domain, positive GR)
    float gr_release_stage_{0.0f};
    float gr_attack_stage_{0.0f};

    // Peak and RMS energy trackers for crest factor calculation
    float running_peak_sq_{0.0f};
    float running_rms_sq_{0.0f};
    float rms_coeff_{0.0f};
    float peak_coeff_{0.0f};

    // Ballistics time constants & coefficients
    float attack_time_sec_{0.0f};
    float attack_coeff_{0.0f};
    float release_time_sec_{0.0f};
    float release_coeff_{0.0f};
    float max_attack_time_{0.0f};
    float max_release_time_{0.0f};

    // Adaptive makeup gain & knee modulation
    float knee_multi_{2.0f};
    float adapt_coeff_{0.0f};
    float adaptive_gain_state_{0.0f};
};