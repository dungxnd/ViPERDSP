#pragma once

#include <cstdint>
#include <cmath>

class FETCompressor {
public:
    FETCompressor();

    // Main stereo interleaved processing: [L0, R0, L1, R1, ...]
    void Process(float *samples, uint32_t size);
    void Reset();

    // Enable / Disable
    void SetEnable(bool enable);
    bool IsEnabled() const { return enable_; }

    // Threshold: accepts normalized [0.0 (0 dB) to 1.0 (-60 dB)] or explicit dB [-60.0 to 0.0]
    void SetThreshold(float value);
    void SetThresholdDb(float db);

    // Ratio: accepts standard ratio (e.g., 4.0 for 4:1) or normalized slope [0.0 (1:1) to 1.0 (inf:1)]
    void SetRatio(float value);

    // Knee Width: accepts normalized [0.0 to 1.0] or explicit dB [0.0 to 30.0 dB]
    void SetKnee(float value);
    void SetKneeAuto(bool enable);

    // Manual Make-Up Gain: accepts normalized [0.0 (0 dB) to 1.0 (+60 dB)] or explicit dB [0.0 to 60.0]
    void SetGain(float value);
    void SetGainDb(float db);
    void SetGainAuto(bool enable);

    // Attack & Release (Normalized 0.0 to 1.0)
    void SetAttack(float value);
    void SetAttackAuto(bool enable);
    void SetRelease(float value);
    void SetReleaseAuto(bool enable);

    // Dynamic Ballistics & Knee Tuning
    void SetKneeMulti(float value);
    void SetMaxAttack(float value);
    void SetMaxRelease(float value);
    void SetCrest(float value);
    void SetAdapt(float value);
    void SetNoClip(bool enable);

    // Sampling Rate
    void SetSamplingRate(uint32_t sampling_rate);

private:
    // Core sidechain computation returning the linear gain multiplier
    double ProcessSidechain(double in);

    // Helper to calculate one-pole filter coefficient (1 - e^(-1 / (tau * fs)))
    static float CalculateAlpha(uint32_t sampling_rate, float time_seconds);

    // State / Config flags
    bool enable_;
    bool auto_knee_;
    bool auto_gain_;
    bool auto_attack_;
    bool auto_release_;
    bool no_clip_;

    uint32_t sampling_rate_;

    // Raw normalized control values
    float attack_raw_;
    float release_raw_;
    float crest_raw_;
    float adapt_raw_;

    // Target parameters in Natural Log (ln) domain
    float target_threshold_ln_;
    float target_gain_ln_;
    float target_knee_ln_;
    float ratio_slope_; // CS = (1 - 1/R) in [0.0, 1.0]

    // Smoothed parameters (zipper-noise prevention)
    float smoothed_threshold_ln_;
    float smoothed_gain_ln_;
    float smoothed_knee_ln_;
    float param_smoothing_coeff_;

    // Giannoulis Smooth Decoupled Detector states (ln domain, positive GR)
    float gr_release_stage_;
    float gr_attack_stage_;

    // Peak and RMS energy trackers for crest factor calculation
    float running_peak_sq_;
    float running_rms_sq_;
    float rms_coeff_;
    float peak_coeff_;

    // Ballistics time constants & coefficients
    float attack_time_sec_;
    float attack_coeff_;
    float release_time_sec_;
    float release_coeff_;
    float max_attack_time_;
    float max_release_time_;

    // Adaptive makeup gain & knee modulation
    float knee_multi_;
    float adapt_coeff_;
    float adaptive_gain_state_;
};