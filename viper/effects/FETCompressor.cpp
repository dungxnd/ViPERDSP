#include "FETCompressor.h"
#include <algorithm>
#include <cmath>

namespace {
// Minimum values to prevent denormals / division by zero
constexpr float kEpsilonLin = 1e-6f;
constexpr float kEpsilonSq  = 1e-12f;
// ln(10) / 20  — converts dB to natural-log domain
constexpr float kDb2Ln      = 2.302585092994046f / 20.0f;
} // namespace

[[nodiscard]] float FETCompressor::CalculateAlpha(
    const uint32_t sampling_rate, const float time_seconds
) noexcept {
    if (time_seconds <= 0.0f) return 1.0f;
    return 1.0f - std::exp(-1.0f / (time_seconds * static_cast<float>(sampling_rate)));
}

FETCompressor::FETCompressor() {
    SetThreshold(0.0f);
    SetRatio(0.0f);
    SetKnee(0.0f);
    SetGain(0.0f);
    SetAttack(attack_raw_);
    SetRelease(release_raw_);
    SetKneeMulti(0.5f);
    SetMaxAttack(0.879450f);
    SetMaxRelease(0.884311f);
    SetCrest(crest_raw_);
    SetAdapt(adapt_raw_);
    Reset();
}

void FETCompressor::Reset() noexcept {
    param_smoothing_coeff_ = CalculateAlpha(sampling_rate_, 0.05f); // 50 ms smoothing
    smoothed_threshold_ln_ = target_threshold_ln_;
    smoothed_gain_ln_      = target_gain_ln_;
    smoothed_knee_ln_      = target_knee_ln_;

    gr_release_stage_    = 0.0f;
    gr_attack_stage_     = 0.0f;
    running_peak_sq_     = kEpsilonSq;
    running_rms_sq_      = kEpsilonSq;
    adaptive_gain_state_ = 0.0f;
}

void FETCompressor::Process(float *samples, const uint32_t size) noexcept {
    if (!enable_ || size == 0 || samples == nullptr) return;

    for (uint32_t i = 0; i < size * 2; i += 2) {
        // Linked stereo sidechain: maximum absolute amplitude
        const double sidechain_in = std::max(
            std::abs(static_cast<double>(samples[i])),
            std::abs(static_cast<double>(samples[i + 1]))
        );

        const double gain_multiplier = ProcessSidechain(sidechain_in);

        samples[i]     *= static_cast<float>(gain_multiplier);
        samples[i + 1] *= static_cast<float>(gain_multiplier);

        // Continuous parameter smoothing
        smoothed_threshold_ln_ += (target_threshold_ln_ - smoothed_threshold_ln_) * param_smoothing_coeff_;
        smoothed_gain_ln_      += (target_gain_ln_      - smoothed_gain_ln_)      * param_smoothing_coeff_;
        smoothed_knee_ln_      += (target_knee_ln_      - smoothed_knee_ln_)      * param_smoothing_coeff_;
    }
}

double FETCompressor::ProcessSidechain(const double in) noexcept {
    float in_lin = static_cast<float>(in);
    float in2 = std::max(in_lin * in_lin, kEpsilonSq);

    // 1. Crest Factor Detection (Peak vs RMS)
    running_rms_sq_ += rms_coeff_ * (in2 - running_rms_sq_);
    if (in2 > running_peak_sq_) {
        running_peak_sq_ = in2; // Instantaneous attack for peak tracker
    } else {
        running_peak_sq_ += peak_coeff_ * (in2 - running_peak_sq_);
    }

    const float crest_ratio = std::max(running_peak_sq_ / (running_rms_sq_ + kEpsilonSq), 1.0f);

    // 2. Program-Dependent Dynamic Attack & Release
    float cur_att_coeff   = attack_coeff_;
    float cur_rel_coeff   = release_coeff_;
    float adaptive_att_time = attack_time_sec_;

    if (auto_attack_) {
        adaptive_att_time = (2.0f * max_attack_time_) / crest_ratio;
        cur_att_coeff = CalculateAlpha(sampling_rate_, adaptive_att_time);
    }

    if (auto_release_) {
        const float adaptive_rel_time = std::max(
            (2.0f * max_release_time_) / crest_ratio - adaptive_att_time, 0.001f
        );
        cur_rel_coeff = CalculateAlpha(sampling_rate_, adaptive_rel_time);
    }

    // 3. Log-Domain Static Gain Computer (Giannoulis Eq. 4)
    const float log_input = std::log(std::max(in_lin, kEpsilonLin));
    const float diff      = log_input - smoothed_threshold_ln_;

    float knee_width      = smoothed_knee_ln_;
    float effective_slope = ratio_slope_;

    if (auto_knee_) {
        const float half_thresh = smoothed_threshold_ln_ * 0.5f;
        const float knee_base   = adaptive_gain_state_ + half_thresh;
        knee_width = std::max(-(knee_base * knee_multi_), 0.0f);
    }

    const float half_knee = knee_width * 0.5f;
    float gain_reduction  = 0.0f; // Positive attenuation magnitude in ln domain

    if (half_knee > 1e-4f) {
        // Soft Knee Characteristic
        if (diff <= -half_knee) {
            gain_reduction = 0.0f;
        } else if (diff >= half_knee) {
            gain_reduction = diff * effective_slope;
        } else {
            const float shifted = diff + half_knee;
            gain_reduction = (shifted * shifted / (2.0f * knee_width)) * effective_slope;
        }
    } else {
        // Hard Knee Characteristic
        if (diff > 0.0f) {
            gain_reduction = diff * effective_slope;
        }
    }

    // 4. Giannoulis Smooth Decoupled Ballistics Detector (AES Eq. 17)
    // Stage 1: Release envelope follower
    gr_release_stage_ = std::max(
        gain_reduction,
        gr_release_stage_ + cur_rel_coeff * (gain_reduction - gr_release_stage_)
    );

    // Stage 2: Attack smoothing
    gr_attack_stage_ += cur_att_coeff * (gr_release_stage_ - gr_attack_stage_);
    const float smoothed_gr = gr_attack_stage_;

    // 5. Adaptive Make-Up Gain & Output Scaling
    const float half_thresh_gr = smoothed_threshold_ln_ * 0.5f;
    const float adapt_target   = -smoothed_gr - half_thresh_gr - adaptive_gain_state_;
    adaptive_gain_state_ += adapt_target * adapt_coeff_;

    if (auto_gain_) {
        float makeup_gain = half_thresh_gr + adaptive_gain_state_;

        if (no_clip_) {
            const float output_level_ln = log_input - smoothed_gr - makeup_gain;
            if (output_level_ln > 0.0f) {
                makeup_gain += output_level_ln; // Prevent clipping over 0 dBFS
                adaptive_gain_state_ = makeup_gain - half_thresh_gr;
            }
        }
        return std::exp(static_cast<double>(-smoothed_gr - makeup_gain));
    }

    return std::exp(static_cast<double>(smoothed_gain_ln_ - smoothed_gr));
}

// ---------------- Parameter Setters ----------------

void FETCompressor::SetEnable(const bool enable) noexcept {
    enable_ = enable;
}

void FETCompressor::SetThreshold(const float value) noexcept {
    SetThresholdDb(value * -60.0f);
}

void FETCompressor::SetThresholdDb(const float db) noexcept {
    target_threshold_ln_ = db * kDb2Ln;
}

void FETCompressor::SetRatio(const float value) noexcept {
    if (value >= 1.0f) {
        // Standard ratio (e.g. 1.0 for 1:1, 4.0 for 4:1)
        ratio_slope_ = 1.0f - (1.0f / value);
    } else {
        // Normalized slope in [0.0, 1.0)
        ratio_slope_ = std::clamp(value, 0.0f, 1.0f);
    }
}

void FETCompressor::SetRatioSlope(const float slope) noexcept {
    ratio_slope_ = std::clamp(slope, 0.0f, 1.0f);
}

void FETCompressor::SetKnee(const float value) noexcept {
    SetKneeWidthDb(value * 60.0f);
}

void FETCompressor::SetKneeWidthDb(const float db) noexcept {
    target_knee_ln_ = db * kDb2Ln;
}

void FETCompressor::SetKneeAuto(const bool enable) noexcept {
    auto_knee_ = enable;
}

void FETCompressor::SetGain(const float value) noexcept {
    SetGainDb(value * 60.0f);
}

void FETCompressor::SetGainDb(const float db) noexcept {
    target_gain_ln_ = db * kDb2Ln;
}

void FETCompressor::SetGainAuto(const bool enable) noexcept {
    auto_gain_ = enable;
}

void FETCompressor::SetAttack(const float value) noexcept {
    attack_raw_      = value;
    attack_time_sec_ = std::exp(value * 7.600903f - 9.21034f);
    attack_coeff_    = CalculateAlpha(sampling_rate_, attack_time_sec_);
}

void FETCompressor::SetAttackAuto(const bool enable) noexcept {
    auto_attack_ = enable;
}

void FETCompressor::SetRelease(const float value) noexcept {
    release_raw_      = value;
    release_time_sec_ = std::exp(value * 5.991465f - 5.298317f);
    release_coeff_    = CalculateAlpha(sampling_rate_, release_time_sec_);
}

void FETCompressor::SetReleaseAuto(const bool enable) noexcept {
    auto_release_ = enable;
}

void FETCompressor::SetKneeMulti(const float value) noexcept {
    knee_multi_ = value * 4.0f;
}

void FETCompressor::SetMaxAttack(const float value) noexcept {
    max_attack_time_ = std::exp(value * 7.600903f - 9.21034f);
}

void FETCompressor::SetMaxRelease(const float value) noexcept {
    max_release_time_ = std::exp(value * 5.991465f - 5.298317f);
}

void FETCompressor::SetCrest(const float value) noexcept {
    crest_raw_  = value;
    const float base_time = std::exp(value * 5.991465f - 5.298317f);
    peak_coeff_ = CalculateAlpha(sampling_rate_, base_time * 0.2f);
    rms_coeff_  = CalculateAlpha(sampling_rate_, base_time);
}

void FETCompressor::SetAdapt(const float value) noexcept {
    adapt_raw_   = value;
    adapt_coeff_ = CalculateAlpha(sampling_rate_, std::exp(value * 1.386294f));
}

void FETCompressor::SetNoClip(const bool enable) noexcept {
    no_clip_ = enable;
}

void FETCompressor::SetSamplingRate(const uint32_t sampling_rate) noexcept {
    if (sampling_rate == 0) return;
    sampling_rate_ = sampling_rate;
    SetAttack(attack_raw_);
    SetRelease(release_raw_);
    SetCrest(crest_raw_);
    SetAdapt(adapt_raw_);
    Reset();
}

void FETCompressor::ProcessPlanar(float* __restrict L, float* __restrict R, const size_t frames) noexcept {
    if (!IsEnabled() || frames == 0u) return;

    for (size_t i = 0u; i < frames; ++i) {
        const double sidechain_in = std::max(
            std::abs(static_cast<double>(L[i])),
            std::abs(static_cast<double>(R[i]))
        );
        const double gain_multiplier = ProcessSidechain(sidechain_in);

        L[i] *= static_cast<float>(gain_multiplier);
        R[i] *= static_cast<float>(gain_multiplier);

        smoothed_threshold_ln_ += (target_threshold_ln_ - smoothed_threshold_ln_) * param_smoothing_coeff_;
        smoothed_gain_ln_      += (target_gain_ln_      - smoothed_gain_ln_)      * param_smoothing_coeff_;
        smoothed_knee_ln_      += (target_knee_ln_      - smoothed_knee_ln_)      * param_smoothing_coeff_;
    }
}
