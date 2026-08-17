#include "FETCompressor.h"
#include "../constants.h"
#include <cmath>
#include <algorithm>

// Minimum values to prevent denormals / division by zero
static constexpr float EPSILON_LIN = 1e-6f;
static constexpr float EPSILON_SQ  = 1e-12f;

float FETCompressor::CalculateAlpha(const uint32_t sampling_rate, const float time_seconds) {
    if (time_seconds <= 0.0f) return 1.0f;
    return 1.0f - std::exp(-1.0f / (time_seconds * static_cast<float>(sampling_rate)));
}

FETCompressor::FETCompressor() :
    enable_(true),
    auto_knee_(true),
    auto_gain_(true),
    auto_attack_(true),
    auto_release_(true),
    no_clip_(true),
    sampling_rate_(VIPER_DEFAULT_SAMPLING_RATE),
    attack_raw_(0.514679f),
    release_raw_(0.384311f),
    crest_raw_(0.615689f),
    adapt_raw_(0.660964f),
    knee_multi_(2.0f) {
    
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

void FETCompressor::Reset() {
    param_smoothing_coeff_ = CalculateAlpha(sampling_rate_, 0.05f); // 50ms smoothing
    smoothed_threshold_ln_ = target_threshold_ln_;
    smoothed_gain_ln_      = target_gain_ln_;
    smoothed_knee_ln_      = target_knee_ln_;

    gr_release_stage_    = 0.0f;
    gr_attack_stage_     = 0.0f;
    running_peak_sq_     = EPSILON_SQ;
    running_rms_sq_      = EPSILON_SQ;
    adaptive_gain_state_ = 0.0f;
}

void FETCompressor::Process(float *samples, const uint32_t size) {
    if (!enable_ || size == 0 || samples == nullptr) return;

    for (uint32_t i = 0; i < size * 2; i += 2) {
        // Linked stereo sidechain: maximum absolute amplitude
        const double in_l = std::abs(samples[i]);
        const double in_r = std::abs(samples[i + 1]);
        const double sidechain_in = std::fmax(in_l, in_r);

        const double gain_multiplier = ProcessSidechain(sidechain_in);

        samples[i]     *= static_cast<float>(gain_multiplier);
        samples[i + 1] *= static_cast<float>(gain_multiplier);

        // Continuous parameter smoothing
        smoothed_threshold_ln_ += (target_threshold_ln_ - smoothed_threshold_ln_) * param_smoothing_coeff_;
        smoothed_gain_ln_      += (target_gain_ln_ - smoothed_gain_ln_) * param_smoothing_coeff_;
        smoothed_knee_ln_      += (target_knee_ln_ - smoothed_knee_ln_) * param_smoothing_coeff_;
    }
}

double FETCompressor::ProcessSidechain(const double in) {
    float in_lin = static_cast<float>(in);
    float in2 = in_lin * in_lin;
    if (in2 < EPSILON_SQ) in2 = EPSILON_SQ;

    // 1. Crest Factor Detection (Peak vs RMS)
    running_rms_sq_ += rms_coeff_ * (in2 - running_rms_sq_);
    if (in2 > running_peak_sq_) {
        running_peak_sq_ = in2; // Instantaneous attack for peak tracker
    } else {
        running_peak_sq_ += peak_coeff_ * (in2 - running_peak_sq_);
    }

    float crest_ratio = running_peak_sq_ / (running_rms_sq_ + EPSILON_SQ);
    if (crest_ratio < 1.0f) crest_ratio = 1.0f;

    // 2. Program-Dependent Dynamic Attack & Release
    float cur_att_coeff = attack_coeff_;
    float cur_rel_coeff = release_coeff_;
    float adaptive_att_time = attack_time_sec_;

    if (auto_attack_) {
        adaptive_att_time = (2.0f * max_attack_time_) / crest_ratio;
        cur_att_coeff = CalculateAlpha(sampling_rate_, adaptive_att_time);
    }

    if (auto_release_) {
        const float adaptive_rel_time = std::fmax(
            (2.0f * max_release_time_) / crest_ratio - adaptive_att_time, 
            0.001f
        );
        cur_rel_coeff = CalculateAlpha(sampling_rate_, adaptive_rel_time);
    }

    // 3. Log-Domain Static Gain Computer (Giannoulis Eq. 4)
    const float log_input = std::log(std::fmax(in_lin, EPSILON_LIN));
    const float diff = log_input - smoothed_threshold_ln_;

    float knee_width = smoothed_knee_ln_;
    float effective_slope = ratio_slope_;

    if (auto_knee_) {
        const float half_thresh = smoothed_threshold_ln_ * 0.5f;
        const float knee_base = adaptive_gain_state_ + half_thresh;
        knee_width = std::fmax(-(knee_base * knee_multi_), 0.0f);
    }

    const float half_knee = knee_width * 0.5f;
    float gain_reduction = 0.0f; // Positive attenuation magnitude in ln domain

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
    // Stage 1: Release envelope follower (releases towards target gain_reduction)
    gr_release_stage_ = std::fmax(
        gain_reduction,
        gr_release_stage_ + cur_rel_coeff * (gain_reduction - gr_release_stage_)
    );

    // Stage 2: Attack smoothing
    gr_attack_stage_ += cur_att_coeff * (gr_release_stage_ - gr_attack_stage_);
    const float smoothed_gr = gr_attack_stage_;

    // 5. Adaptive Make-Up Gain & Output Scaling
    const float half_thresh_gr = smoothed_threshold_ln_ * 0.5f;
    const float adapt_target = -smoothed_gr - half_thresh_gr - adaptive_gain_state_;
    adaptive_gain_state_ += adapt_target * adapt_coeff_;

    if (auto_gain_) {
        float makeup_gain = half_thresh_gr + adaptive_gain_state_;
        
        if (no_clip_) {
            const float output_level_ln = log_input - smoothed_gr - makeup_gain;
            if (output_level_ln > 0.0f) {
                makeup_gain += output_level_ln; // Prevent clipping over 0 dBFS
                adaptive_gain_state_ = output_level_ln;
            }
        }
        return std::exp(-smoothed_gr - makeup_gain);
    }

    return std::exp(smoothed_gain_ln_ - smoothed_gr);
}

// ---------------- Parameter Setters ----------------

void FETCompressor::SetEnable(const bool enable) {
    enable_ = enable;
}

void FETCompressor::SetThreshold(const float value) {
    // Value in [0.0, 1.0] -> 0 dB to -60 dB
    SetThresholdDb(value * -60.0f);
}

void FETCompressor::SetThresholdDb(const float db) {
    // ln(10^(dB / 20)) = dB * ln(10) / 20
    target_threshold_ln_ = db * (2.302585092994046f / 20.0f);
}

void FETCompressor::SetRatio(const float value) {
    if (value > 1.0f) {
        // Provided as standard ratio (e.g. 4.0 for 4:1)
        ratio_slope_ = 1.0f - (1.0f / value);
    } else {
        // Provided as normalized slope factor in [0.0, 1.0]
        ratio_slope_ = std::fmax(0.0f, std::fmin(value, 1.0f));
    }
}

void FETCompressor::SetKnee(const float value) {
    // Value in [0.0, 1.0] -> 0 dB to 60 dB
    SetKneeWidthDb(value * 60.0f);
}

void FETCompressor::SetKneeWidthDb(const float db) {
    target_knee_ln_ = db * (2.302585092994046f / 20.0f);
}

void FETCompressor::SetKneeAuto(const bool enable) {
    auto_knee_ = enable;
}

void FETCompressor::SetGain(const float value) {
    // Value in [0.0, 1.0] -> 0 dB to +60 dB
    SetGainDb(value * 60.0f);
}

void FETCompressor::SetGainDb(const float db) {
    target_gain_ln_ = db * (2.302585092994046f / 20.0f);
}

void FETCompressor::SetGainAuto(const bool enable) {
    auto_gain_ = enable;
}

void FETCompressor::SetAttack(const float value) {
    attack_raw_ = value;
    // Map normalized [0, 1] to ~100us -> ~200ms
    attack_time_sec_ = std::exp(value * 7.600903f - 9.21034f);
    attack_coeff_    = CalculateAlpha(sampling_rate_, attack_time_sec_);
}

void FETCompressor::SetAttackAuto(const bool enable) {
    auto_attack_ = enable;
}

void FETCompressor::SetRelease(const float value) {
    release_raw_ = value;
    // Map normalized [0, 1] to ~5ms -> ~2000ms
    release_time_sec_ = std::exp(value * 5.991465f - 5.298317f);
    release_coeff_    = CalculateAlpha(sampling_rate_, release_time_sec_);
}

void FETCompressor::SetReleaseAuto(const bool enable) {
    auto_release_ = enable;
}

void FETCompressor::SetKneeMulti(const float value) {
    knee_multi_ = value * 4.0f;
}

void FETCompressor::SetMaxAttack(const float value) {
    max_attack_time_ = std::exp(value * 7.600903f - 9.21034f);
}

void FETCompressor::SetMaxRelease(const float value) {
    max_release_time_ = std::exp(value * 5.991465f - 5.298317f);
}

void FETCompressor::SetCrest(const float value) {
    crest_raw_ = value;
    // Fast peak decay (~10ms) and slower RMS integration (~50ms)
    float base_time = std::exp(value * 5.991465f - 5.298317f);
    peak_coeff_ = CalculateAlpha(sampling_rate_, base_time * 0.2f);
    rms_coeff_  = CalculateAlpha(sampling_rate_, base_time);
}

void FETCompressor::SetAdapt(const float value) {
    adapt_raw_ = value;
    float adapt_time = std::exp(value * 1.386294f);
    adapt_coeff_ = CalculateAlpha(sampling_rate_, adapt_time);
}

void FETCompressor::SetNoClip(const bool enable) {
    no_clip_ = enable;
}

void FETCompressor::SetSamplingRate(const uint32_t sampling_rate) {
    if (sampling_rate == 0) return;
    sampling_rate_ = sampling_rate;
    SetAttack(attack_raw_);
    SetRelease(release_raw_);
    SetCrest(crest_raw_);
    SetAdapt(adapt_raw_);
    Reset();
}