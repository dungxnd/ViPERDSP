#include "ViPERDDC.h"
#include "../../include/log.h"
#include <algorithm>

ViPERDDC::ViPERDDC() = default;

void ViPERDDC::Process(float *samples, const uint32_t size) noexcept {
    if (!enable_ || !set_coeffs_ok_ || arr_size_ == 0) return;

    const std::vector<std::array<float, 5>> *coeffs_arr = nullptr;
    if      (sampling_rate_ == 44100u) coeffs_arr = &coeffs_arr44100_;
    else if (sampling_rate_ == 48000u) coeffs_arr = &coeffs_arr48000_;
    else {
        VIPER_LOGD("ViPERDDC: Unsupported sampling rate: %d", sampling_rate_);
        return;
    }

    for (uint32_t i = 0; i < size * 2; i += 2) {
        float sample_l = samples[i];
        float sample_r = samples[i + 1];

        for (uint32_t j = 0; j < arr_size_; j++) {
            const auto& c = (*coeffs_arr)[j];
            const float b0 = c[0];
            const float b1 = c[1];
            const float b2 = c[2];
            const float a1 = c[3];
            const float a2 = c[4];

            const float out_l = sample_l * b0 + x1_l_[j] * b1 + x2_l_[j] * b2
                                + y1_l_[j] * a1 + y2_l_[j] * a2;
            x2_l_[j] = x1_l_[j]; x1_l_[j] = sample_l;
            y2_l_[j] = y1_l_[j]; y1_l_[j] = out_l;
            sample_l  = out_l;

            const float out_r = sample_r * b0 + x1_r_[j] * b1 + x2_r_[j] * b2
                                + y1_r_[j] * a1 + y2_r_[j] * a2;
            x2_r_[j] = x1_r_[j]; x1_r_[j] = sample_r;
            y2_r_[j] = y1_r_[j]; y1_r_[j] = out_r;
            sample_r  = out_r;
        }

        samples[i]     = sample_l;
        samples[i + 1] = sample_r;
    }
}

void ViPERDDC::Reset() noexcept {
    if (!set_coeffs_ok_ || arr_size_ == 0) return;

    std::ranges::fill(x1_l_, 0.0f);
    std::ranges::fill(x1_r_, 0.0f);
    std::ranges::fill(x2_l_, 0.0f);
    std::ranges::fill(x2_r_, 0.0f);
    std::ranges::fill(y1_l_, 0.0f);
    std::ranges::fill(y1_r_, 0.0f);
    std::ranges::fill(y2_l_, 0.0f);
    std::ranges::fill(y2_r_, 0.0f);
}

void ViPERDDC::SetEnable(const bool enable) noexcept {
    if (enable_ != enable) {
        enable_ = enable;
        if (enable_) Reset();
    }
}

void ViPERDDC::SetCoeffs(
    const uint32_t coeffs_size,
    const float *coeffs_44100,
    const float *coeffs_48000
) {
    ReleaseResources();
    if (coeffs_size == 0) return;

    arr_size_ = coeffs_size / 5;
    coeffs_arr44100_.resize(arr_size_);
    coeffs_arr48000_.resize(arr_size_);

    for (uint32_t i = 0; i < arr_size_; i++) {
        std::copy_n(coeffs_44100 + i * 5, 5, coeffs_arr44100_[i].begin());
        std::copy_n(coeffs_48000 + i * 5, 5, coeffs_arr48000_[i].begin());
    }

    x1_l_.assign(arr_size_, 0.0f);
    x1_r_.assign(arr_size_, 0.0f);
    x2_l_.assign(arr_size_, 0.0f);
    x2_r_.assign(arr_size_, 0.0f);
    y1_l_.assign(arr_size_, 0.0f);
    y1_r_.assign(arr_size_, 0.0f);
    y2_l_.assign(arr_size_, 0.0f);
    y2_r_.assign(arr_size_, 0.0f);

    set_coeffs_ok_ = true;
}

void ViPERDDC::SetSamplingRate(const uint32_t sampling_rate) noexcept {
    if (sampling_rate_ != sampling_rate) {
        sampling_rate_ = sampling_rate;
        Reset();
    }
}

void ViPERDDC::ReleaseResources() noexcept {
    set_coeffs_ok_ = false;
    coeffs_arr44100_.clear();
    coeffs_arr48000_.clear();
    x1_l_.clear(); x1_r_.clear();
    x2_l_.clear(); x2_r_.clear();
    y1_l_.clear(); y1_r_.clear();
    y2_l_.clear(); y2_r_.clear();
}

void ViPERDDC::ProcessPlanar(float* __restrict L, float* __restrict R, const size_t frames) noexcept {
    if (!IsEnabled() || frames == 0) return;
    if (!set_coeffs_ok_ || arr_size_ == 0) return;

    const std::vector<std::array<float, 5>> *coeffs_arr = nullptr;
    if      (sampling_rate_ == 44100u) coeffs_arr = &coeffs_arr44100_;
    else if (sampling_rate_ == 48000u) coeffs_arr = &coeffs_arr48000_;
    else return;

    for (size_t f = 0; f < frames; ++f) {
        float sl = L[f];
        float sr = R[f];

        for (uint32_t j = 0; j < arr_size_; ++j) {
            const auto& c = (*coeffs_arr)[j];
            const float b0 = c[0], b1 = c[1], b2 = c[2];
            const float a1 = c[3], a2 = c[4];

            const float ol = sl * b0 + x1_l_[j] * b1 + x2_l_[j] * b2
                             + y1_l_[j] * a1 + y2_l_[j] * a2;
            x2_l_[j] = x1_l_[j]; x1_l_[j] = sl;
            y2_l_[j] = y1_l_[j]; y1_l_[j] = ol;
            sl = ol;

            const float or_ = sr * b0 + x1_r_[j] * b1 + x2_r_[j] * b2
                              + y1_r_[j] * a1 + y2_r_[j] * a2;
            x2_r_[j] = x1_r_[j]; x1_r_[j] = sr;
            y2_r_[j] = y1_r_[j]; y1_r_[j] = or_;
            sr = or_;
        }

        L[f] = sl;
        R[f] = sr;
    }
}
