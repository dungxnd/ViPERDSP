#include "ViPERDDC.h"
#include "../../include/log.h"
#include <algorithm>

ViPERDDC::ViPERDDC() = default;

void ViPERDDC::ConsumeCoeffsSwap() noexcept {
    if (!swap_pending_.load(std::memory_order_acquire)) return;

    std::swap(active_coeffs_, staging_coeffs_);
    arr_size_      = active_coeffs_ ? active_coeffs_->arr_size : 0u;
    set_coeffs_ok_ = (arr_size_ > 0);

    x1_l_.assign(arr_size_, 0.0f);
    x1_r_.assign(arr_size_, 0.0f);
    x2_l_.assign(arr_size_, 0.0f);
    x2_r_.assign(arr_size_, 0.0f);
    y1_l_.assign(arr_size_, 0.0f);
    y1_r_.assign(arr_size_, 0.0f);
    y2_l_.assign(arr_size_, 0.0f);
    y2_r_.assign(arr_size_, 0.0f);

    swap_pending_.store(false, std::memory_order_release);
}

void ViPERDDC::Process(float *samples, const uint32_t size) noexcept {
    if (!IsEnabled() || size == 0) return;
    ConsumeCoeffsSwap();
    if (!set_coeffs_ok_ || arr_size_ == 0) return;

    const std::vector<std::array<float, 5>> *coeffs_arr = nullptr;
    if      (sampling_rate_ == 44100u) coeffs_arr = &active_coeffs_->coeffs_44100;
    else if (sampling_rate_ == 48000u) coeffs_arr = &active_coeffs_->coeffs_48000;
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
    ConsumeCoeffsSwap();
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

void ViPERDDC::SetConfig(const Config& config) noexcept {
    config_ = config;
    SetEnable(config.enable);
}

void ViPERDDC::SetEnable(const bool enable) noexcept {
    if (config_.enable != enable) {
        config_.enable = enable;
        if (enable) Reset();
    }
}

void ViPERDDC::SetCoeffs(
    const uint32_t coeffs_size,
    const float *coeffs_44100,
    const float *coeffs_48000
) {
    auto new_coeffs = std::make_unique<DdcCoeffs>();
    if (coeffs_size > 0 && coeffs_44100 && coeffs_48000) {
        new_coeffs->arr_size = coeffs_size / 5;
        new_coeffs->coeffs_44100.resize(new_coeffs->arr_size);
        new_coeffs->coeffs_48000.resize(new_coeffs->arr_size);

        for (uint32_t i = 0; i < new_coeffs->arr_size; i++) {
            std::copy_n(coeffs_44100 + i * 5, 5, new_coeffs->coeffs_44100[i].begin());
            std::copy_n(coeffs_48000 + i * 5, 5, new_coeffs->coeffs_48000[i].begin());
        }
    }

    const std::lock_guard lock(stage_mutex_);
    while (swap_pending_.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    staging_coeffs_ = std::move(new_coeffs);
    swap_pending_.store(true, std::memory_order_release);
}

void ViPERDDC::SetSamplingRate(const uint32_t sampling_rate) noexcept {
    if (sampling_rate_ != sampling_rate) {
        sampling_rate_ = sampling_rate;
        Reset();
    }
}

void ViPERDDC::ReleaseResources() noexcept {
    SetCoeffs(0, nullptr, nullptr);
}

void ViPERDDC::ProcessPlanar(std::span<float> L, std::span<float> R) noexcept {
    if (!IsEnabled() || L.empty()) return;
    ConsumeCoeffsSwap();
    if (!set_coeffs_ok_ || arr_size_ == 0) return;

    const std::vector<std::array<float, 5>> *coeffs_arr = nullptr;
    if      (sampling_rate_ == 44100u) coeffs_arr = &active_coeffs_->coeffs_44100;
    else if (sampling_rate_ == 48000u) coeffs_arr = &active_coeffs_->coeffs_48000;
    else return;

    for (size_t f = 0; f < L.size(); ++f) {
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
