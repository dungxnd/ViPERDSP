#include "SpeakerCorrection.h"

SpeakerCorrection::SpeakerCorrection() {
    Reset();
}

void SpeakerCorrection::Process(float *samples, const uint32_t size) noexcept {
    if (!enable_) return;

    for (uint32_t i = 0; i < size * 2; i += 2) {
        double out_l = samples[i];
        out_l = low_pass_[0].ProcessSample(out_l);
        out_l = high_pass_[0].ProcessSample(out_l);
        out_l /= 2.0;
        out_l += band_pass_[0].ProcessSample(out_l);
        samples[i] = static_cast<float>(out_l);

        double out_r = samples[i + 1];
        out_r = low_pass_[1].ProcessSample(out_r);
        out_r = high_pass_[1].ProcessSample(out_r);
        out_r /= 2.0;
        out_r += band_pass_[1].ProcessSample(out_r);
        samples[i + 1] = static_cast<float>(out_r);
    }
}

void SpeakerCorrection::Reset() noexcept {
    for (auto& f : low_pass_)   f.Reset();
    for (auto& f : band_pass_)  f.Reset();
    for (auto& f : high_pass_)  f.Reset();

    RefreshHighPass();
    RefreshLowPass();
    RefreshBandPass();
}

void SpeakerCorrection::SetEnable(const bool enable) noexcept {
    if (enable_ != enable) {
        if (enable) Reset();
        enable_ = enable;
    }
}

void SpeakerCorrection::SetSamplingRate(const uint32_t sampling_rate) noexcept {
    if (sampling_rate_ != sampling_rate) {
        sampling_rate_ = sampling_rate;
        Reset();
    }
}

void SpeakerCorrection::SetHighPassCutoff(const uint32_t value) noexcept {
    if (hp_cutoff_ != value) {
        hp_cutoff_ = value;
        RefreshHighPass();
    }
}

void SpeakerCorrection::SetLowPassCutoff(const uint32_t value) noexcept {
    if (lp_cutoff_ != value) {
        lp_cutoff_ = value;
        RefreshLowPass();
    }
}

void SpeakerCorrection::SetBandPassCenter(const uint32_t value) noexcept {
    if (bp_center_ != value) {
        bp_center_ = value;
        RefreshBandPass();
    }
}

void SpeakerCorrection::SetBandPassQ(const float value) noexcept {
    if (bp_q_ != value) {
        bp_q_ = value;
        RefreshBandPass();
    }
}

void SpeakerCorrection::RefreshHighPass() noexcept {
    for (auto& f : high_pass_) {
        f.RefreshFilter(
            MultiBiquad::FilterType::HighPass,
            0.0f,
            static_cast<float>(hp_cutoff_),
            sampling_rate_,
            1.0f,
            false
        );
    }
}

void SpeakerCorrection::RefreshLowPass() noexcept {
    for (auto& f : low_pass_) {
        f.SetLowPassParameter(
            static_cast<float>(lp_cutoff_), sampling_rate_, 1.0f
        );
    }
}

void SpeakerCorrection::RefreshBandPass() noexcept {
    for (auto& f : band_pass_) {
        f.SetBandPassParameter(
            static_cast<float>(bp_center_), sampling_rate_, bp_q_
        );
    }
}

void SpeakerCorrection::ProcessPlanar(float* __restrict L, float* __restrict R, const size_t frames) noexcept {
    if (!IsEnabled() || frames == 0) return;
    const auto n = static_cast<uint32_t>(frames);
    for (size_t i = 0; i < frames; ++i) {
        pp_scratch_[2u * i]      = L[i];
        pp_scratch_[2u * i + 1u] = R[i];
    }
    Process(pp_scratch_.data(), n);
    for (size_t i = 0; i < frames; ++i) {
        L[i] = pp_scratch_[2u * i];
        R[i] = pp_scratch_[2u * i + 1u];
    }
}
