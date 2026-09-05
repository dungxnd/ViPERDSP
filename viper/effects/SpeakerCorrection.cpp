#include "SpeakerCorrection.h"

SpeakerCorrection::SpeakerCorrection() {
    Reset();
}

// True planar in-place processing.
//
// Signal graph (matches the original interleaved Process()):
//   stage1 = LP(input)
//   stage2 = HP(stage1)
//   stage3 = stage2 / 2.0 + BP(stage2 / 2.0)   ← same as original
//
// Iterates directly over L[] and R[] (stride-1), eliminating the interleave
// and deinterleave copies from the legacy shim.
void SpeakerCorrection::ProcessPlanar(std::span<float> L, std::span<float> R) noexcept {
    if (!enable_ || L.empty()) return;

    auto& lp_l = low_pass_[0];   auto& lp_r = low_pass_[1];
    auto& hp_l = high_pass_[0];  auto& hp_r = high_pass_[1];
    auto& bp_l = band_pass_[0];  auto& bp_r = band_pass_[1];

    for (size_t i = 0; i < L.size(); ++i) {
        double s_l = lp_l.ProcessSample(L[i]);
        s_l = hp_l.ProcessSample(s_l);
        s_l /= 2.0;
        s_l += bp_l.ProcessSample(s_l);
        L[i] = static_cast<float>(s_l);

        double s_r = lp_r.ProcessSample(R[i]);
        s_r = hp_r.ProcessSample(s_r);
        s_r /= 2.0;
        s_r += bp_r.ProcessSample(s_r);
        R[i] = static_cast<float>(s_r);
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

void SpeakerCorrection::SetConfig(const Config& config) noexcept {
    config_ = config;
    SetEnable(config.enable);
}

void SpeakerCorrection::SetEnable(const bool enable) noexcept {
    config_.enable = enable;
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
