#include "SpectrumExtend.h"
#include <algorithm>

static constexpr std::array<float, 10> kSpectrumHarmonics{
    0.02f, 0.0f, 0.02f, 0.0f, 0.02f,
    0.0f,  0.02f, 0.0f, 0.02f, 0.0f,
};

SpectrumExtend::SpectrumExtend() {
    Reset();
}

void SpectrumExtend::Process(float *samples, const uint32_t size) noexcept {
    if (!enable_) return;

    for (uint32_t i = 0; i < size * 2; i += 2) {
        double tmp = highpass_[0].ProcessSample(samples[i]);
        tmp = harmonics_[0].Process(tmp);
        tmp = lowpass_[0].ProcessSample(tmp * exciter_);
        samples[i] += static_cast<float>(tmp);

        tmp = highpass_[1].ProcessSample(samples[i + 1]);
        tmp = harmonics_[1].Process(tmp);
        tmp = lowpass_[1].ProcessSample(tmp * exciter_);
        samples[i + 1] += static_cast<float>(tmp);
    }
}

void SpectrumExtend::Reset() noexcept {
    for (auto& f : highpass_) f.Reset();
    for (auto& f : lowpass_)  f.Reset();

    const float max_cutoff = 0.45f * static_cast<float>(sampling_rate_);
    const auto freq  = std::clamp(static_cast<float>(reference_freq_), 20.0f, max_cutoff);
    const auto nyq   = std::clamp(static_cast<float>(sampling_rate_) * 0.5f - 2000.0f, 20.0f, max_cutoff);

    for (auto& f : highpass_) {
        f.RefreshFilter(MultiBiquad::FilterType::HighPass,
                        0.0f, freq, sampling_rate_, 0.717f, false);
    }
    for (auto& f : lowpass_) {
        f.RefreshFilter(MultiBiquad::FilterType::LowPass,
                        0.0f, nyq, sampling_rate_, 0.717f, false);
    }

    for (auto& h : harmonics_) {
        h.Reset();
        h.SetHarmonics(kSpectrumHarmonics);
    }
}

void SpectrumExtend::SetConfig(const Config& config) noexcept {
    config_ = config;
    SetEnable(config.enable);
    SetReferenceFrequency(config.strength);
    SetExciter(config.exciter);
}

void SpectrumExtend::SetEnable(const bool enable) noexcept {
    config_.enable = enable;
    if (enable_ != enable) {
        if (enable) Reset();
        enable_ = enable;
    }
}

void SpectrumExtend::SetExciter(const float value) noexcept {
    exciter_ = value;
}

void SpectrumExtend::SetReferenceFrequency(uint32_t value) noexcept {
    const uint32_t max_freq = sampling_rate_ / 2u - 100u;
    reference_freq_ = std::min(value, max_freq);
    Reset();
}

void SpectrumExtend::SetSamplingRate(const uint32_t sampling_rate) noexcept {
    if (sampling_rate_ != sampling_rate) {
        sampling_rate_ = sampling_rate;
        const uint32_t max_freq = sampling_rate / 2u - 100u;
        reference_freq_ = std::min(reference_freq_, max_freq);
        Reset();
    }
}

void SpectrumExtend::ProcessPlanar(std::span<float> L, std::span<float> R) noexcept {
    if (!IsEnabled() || L.empty()) return;

    for (size_t i = 0u; i < L.size(); ++i) {
        double tmp_l = highpass_[0].ProcessSample(L[i]);
        tmp_l = harmonics_[0].Process(tmp_l);
        tmp_l = lowpass_[0].ProcessSample(tmp_l * exciter_);
        L[i] += static_cast<float>(tmp_l);

        double tmp_r = highpass_[1].ProcessSample(R[i]);
        tmp_r = harmonics_[1].Process(tmp_r);
        tmp_r = lowpass_[1].ProcessSample(tmp_r * exciter_);
        R[i] += static_cast<float>(tmp_r);
    }
}
