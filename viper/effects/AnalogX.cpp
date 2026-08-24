#include "AnalogX.h"
#include <algorithm>
#include <array>
#include <span>
#include <utility>

static constexpr float kOutputScale = 0.8f;

static constexpr std::array<float, 10> kAnalogXHarmonics{
    0.01f, 0.02f, 0.0001f, 0.001f, 0.0f,
    0.0f,  0.0f,  0.0f,    0.0f,   0.0f,
};

AnalogX::AnalogX() {
    Reset();
}

void AnalogX::Process(float* const samples, const uint32_t size) {
    if (!enable_) return;

    for (uint32_t i = 0; i < size * 2; i += 2) {
        const double in_l = samples[i];
        double out_l = high_pass_[0].ProcessSample(in_l);
        out_l = harmonic_[0].Process(out_l);
        out_l = low_pass_[0].ProcessSample(in_l + out_l * gain_);
        out_l = peak_[0].ProcessSample(out_l * kOutputScale);
        samples[i] = static_cast<float>(out_l);

        const double in_r = samples[i + 1];
        double out_r = high_pass_[1].ProcessSample(in_r);
        out_r = harmonic_[1].Process(out_r);
        out_r = low_pass_[1].ProcessSample(in_r + out_r * gain_);
        out_r = peak_[1].ProcessSample(out_r * kOutputScale);
        samples[i + 1] = static_cast<float>(out_r);
    }

    if (freq_range_ < sampling_rate_ / 4) {
        freq_range_ += size;
        std::fill_n(samples, size * 2, 0.0f);
    }
}

void AnalogX::Reset() {
    struct ModelParams { float gain; float cutoff; };
    static constexpr std::array<ModelParams, 3> kModels{{
        {0.6f, 19650.0f},
        {1.2f, 18233.0f},
        {2.4f, 16307.0f},
    }};

    using FT = MultiBiquad::FilterType;

    for (auto& f : high_pass_) {
        f.RefreshFilter(FT::HighPass, {.frequency = 240.0, .sample_rate = sampling_rate_, .q_factor = 0.717});
    }
    for (auto& f : peak_) {
        f.RefreshFilter(FT::Peak, {.gain_db = 0.58, .frequency = 633.0, .sample_rate = sampling_rate_, .q_factor = 6.28, .is_bandwidth = true});
    }
    for (auto& h : harmonic_) {
        h.Reset();
    }

    const auto& m = kModels[static_cast<std::size_t>(std::to_underlying(processing_model_))];
    gain_ = m.gain;
    for (auto& h : harmonic_) {
        h.SetHarmonics(kAnalogXHarmonics);
    }
    for (auto& f : low_pass_) {
        f.RefreshFilter(FT::LowPass, {.frequency = static_cast<double>(m.cutoff), .sample_rate = sampling_rate_, .q_factor = 0.717});
    }

    freq_range_ = 0;
}

void AnalogX::SetEnable(const bool enable) {
    if (enable_ != enable) {
        if (!enable_) {
            Reset();
        }
        enable_ = enable;
    }
}

void AnalogX::SetProcessingModel(const ProcessingModel model) {
    if (processing_model_ != model) {
        processing_model_ = model;
        Reset();
    }
}

void AnalogX::SetSamplingRate(const uint32_t sampling_rate) {
    if (sampling_rate_ != sampling_rate) {
        sampling_rate_ = sampling_rate;
        Reset();
    }
}

void AnalogX::SetProcessingModel(const int model) {
    const auto clamped = static_cast<std::size_t>(std::clamp(model, 0, 2));
    SetProcessingModel(static_cast<ProcessingModel>(clamped));
}

void AnalogX::ProcessPlanar(float* __restrict L, float* __restrict R, const size_t frames) noexcept {
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
