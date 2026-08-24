#include "AnalogX.h"
#include <algorithm>
#include <array>
#include <utility>

static constexpr float kOutputScale = 0.8f;

static constexpr std::array<float, 10> kAnalogXHarmonics{
    0.01f, 0.02f, 0.0001f, 0.001f, 0.0f,
    0.0f,  0.0f,  0.0f,    0.0f,   0.0f,
};

AnalogX::AnalogX() {
    Reset();
}

// True planar in-place processing.
//
// Signal graph (matches the original interleaved Process()):
//   hp_out  = HP(input)
//   harm    = Harmonic(hp_out)
//   lp_out  = LP(input + harm * gain_)   ← LP receives ORIGINAL input, not HP output
//   scaled  = lp_out * kOutputScale
//   output  = Peak(scaled)
//
// We process both channels per sample, iterating over the planar arrays
// directly (stride-1, no interleave copies).
void AnalogX::ProcessPlanar(std::span<float> L, std::span<float> R) noexcept {
    if (!enable_ || L.empty()) return;

    auto& hp_l = high_pass_[0]; auto& hp_r = high_pass_[1];
    auto& lp_l = low_pass_[0];  auto& lp_r = low_pass_[1];
    auto& pk_l = peak_[0];      auto& pk_r = peak_[1];
    auto& hm_l = harmonic_[0];  auto& hm_r = harmonic_[1];

    for (size_t i = 0; i < L.size(); ++i) {
        // Left channel
        const double in_l  = L[i];
        const double hp_l_out = hp_l.ProcessSample(in_l);
        const double harm_l   = hm_l.Process(hp_l_out);
        const double lp_l_out = lp_l.ProcessSample(in_l + harm_l * static_cast<double>(gain_));
        const double pk_l_out = pk_l.ProcessSample(lp_l_out * kOutputScale);
        L[i] = static_cast<float>(pk_l_out);

        // Right channel
        const double in_r  = R[i];
        const double hp_r_out = hp_r.ProcessSample(in_r);
        const double harm_r   = hm_r.Process(hp_r_out);
        const double lp_r_out = lp_r.ProcessSample(in_r + harm_r * static_cast<double>(gain_));
        const double pk_r_out = pk_r.ProcessSample(lp_r_out * kOutputScale);
        R[i] = static_cast<float>(pk_r_out);
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
