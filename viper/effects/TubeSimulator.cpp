#include "TubeSimulator.h"
#include "../constants.h"

// Per-model circuit parameters: { TubeModel, Vdd, Rp, bias }
// 12AX7: high-gain stage, classic V4A topology
// 6N1J:  Soviet medium-gain, warmer H2 character (mu=35, gm=4.35mA/V)
static constexpr struct {
    TubeModel   model;
    double      vdd;
    double      rp;
    double      bias;
} kTubeConfigs[] = {
    { { 1.014e-5, 5.498e-8, 1.076e-5 }, 250.0, 100000.0, -1.5 },  // 0: 12AX7
    { { 5.7349e-5, 5.1490e-7, 3.6043e-5 }, 250.0, 20000.0, -2.0 }, // 1: 6N1J
};
static constexpr int kTubeConfigCount = static_cast<int>(sizeof(kTubeConfigs) / sizeof(kTubeConfigs[0]));

TubeSimulator::TubeSimulator() :
    enable_(false),
    tube_type_(TubeType::k12AX7),
    sampling_rate_(VIPER_DEFAULT_SAMPLING_RATE) {
    Reset();
}

void TubeSimulator::Process(float *buffer, const uint32_t size) {
    if (!enable_) return;

    for (uint32_t i = 0; i < size; i++) {
        const double in_l = buffer[i * 2];
        double harm_l = high_pass_[0].ProcessSample(in_l);
        harm_l = tube_[0].Process(harm_l);
        harm_l = low_pass_[0].ProcessSample(harm_l);
        // Wet/dry mix: preserves original level, blends in tube harmonic colour.
        // kTubeMix = 0.3 adds ~+0.34 dB and 10.2 dB more H2 than the dry signal.
        // Adjust in [0.2, 0.5] to trade harmonic richness vs level transparency.
        static constexpr double kTubeMix = 0.3;
        buffer[i * 2] = static_cast<float>(in_l * (1.0 - kTubeMix) + harm_l * kTubeMix);

        const double in_r = buffer[i * 2 + 1];
        double harm_r = high_pass_[1].ProcessSample(in_r);
        harm_r = tube_[1].Process(harm_r);
        harm_r = low_pass_[1].ProcessSample(harm_r);
        buffer[i * 2 + 1] = static_cast<float>(in_r * (1.0 - kTubeMix) + harm_r * kTubeMix);
    }
}

void TubeSimulator::Reset() {
    const float lp_cutoff = static_cast<float>(sampling_rate_) / 2.0f - 2000.0f;

    const int idx = static_cast<int>(tube_type_);
    const auto &cfg = kTubeConfigs[idx < kTubeConfigCount ? idx : 0];

    for (uint32_t ch = 0; ch < 2; ch++) {
        high_pass_[ch].RefreshFilter(
            MultiBiquad::FilterType::HIGH_PASS,
            0.0f,
            80.0f,
            sampling_rate_,
            0.717f,
            false
        );
        low_pass_[ch].RefreshFilter(
            MultiBiquad::FilterType::LOW_PASS,
            0.0f,
            lp_cutoff,
            sampling_rate_,
            0.717f,
            false
        );
        tube_[ch].SetTubeModel(cfg.model, cfg.vdd, cfg.rp, cfg.bias);
    }
}

void TubeSimulator::SetEnable(const bool enable) {
    if (enable_ != enable) {
        if (!enable_) {
            Reset();
        }
        enable_ = enable;
    }
}

void TubeSimulator::SetTubeType(const int model) {
    const TubeType t = (model >= 0 && model < kTubeConfigCount)
        ? static_cast<TubeType>(model)
        : TubeType::k12AX7;
    if (tube_type_ != t) {
        tube_type_ = t;
        Reset();
    }
}

void TubeSimulator::SetSamplingRate(const uint32_t sampling_rate) {
    if (sampling_rate_ != sampling_rate) {
        sampling_rate_ = sampling_rate;
        Reset();
    }
}
