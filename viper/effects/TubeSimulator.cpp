#include "TubeSimulator.h"
#include "../constants.h"

// Per-model circuit parameters: { TubeModel, Vdd, Rp, bias, output_gain }
// 12AX7: high-gain stage, classic V4A topology
// 6N1J:  Soviet medium-gain, warmer H2 character (mu=35, gm=4.35mA/V)
//
// output_gain normalises each tube to the same perceived loudness at equal drive.
// 12AX7 small-signal gain = 0.5534, 6N1J = 0.2430 (ratio 2.277x / +7.15 dB).
// 6N1J compensating factor = 0.5534 / 0.2430 = 2.277 (12AX7 is the reference at 1.0).
static constexpr struct {
    TubeModel   model;
    double      vdd;
    double      rp;
    double      bias;
    double      output_gain; // level-matching compensation relative to 12AX7
} kTubeConfigs[] = {
    { { 1.014e-5, 5.498e-8, 1.076e-5 },    250.0, 100000.0, -1.5, 1.000 },  // 0: 12AX7 (reference)
    { { 5.7349e-5, 5.1490e-7, 3.6043e-5 }, 250.0,  20000.0, -2.0, 2.277 },  // 1: 6N1J  (+7.15 dB comp)
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

    // Hoist mix coefficients once per block — avoids repeated float→double widening
    // and float subtraction inside the hot loop.
    const double wet_gain = static_cast<double>(mix_amount_);
    const double dry_gain = 1.0 - wet_gain;

    for (uint32_t i = 0; i < size; i++) {
        const double in_l = buffer[i * 2];
        // Phase-matched dry path: same APF poles as HPF + LPF on the wet chain
        const double dry_l = dry_apf_lpf_[0].ProcessSample(
                                 dry_apf_hpf_[0].ProcessSample(in_l));
        double harm_l = high_pass_[0].ProcessSample(in_l);
        harm_l = tube_[0].Process(harm_l);
        harm_l = low_pass_[0].ProcessSample(harm_l);
        buffer[i * 2] = static_cast<float>(dry_l * dry_gain + harm_l * wet_gain);

        const double in_r = buffer[i * 2 + 1];
        const double dry_r = dry_apf_lpf_[1].ProcessSample(
                                 dry_apf_hpf_[1].ProcessSample(in_r));
        double harm_r = high_pass_[1].ProcessSample(in_r);
        harm_r = tube_[1].Process(harm_r);
        harm_r = low_pass_[1].ProcessSample(harm_r);
        buffer[i * 2 + 1] = static_cast<float>(dry_r * dry_gain + harm_r * wet_gain);
    }
}

void TubeSimulator::Reset() {
    const float lp_cutoff = static_cast<float>(sampling_rate_) / 2.0f - 2000.0f;

    const int idx = static_cast<int>(tube_type_);
    const auto &cfg = kTubeConfigs[idx < kTubeConfigCount ? idx : 0];

    for (uint32_t ch = 0; ch < 2; ch++) {
        high_pass_[ch].RefreshFilter(
            MultiBiquad::FilterType::HIGH_PASS,
            0.0f, 120.0f, sampling_rate_, 0.717f, false);
        low_pass_[ch].RefreshFilter(
            MultiBiquad::FilterType::LOW_PASS,
            0.0f, lp_cutoff, sampling_rate_, 0.717f, false);

        // Matched allpass on dry path: same poles as HPF/LPF → cancels phase mismatch
        dry_apf_hpf_[ch].RefreshFilter(
            MultiBiquad::FilterType::ALL_PASS,
            0.0f, 120.0f, sampling_rate_, 0.717f, false);
        dry_apf_lpf_[ch].RefreshFilter(
            MultiBiquad::FilterType::ALL_PASS,
            0.0f, lp_cutoff, sampling_rate_, 0.717f, false);

        tube_[ch].SetTubeModel(cfg.model, cfg.vdd, cfg.rp, cfg.bias, cfg.output_gain);
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

void TubeSimulator::SetTubeMix(const float mix) {
    mix_amount_ = (mix < 0.0f) ? 0.0f : (mix > 1.0f) ? 1.0f : mix;
}

void TubeSimulator::SetTubeDrive(const float drive) {
    tube_[0].SetDrive(static_cast<double>(drive));
    tube_[1].SetDrive(static_cast<double>(drive));
}

void TubeSimulator::SetSamplingRate(const uint32_t sampling_rate) {
    if (sampling_rate_ != sampling_rate) {
        sampling_rate_ = sampling_rate;
        Reset();
    }
}
