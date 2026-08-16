#include "TubeSimulator.h"
#include "../constants.h"

// Per-model circuit parameters: { TubeModel, Vdd, Rp, bias, output_gain }
// 12AX7: high-gain stage, classic V4A topology                    (mu≈98,  gm=1.6mA/V)
// 6N1P:  Soviet medium-gain, warm H2 character                    (mu=30.8, gm=4.5mA/V)
// 12AU7: clean low-distortion triode, high headroom               (mu=18,  gm=3.0mA/V)
//
// output_gain normalises each tube to the same perceived loudness at equal drive.
// Reference: 12AX7 small-signal gain = 0.5534 (drive=1, Rp=100kΩ).
//   6N1P  gain = 0.2186 → comp = 0.5534/0.2186 = 2.531  (+8.07 dB)
//   12AU7 gain = 0.1176 → comp = 0.5534/0.1176 = 4.707  (+13.46 dB)
//
// 6N1P quadric parameters: LS fit over 10 Russian datasheet points
//   (Vp ∈ {100..350}V, Vg ∈ {0,-2,-4,-6,-8}V — Page 2 averaged characteristics)
//   sqrt(ip) = c1*Vp + c2*Vg + c0
//   c1=9.379e-4, c2=2.889e-2, c0=2.599e-2  →  RMS fit error 5.3%
//   kp2=c1²=8.7966e-7, kpg=2*c1*c2=5.4189e-5, kp=2*c1*c0=4.8753e-5, mu=c2/c1=30.8
//   Circuit: Rp=20kΩ, bias=-4.0V → Ia_q=4.25mA, Vpk_q=165V (centred swing)
//
// 12AU7 quadric parameters: LS fit over 8 Brimar datasheet points
//   (Vp ∈ {100..300}V, Vg ∈ {0,-5,-8.5,-10}V — Curve No. 313.20)
//   sqrt(ip) = c1*Vp + c2*Vg + c0
//   c1=7.835e-4, c2=1.411e-2, c0=2.995e-2  →  RMS fit error 5.3%
//   kp2=c1²=6.1383e-7, kpg=2*c1*c2=2.2105e-5, kp=2*c1*c0=4.6931e-5, mu=c2/c1=18.01
//   Circuit: Rp=22kΩ, bias=-8.5V → Ia_q=2.98mA, Vpk_q=184.5V (centred swing)
static constexpr struct {
    TubeModel   model;
    double      vdd;
    double      rp;
    double      bias;
    double      output_gain; // level-matching compensation relative to 12AX7
} kTubeConfigs[] = {
    { { 1.014e-5,   5.498e-8,  1.076e-5  }, 250.0, 100000.0, -1.5, 1.000 },  // 0: 12AX7 (reference)
    { { 4.8753e-5,  8.7966e-7, 5.4189e-5 }, 250.0,  20000.0, -4.0, 2.531 },  // 1: 6N1P  (+8.07 dB comp)
    { { 4.6931e-5,  6.1383e-7, 2.2105e-5 }, 250.0,  22000.0, -8.5, 4.707 },  // 2: 12AU7 (+13.46 dB comp)
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
