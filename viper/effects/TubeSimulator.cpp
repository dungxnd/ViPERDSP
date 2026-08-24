#include "TubeSimulator.h"
#include <algorithm>
#include <array>
#include <cmath>

// Per-model circuit parameters: { TubeModel, Vdd, Rp, bias, output_gain }
// 12AX7: high-gain stage, classic V4A topology                    (mu≈98,  gm=1.6mA/V)
// 6N1P:  Soviet medium-gain, warm H2 character                    (mu=30.8, gm=4.5mA/V)
// 12AU7: clean low-distortion triode, high headroom               (mu=18,  gm=3.0mA/V)
// 12AT7: high-transconductance, medium-high gain                  (mu=63,  gm=5.5mA/V)
// 6DJ8:  high-gm, low-rp frame-grid triode, open/airy character   (mu=33,  gm=12.5mA/V)
//
// output_gain normalises each tube to the same perceived loudness at equal drive.
// Reference: 12AX7 small-signal gain = 0.5534 (drive=1, Rp=100kΩ).
//   6N1P  gain = 0.2186 → comp = 0.5534/0.2186 = 2.531  (+8.07 dB)
//   12AU7 gain = 0.1176 → comp = 0.5534/0.1176 = 4.707  (+13.46 dB)
//   12AT7 gain = 0.5101 → comp = 0.5534/0.5101 = 1.085  (+0.71 dB)
//   6DJ8  gain = 0.3269 → comp = 0.5534/0.3269 = 1.693  (+4.57 dB)
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
//
// 12AT7 quadric parameters: LS fit over 11 Brimar datasheet points
//   (Vp ∈ {100..300}V, Vg ∈ {0,-1,-2,-3}V — Curve No. 313.13)
//   sqrt(ip) = c1*Vp + c2*Vg + c0
//   c1=6.688e-4, c2=4.211e-2, c0=3.804e-2  →  RMS fit error 8.5%
//   kp2=c1²=4.4729e-7, kpg=2*c1*c2=5.6324e-5, kp=2*c1*c0=5.0885e-5, mu=c2/c1=63.0
//   Circuit: Rp=100kΩ, bias=-2.5V → Ia_q=1.02mA, Vpk_q=148V (centred swing)
//
// 6DJ8 quadric parameters: LS fit over 19 Philips ECC88 datasheet points
//   (Vp ∈ {50..150}V, Vg ∈ {0,-1,-2,-3}V — Philips plate curves, Curve 313.21)
//   sqrt(ip) = c1*Vp + c2*Vg + c0
//   c1=4.812e-4, c2=2.949e-2, c0=1.231e-1  →  RMS fit error 6.2%
//   kp2=c1²=2.3156e-7, kpg=2*c1*c2=2.8386e-5, kp=2*c1*c0=1.1844e-4, mu_local≈28.6
//   Circuit: Rp=9.1kΩ, bias=-1.3V → Ia_q=17.0mA, Vpk_q=95V (centred swing)
struct TubeConfig {
    TubeModel model;
    double    vdd;
    double    rp;
    double    bias;
    double    output_gain; // level-matching compensation relative to 12AX7
};

static constexpr std::array<TubeConfig, 5> kTubeConfigs{{
    { { 1.014e-5,   5.498e-8,  1.076e-5  }, 250.0, 100000.0, -1.5, 1.000 },  // 0: 12AX7 (reference)
    { { 4.8753e-5,  8.7966e-7, 5.4189e-5 }, 250.0,  20000.0, -4.0, 2.531 },  // 1: 6N1P  (+8.07 dB comp)
    { { 4.6931e-5,  6.1383e-7, 2.2105e-5 }, 250.0,  22000.0, -8.5, 4.707 },  // 2: 12AU7 (+13.46 dB comp)
    { { 5.0885e-5,  4.4729e-7, 5.6324e-5 }, 250.0, 100000.0, -2.5, 1.085 },  // 3: 12AT7 (+0.71 dB comp)
    { { 1.1844e-4,  2.3156e-7, 2.8386e-5 }, 250.0,   9100.0, -1.3, 1.693 },  // 4: 6DJ8  (+4.57 dB comp)
}};

TubeSimulator::TubeSimulator() {
    Reset();
}

void TubeSimulator::Process(float *buffer, const uint32_t size) noexcept {
    if (!enable_) return;

    // Hoist mix coefficients once per block — avoids repeated float→double widening
    // and float subtraction inside the hot loop.
    const double wet_gain = static_cast<double>(mix_amount_);
    const double dry_gain = 1.0 - wet_gain;

    for (uint32_t i = 0; i < size; i++) {
        // Sanitize: Inf/NaN on input would corrupt IIR state permanently.
        if (!std::isfinite(buffer[i * 2]))     buffer[i * 2]     = 0.0f;
        if (!std::isfinite(buffer[i * 2 + 1])) buffer[i * 2 + 1] = 0.0f;

        const double in_l = buffer[i * 2];
        // Phase-matched dry path: same APF poles as HPF + LPF on the wet chain
        const double dry_l = dry_apf_lpf_[0].ProcessSample(
                                 dry_apf_hpf_[0].ProcessSample(in_l));
        double harm_l = high_pass_[0].ProcessSample(in_l);
        harm_l = (tube_mode_ == TubeMode::kStatic)
            ? tube_[0].Process(harm_l)
            : tube_wdf_[0].Process(harm_l);
        harm_l = low_pass_[0].ProcessSample(harm_l);
        buffer[i * 2] = static_cast<float>(dry_l * dry_gain + harm_l * wet_gain);

        const double in_r = buffer[i * 2 + 1];
        const double dry_r = dry_apf_lpf_[1].ProcessSample(
                                 dry_apf_hpf_[1].ProcessSample(in_r));
        double harm_r = high_pass_[1].ProcessSample(in_r);
        harm_r = (tube_mode_ == TubeMode::kStatic)
            ? tube_[1].Process(harm_r)
            : tube_wdf_[1].Process(harm_r);
        harm_r = low_pass_[1].ProcessSample(harm_r);
        buffer[i * 2 + 1] = static_cast<float>(dry_r * dry_gain + harm_r * wet_gain);
    }
}

void TubeSimulator::Reset() noexcept {
    const float lp_cutoff = static_cast<float>(sampling_rate_) / 2.0f - 2000.0f;

    const int idx       = static_cast<int>(tube_type_);
    const auto& cfg     = kTubeConfigs[static_cast<std::size_t>(idx) < kTubeConfigs.size()
                                        ? idx : 0];

    for (std::size_t ch = 0; ch < 2; ch++) {
        high_pass_[ch].RefreshFilter(
            MultiBiquad::FilterType::HighPass,
            0.0f, hpf_cutoff_hz_, sampling_rate_, 0.717f, false);
        low_pass_[ch].RefreshFilter(
            MultiBiquad::FilterType::LowPass,
            0.0f, lp_cutoff, sampling_rate_, 0.717f, false);

        // Matched allpass on dry path: same poles as HPF/LPF → cancels phase mismatch
        dry_apf_hpf_[ch].RefreshFilter(
            MultiBiquad::FilterType::AllPass,
            0.0f, hpf_cutoff_hz_, sampling_rate_, 0.717f, false);
        dry_apf_lpf_[ch].RefreshFilter(
            MultiBiquad::FilterType::AllPass,
            0.0f, lp_cutoff, sampling_rate_, 0.717f, false);

        tube_[ch].SetTubeModel(cfg.model, cfg.vdd, cfg.rp, cfg.bias, cfg.output_gain);
        tube_wdf_[ch].SetTubeModel(cfg.model, cfg.vdd, cfg.rp, cfg.bias, cfg.output_gain);
    }
}

void TubeSimulator::SetEnable(const bool enable) noexcept {
    if (enable_ != enable) {
        if (!enable_) Reset();
        enable_ = enable;
    }
}

void TubeSimulator::SetTubeType(const int model) noexcept {
    const TubeType t = (model >= 0 && static_cast<std::size_t>(model) < kTubeConfigs.size())
        ? static_cast<TubeType>(model)
        : TubeType::k12AX7;
    if (tube_type_ != t) {
        tube_type_ = t;
        Reset();
    }
}

void TubeSimulator::SetTubeMode(const int mode) noexcept {
    const TubeMode t = (mode == 1) ? TubeMode::kWDF : TubeMode::kStatic;
    if (tube_mode_ != t) {
        tube_mode_ = t;
        Reset();
    }
}

void TubeSimulator::SetTubeMix(const float mix) noexcept {
    mix_amount_ = std::clamp(mix, 0.0f, 1.0f);
}

void TubeSimulator::SetTubeDrive(const float drive) noexcept {
    const double d = static_cast<double>(drive);
    for (auto& t : tube_)     t.SetDrive(d);
    for (auto& t : tube_wdf_) t.SetDrive(d);
}

void TubeSimulator::SetTubeHpfCutoff(const float cutoff_hz) noexcept {
    // Clamp to safe operating range [20 Hz – 250 Hz].
    const float clamped = std::clamp(cutoff_hz, 20.0f, 250.0f);
    if (hpf_cutoff_hz_ == clamped) return;
    hpf_cutoff_hz_ = clamped;
    // Update wet HPF and dry APF synchronously — tube DC-blocker state is untouched.
    for (std::size_t ch = 0; ch < 2; ch++) {
        high_pass_[ch].RefreshFilter(
            MultiBiquad::FilterType::HighPass,
            0.0f, hpf_cutoff_hz_, sampling_rate_, 0.717f, false);
        dry_apf_hpf_[ch].RefreshFilter(
            MultiBiquad::FilterType::AllPass,
            0.0f, hpf_cutoff_hz_, sampling_rate_, 0.717f, false);
    }
}

void TubeSimulator::SetSamplingRate(const uint32_t sampling_rate) noexcept {
    if (sampling_rate_ != sampling_rate) {
        sampling_rate_ = sampling_rate;
        Reset();
    }
}

void TubeSimulator::ProcessPlanar(float* __restrict L, float* __restrict R, const size_t frames) noexcept {
    if (!IsEnabled() || frames == 0) return;

    const double wet_gain = static_cast<double>(mix_amount_);
    const double dry_gain = 1.0 - wet_gain;

    for (size_t i = 0u; i < frames; ++i) {
        if (!std::isfinite(L[i])) L[i] = 0.0f;
        if (!std::isfinite(R[i])) R[i] = 0.0f;

        const double in_l  = L[i];
        const double dry_l = dry_apf_lpf_[0].ProcessSample(dry_apf_hpf_[0].ProcessSample(in_l));
        double harm_l      = high_pass_[0].ProcessSample(in_l);
        harm_l = (tube_mode_ == TubeMode::kStatic) ? tube_[0].Process(harm_l) : tube_wdf_[0].Process(harm_l);
        harm_l = low_pass_[0].ProcessSample(harm_l);
        L[i] = static_cast<float>(dry_l * dry_gain + harm_l * wet_gain);

        const double in_r  = R[i];
        const double dry_r = dry_apf_lpf_[1].ProcessSample(dry_apf_hpf_[1].ProcessSample(in_r));
        double harm_r      = high_pass_[1].ProcessSample(in_r);
        harm_r = (tube_mode_ == TubeMode::kStatic) ? tube_[1].Process(harm_r) : tube_wdf_[1].Process(harm_r);
        harm_r = low_pass_[1].ProcessSample(harm_r);
        R[i] = static_cast<float>(dry_r * dry_gain + harm_r * wet_gain);
    }
}
