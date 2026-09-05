#include "PsychoacousticBass.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <span>

namespace {

constexpr std::array<float, 10> kHarmonicOrder2{
    0.0f, 1.0f,  0.0f, 0.0f, 0.0f,
    0.0f, 0.0f,  0.0f, 0.0f, 0.0f,
};
constexpr std::array<float, 10> kHarmonicOrder3{
    0.0f, 0.7f,  0.3f, 0.0f, 0.0f,
    0.0f, 0.0f,  0.0f, 0.0f, 0.0f,
};
constexpr std::array<float, 10> kHarmonicOrder4{
    0.0f, 0.5f,  0.3f, 0.2f, 0.0f,
    0.0f, 0.0f,  0.0f, 0.0f, 0.0f,
};
constexpr std::array<float, 10> kHarmonicOrder5{
    0.0f, 0.4f,  0.25f, 0.2f, 0.15f,
    0.0f, 0.0f,  0.0f,  0.0f, 0.0f,
};

// Index 0 unused; order N maps to index N (valid range 2–5).
constexpr std::array<const std::array<float, 10>*, 6> kHarmonicTables{{
    nullptr,          // 0 — invalid
    nullptr,          // 1 — invalid
    &kHarmonicOrder2,
    &kHarmonicOrder3,
    &kHarmonicOrder4,
    &kHarmonicOrder5,
}};

constexpr float kFilterQ = 0.717f;

} // anonymous namespace

// ---------------------------------------------------------------------------

PsychoacousticBass::PsychoacousticBass() {
    // All members initialized by in-class defaults.
    Reset();
}

void PsychoacousticBass::Process(float* const samples, const uint32_t size) noexcept {
    if (!enable_) return;

    for (uint32_t i = 0; i < size * 2; i += 2) {
        const double bass_l = lowpass_[0].ProcessSample(samples[i]);
        const double bass_r = lowpass_[1].ProcessSample(samples[i + 1]);

        const double peak = std::max(std::abs(bass_l), std::abs(bass_r));
        envelope_ += (peak > envelope_) ? att_coeff_ * (peak - envelope_)
                                        : rel_coeff_ * (peak - envelope_);
        envelope_ = std::max(envelope_, 1e-10);

        const double norm_l = std::clamp(bass_l / envelope_, -1.0, 1.0);
        const double norm_r = std::clamp(bass_r / envelope_, -1.0, 1.0);

        double harmonic_l = harmonics_[0].Process(norm_l) * envelope_;
        double harmonic_r = harmonics_[1].Process(norm_r) * envelope_;

        harmonic_l = highpass_[0].ProcessSample(harmonic_l);
        harmonic_r = highpass_[1].ProcessSample(harmonic_r);

        samples[i]     += static_cast<float>(harmonic_l * intensity_);
        samples[i + 1] += static_cast<float>(harmonic_r * intensity_);

        if (original_bass_level_ < 1.0f) {
            const float scaled_l = static_cast<float>(bass_l) * original_bass_level_;
            const float scaled_r = static_cast<float>(bass_r) * original_bass_level_;
            samples[i]     = (samples[i]     - static_cast<float>(bass_l)) + scaled_l;
            samples[i + 1] = (samples[i + 1] - static_cast<float>(bass_r)) + scaled_r;
        }
    }
}

void PsychoacousticBass::Reset() noexcept {
    envelope_ = 1e-10;
    for (uint32_t ch = 0; ch < 2; ++ch) {
        lowpass_[ch].Reset();
        highpass_[ch].Reset();
    }
    RefreshTimeConstants();
    RefreshFilters();
    ApplyHarmonicCoeffs();
}

void PsychoacousticBass::SetConfig(const Config& config) noexcept {
    config_ = config;
    SetEnable(config.enable);
    SetCutoff(config.cutoff);
    SetIntensity(config.intensity);
    SetHarmonicOrder(config.harmonic_order);
    SetOriginalBassLevel(config.original_level);
}

void PsychoacousticBass::SetEnable(const bool enable) noexcept {
    config_.enable = enable;
    if (enable_ != enable) {
        if (enable) Reset();
        enable_ = enable;
    }
}

void PsychoacousticBass::SetSamplingRate(const uint32_t sampling_rate) noexcept {
    if (sampling_rate_ == sampling_rate) return;
    sampling_rate_ = sampling_rate;
    Reset(); // calls RefreshTimeConstants() + RefreshFilters()
}

void PsychoacousticBass::SetCutoff(const uint32_t value) noexcept {
    const uint32_t clamped = std::clamp(value, 60u, 150u);
    if (cutoff_ == clamped) return;
    cutoff_ = clamped;
    RefreshFilters();
}

void PsychoacousticBass::SetIntensity(const uint32_t value) noexcept {
    intensity_ = static_cast<float>(std::min(value, 100u)) / 100.0f;
}

void PsychoacousticBass::SetHarmonicOrder(const uint32_t value) noexcept {
    const uint32_t clamped = std::clamp(value, 2u, 5u);
    if (harmonic_order_ == clamped) return;
    harmonic_order_ = clamped;
    ApplyHarmonicCoeffs();
}

void PsychoacousticBass::SetOriginalBassLevel(const uint32_t value) noexcept {
    original_bass_level_ = static_cast<float>(std::min(value, 100u)) / 100.0f;
}

void PsychoacousticBass::RefreshFilters() noexcept {
    const auto freq = static_cast<float>(cutoff_);
    for (uint32_t ch = 0; ch < 2; ++ch) {
        lowpass_[ch].RefreshFilter(
            MultiBiquad::LOW_PASS,  0.0f, freq, sampling_rate_, kFilterQ, false
        );
        highpass_[ch].RefreshFilter(
            MultiBiquad::HIGH_PASS, 0.0f, freq, sampling_rate_, kFilterQ, false
        );
    }
}

// Compute sample-rate-dependent envelope follower time constants.
// attack  τ = 2.5 ms,  release τ = 250 ms.
// α = 1 − e^(−1 / (τ · fs))
void PsychoacousticBass::RefreshTimeConstants() noexcept {
    const double fs = static_cast<double>(sampling_rate_);
    att_coeff_ = 1.0 - std::exp(-1.0 / (0.0025 * fs));
    rel_coeff_ = 1.0 - std::exp(-1.0 / (0.2500 * fs));
}

void PsychoacousticBass::ApplyHarmonicCoeffs() noexcept {
    const auto* const coeffs = kHarmonicTables[harmonic_order_];
    for (auto& h : harmonics_) {
        h.SetHarmonics(*coeffs);
    }
}

void PsychoacousticBass::ProcessPlanar(std::span<float> L, std::span<float> R) noexcept {
    if (!IsEnabled() || L.empty()) return;

    for (size_t i = 0u; i < L.size(); ++i) {
        const double bass_l = lowpass_[0].ProcessSample(L[i]);
        const double bass_r = lowpass_[1].ProcessSample(R[i]);

        const double peak = std::max(std::abs(bass_l), std::abs(bass_r));
        envelope_ += (peak > envelope_) ? att_coeff_ * (peak - envelope_)
                                        : rel_coeff_ * (peak - envelope_);
        envelope_ = std::max(envelope_, 1e-10);

        const double norm_l = std::clamp(bass_l / envelope_, -1.0, 1.0);
        const double norm_r = std::clamp(bass_r / envelope_, -1.0, 1.0);

        double harmonic_l = harmonics_[0].Process(norm_l) * envelope_;
        double harmonic_r = harmonics_[1].Process(norm_r) * envelope_;

        harmonic_l = highpass_[0].ProcessSample(harmonic_l);
        harmonic_r = highpass_[1].ProcessSample(harmonic_r);

        L[i] += static_cast<float>(harmonic_l * intensity_);
        R[i] += static_cast<float>(harmonic_r * intensity_);

        if (original_bass_level_ < 1.0f) {
            L[i] = (L[i] - static_cast<float>(bass_l)) + static_cast<float>(bass_l) * original_bass_level_;
            R[i] = (R[i] - static_cast<float>(bass_r)) + static_cast<float>(bass_r) * original_bass_level_;
        }
    }
}
