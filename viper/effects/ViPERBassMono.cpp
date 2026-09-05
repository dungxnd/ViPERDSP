#include "ViPERBassMono.h"
#include <algorithm>
#include <cmath>
#include <mdspan>
#include <numbers>
#include <ranges>
#include <utility>

ViPERBassMono::ViPERBassMono() {
    // scratch_buffer_ must be valid before any Process() call, including at the
    // default 44100 Hz rate where SetSamplingRate() is never invoked.
    // PureBassPlus needs 4*frames floats (2*frames FIR input + 2*frames delayed dry).
    // Size to 4096*4 for headroom consistency with ViPERBass.
    scratch_buffer_.resize(4096u * 4u, 0.0f);
    biquad_.Reset();
    biquad_.SetLowPassParameter(static_cast<float>(frequency_), sampling_rate_, 0.53f);
    subwoofer_.SetBassGain(sampling_rate_, 0.0f);
    Reset();
}

void ViPERBassMono::ShapeMix(const float bass, float& left, float& right) noexcept {
    // 1. Soft-clip the mono bass component to generate low-order warm harmonics.
    const float shaped = BassSoftClip(bass, 0.8f);

    // 2. DC-block only the mono bass path.
    //    Applying the DC blocker to the mixed stereo signal (as the previous code
    //    did) leaked the Left channel's low-frequency content into the Right
    //    channel via `out_r = mix_r - mix_l + out_l`, destroying stereo separation.
    //    Blocking only the mono `shaped` signal here eliminates inter-channel
    //    crosstalk while still removing any DC offset introduced by soft-clipping.
    const float dc_blocked_bass = dc_block_coeff_ * (dc_y1_ + shaped - dc_x1_);
    dc_x1_ = shaped;
    dc_y1_ = dc_blocked_bass;

    // 3. Add the DC-blocked mono bass cleanly to both channels.
    //    No outer soft-clip: in the complementary crossover architecture
    //    High + Bass = Dry at unity gain, so headroom is already bounded.
    left  = left  + dc_blocked_bass;
    right = right + dc_blocked_bass;
}

void ViPERBassMono::ProcessNaturalBass(float* const L, float* const R,
                                        const size_t frames) noexcept {
    for (size_t f = 0; f < frames; ++f) {
        bass_factor_smoothed_ +=
            (bass_factor_ - bass_factor_smoothed_) * smoothing_coeff_;
        const double sample = (static_cast<double>(L[f])
                               + static_cast<double>(R[f])) * 0.5;
        float bass = static_cast<float>(biquad_.ProcessSample(sample))
                     * bass_factor_smoothed_;
        if (anti_pop_ < 1.0f) {
            bass      *= anti_pop_;
            anti_pop_  = std::min(anti_pop_ + anti_pop_step_, 1.0f);
        }
        ShapeMix(bass, L[f], R[f]);
    }
}

// PureBass+ mono — complementary linear-phase crossover with transparent saturation:
//
//   x[n] ──► [ ring delay D=31 ] ────────────────────────────────────────────────────────► Dry[n-31]
//                                                                                               │
//   mid(x[n]) ──► [ Linear-Phase FIR (D=31) ] ──► Bass[n-31] ──► [ × (1+factor) ] ──► delta ──► DC-block
//                                                                                               │
//                                                                           y[n] = Dry + DC_Blocked_Delta
//
// Same correctness guarantee as ViPERBass::ProcessPureBassPlus:
//   At factor=0: boost_delta = SoftClip(bass_linear) - bass_linear ≈ 0 → output ≡ Dry[n-31].
//   DC blocker runs only on the non-linear increment, never on the direct signal path.
void ViPERBassMono::ProcessPureBassPlus(float* const L, float* const R,
                                         const size_t frames) noexcept {
    // scratch_buffer_ planar layout: [fir_l(f), fir_r(f), dry_l(f), dry_r(f)]
    // fir_l and fir_r are SEPARATE buffers (not aliased) so ProcessPlanar's
    // __restrict contract is satisfied for the in-place call below.
    float* const fir_l = scratch_buffer_.data();              // [0  ..  f-1]
    float* const fir_r = scratch_buffer_.data() + frames;     // [f  .. 2f-1]
    float* const dry_l = scratch_buffer_.data() + frames * 2u;// [2f .. 3f-1]
    float* const dry_r = scratch_buffer_.data() + frames * 3u;// [3f .. 4f-1]

    // ── Step 1: delay raw stereo dry by kLatency, fill mono FIR inputs ──────
    for (size_t f = 0; f < frames; ++f) {
        bass_delay_[delay_write_idx_ * 2u]      = L[f];
        bass_delay_[delay_write_idx_ * 2u + 1u] = R[f];

        const size_t read_idx = (delay_write_idx_ - Polyphase::kLatency) & kDelayMask;
        delay_write_idx_ = (delay_write_idx_ + 1u) & kDelayMask;

        dry_l[f] = bass_delay_[read_idx * 2u];
        dry_r[f] = bass_delay_[read_idx * 2u + 1u];

        // Mono mix fed to both FIR channels (identical data, separate buffers).
        // The linear-phase FIR is the sole frequency-selective element.
        const float mono = (L[f] + R[f]) * 0.5f;
        fir_l[f] = mono;
        fir_r[f] = mono;
    }

    // ── Step 2: linear-phase FIR extracts Bass[n-31] (group delay = 31) ────
    // fir_l/fir_r are separate buffers; in-place (in==out) per channel is safe.
    polyphase_.ProcessPlanar(fir_l, fir_r, fir_l, fir_r, frames);

    // ── Step 3: transparent saturation blend — DC-block only the boost delta ─
    for (size_t f = 0; f < frames; ++f) {
        bass_factor_smoothed_ +=
            (bass_factor_ - bass_factor_smoothed_) * smoothing_coeff_;

        // Mono bass (fir_l == fir_r after FIR since input was identical mono).
        const float bass_linear = fir_l[f];

        float boosted = bass_linear * (1.0f + bass_factor_smoothed_);
        if (anti_pop_ < 1.0f) {
            boosted   *= anti_pop_;
            anti_pop_  = std::min(anti_pop_ + anti_pop_step_, 1.0f);
        }

        const float shaped      = BassSoftClip(boosted, 0.8f);
        const float boost_delta = shaped - bass_linear;

        const float dc_blocked_delta =
            dc_block_coeff_ * (dc_y1_ + boost_delta - dc_x1_);
        dc_x1_ = boost_delta;
        dc_y1_ = dc_blocked_delta;

        L[f] = dry_l[f] + dc_blocked_delta;
        R[f] = dry_r[f] + dc_blocked_delta;
    }
}

void ViPERBassMono::ProcessSubwoofer(std::span<float> samples, StereoView audio) noexcept {
    if (anti_pop_ >= 1.0f) [[likely]] {
        subwoofer_.Process(samples.data(), static_cast<uint32_t>(audio.extent(0)));
        return;
    }

    // Crossfade dry→wet over anti_pop_ ramp to eliminate the onset transient.
    const uint32_t size         = static_cast<uint32_t>(audio.extent(0));
    const uint32_t sample_count = size * 2u;
    if (sample_count > staging_buffer_.size()) return;

    std::copy_n(samples.data(), sample_count, staging_buffer_.data());
    subwoofer_.Process(samples.data(), size);

    const StereoView dry(staging_buffer_.data(), size, 2u);
    const StereoView wet(samples.data(), size, 2u);
    for (size_t f = 0; f < size; ++f) {
        audio[f, 0] = dry[f, 0] + anti_pop_ * (wet[f, 0] - dry[f, 0]);
        audio[f, 1] = dry[f, 1] + anti_pop_ * (wet[f, 1] - dry[f, 1]);
        anti_pop_ = std::min(anti_pop_ + anti_pop_step_, 1.0f);
    }
}

void ViPERBassMono::Process(std::span<float> samples) noexcept {
    if (!enable_ || samples.empty()) return;
    const size_t size = samples.size();
    [[assume(size % 2 == 0)]];

    // Subwoofer requires interleaved layout — bounce through scratch.
    // NaturalBass and PureBassPlus are planar-native; deinterleave temporarily.
    const size_t frames = size / 2u;
    if (frames > kMaxFrames) return; // staging_buffer_ is sized for kMaxFrames
    float* const sc_l = staging_buffer_.data();
    float* const sc_r = staging_buffer_.data() + frames;

    using enum ProcessMode;
    switch (process_mode_) {
        case NaturalBass:
            for (size_t f = 0; f < frames; ++f) {
                sc_l[f] = samples[f * 2u];
                sc_r[f] = samples[f * 2u + 1u];
            }
            ProcessNaturalBass(sc_l, sc_r, frames);
            for (size_t f = 0; f < frames; ++f) {
                samples[f * 2u]      = sc_l[f];
                samples[f * 2u + 1u] = sc_r[f];
            }
            break;
        case PureBassPlus: {
            for (size_t f = 0; f < frames; ++f) {
                sc_l[f] = samples[f * 2u];
                sc_r[f] = samples[f * 2u + 1u];
            }
            ProcessPureBassPlus(sc_l, sc_r, frames);
            for (size_t f = 0; f < frames; ++f) {
                samples[f * 2u]      = sc_l[f];
                samples[f * 2u + 1u] = sc_r[f];
            }
            break;
        }
        case Subwoofer: {
            StereoView audio(samples.data(), frames, 2u);
            ProcessSubwoofer(samples, audio);
            break;
        }
        default: return;
    }
}

void ViPERBassMono::Reset() noexcept {
    polyphase_.SetCutoffFrequency(static_cast<float>(frequency_));
    polyphase_.SetSamplingRate(sampling_rate_);
    polyphase_.Reset();
    subwoofer_.Reset(); // clear biquad delay-state before reconfiguring coefficients
    subwoofer_.SetBassGain(sampling_rate_, bass_factor_ * 2.5f);
    biquad_.SetLowPassParameter(static_cast<float>(frequency_), sampling_rate_, 0.53f);

    std::ranges::fill(bass_delay_, 0.0f);
    delay_write_idx_ = 0u;

    const auto sr_f       = static_cast<float>(sampling_rate_);
    anti_pop_step_        = 1.0f / (0.020f * sr_f);
    anti_pop_             = 0.0f;
    smoothing_coeff_      = 1.0f - std::exp(-1.0f / (0.030f * sr_f));
    bass_factor_smoothed_ = bass_factor_;
    dc_block_coeff_       = std::exp(-2.0f * std::numbers::pi_v<float> * 18.0f / sr_f);
    dc_x1_ = 0.0f;
    dc_y1_ = 0.0f;
}

void ViPERBassMono::SetConfig(const Config& config) noexcept {
    config_ = config;
    SetEnable(config.enable);
    SetProcessMode(static_cast<ProcessMode>(config.mode));
    SetFrequency(config.frequency);
    SetBassFactor(config.gain);
    SetAntiPop(config.anti_pop);
}

void ViPERBassMono::SetEnable(const bool enable) noexcept {
    config_.enable = enable;
    if (enable_ != enable) {
        if (enable) Reset();
        enable_ = enable;
    }
}

void ViPERBassMono::SetProcessMode(const ProcessMode mode) noexcept {
    const auto safe_mode = static_cast<ProcessMode>(
        std::clamp(std::to_underlying(mode), uint8_t{0}, uint8_t{2}));
    config_.mode = std::to_underlying(safe_mode);
    if (process_mode_ != safe_mode) {
        process_mode_ = safe_mode;
        Reset();
    }
}

void ViPERBassMono::SetBassFactor(const float value) noexcept {
    config_.gain = value;
    if (bass_factor_ != value) {
        bass_factor_ = value;
        subwoofer_.SetBassGain(sampling_rate_, bass_factor_ * 2.5f);
    }
}

void ViPERBassMono::SetFrequency(const uint32_t value) noexcept {
    const uint32_t safe_val = std::clamp(value, 30u, 300u);
    config_.frequency = safe_val;
    if (frequency_ != safe_val) {
        frequency_ = safe_val;
        // Update the linear-phase FIR crossover cutoff for PureBass+ mode.
        polyphase_.SetCutoffFrequency(static_cast<float>(safe_val));
        // Keep biquad in sync for NaturalBass mode.
        biquad_.SetLowPassParameter(
            static_cast<float>(frequency_), sampling_rate_, 0.53f);
    }
}

void ViPERBassMono::SetAntiPop(const bool enable) noexcept {
    anti_pop_ = enable ? 0.0f : 1.0f;
}

void ViPERBassMono::SetSamplingRate(const uint32_t sampling_rate) noexcept {
    if (sampling_rate_ != sampling_rate) {
        sampling_rate_ = sampling_rate;
        scratch_buffer_.resize(4096u * 6u);
        Reset();
    }
}

void ViPERBassMono::ProcessPlanar(std::span<float> L, std::span<float> R) noexcept {
    if (!IsEnabled() || L.empty()) return;
    const size_t frames = L.size();

    using enum ProcessMode;
    switch (process_mode_) {
        case NaturalBass:
            ProcessNaturalBass(L.data(), R.data(), frames);
            break;
        case PureBassPlus:
            ProcessPureBassPlus(L.data(), R.data(), frames);
            break;
        case Subwoofer: {
            // Subwoofer requires interleaved layout — minimal bounce through scratch.
            float* const sc = scratch_buffer_.data();
            for (size_t i = 0; i < frames; ++i) {
                sc[i * 2u]      = L[i];
                sc[i * 2u + 1u] = R[i];
            }
            StereoView audio(sc, frames, 2u);
            ProcessSubwoofer(std::span<float>{sc, frames * 2u}, audio);
            for (size_t i = 0; i < frames; ++i) {
                L[i] = sc[i * 2u];
                R[i] = sc[i * 2u + 1u];
            }
            break;
        }
        default: return;
    }
}
