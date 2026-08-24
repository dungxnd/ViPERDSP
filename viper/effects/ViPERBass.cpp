#include "ViPERBass.h"
#include <algorithm>
#include <cmath>
#include <mdspan>
#include <numbers>
#include <ranges>
#include <utility>

ViPERBass::ViPERBass() {
    // scratch_buffer_ must be valid before any Process() call, including at the
    // default 44100 Hz rate where SetSamplingRate() is never invoked.
    // PureBassPlus needs 4*frames floats (2*frames FIR + 2*frames delayed bass).
    // Size to 4096*4 so any block size up to 4096 frames is safe without RT realloc.
    scratch_buffer_.resize(4096u * 4u, 0.0f);
    for (auto& bq : biquad_) {
        bq.Reset();
        bq.SetLowPassParameter(static_cast<float>(frequency_), sampling_rate_, 0.53f);
    }
    subwoofer_.SetBassGain(sampling_rate_, 0.0f);
    Reset();
}

void ViPERBass::ShapeMix(float& left, float& right, float bass_l, float bass_r) noexcept {
    // 1. Soft-clip only the bass component to generate low-order warm harmonics.
    const float shaped_l = BassSoftClip(bass_l, 0.8f);
    const float shaped_r = BassSoftClip(bass_r, 0.8f);

    // 2. DC-block only the bass path.
    //    The DC blocker is applied before mixing so that any DC offset introduced
    //    by soft-clipping is removed from the bass signal alone — not from the
    //    full-range dry signal.  This prevents high-pass filtering of the dry path
    //    and eliminates the cross-channel bleed that occurred when the blocker ran
    //    on the already-mixed signal.
    const float out_bass_l = dc_block_coeff_ * (dc_y1_[0] + shaped_l - dc_x1_[0]);
    dc_x1_[0] = shaped_l;  dc_y1_[0] = out_bass_l;

    const float out_bass_r = dc_block_coeff_ * (dc_y1_[1] + shaped_r - dc_x1_[1]);
    dc_x1_[1] = shaped_r;  dc_y1_[1] = out_bass_r;

    // 3. Add the DC-blocked bass cleanly to the (high-pass) dry signal.
    //    No outer soft-clip: in the complementary crossover architecture
    //    High + Bass = Dry at unity gain, so headroom is already bounded.
    left  = left  + out_bass_l;
    right = right + out_bass_r;
}

void ViPERBass::ApplyAntiPop(float& bass_l, float& bass_r) noexcept {
    if (anti_pop_ >= 1.0f) [[likely]] return;
    bass_l   *= anti_pop_;
    bass_r   *= anti_pop_;
    anti_pop_ = std::min(anti_pop_ + anti_pop_step_, 1.0f);
}

void ViPERBass::ProcessNaturalBass(StereoView audio) noexcept {
    for (size_t f = 0; f < audio.extent(0); ++f) {
        bass_factor_smoothed_ +=
            (bass_factor_ - bass_factor_smoothed_) * smoothing_coeff_;
        float bass_l = static_cast<float>(biquad_[0].ProcessSample(audio[f, 0]))
                       * bass_factor_smoothed_;
        float bass_r = static_cast<float>(biquad_[1].ProcessSample(audio[f, 1]))
                       * bass_factor_smoothed_;
        ApplyAntiPop(bass_l, bass_r);
        ShapeMix(audio[f, 0], audio[f, 1], bass_l, bass_r);
    }
}

// PureBass+ complementary linear-phase crossover with transparent saturation:
//
//   x[n] ──► [ ring delay D=31 ] ──────────────────────────────────────────────────────────► Dry[n-31]
//                                                                                                │
//   x[n] ──► [ Linear-Phase FIR (D=31) ] ──► Bass[n-31] ──► [ × (1+factor) ] ──► shaped ──► delta ──► DC-block
//                                                                                                │
//                                                                             y[n] = Dry + DC_Blocked_Delta
//
// Correctness proof at factor = 0:
//   boost_delta = SoftClip(bass_linear) - bass_linear  ≈ 0 (sub-bass level << clip threshold)
//   output      = Dry[n-31] + DCBlock(0) = Dry[n-31]   — bit-perfect pass-through, zero phase error.
//
// Correctness proof at factor > 0:
//   Only the non-linear harmonic increment (SoftClip(boosted) - bass_linear) is DC-blocked
//   and injected.  The linear bass and dry paths combine as Dry = High + Bass without any
//   phase-shifted component in the direct path — no comb-filter dip at the crossover.
void ViPERBass::ProcessPureBassPlus(StereoView audio) noexcept {
    const size_t frames = audio.extent(0);

    // scratch_buffer_ layout:
    //   [0  .. 2f-1]: fir_buf — raw stereo input/output for the linear-phase FIR
    //   [2f .. 4f-1]: dry_buf — clean dry stereo delayed by kLatency frames
    // Total: 4 * frames.  Pre-sized to 4096 * 4 in ctor / SetSamplingRate().
    float* const fir_buf = scratch_buffer_.data();                 // [0  .. 2f-1]
    float* const dry_buf = scratch_buffer_.data() + frames * 2u;  // [2f .. 4f-1]

    // ── Step 1: delay raw stereo dry by kLatency, feed raw audio to FIR ──────
    for (size_t f = 0; f < frames; ++f) {
        // Store raw dry input in ring delay; read back kLatency samples ago.
        bass_delay_[0][delay_write_idx_] = audio[f, 0];
        bass_delay_[1][delay_write_idx_] = audio[f, 1];

        const size_t read_idx = (delay_write_idx_ - Polyphase::kLatency) & kDelayMask;
        delay_write_idx_ = (delay_write_idx_ + 1u) & kDelayMask;

        dry_buf[f * 2u]      = bass_delay_[0][read_idx];
        dry_buf[f * 2u + 1u] = bass_delay_[1][read_idx];

        // Feed the raw (unfiltered) stereo signal to the FIR.
        // No IIR pre-filter: the linear-phase FIR is the sole frequency-selective
        // element, so group delay is constant (31 samples) at all frequencies.
        fir_buf[f * 2u]      = audio[f, 0];
        fir_buf[f * 2u + 1u] = audio[f, 1];
    }

    // ── Step 2: linear-phase FIR extracts Bass[n-31] (group delay = 31) ──────
    polyphase_.Process(fir_buf, static_cast<uint32_t>(frames));

    // ── Step 3: transparent saturation blend — DC-block only the boost delta ─
    //
    // The previous approach called ShapeMix(high, boosted), which DC-blocked the
    // entire boosted bass signal before adding it to `high`.  At factor=0 this
    // caused a spurious phase lead from the DC blocker on the linear bass path:
    //
    //   output = (Dry - Bass) + DCBlock(SoftClip(Bass))
    //          = Dry + (DCBlock(SoftClip(Bass)) - Bass)   ← phase-shifted residual ≠ 0
    //
    // The corrected formulation isolates only the non-linear harmonic increment:
    //
    //   boost_delta = SoftClip(boosted) - bass_linear        ← zero at unity gain
    //   output      = Dry[n-31] + DCBlock(boost_delta)       ← exact identity at factor=0
    for (size_t f = 0; f < frames; ++f) {
        bass_factor_smoothed_ +=
            (bass_factor_ - bass_factor_smoothed_) * smoothing_coeff_;

        const float dry_l = dry_buf[f * 2u];
        const float dry_r = dry_buf[f * 2u + 1u];

        const float bass_linear_l = fir_buf[f * 2u];
        const float bass_linear_r = fir_buf[f * 2u + 1u];

        // Boosted bass: (1 + factor) so factor=0 → unity (bass_linear unchanged).
        float boosted_l = bass_linear_l * (1.0f + bass_factor_smoothed_);
        float boosted_r = bass_linear_r * (1.0f + bass_factor_smoothed_);
        ApplyAntiPop(boosted_l, boosted_r);

        // Soft-clip only the boosted bass; subtract the original linear component
        // to isolate just the saturation / harmonic-boost increment.
        const float shaped_l = BassSoftClip(boosted_l, 0.8f);
        const float shaped_r = BassSoftClip(boosted_r, 0.8f);

        const float boost_delta_l = shaped_l - bass_linear_l;
        const float boost_delta_r = shaped_r - bass_linear_r;

        // DC-block only the isolated non-linear delta so that any DC component
        // introduced by asymmetric saturation is removed without touching the
        // dry signal or the linear bass reconstruction path.
        const float dc_blocked_delta_l =
            dc_block_coeff_ * (dc_y1_[0] + boost_delta_l - dc_x1_[0]);
        dc_x1_[0] = boost_delta_l;
        dc_y1_[0] = dc_blocked_delta_l;

        const float dc_blocked_delta_r =
            dc_block_coeff_ * (dc_y1_[1] + boost_delta_r - dc_x1_[1]);
        dc_x1_[1] = boost_delta_r;
        dc_y1_[1] = dc_blocked_delta_r;

        // Perfect reconstruction: Dry[n-31] + harmonic boost only.
        // At factor=0: boost_delta≈0 → output≡Dry[n-31], zero phase error.
        audio[f, 0] = dry_l + dc_blocked_delta_l;
        audio[f, 1] = dry_r + dc_blocked_delta_r;
    }
}

void ViPERBass::ProcessSubwoofer(std::span<float> samples, StereoView audio) noexcept {
    if (anti_pop_ >= 1.0f) [[likely]] {
        subwoofer_.Process(samples);
        return;
    }

    // Crossfade dry→wet over anti_pop_ ramp to eliminate the onset transient.
    const uint32_t size         = static_cast<uint32_t>(audio.extent(0));
    const uint32_t sample_count = size * 2u;
    std::copy_n(samples.data(), sample_count, scratch_buffer_.data());
    subwoofer_.Process(scratch_buffer_.data(), size);

    const StereoView wet(scratch_buffer_.data(), size, 2u);
    for (size_t f = 0; f < size; ++f) {
        audio[f, 0] += anti_pop_ * (wet[f, 0] - audio[f, 0]);
        audio[f, 1] += anti_pop_ * (wet[f, 1] - audio[f, 1]);
        anti_pop_ = std::min(anti_pop_ + anti_pop_step_, 1.0f);
    }
}

void ViPERBass::Process(std::span<float> samples) noexcept {
    if (!enable_ || samples.empty()) return;
    const size_t size = samples.size();
    [[assume(size % 2 == 0)]];

    StereoView audio(samples.data(), size / 2, 2u);

    using enum ProcessMode;
    switch (process_mode_) {
        case NaturalBass:  ProcessNaturalBass (audio);          break;
        case PureBassPlus: ProcessPureBassPlus(audio);          break;
        case Subwoofer:    ProcessSubwoofer   (samples, audio); break;
        default:           return; // guard against corrupt IPC value
    }
}

void ViPERBass::Reset() noexcept {
    polyphase_.SetCutoffFrequency(static_cast<float>(frequency_));
    polyphase_.SetSamplingRate(sampling_rate_);
    polyphase_.Reset();
    subwoofer_.Reset(); // clear biquad delay-state before reconfiguring coefficients
    subwoofer_.SetBassGain(sampling_rate_, bass_factor_ * 2.5f);

    for (auto& ch : bass_delay_) {
        std::ranges::fill(ch, 0.0f);
    }
    delay_write_idx_ = 0u;

    const auto sr_f = static_cast<float>(sampling_rate_);
    for (auto& bq : biquad_) {
        bq.SetLowPassParameter(static_cast<float>(frequency_), sampling_rate_, 0.53f);
    }
    anti_pop_step_        = 1.0f / (0.020f * sr_f);
    anti_pop_             = 0.0f;
    smoothing_coeff_      = 1.0f - std::exp(-1.0f / (0.030f * sr_f));
    bass_factor_smoothed_ = bass_factor_;
    dc_block_coeff_       = std::exp(-2.0f * std::numbers::pi_v<float> * 18.0f / sr_f);
    dc_x1_.fill(0.0f);
    dc_y1_.fill(0.0f);
}

void ViPERBass::SetEnable(const bool enable) noexcept {
    if (enable_ != enable) {
        if (enable) Reset();
        enable_ = enable;
    }
}

void ViPERBass::SetProcessMode(const ProcessMode mode) noexcept {
    const auto safe_mode = static_cast<ProcessMode>(
        std::clamp(std::to_underlying(mode), uint8_t{0}, uint8_t{2}));
    if (process_mode_ != safe_mode) {
        process_mode_ = safe_mode;
        Reset();
    }
}

void ViPERBass::SetBassFactor(const float value) noexcept {
    if (bass_factor_ != value) {
        bass_factor_ = value;
        subwoofer_.SetBassGain(sampling_rate_, bass_factor_ * 2.5f);
    }
}

void ViPERBass::SetFrequency(const uint32_t value) noexcept {
    if (frequency_ != value) {
        frequency_ = value;
        // Update the linear-phase FIR crossover cutoff for PureBass+ mode.
        polyphase_.SetCutoffFrequency(static_cast<float>(value));
        // Keep biquad in sync for NaturalBass mode.
        for (auto& bq : biquad_) {
            bq.SetLowPassParameter(
                static_cast<float>(frequency_), sampling_rate_, 0.53f);
        }
    }
}

void ViPERBass::SetAntiPop(const bool enable) noexcept {
    anti_pop_ = enable ? 0.0f : 1.0f;
}

void ViPERBass::SetSamplingRate(const uint32_t sampling_rate) noexcept {
    if (sampling_rate_ != sampling_rate) {
        sampling_rate_ = sampling_rate;
        scratch_buffer_.resize(4096u * 4u);
        Reset();
    }
}

void ViPERBass::ProcessPlanar(float* __restrict L, float* __restrict R, const size_t frames) noexcept {
    if (!IsEnabled() || frames == 0) return;
    for (size_t i = 0; i < frames; ++i) {
        pp_scratch_[2u * i]      = L[i];
        pp_scratch_[2u * i + 1u] = R[i];
    }
    Process(std::span<float>{pp_scratch_.data(), frames * 2u});
    for (size_t i = 0; i < frames; ++i) {
        L[i] = pp_scratch_[2u * i];
        R[i] = pp_scratch_[2u * i + 1u];
    }
}
