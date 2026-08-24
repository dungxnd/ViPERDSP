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

void ViPERBassMono::ProcessNaturalBass(StereoView audio) noexcept {
    for (size_t f = 0; f < audio.extent(0); ++f) {
        bass_factor_smoothed_ +=
            (bass_factor_ - bass_factor_smoothed_) * smoothing_coeff_;
        const double sample = (static_cast<double>(audio[f, 0])
                               + static_cast<double>(audio[f, 1])) * 0.5;
        float bass = static_cast<float>(biquad_.ProcessSample(sample))
                     * bass_factor_smoothed_;
        if (anti_pop_ < 1.0f) {
            bass      *= anti_pop_;
            anti_pop_  = std::min(anti_pop_ + anti_pop_step_, 1.0f);
        }
        ShapeMix(bass, audio[f, 0], audio[f, 1]);
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
void ViPERBassMono::ProcessPureBassPlus(StereoView audio) noexcept {
    const size_t frames = audio.extent(0);

    // scratch_buffer_ layout:
    //   [0  .. 2f-1]: fir_buf — mono-mix input/output for the linear-phase FIR
    //   [2f .. 4f-1]: dry_buf — clean dry stereo delayed by kLatency frames
    // Total: 4 * frames.  Pre-sized to 4096 * 4 in ctor / SetSamplingRate().
    float* const fir_buf = scratch_buffer_.data();                 // [0  .. 2f-1]
    float* const dry_buf = scratch_buffer_.data() + frames * 2u;  // [2f .. 4f-1]

    // ── Step 1: delay raw stereo dry by kLatency, feed mono mix to FIR ───────
    for (size_t f = 0; f < frames; ++f) {
        // Store raw stereo dry input interleaved in ring delay; read back kLatency ago.
        bass_delay_[delay_write_idx_ * 2u]      = audio[f, 0];
        bass_delay_[delay_write_idx_ * 2u + 1u] = audio[f, 1];

        const size_t read_idx = (delay_write_idx_ - Polyphase::kLatency) & kDelayMask;
        delay_write_idx_ = (delay_write_idx_ + 1u) & kDelayMask;

        dry_buf[f * 2u]      = bass_delay_[read_idx * 2u];
        dry_buf[f * 2u + 1u] = bass_delay_[read_idx * 2u + 1u];

        // Mono-mix raw input (no IIR pre-filter) and duplicate to both FIR channels.
        // The linear-phase FIR is the sole frequency-selective element.
        const float mono = (audio[f, 0] + audio[f, 1]) * 0.5f;
        fir_buf[f * 2u]      = mono;
        fir_buf[f * 2u + 1u] = mono;
    }

    // ── Step 2: linear-phase FIR extracts Bass[n-31] (group delay = 31) ──────
    polyphase_.Process(fir_buf, static_cast<uint32_t>(frames));

    // ── Step 3: transparent saturation blend — DC-block only the boost delta ─
    for (size_t f = 0; f < frames; ++f) {
        bass_factor_smoothed_ +=
            (bass_factor_ - bass_factor_smoothed_) * smoothing_coeff_;

        const float dry_l = dry_buf[f * 2u];
        const float dry_r = dry_buf[f * 2u + 1u];

        // Mono bass from FIR (L == R since input was duplicated mono mid-mix).
        const float bass_linear = fir_buf[f * 2u];

        // Boosted mono bass: (1 + factor) so factor=0 → unity (bass_linear unchanged).
        float boosted = bass_linear * (1.0f + bass_factor_smoothed_);
        if (anti_pop_ < 1.0f) {
            boosted   *= anti_pop_;
            anti_pop_  = std::min(anti_pop_ + anti_pop_step_, 1.0f);
        }

        // Isolate the non-linear saturation increment only.
        // At factor=0 and sub-bass levels: SoftClip(x) ≈ x → boost_delta ≈ 0.
        const float shaped      = BassSoftClip(boosted, 0.8f);
        const float boost_delta = shaped - bass_linear;

        // DC-block only the non-linear delta; the dry and bass_linear paths
        // recombine without any phase-shifted component in the direct signal path.
        const float dc_blocked_delta =
            dc_block_coeff_ * (dc_y1_ + boost_delta - dc_x1_);
        dc_x1_ = boost_delta;
        dc_y1_ = dc_blocked_delta;

        // Perfect reconstruction: Dry[n-31] + harmonic boost only.
        // At factor=0: boost_delta≈0 → output≡Dry[n-31], zero phase error.
        audio[f, 0] = dry_l + dc_blocked_delta;
        audio[f, 1] = dry_r + dc_blocked_delta;
    }
}

void ViPERBassMono::ProcessSubwoofer(std::span<float> samples, StereoView audio) noexcept {
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

void ViPERBassMono::Process(std::span<float> samples) noexcept {
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

void ViPERBassMono::SetEnable(const bool enable) noexcept {
    if (enable_ != enable) {
        if (enable) Reset();
        enable_ = enable;
    }
}

void ViPERBassMono::SetProcessMode(const ProcessMode mode) noexcept {
    const auto safe_mode = static_cast<ProcessMode>(
        std::clamp(std::to_underlying(mode), uint8_t{0}, uint8_t{2}));
    if (process_mode_ != safe_mode) {
        process_mode_ = safe_mode;
        Reset();
    }
}

void ViPERBassMono::SetBassFactor(const float value) noexcept {
    if (bass_factor_ != value) {
        bass_factor_ = value;
        subwoofer_.SetBassGain(sampling_rate_, bass_factor_ * 2.5f);
    }
}

void ViPERBassMono::SetFrequency(const uint32_t value) noexcept {
    if (frequency_ != value) {
        frequency_ = value;
        // Update the linear-phase FIR crossover cutoff for PureBass+ mode.
        polyphase_.SetCutoffFrequency(static_cast<float>(value));
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
        scratch_buffer_.resize(4096u * 4u);
        Reset();
    }
}

void ViPERBassMono::ProcessPlanar(float* __restrict L, float* __restrict R, const size_t frames) noexcept {
    if (!IsEnabled() || frames == 0) return;
    float* const sc = planar_scratch_.data();
    for (size_t i = 0; i < frames; ++i) {
        sc[2u * i]      = L[i];
        sc[2u * i + 1u] = R[i];
    }
    Process(std::span<float>{sc, frames * 2u});
    for (size_t i = 0; i < frames; ++i) {
        L[i] = sc[2u * i];
        R[i] = sc[2u * i + 1u];
    }
}
