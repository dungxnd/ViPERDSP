#include "ViPERBassMono.h"
#include <algorithm>
#include <cmath>
#include <numbers>
#include <utility>

ViPERBassMono::ViPERBassMono() {
    biquad_.Reset();
    biquad_.SetLowPassParameter(static_cast<float>(frequency_), sampling_rate_, 0.53f);
    subwoofer_.SetBassGain(sampling_rate_, 0.0f);
    Reset();
}

// DC-block + soft-clip shaping: applies the mono bass signal to both L and R channels.
void ViPERBassMono::ShapeMix(const float bass, const uint32_t i,
                              std::span<float> samples) noexcept {
    const float y = dc_block_coeff_ * (dc_y1_ + bass - dc_x1_);
    dc_x1_ = bass;
    dc_y1_ = y;
    const float clipped = BassSoftClip(y, 0.8f);
    samples[i]     = BassSoftClip(samples[i]     + clipped, 0.95f);
    samples[i + 1] = BassSoftClip(samples[i + 1] + clipped, 0.95f);
}

// ── ProcessNaturalBass ──────────────────────────────────────────────────────
// Bug fix: anti-pop now ramps ONLY the wet (bass) signal, not the entire dry
// stream. Previously the top of Process() would silently fade in the whole
// audio when anti_pop_ < 1.0, causing a 1-second mute on effect enable.
void ViPERBassMono::ProcessNaturalBass(std::span<float> samples) noexcept {
    const uint32_t size = static_cast<uint32_t>(samples.size() / 2);
    [[assume(samples.size() % 2 == 0)]];
    for (uint32_t i = 0; i < size * 2; i += 2) {
        bass_factor_smoothed_ +=
            (bass_factor_ - bass_factor_smoothed_) * smoothing_coeff_;
        const double sample = (static_cast<double>(samples[i])
                               + static_cast<double>(samples[i + 1])) * 0.5;
        float bass = static_cast<float>(biquad_.ProcessSample(sample))
                     * bass_factor_smoothed_;
        // Ramp only the wet bass component — dry signal is unaffected.
        if (anti_pop_ < 1.0f) {
            bass      *= anti_pop_;
            anti_pop_  = std::min(anti_pop_ + anti_pop_step_, 1.0f);
        }
        ShapeMix(bass, i, samples);
    }
}

// ── ProcessPureBassPlus ─────────────────────────────────────────────────────
// Bug fixes:
// 1. wave_buffer_ is now channels=2 (was 1), so PushSamples correctly copies
//    all size*2 interleaved samples — previously only the first `size` floats
//    were copied (half the stereo data), corrupting the delay line.
// 2. PopSamples is called UNCONDITIONALLY (after the polyphase check) to keep
//    the delay line in sync even when polyphase hasn't produced output yet.
//    Previously, failure to pop when polyphase returned 0 caused wave_buffer_
//    to accumulate unboundedly, breaking the dry/wet alignment.
// 3. Biquad write offset: now uses * 2u (stereo stride) to match wave_buffer_
//    channels=2 layout. Index into bass delay is i (stereo) → channel 0 only
//    is the mono-mixed biquad signal; we write to even indices.
void ViPERBassMono::ProcessPureBassPlus(std::span<float> samples) noexcept {
    const uint32_t size = static_cast<uint32_t>(samples.size() / 2);

    if (!wave_buffer_.PushSamples(samples.data(), size)) return;

    // Write mono-mixed biquad output into channel-0 (even) slots of the delay buffer.
    // wave_buffer_ channels=2, so each frame occupies 2 floats. We use slot [f*2+0]
    // for the biquad result; slot [f*2+1] is unused (zeroed from PushSamples).
    float* const buffer         = wave_buffer_.GetBuffer();
    const uint32_t write_offset = (wave_buffer_.GetBufferOffset() - size) * 2u;

    for (uint32_t i = 0; i < size * 2; i += 2) {
        const double sample = (static_cast<double>(samples[i])
                               + static_cast<double>(samples[i + 1])) * 0.5;
        buffer[write_offset + i] = static_cast<float>(biquad_.ProcessSample(sample));
        // channel-1 slot intentionally left as-is (dry data from PushSamples,
        // not used during bass read-back below).
    }

    // Polyphase processes the full stereo signal for timing/pitch analysis.
    // Only mix bass when it has produced a full block.
    const bool polyphase_ready = (polyphase_.Process(samples.data(), size) == size);

    if (polyphase_ready) {
        for (uint32_t i = 0; i < size * 2; i += 2) {
            bass_factor_smoothed_ +=
                (bass_factor_ - bass_factor_smoothed_) * smoothing_coeff_;
            float bass = buffer[i] * bass_factor_smoothed_; // channel-0 = biquad output
            if (anti_pop_ < 1.0f) {
                bass      *= anti_pop_;
                anti_pop_  = std::min(anti_pop_ + anti_pop_step_, 1.0f);
            }
            ShapeMix(bass, i, samples);
        }
    }
    // Always pop to keep wave_buffer_ delay line in sync with the input stream.
    wave_buffer_.PopSamples(size, true);
}

// ── ProcessSubwoofer ────────────────────────────────────────────────────────
// Bug fix: replaced RT heap allocation (std::vector wet{...}) with a
// pre-allocated scratch_buffer_ sized in SetSamplingRate().
void ViPERBassMono::ProcessSubwoofer(std::span<float> samples) noexcept {
    const uint32_t size = static_cast<uint32_t>(samples.size() / 2);

    if (anti_pop_ >= 1.0f) [[likely]] {
        subwoofer_.Process(samples.data(), size);
        return;
    }

    const uint32_t sample_count = size * 2u;
    std::copy_n(samples.data(), sample_count, scratch_buffer_.data());
    subwoofer_.Process(scratch_buffer_.data(), size);

    for (uint32_t i = 0; i < sample_count; i += 2) {
        samples[i]     += anti_pop_ * (scratch_buffer_[i]     - samples[i]);
        samples[i + 1] += anti_pop_ * (scratch_buffer_[i + 1] - samples[i + 1]);
        anti_pop_ = std::min(anti_pop_ + anti_pop_step_, 1.0f);
    }
}

void ViPERBassMono::Process(std::span<float> samples) noexcept {
    if (!enable_ || samples.empty()) return;
    [[assume(samples.size() % 2 == 0)]];

    using enum ProcessMode;
    switch (process_mode_) {
        case NaturalBass:  ProcessNaturalBass (samples); break;
        case PureBassPlus: ProcessPureBassPlus(samples); break;
        case Subwoofer:    ProcessSubwoofer   (samples); break;
        default:           std::unreachable();
    }
}

void ViPERBassMono::Reset() noexcept {
    polyphase_.SetSamplingRate(sampling_rate_);
    polyphase_.Reset();
    wave_buffer_.Reset();
    wave_buffer_.PushZeros(polyphase_.GetLatency());
    subwoofer_.SetBassGain(sampling_rate_, bass_factor_ * 2.5f);
    biquad_.SetLowPassParameter(static_cast<float>(frequency_), sampling_rate_, 0.53f);

    const auto sr_f       = static_cast<float>(sampling_rate_);
    // 20 ms anti-pop ramp (down from 1000 ms).
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
    if (process_mode_ != mode) {
        process_mode_ = mode;
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
        // Pre-size the scratch buffer to avoid RT-thread allocation.
        scratch_buffer_.resize(4096u * 2u);
        // Reset() recalculates all SR-dependent state.
        Reset();
    }
}
