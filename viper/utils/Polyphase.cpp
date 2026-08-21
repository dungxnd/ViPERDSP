#include "Polyphase.h"
#include <algorithm>
#include <cmath>
#include <numbers>

// ─── DesignLinearPhaseFilter ──────────────────────────────────────────────────
// Windowed-sinc (Blackman window) linear-phase lowpass FIR.
//
// Properties that make this correct for PureBass+ complementary crossover:
//
//   1. Symmetric Type-I FIR (odd length 63, centre at tap 31):
//      h[k] == h[62-k] for all k.  This gives exactly linear phase —
//      constant group delay of 31 samples at ALL frequencies, identical to
//      the ring-buffer dry-path delay.
//
//   2. Unity DC gain at any sample rate:
//      Coefficients are normalised so Σh[k] = 1.0 exactly.
//      At bass_factor = 0, High[n-31] + Bass[n-31] = Dry[n-31] — bit-perfect
//      pass-through with zero phase cancellation and zero volume loss.
//
//   3. Blackman window: ≥74 dB stopband rejection, −6 dB at f_c.
//
// Replacing the old static lookup tables (which had Σh ≈ 0.25, −12 dB DC gain,
// and were never re-designed when the sample rate changed) removes the primary
// causes of volume loss and comb-filter distortion in PureBass+ mode.
void Polyphase::DesignLinearPhaseFilter() noexcept {
    constexpr int   M   = static_cast<int>(kNumTaps - 1u); // 62
    constexpr float mid = static_cast<float>(M) / 2.0f;    // 31.0

    // Normalised cutoff: clamp well away from 0 and Nyquist.
    const float fc = std::clamp(cutoff_hz_ / static_cast<float>(sampling_rate_),
                                0.001f, 0.45f);

    float sum = 0.0f;
    for (int i = 0; i <= M; ++i) {
        const float n = static_cast<float>(i) - mid;

        // Windowed-sinc: sinc(2πfc·n)
        const float sinc = (n == 0.0f)
            ? 2.0f * std::numbers::pi_v<float> * fc
            : std::sin(2.0f * std::numbers::pi_v<float> * fc * n) / n;

        // Exact Blackman window coefficients (>74 dB stopband).
        constexpr float a0 = 0.42f;
        constexpr float a1 = 0.50f;
        constexpr float a2 = 0.08f;
        const float w = a0
            - a1 * std::cos(2.0f * std::numbers::pi_v<float>
                            * static_cast<float>(i) / static_cast<float>(M))
            + a2 * std::cos(4.0f * std::numbers::pi_v<float>
                            * static_cast<float>(i) / static_cast<float>(M));

        coeffs_[static_cast<uint32_t>(i)] = sinc * w;
        sum += coeffs_[static_cast<uint32_t>(i)];
    }

    // Normalise to exact unity DC gain (0 dB at 0 Hz) at any sample rate.
    const float inv_sum = 1.0f / sum;
    for (float& c : coeffs_) c *= inv_sum;
}

// ─── Constructor ─────────────────────────────────────────────────────────────
Polyphase::Polyphase(const int /*coeff_type*/) noexcept {
    DesignLinearPhaseFilter();
    Reset();
}

// ─── SetSamplingRate ──────────────────────────────────────────────────────────
// Redesigns coefficients (cutoff now recalculated relative to new Fs) and
// resets history so stale samples at the old rate do not corrupt the output.
void Polyphase::SetSamplingRate(const uint32_t sampling_rate) noexcept {
    if (sampling_rate_ != sampling_rate && sampling_rate > 0u) {
        sampling_rate_ = sampling_rate;
        DesignLinearPhaseFilter();
        Reset();
    }
}

// ─── SetCutoffFrequency ───────────────────────────────────────────────────────
// Redesigns coefficients for the new cutoff.  Does NOT reset history — the
// transition is sample-accurate and artifact-free for slow UI changes.
void Polyphase::SetCutoffFrequency(const float cutoff_hz) noexcept {
    if (std::abs(cutoff_hz_ - cutoff_hz) > 1.0f) {
        cutoff_hz_ = cutoff_hz;
        DesignLinearPhaseFilter();
    }
}

// ─── Reset ───────────────────────────────────────────────────────────────────
void Polyphase::Reset() noexcept {
    for (auto& ch : history_) ch.fill(0.0f);
    history_idx_ = 0u;
}

// ─── Process ─────────────────────────────────────────────────────────────────
// Direct-form FIR convolution with a double-buffered delay line.
//
// Classical circular-buffer FIR requires a masked index inside the inner loop:
//   idx = (history_idx_ + k) & mask
// This prevents auto-vectorisation because the loads are non-contiguous.
//
// Double-buffer technique: each incoming sample is written to two positions
// separated by kHistCap (64), so the k-loop always reads a contiguous slice
// of kNumTaps (63) floats starting at history_idx_.  Clang/GCC with -O2
// -ffast-math can then emit NEON vmlaq_f32 / AVX _mm256_fmadd_ps instructions
// across the entire 63-tap accumulation.
//
// Coefficient order: coeffs_[0] multiplies the NEWEST sample (x[n]),
// coeffs_[k] multiplies x[n-k].  Tap 31 is the centre (peak) of the
// symmetric Blackman-sinc kernel.
void Polyphase::Process(float* const samples, const uint32_t size) noexcept {
    if (!samples || size == 0u) return;

    const float* const __restrict c = coeffs_.data();

    for (uint32_t i = 0u; i < size; ++i) {
        // Power-of-two wrap: visits all 64 slots without skipping any index.
        history_idx_ = (history_idx_ - 1u) & kHistIdxMask;

        const float in_l = samples[i * 2u];
        const float in_r = samples[i * 2u + 1u];

        // Write to both the circular slot and its mirror kHistCap slots ahead.
        // This guarantees history_[ch][history_idx_ .. +kNumTaps-1] is always
        // a valid contiguous sequence of the last kNumTaps samples.
        history_[0][history_idx_]            = in_l;
        history_[0][history_idx_ + kHistCap] = in_l;
        history_[1][history_idx_]            = in_r;
        history_[1][history_idx_ + kHistCap] = in_r;

        // Contiguous dot-product — no masked index in inner loop.
        const float* const __restrict hist_l = &history_[0][history_idx_];
        const float* const __restrict hist_r = &history_[1][history_idx_];

        float out_l = 0.0f;
        float out_r = 0.0f;

#pragma clang loop vectorize(enable)
        for (uint32_t k = 0u; k < kNumTaps; ++k) {
            out_l += c[k] * hist_l[k];
            out_r += c[k] * hist_r[k];
        }

        samples[i * 2u]      = out_l;
        samples[i * 2u + 1u] = out_r;
    }
}
