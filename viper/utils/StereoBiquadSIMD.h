#pragma once
// StereoBiquadSIMD.h — Single-precision Transposed Direct Form II biquad bank.
//
// Provides dual-channel (L+R) bulk processing on planar buffers.
//
// SIMD strategy — "vectorize across channels":
//   A TDF-II IIR loop has a loop-carried dependency on its state registers
//   (s1, s2), so auto-vectorization across time is impossible.  Instead,
//   Process() packs L and R into a float32x2_t (lane 0 = L, lane 1 = R) and
//   runs a SINGLE loop that computes both channels in parallel per sample via
//   ARM NEON.  This is the standard production-DSP technique (used in JUCE,
//   Apple vDSP, etc.) and gives the full 2× throughput gain on arm64.
//   A portable scalar fallback is compiled for non-NEON targets (x86, etc.).
//
// TDF-II recurrence (no feedback from previous output, maximally immune to
// coefficient-quantisation limit cycles):
//   y[n]  = b0 * x[n] + s1
//   s1    = s2 + b1*x[n] - a1*y[n]
//   s2    = b2*x[n] - a2*y[n]
//
// Thread-safety: coefficients are read-only during Process(); state is
// per-channel and must not be shared across threads.

#include <cmath>
#include <cstddef>
#include <numbers>
#if defined(__ARM_NEON)
#  include <arm_neon.h>
#endif

namespace viper::dsp {

struct StereoBiquadCoeffs {
    float b0{1.0f}, b1{0.0f}, b2{0.0f};
    float        a1{0.0f}, a2{0.0f};  // a0-normalised; sign convention: y = b0*x + b1*x1 + b2*x2 - a1*y1 - a2*y2

    // ── Factory helpers ────────────────────────────────────────────────────

    [[nodiscard]] static StereoBiquadCoeffs MakeIdentity() noexcept {
        return {};  // b0=1, rest 0 → pure pass-through
    }

    [[nodiscard]] static StereoBiquadCoeffs MakeLowPass(float freq_hz, float q, float fs) noexcept {
        const float w0    = 2.0f * std::numbers::pi_v<float> * freq_hz / fs;
        const float cos_w = std::cos(w0);
        const float sin_w = std::sin(w0);
        const float alpha = sin_w / (2.0f * q);
        const float a0    = 1.0f + alpha;
        return {
            .b0 = ((1.0f - cos_w) * 0.5f) / a0,
            .b1 = (1.0f - cos_w) / a0,
            .b2 = ((1.0f - cos_w) * 0.5f) / a0,
            .a1 = (-2.0f * cos_w) / a0,
            .a2 = (1.0f - alpha) / a0,
        };
    }

    [[nodiscard]] static StereoBiquadCoeffs MakeHighPass(float freq_hz, float q, float fs) noexcept {
        const float w0    = 2.0f * std::numbers::pi_v<float> * freq_hz / fs;
        const float cos_w = std::cos(w0);
        const float sin_w = std::sin(w0);
        const float alpha = sin_w / (2.0f * q);
        const float a0    = 1.0f + alpha;
        return {
            .b0 =  ((1.0f + cos_w) * 0.5f) / a0,
            .b1 = -(1.0f + cos_w) / a0,
            .b2 =  ((1.0f + cos_w) * 0.5f) / a0,
            .a1 = (-2.0f * cos_w) / a0,
            .a2 = (1.0f - alpha) / a0,
        };
    }

    [[nodiscard]] static StereoBiquadCoeffs MakeBandPass(float freq_hz, float q, float fs) noexcept {
        const float w0    = 2.0f * std::numbers::pi_v<float> * freq_hz / fs;
        const float sin_w = std::sin(w0);
        const float cos_w = std::cos(w0);
        const float alpha = sin_w / (2.0f * q);
        const float a0    = 1.0f + alpha;
        // Constant skirt gain, peak gain = Q
        return {
            .b0 =  (sin_w * 0.5f) / a0,
            .b1 =  0.0f,
            .b2 = -(sin_w * 0.5f) / a0,
            .a1 = (-2.0f * cos_w) / a0,
            .a2 = (1.0f - alpha) / a0,
        };
    }

    [[nodiscard]] static StereoBiquadCoeffs MakePeakEQ(float freq_hz, float q, float db_gain, float fs) noexcept {
        const float A     = std::pow(10.0f, db_gain / 40.0f);
        const float w0    = 2.0f * std::numbers::pi_v<float> * freq_hz / fs;
        const float sin_w = std::sin(w0);
        const float cos_w = std::cos(w0);
        const float alpha = sin_w / (2.0f * q);
        const float a0    = 1.0f + alpha / A;
        return {
            .b0 = (1.0f + alpha * A) / a0,
            .b1 = (-2.0f * cos_w) / a0,
            .b2 = (1.0f - alpha * A) / a0,
            .a1 = (-2.0f * cos_w) / a0,
            .a2 = (1.0f - alpha / A) / a0,
        };
    }

    // ── AllPass (2nd-order, RBJ cookbook) ─────────────────────────────────
    // Standard 2nd-order allpass: |H(e^jω)| = 1 at all frequencies.
    // Coefficients satisfy the reversed-denominator condition b[k] = a[N-k]:
    //   b0 = a2 = (1-α)/a0,  b1 = a1 = -2cos/a0,  b2 = 1  (a0-normalised).
    //
    // Used for phase-synchronisation in LinkwitzRileyCrossover:
    //   LP_LR4(z) + HP_LR4(z) = AP_2nd(z)  (Sophie Germain identity on numerator)
    // The AP_2nd is exactly ONE biquad stage — NOT two cascaded stages.
    //
    // NOTE: b2 = 1.0f (not divided by a0).  All other factory methods
    //       divide b2 by a0; this method intentionally does not, because
    //       the a0-normalised allpass form requires b2 = (1+α)/a0 = 1.
    [[nodiscard]] static StereoBiquadCoeffs MakeAllPass(float freq_hz, float q, float fs) noexcept {
        const float w0    = 2.0f * std::numbers::pi_v<float> * freq_hz / fs;
        const float cos_w = std::cos(w0);
        const float sin_w = std::sin(w0);
        const float alpha = sin_w / (2.0f * q);
        const float a0    = 1.0f + alpha;
        return {
            .b0 = (1.0f - alpha) / a0,
            .b1 = (-2.0f * cos_w) / a0,
            .b2 =  1.0f,            // (1+alpha)/a0 = 1 after a0-normalization
            .a1 = (-2.0f * cos_w) / a0,
            .a2 = (1.0f - alpha) / a0,
        };
    }
};

// Per-channel TDF-II state (two delay registers per filter).
struct StereoBiquadState {
    float s1_l{0.0f}, s2_l{0.0f};
    float s1_r{0.0f}, s2_r{0.0f};

    void Reset() noexcept { s1_l = s2_l = s1_r = s2_r = 0.0f; }
};

// Stateless bulk processor — wraps TDF-II recurrence over planar L[]/R[].
struct StereoBiquadBank {
    // Process `frames` samples in-place on L and R using TDF-II.
    // Both channels share the same coefficients (same filter per channel),
    // but have independent state registers.
    //
    // On arm64 (NEON): packs L[i] and R[i] into float32x2_t and runs a single
    // loop — both channels computed in parallel per sample.  This is true SIMD
    // (across-channel vectorization) and avoids any auto-vectorization request.
    static void Process(
        const StereoBiquadCoeffs& c,
        StereoBiquadState&        s,
        float* __restrict         L,
        float* __restrict         R,
        size_t                    frames
    ) noexcept {
#if defined(__ARM_NEON)
        // ── NEON path: lane 0 = L, lane 1 = R ──────────────────────────────
        // Pack state and coefficients into 2-lane vectors.
        float32x2_t vs1 = {s.s1_l, s.s1_r};
        float32x2_t vs2 = {s.s2_l, s.s2_r};
        const float32x2_t vb0 = vdup_n_f32(c.b0);
        const float32x2_t vb1 = vdup_n_f32(c.b1);
        const float32x2_t vb2 = vdup_n_f32(c.b2);
        const float32x2_t va1 = vdup_n_f32(c.a1);
        const float32x2_t va2 = vdup_n_f32(c.a2);

        for (size_t i = 0; i < frames; ++i) {
            // x = {L[i], R[i]}
            const float32x2_t vx = {L[i], R[i]};
            // y  = b0*x + s1
            const float32x2_t vy = vmla_f32(vs1, vb0, vx);
            // s1 = b1*x - a1*y + s2
            vs1 = vadd_f32(vmls_f32(vmul_f32(vb1, vx), va1, vy), vs2);
            // s2 = b2*x - a2*y
            vs2 = vmls_f32(vmul_f32(vb2, vx), va2, vy);
            L[i] = vget_lane_f32(vy, 0);
            R[i] = vget_lane_f32(vy, 1);
        }

        s.s1_l = vget_lane_f32(vs1, 0); s.s1_r = vget_lane_f32(vs1, 1);
        s.s2_l = vget_lane_f32(vs2, 0); s.s2_r = vget_lane_f32(vs2, 1);
#else
        // ── Scalar fallback (x86, etc.) ─────────────────────────────────────
        float s1_l = s.s1_l, s2_l = s.s2_l;
        float s1_r = s.s1_r, s2_r = s.s2_r;
        const float b0 = c.b0, b1 = c.b1, b2 = c.b2;
        const float a1 = c.a1, a2 = c.a2;

        for (size_t i = 0; i < frames; ++i) {
            const float xl = L[i];
            const float yl = b0 * xl + s1_l;
            s1_l = b1 * xl - a1 * yl + s2_l;
            s2_l = b2 * xl - a2 * yl;
            L[i] = yl;
        }
        for (size_t i = 0; i < frames; ++i) {
            const float xr = R[i];
            const float yr = b0 * xr + s1_r;
            s1_r = b1 * xr - a1 * yr + s2_r;
            s2_r = b2 * xr - a2 * yr;
            R[i] = yr;
        }

        s.s1_l = s1_l; s.s2_l = s2_l;
        s.s1_r = s1_r; s.s2_r = s2_r;
#endif
    }

    // Variant that processes a single channel in-place.
    // NEON does not help here (one channel = one scalar stream with carried
    // dependency), so this stays scalar on all targets.
    static void ProcessChannel(
        const StereoBiquadCoeffs& c,
        float&                    s1, float& s2,
        float* __restrict         buf,
        size_t                    frames
    ) noexcept {
        const float b0 = c.b0, b1 = c.b1, b2 = c.b2;
        const float a1 = c.a1, a2 = c.a2;
        float _s1 = s1, _s2 = s2;
        for (size_t i = 0; i < frames; ++i) {
            const float x = buf[i];
            const float y = b0 * x + _s1;
            _s1 = b1 * x - a1 * y + _s2;
            _s2 = b2 * x - a2 * y;
            buf[i] = y;
        }
        s1 = _s1; s2 = _s2;
    }
};

} // namespace viper::dsp
