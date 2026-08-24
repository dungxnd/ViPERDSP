#pragma once
// Vectorised Transposed Direct Form II (TDF-II) stereo biquad.
//
// Processes L and R channels simultaneously in a single loop using pure FP32
// arithmetic.  The loop body is dependency-chain-free per channel, letting
// the compiler schedule NEON/SVE instructions efficiently.
//
// Drop-in for any 2-channel biquad stage where a ProcessPlanar() interface
// is preferred over the legacy per-sample interleaved path.
//
// Example:
//   StereoBiquadCoeffs  c = MakePeakEQ(fc, Q, gain_db, sr);
//   StereoBiquadBank::State st{};
//   StereoBiquadBank::ProcessPlanar(c, st, left_buf, right_buf, frames);

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>

namespace viper::dsp {

// Biquad coefficients (normalised: a0 = 1).
// Store b0/b1/b2 (feed-forward) and a1/a2 (feedback, sign-convention: y[n] =
// b0·x[n] + b1·x[n-1] + b2·x[n-2] − a1·y[n-1] − a2·y[n-2]).
struct alignas(16) StereoBiquadCoeffs {
    float b0{1.0f}, b1{0.0f}, b2{0.0f};
    float a1{0.0f}, a2{0.0f};
};

class StereoBiquadBank {
public:
    // Per-channel TDF-II delay state (s1, s2 per channel).
    struct alignas(16) State {
        float s1_l{0.0f}, s2_l{0.0f};
        float s1_r{0.0f}, s2_r{0.0f};
    };

    // Process `frames` samples of L/R planar buffers in-place.
    // All pointers must be valid and non-null.
    // __restrict__ hints allow the compiler to elide aliasing checks and emit
    // wider SIMD loads/stores.
    static void ProcessPlanar(
        const StereoBiquadCoeffs& __restrict c,
        State&                   __restrict st,
        float* __restrict                   left,
        float* __restrict                   right,
        size_t                              frames) noexcept {

        // Hoist state to locals — avoids repeated struct-member indirection
        // and lets the register allocator keep them in FP registers.
        float s1_l = st.s1_l, s2_l = st.s2_l;
        float s1_r = st.s1_r, s2_r = st.s2_r;

        // TDF-II recurrence (per channel, independent → dual-issue on NEON):
        //   out[n] = b0·in[n] + s1[n-1]
        //   s1[n]  = b1·in[n] − a1·out[n] + s2[n-1]
        //   s2[n]  = b2·in[n] − a2·out[n]
#pragma clang loop vectorize(enable)
        for (size_t i = 0; i < frames; ++i) {
            const float in_l  = left[i];
            const float out_l = c.b0 * in_l + s1_l;
            s1_l              = c.b1 * in_l - c.a1 * out_l + s2_l;
            s2_l              = c.b2 * in_l - c.a2 * out_l;
            left[i]           = out_l;

            const float in_r  = right[i];
            const float out_r = c.b0 * in_r + s1_r;
            s1_r              = c.b1 * in_r - c.a1 * out_r + s2_r;
            s2_r              = c.b2 * in_r - c.a2 * out_r;
            right[i]          = out_r;
        }

        st.s1_l = s1_l; st.s2_l = s2_l;
        st.s1_r = s1_r; st.s2_r = s2_r;
    }

    // ── Coefficient factories ────────────────────────────────────────────

    // Peak EQ: centre `fc` Hz, bandwidth `Q`, `gain_db` dB boost/cut.
    [[nodiscard]] static StereoBiquadCoeffs MakePeakEQ(
        float fc, float Q, float gain_db, float sr) noexcept {
        const float A  = std::pow(10.0f, gain_db / 40.0f);
        const float w0 = 2.0f * std::numbers::pi_v<float> * fc / sr;
        const float cw = std::cos(w0);
        const float sw = std::sin(w0);
        const float al = sw / (2.0f * Q);
        const float b0 = 1.0f + al * A;
        const float b1 = -2.0f * cw;
        const float b2 = 1.0f - al * A;
        const float a0 = 1.0f + al / A;
        const float a1 = -2.0f * cw;
        const float a2 = 1.0f - al / A;
        return {b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0};
    }

    // Identity (pass-through) coefficients — useful for disabled slots.
    [[nodiscard]] static StereoBiquadCoeffs MakeIdentity() noexcept {
        return {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }
};

} // namespace viper::dsp
