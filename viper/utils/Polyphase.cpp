#include "Polyphase.h"
#include <ranges>

namespace {

constexpr std::array<float, 63> kPolyphaseCoefficients2 = {
    -0.002339f, -0.002073f, -0.001940f, -0.001675f, -0.001515f, -0.001329f, -0.001223f,
    -0.001037f, -0.000904f, -0.000851f, -0.000532f, -0.000851f, -0.000106f, -0.001010f,
     0.000558f, -0.001435f,  0.001302f, -0.001967f,  0.002259f, -0.002605f,  0.003216f,
    -0.003562f,  0.004784f, -0.005475f,  0.007655f, -0.008506f,  0.017622f, -0.024639f,
     0.028679f, -0.017303f, -0.032507f,  0.623321f,  0.184702f, -0.166867f,  0.025729f,
    -0.078490f, -0.015735f, -0.041199f, -0.023151f, -0.031524f, -0.020121f, -0.024985f,
    -0.017303f, -0.019616f, -0.015018f, -0.015204f, -0.012838f, -0.011881f, -0.010951f,
    -0.009516f, -0.009090f, -0.007788f, -0.007442f, -0.006353f, -0.006087f, -0.005183f,
    -0.004970f, -0.004253f, -0.003987f, -0.003482f, -0.003216f, -0.002871f, -0.002578f
};

constexpr std::array<float, 63> kPolyphaseCoefficientsOther = {
    -0.014194f, -0.002339f, -0.006220f, -0.019722f, -0.020626f, -0.014885f, -0.012240f,
    -0.012386f, -0.011801f, -0.011376f, -0.016293f, -0.018845f, -0.018327f, -0.013902f,
    -0.014951f, -0.015895f, -0.019044f, -0.017928f, -0.020094f, -0.017715f, -0.018845f,
    -0.015377f, -0.018354f, -0.016665f, -0.018951f, -0.011416f, -0.019469f, -0.017250f,
     0.003549f, -0.076045f,  0.288350f,  0.267751f, -0.041212f, -0.005130f, -0.088418f,
    -0.089348f, -0.087686f, -0.065625f, -0.041305f, -0.013343f,  0.001422f,  0.010313f,
     0.005834f, -0.001170f, -0.014499f, -0.021822f, -0.030792f, -0.029331f, -0.031071f,
    -0.018407f, -0.027271f, -0.008373f, -0.010791f, -0.040680f,  0.229171f,  0.080324f,
    -0.070955f,  0.021689f, -0.046607f, -0.025011f, -0.026886f, -0.027271f, -0.032919f
};

} // anonymous namespace

Polyphase::Polyphase(const int coeff_type) noexcept {
    coeffs_ = (coeff_type == 2) ? &kPolyphaseCoefficients2 : &kPolyphaseCoefficientsOther;
    Reset();
}

void Polyphase::Reset() noexcept {
    for (auto& ch : history_) {
        std::ranges::fill(ch, 0.0f);
    }
    history_idx_ = 0u;
}

// Direct-form FIR convolution with a circular delay line.
// Processes any block size ≥ 1; zero start-up latency mismatch.
// The output is delayed by kLatency (31) frames relative to the input,
// which equals the peak-tap index of the 63-tap symmetric impulse response.
void Polyphase::Process(float* const samples, const uint32_t size) noexcept {
    if (!samples || size == 0u) return;

    const auto& c = *coeffs_;

    for (uint32_t i = 0u; i < size; ++i) {
        // Step head backward so index 0 is always "newest" sample.
        history_idx_ = (history_idx_ - 1u) & kHistoryMask;

        history_[0][history_idx_] = samples[i * 2u];
        history_[1][history_idx_] = samples[i * 2u + 1u];

        float out_l = 0.0f;
        float out_r = 0.0f;

        for (uint32_t k = 0u; k < kNumTaps; ++k) {
            const uint32_t idx = (history_idx_ + k) & kHistoryMask;
            out_l += c[k] * history_[0][idx];
            out_r += c[k] * history_[1][idx];
        }

        samples[i * 2u]      = out_l;
        samples[i * 2u + 1u] = out_r;
    }
}
