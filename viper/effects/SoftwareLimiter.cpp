#include "SoftwareLimiter.h"
#include <cmath>
#include <algorithm>

static constexpr uint32_t kLookahead    = 256;
static constexpr float    kReleaseTauSec = 0.080f;  // 80 ms release time constant
static constexpr float    kDenormFix    = 1e-25f;

SoftwareLimiter::SoftwareLimiter() {
    // SetSamplingRate initialises release_coeff_ from the default 48 kHz rate,
    // then Reset() clears all runtime state.
    SetSamplingRate(sampling_rate_);
    Reset();
}

float SoftwareLimiter::Process(float sample) noexcept {
    if (!std::isfinite(sample)) sample = 0.0f;

    const uint32_t wi = write_index_;

    const float delayed     = arr256_[wi];
    const float window_peak = arr512_[1];

    const float target_gain = window_peak > gate_ ? gate_ / window_peak : 1.0f;
    const float released    =
        gain_envelope_ + release_coeff_ * (1.0f - gain_envelope_) + kDenormFix;
    float gain = std::min(target_gain, released);
    if (gain > 1.0f) gain = 1.0f;
    gain_envelope_ = gain;

    uint32_t node = kLookahead + wi;
    arr512_[node] = std::fabs(sample);
    while (node > 1) {
        const uint32_t parent  = node >> 1;
        const uint32_t sibling = node ^ 1;
        arr512_[parent] = std::max(arr512_[node], arr512_[sibling]);
        node = parent;
    }
    arr256_[wi] = sample;
    write_index_ = (wi + 1) & (kLookahead - 1);

    return delayed * gain;
}

void SoftwareLimiter::Reset() noexcept {
    arr256_.fill(0.0f);
    arr512_.fill(0.0f);
    ready_         = false;
    write_index_   = 0;
    gain_envelope_ = 1.0f;
    smoothed_gain_ = 1.0f;
    target_gain_   = 1.0f;
}

void SoftwareLimiter::SetGate(const float gate) noexcept {
    gate_ = gate;
}

void SoftwareLimiter::SetSamplingRate(const uint32_t sampling_rate) noexcept {
    // Guard against divide-by-zero for pathological rates; clamp to 1 Hz minimum.
    const uint32_t fs = sampling_rate > 0u ? sampling_rate : 1u;
    sampling_rate_ = fs;
    // One-pole coefficient for a first-order IIR with time constant τ:
    //   c = 1 − exp(−1 / (τ · fs))
    // At 44.1 kHz: c ≈ 2.837e-4; at 48 kHz: c ≈ 2.604e-4.
    release_coeff_ = 1.0f - std::exp(-1.0f / (kReleaseTauSec * static_cast<float>(fs)));
}
