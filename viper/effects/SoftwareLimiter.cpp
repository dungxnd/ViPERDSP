#include "SoftwareLimiter.h"
#include <cmath>
#include <algorithm>

static constexpr uint32_t kLookahead = 256;
static constexpr float kReleaseTauSec = 0.080f;
static constexpr float kDenormFix = 1e-25f;
static const float kReleaseCoeff =
    1.0f - std::exp(-1.0f / (kReleaseTauSec * 44100.0f));

SoftwareLimiter::SoftwareLimiter() {
    // all members have in-class defaults; Reset() clears the arrays
    Reset();
}

float SoftwareLimiter::Process(float sample) noexcept {
    if (!std::isfinite(sample)) sample = 0.0f;

    const uint32_t wi = write_index_;

    const float delayed     = arr256_[wi];
    const float window_peak = arr512_[1];

    const float target_gain = window_peak > gate_ ? gate_ / window_peak : 1.0f;
    const float released    =
        gain_envelope_ + kReleaseCoeff * (1.0f - gain_envelope_) + kDenormFix;
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
