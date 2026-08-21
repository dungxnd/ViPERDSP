#include "LUFSTargeting.h"
#include "../utils/FastAudioMath.h"
#include <algorithm>
#include <cmath>

// ---------------------------------------------------------------------------
// K-weighting TDF-II coefficient tables (ITU-R BS.1770), a0-normalized.
// ---------------------------------------------------------------------------
namespace {

struct BiquadCoeffs {
    float b0;
    float b1;
    float b2;
    float a1;
    float a2;
};
struct KWeightCoeffs {
    BiquadCoeffs stage1;
    BiquadCoeffs stage2;
};

constexpr KWeightCoeffs kKWeight48k = {
    { 1.5351248f, -2.6916962f, 1.1983928f, -1.6906593f, 0.7324808f },
    { 1.0f,       -2.0f,       1.0f,       -1.9900475f, 0.9900723f }
};

constexpr KWeightCoeffs kKWeight44k1 = {
    { 1.5308412f, -2.6509800f, 1.1690791f, -1.6636551f, 0.7125954f },
    { 1.0f,       -2.0f,       1.0f,       -1.9891697f, 0.9891990f }
};

struct SpeedParams { float attack_ms; float release_ms; };
constexpr std::array<SpeedParams, 3> kSpeedTable = {{
    { 200.0f, 1000.0f },  // 0 = slow
    { 100.0f,  500.0f },  // 1 = medium (default)
    {  50.0f,  200.0f },  // 2 = fast
}};

} // anonymous namespace

// ---------------------------------------------------------------------------

LUFSTargeting::LUFSTargeting() {
    ConfigureFilters();
    window_size_ = static_cast<uint32_t>(sampling_rate_ * 0.4f);
    step_size_   = window_size_ / 4u;
}

// ---------------------------------------------------------------------------
// UpdateWindow — fires at ~100 ms step boundary.
// Maintains running_power_sum_ in O(1): subtract evicted slot, add new slot.
// MeasureLUFS() then requires zero iteration — single division + FastLog2.
// ---------------------------------------------------------------------------
void LUFSTargeting::UpdateWindow() noexcept {
    if (window_sample_count_ < window_size_) return;

    if (const float mean_square = window_accumulator_ / static_cast<float>(window_sample_count_);
        mean_square > kGateThreshold) {
        // O(1) running sum: remove the value about to be overwritten, add new
        running_power_sum_ -= window_power_[window_write_idx_];
        window_power_[window_write_idx_] = mean_square;
        running_power_sum_ += mean_square;

        window_write_idx_ = (window_write_idx_ + 1u) % kMaxWindows;
        if (window_count_ < kMaxWindows) ++window_count_;
    }

    const auto shift_samples = static_cast<float>(window_size_ - step_size_);
    window_accumulator_  *= shift_samples / static_cast<float>(window_sample_count_);
    window_sample_count_  = static_cast<uint32_t>(shift_samples);

    cached_lufs_ = MeasureLUFS();
}

// O(1): no loop, one division, one FastLog2 (3 clock cycles)
float LUFSTargeting::MeasureLUFS() const noexcept {
    if (window_count_ == 0u) return -70.0f;
    const float gated_mean = running_power_sum_ / static_cast<float>(window_count_);
    return (gated_mean > 1e-12f)
               ? -0.691f + viper::dsp::FastPowerToDb(gated_mean)
               : -70.0f;
}

void LUFSTargeting::Process(std::span<float> samples) noexcept {
    if (!enable_ || samples.empty()) return;
    const size_t size = samples.size();
    [[assume(size % 2 == 0)]];

    const size_t frame_count = size / 2u;

    // ----------------------------------------------------------------
    // 1. Stereo K-weighting + energy accumulation
    //    stage1/stage2 each process L & R in parallel (2-lane SIMD).
    //    Zero transcendental calls in this loop.
    // ----------------------------------------------------------------
    for (size_t i = 0u; i < frame_count; ++i) {
        float k_l;
        float k_r;
        stage1_.Process(samples[i * 2u], samples[i * 2u + 1u], k_l, k_r);
        stage2_.Process(k_l, k_r, k_l, k_r);

        window_accumulator_ += k_l * k_l + k_r * k_r;
        ++window_sample_count_;

        if (++sample_counter_ >= step_size_) [[unlikely]] {
            sample_counter_ = 0u;
            UpdateWindow();
        }
    }

    // ----------------------------------------------------------------
    // 2. Block-rate gain computation (once per Process() call).
    //    Time constant scaled to block duration dt = frame_count/sr so
    //    alpha_block = 1 - exp(-dt/tau) — exact regardless of block size.
    // ----------------------------------------------------------------
    const float desired_gain_db = std::clamp(
        target_lufs_ - cached_lufs_,
        -max_gain_db_,
        max_gain_db_
    );

    const auto [attack_ms, release_ms] = kSpeedTable[static_cast<size_t>(speed_)];
    const float tau_sec  = (desired_gain_db > current_gain_db_ ? attack_ms : release_ms) * 0.001f;
    const float block_dt = static_cast<float>(frame_count) / static_cast<float>(sampling_rate_);
    const float block_coeff = 1.0f - std::exp(-block_dt / tau_sec);

    current_gain_db_ += block_coeff * (desired_gain_db - current_gain_db_);

    const float target_gain_linear = viper::dsp::FastDbToLinear(current_gain_db_);

    // ----------------------------------------------------------------
    // 3. Linear gain ramp — two MACs per frame, zero transcendentals
    // ----------------------------------------------------------------
    const float gain_start = current_gain_linear_;
    const float gain_step  = (target_gain_linear - gain_start)
                             / static_cast<float>(frame_count);

    float g = gain_start;
#pragma clang loop vectorize(enable)
    for (size_t i = 0u; i < frame_count; ++i) {
        g += gain_step;
        samples[i * 2u]      *= g;
        samples[i * 2u + 1u] *= g;
    }
    current_gain_linear_ = target_gain_linear;
}

void LUFSTargeting::Reset() noexcept {
    stage1_.Reset();
    stage2_.Reset();
    ConfigureFilters();
    current_gain_db_     = 0.0f;
    current_gain_linear_ = 1.0f;
    sample_counter_      = 0u;
    window_accumulator_  = 0.0f;
    window_sample_count_ = 0u;
    window_write_idx_    = 0u;
    window_count_        = 0u;
    running_power_sum_   = 0.0f;
    cached_lufs_         = -70.0f;
    window_power_.fill(0.0f);
    window_size_ = static_cast<uint32_t>(sampling_rate_ * 0.4f);
    step_size_   = window_size_ / 4u;
}

void LUFSTargeting::SetEnable(const bool enable) noexcept {
    if (enable_ != enable) {
        if (enable) Reset();
        enable_ = enable;
    }
}

void LUFSTargeting::SetTargetLUFS(const float value) noexcept { target_lufs_ = value; }
void LUFSTargeting::SetMaxGain(const float value)    noexcept { max_gain_db_ = value; }

void LUFSTargeting::SetSpeed(const int value) noexcept {
    speed_ = std::clamp(value, 0, 2);
    // kSpeedTable is read directly in Process() at block rate — no coefficients to precompute
}

void LUFSTargeting::SetSamplingRate(const uint32_t sampling_rate) {
    if (sampling_rate_ != sampling_rate && sampling_rate > 0u) {
        sampling_rate_ = sampling_rate;
        Reset();
    }
}

void LUFSTargeting::ConfigureFilters() noexcept {
    const KWeightCoeffs& kw = (sampling_rate_ == 48000u) ? kKWeight48k : kKWeight44k1;

    stage1_.b0 = kw.stage1.b0; stage1_.b1 = kw.stage1.b1; stage1_.b2 = kw.stage1.b2;
    stage1_.a1 = kw.stage1.a1; stage1_.a2 = kw.stage1.a2;

    stage2_.b0 = kw.stage2.b0; stage2_.b1 = kw.stage2.b1; stage2_.b2 = kw.stage2.b2;
    stage2_.a1 = kw.stage2.a1; stage2_.a2 = kw.stage2.a2;
}

