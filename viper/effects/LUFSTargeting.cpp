#include "LUFSTargeting.h"
#include <algorithm>
#include <array>
#include <cmath>

// ---------------------------------------------------------------------------
// K-weighting biquad coefficient tables (ITU-R BS.1770)
// ---------------------------------------------------------------------------
namespace {

struct BiquadCoeffs {
    double a0, a1, a2, b0, b1, b2;
};

struct KWeightCoeffs {
    BiquadCoeffs stage1;
    BiquadCoeffs stage2;
};

// Precomputed for 48 kHz (ITU-R BS.1770 Table 2)
constexpr KWeightCoeffs kKWeight48k = {
    { 1.0, -1.69065929318241,  0.73248077421585,
      1.53512485958697, -2.69169618940638, 1.19839281085285 },
    { 1.0, -1.99004745483398,  0.99007225036621,
      1.0, -2.0, 1.0 }
};

// Precomputed for 44.1 kHz
constexpr KWeightCoeffs kKWeight44k1 = {
    { 1.0, -1.6636551132560204, 0.7125954280732254,
      1.5308412300503478, -2.6509799951547297, 1.1690790799215869 },
    { 1.0, -1.9891696736297957, 0.9891990357870394,
      1.0, -2.0, 1.0 }
};

// Smoothing time constants per speed setting (attack_ms, release_ms)
struct SpeedParams { double attack_ms; double release_ms; };
constexpr std::array<SpeedParams, 3> kSpeedTable = {{
    { 200.0, 1000.0 },  // 0 = slow
    { 100.0,  500.0 },  // 1 = medium (default)
    {  50.0,  200.0 },  // 2 = fast
}};

} // anonymous namespace

// ---------------------------------------------------------------------------

LUFSTargeting::LUFSTargeting() {
    // window_power_{} already zero-init via in-class default; buf indices and
    // accumulators likewise. ConfigureFilters/UpdateSmoothingCoeffs finish setup.
    ConfigureFilters();
    UpdateSmoothingCoeffs();
    window_size_ = static_cast<uint32_t>(sampling_rate_ * 0.4);
    step_size_   = window_size_ / 4;
}

void LUFSTargeting::UpdateWindow(const double gate_threshold) noexcept {
    if (window_sample_count_ < window_size_) return;

    const double mean_square =
        window_accumulator_ / static_cast<double>(window_sample_count_);

    if (mean_square > gate_threshold) {
        window_power_[window_write_idx_] = mean_square;
        window_write_idx_ = (window_write_idx_ + 1) % kMaxWindows;
        if (window_count_ < kMaxWindows) ++window_count_;
    }

    const auto shift_samples = static_cast<double>(window_size_ - step_size_);
    window_accumulator_  *= shift_samples / static_cast<double>(window_sample_count_);
    window_sample_count_  = static_cast<uint32_t>(shift_samples);
}

double LUFSTargeting::MeasureLUFS() const noexcept {
    if (window_count_ == 0) return -70.0;

    double sum = 0.0;
    for (uint32_t w = 0; w < window_count_; ++w) sum += window_power_[w];
    const double gated_mean = sum / static_cast<double>(window_count_);
    return (gated_mean > 1e-20) ? -0.691 + 10.0 * std::log10(gated_mean) : -70.0;
}

void LUFSTargeting::Process(float* const samples, const uint32_t size) noexcept {
    if (!enable_ || size == 0) return;

    const double gate_threshold = std::pow(10.0, (kAbsoluteGateLufs + 0.691) / 10.0);

    for (uint32_t i = 0; i < size; ++i) {
        const double left  = samples[i * 2];
        const double right = samples[i * 2 + 1];

        double kLeft = k_weight_stage1_l_.ProcessSample(left);
        kLeft        = k_weight_stage2_l_.ProcessSample(kLeft);

        double kRight = k_weight_stage1_r_.ProcessSample(right);
        kRight        = k_weight_stage2_r_.ProcessSample(kRight);

        window_accumulator_ += kLeft * kLeft + kRight * kRight;
        ++window_sample_count_;

        if (++sample_counter_ >= step_size_) {
            sample_counter_ = 0;
            UpdateWindow(gate_threshold);
        }

        const double measured_lufs = MeasureLUFS();

        const double desired_gain_db = std::clamp(
            static_cast<double>(target_lufs_) - measured_lufs,
            static_cast<double>(-max_gain_db_),
            static_cast<double>(max_gain_db_)
        );

        const double coeff = (desired_gain_db > smoothed_gain_db_)
                             ? attack_coeff_ : release_coeff_;
        smoothed_gain_db_ += coeff * (desired_gain_db - smoothed_gain_db_);

        const auto gain_linear = static_cast<float>(
            std::pow(10.0, smoothed_gain_db_ / 20.0)
        );
        samples[i * 2]     *= gain_linear;
        samples[i * 2 + 1] *= gain_linear;
    }
}

void LUFSTargeting::Reset() noexcept {
    k_weight_stage1_l_.Reset();
    k_weight_stage1_r_.Reset();
    k_weight_stage2_l_.Reset();
    k_weight_stage2_r_.Reset();
    ConfigureFilters();
    UpdateSmoothingCoeffs();
    smoothed_gain_db_    = 0.0;
    sample_counter_      = 0;
    window_accumulator_  = 0.0;
    window_sample_count_ = 0;
    window_write_idx_    = 0;
    window_count_        = 0;
    window_power_.fill(0.0);
    window_size_ = static_cast<uint32_t>(sampling_rate_ * 0.4);
    step_size_   = window_size_ / 4;
}

void LUFSTargeting::SetEnable(const bool enable) noexcept {
    if (enable_ != enable) {
        if (enable) Reset();
        enable_ = enable;
    }
}

void LUFSTargeting::SetTargetLUFS(const float value) noexcept {
    target_lufs_ = value;
}

void LUFSTargeting::SetMaxGain(const float value) noexcept {
    max_gain_db_ = value;
}

void LUFSTargeting::SetSpeed(const int value) noexcept {
    speed_ = std::clamp(value, 0, 2);
    UpdateSmoothingCoeffs();
}

void LUFSTargeting::SetSamplingRate(const uint32_t sampling_rate) {
    if (sampling_rate_ != sampling_rate) {
        sampling_rate_ = sampling_rate;
        Reset();
    }
}

void LUFSTargeting::ConfigureFilters() noexcept {
    const KWeightCoeffs& kw = (sampling_rate_ == 48000) ? kKWeight48k : kKWeight44k1;

    k_weight_stage1_l_.SetCoeffs(kw.stage1.a0, kw.stage1.a1, kw.stage1.a2,
                                  kw.stage1.b0, kw.stage1.b1, kw.stage1.b2);
    k_weight_stage1_r_.SetCoeffs(kw.stage1.a0, kw.stage1.a1, kw.stage1.a2,
                                  kw.stage1.b0, kw.stage1.b1, kw.stage1.b2);
    k_weight_stage2_l_.SetCoeffs(kw.stage2.a0, kw.stage2.a1, kw.stage2.a2,
                                  kw.stage2.b0, kw.stage2.b1, kw.stage2.b2);
    k_weight_stage2_r_.SetCoeffs(kw.stage2.a0, kw.stage2.a1, kw.stage2.a2,
                                  kw.stage2.b0, kw.stage2.b1, kw.stage2.b2);
}

void LUFSTargeting::UpdateSmoothingCoeffs() noexcept {
    const auto [attack_ms, release_ms] = kSpeedTable[static_cast<size_t>(speed_)];
    const auto sr = static_cast<double>(sampling_rate_);
    attack_coeff_  = 1.0 - std::exp(-1.0 / (sr * attack_ms  / 1000.0));
    release_coeff_ = 1.0 - std::exp(-1.0 / (sr * release_ms / 1000.0));
}
