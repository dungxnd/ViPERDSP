#include "StereoImager.h"
#include <array>

static constexpr float kButterworthQ = 0.7071f;

StereoImager::StereoImager() {
    ConfigureCrossovers();
}

void StereoImager::Process(float *samples, const uint32_t size) noexcept {
    if (!enable_ || size == 0) return;

    const uint32_t frame_count = size * 2;

    for (auto& buf : band_buffers_) {
        if (buf.size() < frame_count) buf.resize(frame_count);
    }

    for (uint32_t b = 0; b < kNumBands; b++) {
        for (uint32_t i = 0; i < frame_count; i += 2) {
            double sample_l = samples[i];
            double sample_r = samples[i + 1];

            if (b == 0) {
                sample_l = lowpass_la_[0].ProcessSample(sample_l);
                sample_l = lowpass_lb_[0].ProcessSample(sample_l);
                sample_r = lowpass_ra_[0].ProcessSample(sample_r);
                sample_r = lowpass_rb_[0].ProcessSample(sample_r);
            } else if (b == 2) {
                sample_l = highpass_la_[1].ProcessSample(sample_l);
                sample_l = highpass_lb_[1].ProcessSample(sample_l);
                sample_r = highpass_ra_[1].ProcessSample(sample_r);
                sample_r = highpass_rb_[1].ProcessSample(sample_r);
            } else {
                sample_l = highpass_la_[0].ProcessSample(sample_l);
                sample_l = highpass_lb_[0].ProcessSample(sample_l);
                sample_l = lowpass_la_[1].ProcessSample(sample_l);
                sample_l = lowpass_lb_[1].ProcessSample(sample_l);
                sample_r = highpass_ra_[0].ProcessSample(sample_r);
                sample_r = highpass_rb_[0].ProcessSample(sample_r);
                sample_r = lowpass_ra_[1].ProcessSample(sample_r);
                sample_r = lowpass_rb_[1].ProcessSample(sample_r);
            }

            const auto f_l = static_cast<float>(sample_l);
            const auto f_r = static_cast<float>(sample_r);

            const float mid  = (f_l + f_r) * 0.5f;
            const float side = (f_l - f_r) * 0.5f * band_widths_[b];

            band_buffers_[b][i]     = mid + side;
            band_buffers_[b][i + 1] = mid - side;
        }
    }

    for (uint32_t i = 0; i < frame_count; i += 2) {
        float sum_l = 0.0f;
        float sum_r = 0.0f;
        for (uint32_t b = 0; b < kNumBands; b++) {
            sum_l += band_buffers_[b][i];
            sum_r += band_buffers_[b][i + 1];
        }
        samples[i]     = sum_l;
        samples[i + 1] = sum_r;
    }
}

void StereoImager::Reset() noexcept {
    // Collect all filter banks for a single reset sweep
    std::array<MultiBiquad*, kNumCrossovers * 8> all_filters;
    for (uint32_t i = 0; i < kNumCrossovers; i++) {
        all_filters[i * 8 + 0] = &lowpass_la_[i];
        all_filters[i * 8 + 1] = &lowpass_lb_[i];
        all_filters[i * 8 + 2] = &lowpass_ra_[i];
        all_filters[i * 8 + 3] = &lowpass_rb_[i];
        all_filters[i * 8 + 4] = &highpass_la_[i];
        all_filters[i * 8 + 5] = &highpass_lb_[i];
        all_filters[i * 8 + 6] = &highpass_ra_[i];
        all_filters[i * 8 + 7] = &highpass_rb_[i];
    }
    for (auto* f : all_filters) f->Reset();

    ConfigureCrossovers();
}

void StereoImager::SetEnable(const bool enable) noexcept {
    if (enable_ != enable) {
        if (enable) Reset();
        enable_ = enable;
    }
}

void StereoImager::SetLowWidth(const float value) noexcept {
    band_widths_[0] = value / 100.0f;
}

void StereoImager::SetMidWidth(const float value) noexcept {
    band_widths_[1] = value / 100.0f;
}

void StereoImager::SetHighWidth(const float value) noexcept {
    band_widths_[2] = value / 100.0f;
}

void StereoImager::SetLowCrossover(const float value) noexcept {
    if (crossover_freqs_[0] != value) {
        crossover_freqs_[0] = value;
        ConfigureCrossovers();
    }
}

void StereoImager::SetHighCrossover(const float value) noexcept {
    if (crossover_freqs_[1] != value) {
        crossover_freqs_[1] = value;
        ConfigureCrossovers();
    }
}

void StereoImager::SetSamplingRate(const uint32_t sampling_rate) noexcept {
    if (sampling_rate_ != sampling_rate) {
        sampling_rate_ = sampling_rate;
        ConfigureCrossovers();
    }
}

void StereoImager::ConfigureCrossovers() noexcept {
    // Per crossover index: LP and HP banks for both channels, both cascade stages
    using BankPair = std::array<MultiBiquad*, 4>;
    for (uint32_t i = 0; i < kNumCrossovers; i++) {
        const float freq = crossover_freqs_[i];

        const BankPair lp_banks{&lowpass_la_[i],  &lowpass_lb_[i],
                                &lowpass_ra_[i],  &lowpass_rb_[i]};
        for (auto* f : lp_banks) {
            f->RefreshFilter(MultiBiquad::LOW_PASS, 1.0f, freq,
                             sampling_rate_, kButterworthQ, false);
        }

        const BankPair hp_banks{&highpass_la_[i], &highpass_lb_[i],
                                &highpass_ra_[i], &highpass_rb_[i]};
        for (auto* f : hp_banks) {
            f->RefreshFilter(MultiBiquad::HIGH_PASS, 1.0f, freq,
                             sampling_rate_, kButterworthQ, false);
        }
    }
}
