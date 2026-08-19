#pragma once

#include "../utils/MultiBiquad.h"
#include <array>
#include <cstdint>

class DynamicEQ {
public:
    static constexpr uint32_t kMaxBands = 10;

    DynamicEQ();

    void Process(float *samples, uint32_t size);
    void Reset();

    void SetEnable(bool enable);
    void SetBandCount(uint32_t count);
    void SetSamplingRate(uint32_t sampling_rate);

    void SetBandFrequency(uint32_t band, float value);
    void SetBandGain(uint32_t band, float value);
    void SetBandQ(uint32_t band, float value);
    void SetBandThreshold(uint32_t band, float value);
    void SetBandAttack(uint32_t band, float value);
    void SetBandRelease(uint32_t band, float value);
    void SetBandFilterType(uint32_t band, int value);

private:
    struct BandParam {
        float frequency{1000.0f};
        float q{1.0f};
        float target_gain_db{0.0f};
        float threshold_db{-24.0f};
        float attack_ms{10.0f};
        float release_ms{100.0f};
        MultiBiquad::FilterType filter_type{MultiBiquad::FilterType::PEAK};
    };

    struct BandState {
        double envelope_l{0.0};
        double envelope_r{0.0};
        double smoothed_gain_db{0.0};
        float  last_applied_gain_db{0.0f};
        double attack_coeff{0.0};
        double release_coeff{0.0};
    };

    bool enable_{false};

    uint32_t sampling_rate_{44100u};
    uint32_t band_count_{0};

    std::array<BandParam,    kMaxBands> params_{};
    std::array<BandState,    kMaxBands> state_{};
    std::array<MultiBiquad,  kMaxBands> apply_l_{};
    std::array<MultiBiquad,  kMaxBands> apply_r_{};

    void RecalcAttackRelease(uint32_t band);
    void ConfigureApplicationFilter(uint32_t band, float gain_db);
};
