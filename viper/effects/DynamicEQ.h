#pragma once

#include <array>
#include <cstdint>
#include <mdspan>
#include <span>

class DynamicEQ {
public:
    static constexpr uint32_t kMaxBands      = 10u;
    static constexpr uint32_t kControlPeriod = 16u; // Sub-block control rate (samples)

    DynamicEQ();

    void Process(std::span<float> samples) noexcept;
    void Process(float* samples, uint32_t size) noexcept {
        if (samples) Process(std::span<float>(samples, size * 2u));
    }

    void Reset() noexcept;

    void SetEnable(bool enable) noexcept;
    void SetBandCount(uint32_t count) noexcept;
    void SetSamplingRate(uint32_t sampling_rate) noexcept;

    void SetBandFrequency(uint32_t band, float value) noexcept;
    void SetBandGain(uint32_t band, float value) noexcept;
    void SetBandQ(uint32_t band, float value) noexcept;
    void SetBandThreshold(uint32_t band, float value) noexcept;
    void SetBandAttack(uint32_t band, float value) noexcept;
    void SetBandRelease(uint32_t band, float value) noexcept;
    void SetBandFilterType(uint32_t band, int value) noexcept;

private:
    // C++23 mdspan view over the interleaved stereo buffer: audio[frame, channel]
    using StereoView = std::mdspan<float, std::dextents<size_t, 2>, std::layout_right>;

    struct alignas(8) BiquadState {
        float s1{0.0f};
        float s2{0.0f};
    };

    struct FilterCoeffs {
        float b0{1.0f};
        float b1{0.0f};
        float b2{0.0f};
        float a1{0.0f};
        float a2{0.0f};
    };

    // Trig invariants cached on parameter change; hot path uses zero sin/cos.
    struct PrecomputedTrig {
        float cos_w0{1.0f};
        float sin_w0{0.0f};
        float alpha{0.0f};        // sin_w0 / (2*q)
        float neg_2_cos_w0{-2.0f}; // b1 == a1 in Peak filter — cache once
    };

    struct BandParam {
        float frequency{1000.0f};
        float q{1.0f};
        float target_gain_db{0.0f};
        float threshold_db{-24.0f};
        float linear_threshold_sq{0.003981f}; // 10^(threshold_db/10), precomputed
        float attack_ms{10.0f};
        float release_ms{100.0f};
        int   filter_type{0}; // 0=Peak, 1=LowShelf, 2=HighShelf
    };

    struct BandState {
        float env_l{0.0f};
        float env_r{0.0f};
        // Per-sample coefficients — used by audio-rate envelope follower
        float attack_coeff{0.1f};
        float release_coeff{0.01f};
        // Sub-block coefficients — used by kControlPeriod-rate gain smoother
        // α_sub = 1 - exp(-K/τ·fs) = 1 - (1 - α_sample)^K
        float subblock_attack_coeff{0.1f};
        float subblock_release_coeff{0.01f};
        float current_gain_db{0.0f};
    };

    bool     enable_{false};
    uint32_t sampling_rate_{44100u};
    uint32_t band_count_{0u};

    std::array<BandParam,       kMaxBands>                  params_{};
    std::array<BandState,       kMaxBands>                  state_{};
    std::array<PrecomputedTrig, kMaxBands>                  trig_cache_{};
    std::array<FilterCoeffs,    kMaxBands>                  coeffs_{};
    std::array<std::array<BiquadState, kMaxBands>, 2>       filter_state_{}; // [ch][band]

    void PrecomputeTrigConstants(uint32_t band) noexcept;
    void RecalcAttackRelease(uint32_t band) noexcept;
    void FastUpdateBandCoeffs(uint32_t band, float gain_db) noexcept;

    // Sub-functions extracted from Process() to satisfy ≤3 nesting depth
    void TrackEnvelope(StereoView& audio, size_t frame_offset,
                       size_t chunk, BandState& st) noexcept;
    void UpdateSubBlockGain(const BandParam& p, BandState& st) noexcept;
    void ApplyBiquadBlock(StereoView& audio, size_t frame_offset,
                          size_t chunk, uint32_t band) noexcept;
};
