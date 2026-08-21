#pragma once

#include <array>
#include <cstdint>
#include <span>

class LUFSTargeting {
public:
    LUFSTargeting();

    void Process(std::span<float> samples) noexcept;
    void Process(float* samples, uint32_t size) noexcept {
        if (samples) Process(std::span<float>(samples, size * 2u));
    }

    void Reset() noexcept;

    void SetEnable(bool enable) noexcept;
    void SetTargetLUFS(float value) noexcept;
    void SetMaxGain(float value) noexcept;
    void SetSpeed(int value) noexcept;
    void SetSamplingRate(uint32_t sampling_rate);

private:
    static constexpr uint32_t kMaxWindows    = 40u;
    static constexpr float    kGateThreshold = 1.1724644e-7f; // 10^((-70+0.691)/10)

    // Stereo-interleaved TDF-II biquad: shared coefficients, independent L/R state.
    // alignas(16) lets the compiler pack s1_l/s1_r/s2_l/s2_r into a 128-bit register.
    struct alignas(16) StereoTDF2Biquad {
        float b0{1.0f};
        float b1{0.0f};
        float b2{0.0f};
        float a1{0.0f};
        float a2{0.0f};
        // State registers — laid out for 2-lane SIMD auto-vectorization
        float s1_l{0.0f};
        float s1_r{0.0f};
        float s2_l{0.0f};
        float s2_r{0.0f};

        inline void Process(float in_l, float in_r,
                            float& out_l, float& out_r) noexcept {
            out_l = b0 * in_l + s1_l;
            s1_l  = b1 * in_l - a1 * out_l + s2_l;
            s2_l  = b2 * in_l - a2 * out_l;

            out_r = b0 * in_r + s1_r;
            s1_r  = b1 * in_r - a1 * out_r + s2_r;
            s2_r  = b2 * in_r - a2 * out_r;
        }

        void Reset() noexcept {
            s1_l = 0.0f; s1_r = 0.0f;
            s2_l = 0.0f; s2_r = 0.0f;
        }
    };

    bool     enable_{false};
    int      speed_{1};
    uint32_t sampling_rate_{44100u};
    uint32_t window_size_{0u};
    uint32_t step_size_{0u};
    uint32_t sample_counter_{0u};
    uint32_t window_sample_count_{0u};
    uint32_t window_write_idx_{0u};
    uint32_t window_count_{0u};

    float target_lufs_{-14.0f};
    float max_gain_db_{6.0f};

    float current_gain_db_{0.0f};
    float current_gain_linear_{1.0f};
    float window_accumulator_{0.0f};
    float cached_lufs_{-70.0f};
    float running_power_sum_{0.0f}; // O(1) sliding window sum — no iteration in MeasureLUFS

    std::array<float, kMaxWindows> window_power_{};

    StereoTDF2Biquad stage1_{};
    StereoTDF2Biquad stage2_{};

    void ConfigureFilters() noexcept;
    void UpdateWindow() noexcept;
    [[nodiscard]] float MeasureLUFS() const noexcept;
};
