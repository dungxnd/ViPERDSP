#pragma once

#include <span>


#include "../include/ViPERParams.h"
#include "../utils/MultiBiquad.h"
#include <array>
#include <cstdint>

class DiffSurround {
public:
    using Config = viper::DiffSurroundParams;

    DiffSurround();

    void ProcessPlanar(std::span<float> L, std::span<float> R) noexcept;
    void Process(float* samples, uint32_t size) noexcept;
    void Reset();

    [[nodiscard]] bool IsEnabled() const noexcept { return config_.enable; }
    void SetConfig(const Config& config) noexcept;
    [[nodiscard]] const Config& GetConfig() const noexcept { return config_; }

    void SetEnable(bool enable);
    void SetDelayTime(float value);
    void SetReverse(bool value);
    void SetWetDryMix(float value);
    void SetLPCutoff(float value);
    void SetSamplingRate(uint32_t sampling_rate);

private:
    Config config_{};
    // Zero-copy power-of-two ring delay line — O(1) read/write, no memmove.
    struct RingDelay {
        static constexpr uint32_t kCap  = 8192u;
        static constexpr uint32_t kMask = kCap - 1u;

        void Reset() noexcept { ring_.fill(0.0f); write_pos_ = 0u; delay_ = 0u; }

        void SetDelay(uint32_t delay_samples) noexcept {
            delay_ = delay_samples & kMask;
        }

        inline float Process(float input) noexcept {
            ring_[write_pos_] = input;
            const uint32_t read_pos = (write_pos_ - delay_) & kMask;
            write_pos_ = (write_pos_ + 1u) & kMask;
            return ring_[read_pos];
        }

        alignas(64) std::array<float, kCap> ring_{};
        uint32_t write_pos_{0u};
        uint32_t delay_{0u};
    };

    bool enable_{false};
    bool reverse_{false};

    uint32_t sampling_rate_{44100u};

    float delay_time_{0.0f};
    float wet_dry_mix_{1.0f};
    float lp_cutoff_{0.0f};

    std::array<RingDelay, 2> ring_{};
    MultiBiquad lp_filter_{};
    alignas(64) std::array<float, 4096u * 2u> scratch_{};
};
