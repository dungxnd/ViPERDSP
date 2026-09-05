#pragma once

#include "ViPERBassCommon.h"
#include "../include/ViPERParams.h"
#include "../utils/Biquad.h"
#include "../utils/Polyphase.h"
#include "../utils/Subwoofer.h"
#include <array>
#include <cstdint>
#include <mdspan>
#include <span>
#include <vector>

class ViPERBass {
public:
    using Config = viper::BassParams;
    using ProcessMode = BassProcessMode;
    // 2-D view over interleaved stereo audio: [frame, channel].
    using StereoView  = std::mdspan<float, std::dextents<size_t, 2>, std::layout_right>;

    ViPERBass();

    void ProcessPlanar(std::span<float> L, std::span<float> R) noexcept;
    void Reset() noexcept;

    [[nodiscard]] bool IsEnabled() const noexcept { return config_.enable; }
    void SetConfig(const Config& config) noexcept;
    [[nodiscard]] const Config& GetConfig() const noexcept { return config_; }

    void SetEnable(bool enable) noexcept;
    void SetProcessMode(ProcessMode mode) noexcept;
    void SetBassFactor(float value) noexcept;
    void SetFrequency(uint32_t value) noexcept;
    void SetAntiPop(bool enable) noexcept;
    void SetSamplingRate(uint32_t sampling_rate) noexcept;

private:
    // samples: interleaved stereo, size = frame count (not sample count).
    void Process(std::span<float> samples) noexcept;

    Config      config_{};
    bool        enable_{false};
    ProcessMode process_mode_{ProcessMode::NaturalBass};

    uint32_t sampling_rate_{44100u};
    uint32_t frequency_{60u};

    float anti_pop_{0.0f};
    // Step per frame for 20 ms fade-in ramp: 1 / (0.020 * sr).
    float anti_pop_step_{0.0f};
    float bass_factor_{0.0f};
    float bass_factor_smoothed_{0.0f};
    float smoothing_coeff_{0.0f};

    float dc_block_coeff_{0.0f};
    std::array<float, 2> dc_x1_{};
    std::array<float, 2> dc_y1_{};

    Polyphase             polyphase_{2};
    std::array<Biquad, 2> biquad_{};
    Subwoofer             subwoofer_;

    // Allocation-free circular delay line for PureBass+ dry-path delay.
    // Stores the raw input and delays it by kLatency (31) samples.
    // Must be a power-of-two >= Polyphase::kLatency (31) — 64 satisfies this.
    static constexpr size_t kDelayCapacity = 64u;
    static constexpr size_t kDelayMask     = kDelayCapacity - 1u;
    // Largest frame block the interleaved Process() path accepts; matches the
    // planar pipeline's AudioProcessContext<4096> block size.
    static constexpr size_t kMaxFrames = 4096u;

    std::array<std::array<float, kDelayCapacity>, 2> bass_delay_{};
    size_t delay_write_idx_{0u};

    // Pre-allocated scratch buffer for ProcessSubwoofer anti-pop blend and
    // ProcessPureBassPlus mid/high FIR pass.  Sized in ctor / SetSamplingRate().
    std::vector<float> scratch_buffer_;

    // Dedicated staging for the interleaved Process() path (NaturalBass /
    // PureBassPlus deinterleave staging + Subwoofer planar bounce).  Separate
    // from scratch_buffer_ so the planar region [0..4*frames) is never
    // aliased by the interleaved staging at [4*frames..6*frames) — the ctor
    // only sized 4*frames and SetSamplingRate() is not called at 44.1 kHz.
    std::array<float, kMaxFrames * 2u> staging_buffer_{};

    void ShapeMix(float& left, float& right, float bass_l, float bass_r) noexcept;
    void ApplyAntiPop(float& bass_l, float& bass_r) noexcept;
    void ProcessNaturalBass (float* L, float* R, size_t frames) noexcept;
    void ProcessPureBassPlus(float* L, float* R, size_t frames) noexcept;
    void ProcessSubwoofer   (std::span<float> samples, StereoView audio) noexcept;
};
