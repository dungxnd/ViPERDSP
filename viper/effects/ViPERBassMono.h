#pragma once

#include "ViPERBassCommon.h"
#include "../utils/Biquad.h"
#include "../utils/Polyphase.h"
#include "../utils/Subwoofer.h"
#include <array>
#include <cstdint>
#include <mdspan>
#include <span>
#include <vector>

class ViPERBassMono {
public:
    using ProcessMode = BassProcessMode;
    // 2-D view over interleaved stereo audio: [frame, channel].
    using StereoView = std::mdspan<float, std::dextents<size_t, 2>, std::layout_right>;

    ViPERBassMono();

    // samples: interleaved stereo, size = frame count (not sample count).
    void Process(std::span<float> samples) noexcept;
    void ProcessPlanar(float* __restrict L, float* __restrict R, size_t frames) noexcept;
    void Reset() noexcept;

    [[nodiscard]] bool IsEnabled() const noexcept { return enable_; }
    void SetEnable(bool enable) noexcept;
    void SetProcessMode(ProcessMode mode) noexcept;
    void SetBassFactor(float value) noexcept;
    void SetFrequency(uint32_t value) noexcept;
    void SetAntiPop(bool enable) noexcept;
    void SetSamplingRate(uint32_t sampling_rate) noexcept;

private:
    bool enable_{false};

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
    float dc_x1_{0.0f};
    float dc_y1_{0.0f};

    Polyphase  polyphase_{2};
    Biquad     biquad_;
    Subwoofer  subwoofer_;

    // Allocation-free circular delay line for PureBass+ dry-path delay.
    // Stores interleaved stereo (L,R) raw input, delayed by kLatency (31) samples.
    // Capacity is kDelayCapacity frames * 2 channels.
    // Power-of-two frame capacity >= Polyphase::kLatency (31).
    static constexpr size_t kDelayCapacity = 64u;
    static constexpr size_t kDelayMask     = kDelayCapacity - 1u;

    std::array<float, kDelayCapacity * 2u> bass_delay_{};
    size_t delay_write_idx_{0u};

    // Pre-allocated scratch buffer for ProcessSubwoofer anti-pop blend and
    // ProcessPureBassPlus FIR pass.  Sized in ctor / SetSamplingRate().
    std::vector<float> scratch_buffer_;
    alignas(64) std::array<float, 4096u * 2u> pp_scratch_{};

    void ShapeMix(float bass, float& left, float& right) noexcept;
    void ProcessNaturalBass (StereoView audio) noexcept;
    void ProcessPureBassPlus(StereoView audio) noexcept;
    void ProcessSubwoofer   (std::span<float> samples, StereoView audio) noexcept;
};
