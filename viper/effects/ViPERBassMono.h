#pragma once

#include "ViPERBassCommon.h"
#include "../utils/Biquad.h"
#include "../utils/Polyphase.h"
#include "../utils/Subwoofer.h"
#include "../utils/WaveBuffer.h"
#include <cstdint>
#include <span>
#include <vector>

class ViPERBassMono {
public:
    using ProcessMode = BassProcessMode;

    ViPERBassMono();

    // samples: interleaved stereo, size = frame count (not sample count).
    void Process(std::span<float> samples) noexcept;
    void Reset() noexcept;

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
    // Stereo wave_buffer_ (channels=2) so PushSamples correctly handles
    // interleaved stereo input, matching the biquad delay-line indexing.
    WaveBuffer wave_buffer_{2, 4096};

    // Pre-allocated scratch buffer for ProcessSubwoofer anti-pop blend.
    std::vector<float> scratch_buffer_;

    void ShapeMix(float bass, uint32_t i, std::span<float> samples) noexcept;
    void ProcessNaturalBass (std::span<float> samples) noexcept;
    void ProcessPureBassPlus(std::span<float> samples) noexcept;
    void ProcessSubwoofer   (std::span<float> samples) noexcept;
};
