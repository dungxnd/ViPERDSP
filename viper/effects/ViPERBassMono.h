#pragma once

#include "../utils/Biquad.h"
#include "../utils/Polyphase.h"
#include "../utils/Subwoofer.h"
#include "../utils/WaveBuffer.h"
#include <cstdint>

class ViPERBassMono {
public:
    enum class ProcessMode {
        NaturalBass    = 0,
        PureBassPlus   = 1,
        Subwoofer      = 2,
        // ALL_CAPS aliases for source compatibility
        NATURAL_BASS   = NaturalBass,
        PURE_BASS_PLUS = PureBassPlus,
        SUBWOOFER      = Subwoofer,
    };

    ViPERBassMono();

    void Process(float *samples, uint32_t size) noexcept;
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

    float sampling_rate_period_{1.0f / 44100.0f};
    float anti_pop_{0.0f};
    float bass_factor_{0.0f};
    float bass_factor_smoothed_{0.0f};
    float smoothing_coeff_{0.0f};

    float dc_block_coeff_{0.0f};
    float dc_x1_{0.0f};
    float dc_y1_{0.0f};

    Polyphase  polyphase_{2};
    Biquad     biquad_;
    Subwoofer  subwoofer_;
    WaveBuffer wave_buffer_{1, 4096};
};
