#pragma once

#include "../utils/HiFi.h"
#include "../utils/HighShelf.h"
#include "../utils/NoiseSharpening.h"
#include <array>
#include <cstdint>

class ViPERClarity {
public:
    enum class ClarityMode {
        Natural = 0,
        Ozone   = 1,
        XHiFi   = 2,
        // ALL_CAPS aliases for source compatibility
        NATURAL = Natural,
        OZONE   = Ozone,
        XHIFI   = XHiFi,
    };

    ViPERClarity();

    void ProcessPlanar(float* __restrict L, float* __restrict R, size_t frames) noexcept;
    void Reset() noexcept;

    [[nodiscard]] bool IsEnabled() const noexcept { return enable_; }
    void SetEnable(bool enable) noexcept;
    void SetProcessMode(ClarityMode mode) noexcept;
    void SetClarityGain(float value) noexcept;
    void SetClarityToFilter() noexcept;
    void SetSamplingRate(uint32_t sampling_rate) noexcept;

private:
    void Process(float *samples, uint32_t size) noexcept;

    bool enable_{false};

    ClarityMode process_mode_{ClarityMode::Natural};

    uint32_t sampling_rate_{44100u};

    float gain_{0.0f};

    NoiseSharpening         noise_sharpening_;
    std::array<HighShelf, 2> high_shelf_{};
    HiFi                    hifi_;
    alignas(64) std::array<float, 4096u * 2u> scratch_{};
};
