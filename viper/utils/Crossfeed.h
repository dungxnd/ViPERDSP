#pragma once

#include <array>
#include <cstdint>

class Crossfeed {
public:
    struct Preset {
        uint32_t cutoff;
        uint32_t feedback;
    };

    Crossfeed();

    void ProcessFrames(float *buffer, uint32_t size) noexcept;
    void Reset() noexcept;

    [[nodiscard]] uint32_t GetCutoff() const noexcept;
    [[nodiscard]] float GetFeedback() const noexcept;
    [[nodiscard]] float GetLevelDelay() const noexcept;
    [[nodiscard]] Preset GetPreset() const noexcept;

    void SetCutoff(uint32_t value) noexcept;
    void SetFeedback(float value) noexcept;
    void SetPreset(Preset preset) noexcept;
    void SetSamplingRate(uint32_t sampling_rate) noexcept;

    void FilterSample(float *sample) noexcept;

private:
    uint32_t sampling_rate_{44100u};

    float a0_lo_{0.0f};
    float b1_lo_{0.0f};
    float a0_hi_{0.0f};
    float b1_hi_{0.0f};
    float a1_hi_{0.0f};
    float gain_{0.0f};

    struct Lfs {
        std::array<float, 2> asis{};
        std::array<float, 2> lo{};
        std::array<float, 2> hi{};
    } lfs_{};

    Preset preset_{700, 45};

    [[nodiscard]] float ApplyLoFilter(float in, float out_1) const noexcept {
        return a0_lo_ * in + b1_lo_ * out_1;
    }

    [[nodiscard]] float ApplyHiFilter(float in, float in_1, float out_1) const noexcept {
        return a0_hi_ * in + a1_hi_ * in_1 + b1_hi_ * out_1;
    }
};
