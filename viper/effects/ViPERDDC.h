#pragma once

#include <array>
#include <vector>
#include <cstdint>

class ViPERDDC {
public:
    ViPERDDC();

    void ProcessPlanar(float* __restrict L, float* __restrict R, size_t frames) noexcept;
    void Reset() noexcept;

    [[nodiscard]] bool IsEnabled() const noexcept { return enable_; }
    void SetEnable(bool enable) noexcept;
    void SetCoeffs(
        uint32_t coeffs_size, const float *coeffs_44100, const float *coeffs_48000
    );
    void SetSamplingRate(uint32_t sampling_rate) noexcept;

private:
    void Process(float *samples, uint32_t size) noexcept;

    bool enable_{false};
    bool set_coeffs_ok_{false};

    uint32_t sampling_rate_{44100u};
    uint32_t arr_size_{0u};

    std::vector<std::array<float, 5>> coeffs_arr44100_;
    std::vector<std::array<float, 5>> coeffs_arr48000_;
    std::vector<float> x1_l_;
    std::vector<float> x1_r_;
    std::vector<float> x2_l_;
    std::vector<float> x2_r_;
    std::vector<float> y1_l_;
    std::vector<float> y1_r_;
    std::vector<float> y2_l_;
    std::vector<float> y2_r_;

    void ReleaseResources() noexcept;
};
