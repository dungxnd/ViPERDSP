#pragma once

#include "../include/ViPERParams.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

struct DdcCoeffs {
    uint32_t arr_size{0u};
    std::vector<std::array<float, 5>> coeffs_44100;
    std::vector<std::array<float, 5>> coeffs_48000;
};

class ViPERDDC {
public:
    using Config = viper::DdcParams;

    ViPERDDC();

    void ProcessPlanar(std::span<float> L, std::span<float> R) noexcept;
    void Reset() noexcept;

    [[nodiscard]] bool IsEnabled() const noexcept { return config_.enable; }
    void SetConfig(const Config& config) noexcept;
    [[nodiscard]] const Config& GetConfig() const noexcept { return config_; }

    void SetEnable(bool enable) noexcept;
    void SetCoeffs(
        uint32_t coeffs_size, const float *coeffs_44100, const float *coeffs_48000
    );
    void SetSamplingRate(uint32_t sampling_rate) noexcept;

private:
    void Process(float *samples, uint32_t size) noexcept;
    void ConsumeCoeffsSwap() noexcept;

    Config config_{};
    bool set_coeffs_ok_{false};

    uint32_t sampling_rate_{44100u};
    uint32_t arr_size_{0u};

    std::unique_ptr<DdcCoeffs> active_coeffs_{std::make_unique<DdcCoeffs>()};
    std::unique_ptr<DdcCoeffs> staging_coeffs_{std::make_unique<DdcCoeffs>()};
    std::atomic<bool> swap_pending_{false};
    std::mutex stage_mutex_;

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
