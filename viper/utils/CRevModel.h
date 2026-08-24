#pragma once

#include "CAllPassFilter.h"
#include "CCombFilter.h"
#include <array>
#include <cstdint>
#include <memory>

class CRevModel {
public:
    CRevModel();

    // Rule of Zero — unique_ptr manages buffer_pool_; copy deleted, move = default.
    CRevModel(const CRevModel&)            = delete;
    CRevModel& operator=(const CRevModel&) = delete;
    CRevModel(CRevModel&&)                 = default;
    CRevModel& operator=(CRevModel&&)      = default;

    void ProcessReplace(float* buf_l, float* buf_r, uint32_t size) noexcept;
    void Mute()  const noexcept;
    void Reset() const noexcept;

    // Scale all delay-line lengths proportionally to the host sample rate.
    // Must be called (once) before the first ProcessReplace when the host rate
    // is not 44100 Hz.  Reallocates buffer_pool_; re-wires all filter views.
    void SetSamplingRate(uint32_t sampling_rate) noexcept;

    void SetRoomSize(float value) noexcept;
    void SetDamp(float value)     noexcept;
    void SetWet(float value)      noexcept;
    void SetDry(float value)      noexcept;
    void SetWidth(float value)    noexcept;

    [[nodiscard]] float GetRoomSize() const noexcept;
    [[nodiscard]] float GetDamp()     const noexcept;
    [[nodiscard]] float GetWet()      const noexcept;
    [[nodiscard]] float GetDry()      const noexcept;
    [[nodiscard]] float GetWidth()    const noexcept;

    void UpdateCoeffs() noexcept;

private:
    static constexpr uint32_t kNumCombs   = 8;
    static constexpr uint32_t kNumAllPass = 4;

    // Reference buffer sizes at 44100 Hz — scaled on SetSamplingRate().
    static constexpr std::array<uint32_t, 24> kBufferSizes44k = {
        1116, 1139, 1188, 1211, 1277, 1300, 1356, 1379,
        1422, 1445, 1491, 1514, 1557, 1580, 1617, 1640,
         556,  579,  441,  464,  341,  364,  225,  248,
    };

    // Rebuilds buffer_pool_ and re-wires comb/allpass filters for scaled sizes.
    void ReallocateBuffers(float sr_scale) noexcept;

    float gain_{0.0f};
    float room_size_{0.0f};
    float internal_room_size_{0.0f};
    float damp_{0.0f};
    float internal_damp_{0.0f};
    float wet_{0.0f};
    float wet1_{0.0f};
    float wet2_{0.0f};
    float dry_{0.0f};
    float width_{0.0f};

    std::array<CCombFilter,    kNumCombs>   comb_l_;
    std::array<CCombFilter,    kNumCombs>   comb_r_;
    std::array<CAllPassFilter, kNumAllPass> allpass_l_;
    std::array<CAllPassFilter, kNumAllPass> allpass_r_;

    std::unique_ptr<float[]> buffer_pool_;
    // Non-owning views into buffer_pool_ — valid as long as pool is alive.
    std::array<float*, 24>   buffers_{}; // NOLINT(modernize-avoid-c-arrays) — non-owning view pointers
    // Scaled sizes matching current buffer_pool_ allocation.
    std::array<uint32_t, 24> buffer_sizes_{kBufferSizes44k};
};
