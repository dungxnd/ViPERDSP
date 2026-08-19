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
};
