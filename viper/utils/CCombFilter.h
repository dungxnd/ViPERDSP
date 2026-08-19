#pragma once

#include <cstdint>

class CCombFilter {
public:
    CCombFilter() noexcept = default;

    float Process(float sample) noexcept;
    void  Mute() const noexcept;

    void SetBuffer(float* buffer, uint32_t size) noexcept;
    void SetDamp(float value) noexcept;
    void SetFeedback(float value) noexcept;

    [[nodiscard]] float GetDamp()     const noexcept;
    [[nodiscard]] float GetFeedback() const noexcept;

private:
    uint32_t buffer_size_{0};
    uint32_t buffer_index_{0};
    float    feedback_{0.0f};
    float    filter_store_{0.0f};
    float    damp_{0.0f};
    float    damp2_{0.0f};
    float*   buffer_{nullptr};
};
