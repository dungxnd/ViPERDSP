#pragma once

#include <cstdint>

class CAllPassFilter {
public:
    CAllPassFilter() noexcept = default;

    float Process(float sample) noexcept;
    void  Mute() const noexcept;

    void SetBuffer(float* buffer, uint32_t size) noexcept;
    void SetFeedback(float value) noexcept;

    [[nodiscard]] float GetFeedback() const noexcept;

private:
    uint32_t buffer_size_{0};
    uint32_t buffer_index_{0};
    float    feedback_{0.0f};
    float*   buffer_{nullptr};
};
