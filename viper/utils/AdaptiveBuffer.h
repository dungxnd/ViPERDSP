#pragma once

#include <cstdint>
#include <vector>

class AdaptiveBuffer {
public:
    AdaptiveBuffer(uint32_t channels, uint32_t length);

    [[nodiscard]] uint32_t  GetBufferLength() const noexcept;
    [[nodiscard]] uint32_t  GetBufferOffset() const noexcept;
    [[nodiscard]] uint32_t  GetChannels()     const noexcept;
    float*                  GetBuffer()       noexcept;

    void SetBufferOffset(uint32_t value) noexcept;

    void PanFrames(float left, float right) noexcept;
    int  PopFrames(float* frames, uint32_t length) noexcept;
    int  PushFrames(const float* frames, uint32_t length);
    void ScaleFrames(float scale) noexcept;
    int  PushZero(uint32_t length);
    void FlushBuffer() noexcept;

private:
    uint32_t           length_;
    uint32_t           offset_{0};
    uint32_t           channels_;
    std::vector<float> buffer_;
};
