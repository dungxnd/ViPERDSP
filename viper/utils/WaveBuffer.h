#pragma once

#include <vector>
#include <cstdint>

class WaveBuffer {
public:
    WaveBuffer(uint32_t channels, uint32_t length);

    void Reset() noexcept;

    [[nodiscard]] uint32_t GetBufferOffset() const noexcept;
    [[nodiscard]] uint32_t GetBufferSize() const noexcept;
    [[nodiscard]] float *GetBuffer() noexcept;

    void SetBufferOffset(uint32_t offset) noexcept;

    uint32_t PopSamples(uint32_t size, bool reset_idx) noexcept;
    uint32_t PopSamples(float *dest, uint32_t size, bool reset_idx) noexcept;
    int PushSamples(const float *source, uint32_t size);
    int PushZeros(uint32_t size);
    float *PushZerosGetBuffer(uint32_t size);

private:
    uint32_t index_{0};
    uint32_t channels_;

    std::vector<float> buffer_;
};
