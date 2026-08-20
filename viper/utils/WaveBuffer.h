#pragma once

#include <cstdint>
#include <span>
#include <vector>

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
    bool PushSamples(const float *source, uint32_t size);
    bool PushZeros(uint32_t size);
    float *PushZerosGetBuffer(uint32_t size);

    bool PushSamples(std::span<const float> source) {
        return PushSamples(source.data(),
                           static_cast<uint32_t>(source.size() / channels_));
    }

private:
    uint32_t index_{0};
    uint32_t channels_;

    std::vector<float> buffer_;
};
