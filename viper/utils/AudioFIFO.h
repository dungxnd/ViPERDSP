#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <vector>

/// Single-producer / single-consumer power-of-two circular ring buffer.
/// All operations are allocation-free after Init(). Not thread-safe across
/// simultaneous Read + Write unless external synchronisation is provided;
/// the audio thread here is strictly single-threaded per channel.
class AudioFIFO {
public:
    AudioFIFO() = default;
    explicit AudioFIFO(std::size_t capacity) { Init(capacity); }

    void Init(std::size_t capacity) {
        // Round up to next power of two (minimum 16)
        const std::size_t cap = std::max(std::bit_ceil(capacity), std::size_t{16});
        buffer_.assign(cap, 0.0f);
        mask_      = cap - 1u;
        write_pos_ = 0u;
        read_pos_  = 0u;
    }

    void Reset() noexcept {
        std::ranges::fill(buffer_, 0.0f);
        write_pos_ = 0u;
        read_pos_  = 0u;
    }

    [[nodiscard]] std::size_t AvailableRead() const noexcept {
        return write_pos_ - read_pos_;
    }

    [[nodiscard]] std::size_t AvailableWrite() const noexcept {
        return buffer_.size() - AvailableRead();
    }

    void Write(const float* src, std::size_t count) noexcept {
        for (std::size_t i = 0; i < count; ++i) {
            buffer_[write_pos_++ & mask_] = src[i];
        }
    }

    void WriteZeros(std::size_t count) noexcept {
        for (std::size_t i = 0; i < count; ++i) {
            buffer_[write_pos_++ & mask_] = 0.0f;
        }
    }

    void Read(float* dst, std::size_t count) noexcept {
        for (std::size_t i = 0; i < count; ++i) {
            dst[i] = buffer_[read_pos_++ & mask_];
        }
    }

private:
    std::vector<float> buffer_;
    std::size_t        mask_{0u};
    std::size_t        write_pos_{0u};
    std::size_t        read_pos_{0u};
};
