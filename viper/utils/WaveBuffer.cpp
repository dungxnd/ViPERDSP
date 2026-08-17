#include "WaveBuffer.h"
#include <algorithm>
#include <cstdint>

WaveBuffer::WaveBuffer(const uint32_t channels, const uint32_t length)
    : channels_(channels), buffer_(static_cast<std::size_t>(length) * channels) {}

void WaveBuffer::Reset() noexcept {
    index_ = 0;
    std::fill(buffer_.begin(), buffer_.end(), 0.0f);
}

uint32_t WaveBuffer::GetBufferOffset() const noexcept {
    return index_ / channels_;
}

uint32_t WaveBuffer::GetBufferSize() const noexcept {
    return static_cast<uint32_t>(buffer_.size()) / channels_;
}

float *WaveBuffer::GetBuffer() noexcept {
    return buffer_.data();
}

void WaveBuffer::SetBufferOffset(const uint32_t offset) noexcept {
    const uint32_t max_offset = static_cast<uint32_t>(buffer_.size()) / channels_;
    if (offset <= max_offset) {
        index_ = offset * channels_;
    }
}

uint32_t WaveBuffer::PopSamples(const uint32_t size, const bool reset_idx) noexcept {
    if (buffer_.empty()) return 0;

    const uint32_t needed = channels_ * size;
    if (needed <= index_) {
        index_ -= needed;
        std::copy_n(buffer_.data() + needed, index_, buffer_.data());
        return size;
    }

    if (reset_idx) {
        const uint32_t ret = index_ / channels_;
        index_ = 0;
        return ret;
    }

    return 0;
}

uint32_t WaveBuffer::PopSamples(float *dest, const uint32_t size, const bool reset_idx) noexcept {
    if (buffer_.empty() || dest == nullptr) return 0;

    const uint32_t needed = channels_ * size;
    if (needed <= index_) {
        std::copy_n(buffer_.data(), needed, dest);
        index_ -= needed;
        std::copy_n(buffer_.data() + needed, index_, buffer_.data());
        return size;
    }

    if (reset_idx) {
        const uint32_t ret = index_ / channels_;
        std::copy_n(buffer_.data(), index_, dest);
        index_ = 0;
        return ret;
    }

    return 0;
}

int WaveBuffer::PushSamples(const float *source, const uint32_t size) {
    if (size > 0) {
        const std::size_t required = static_cast<std::size_t>(channels_) * size + index_;
        if (required > buffer_.size()) buffer_.resize(required);
        std::copy_n(source, channels_ * size, buffer_.data() + index_);
        index_ += channels_ * size;
    }
    return 1;
}

int WaveBuffer::PushZeros(const uint32_t size) {
    if (size > 0) {
        const std::size_t required = static_cast<std::size_t>(channels_) * size + index_;
        if (required > buffer_.size()) buffer_.resize(required);
        std::fill_n(buffer_.data() + index_, channels_ * size, 0.0f);
        index_ += channels_ * size;
    }
    return 1;
}

float *WaveBuffer::PushZerosGetBuffer(const uint32_t size) {
    const uint32_t old_idx = index_;
    if (size > 0) {
        const std::size_t required = static_cast<std::size_t>(channels_) * size + index_;
        if (required > buffer_.size()) buffer_.resize(required);
        std::fill_n(buffer_.data() + index_, channels_ * size, 0.0f);
        index_ += channels_ * size;
    }
    return buffer_.data() + old_idx;
}
