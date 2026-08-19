#include "AdaptiveBuffer.h"
#include <algorithm>

AdaptiveBuffer::AdaptiveBuffer(const uint32_t channels, const uint32_t length)
    : length_(length), channels_(channels), buffer_(channels * length) {}

uint32_t AdaptiveBuffer::GetBufferLength() const noexcept { return length_;   }
uint32_t AdaptiveBuffer::GetBufferOffset() const noexcept { return offset_;   }
uint32_t AdaptiveBuffer::GetChannels()     const noexcept { return channels_; }
float*   AdaptiveBuffer::GetBuffer()       noexcept       { return buffer_.data(); }

void AdaptiveBuffer::SetBufferOffset(const uint32_t value) noexcept {
    offset_ = value;
}

void AdaptiveBuffer::PanFrames(const float left, const float right) noexcept {
    if (channels_ != 2) return;
    for (uint32_t i = 0; i < offset_ * channels_; i += 2) {
        buffer_[i]     *= left;
        buffer_[i + 1] *= right;
    }
}

int AdaptiveBuffer::PopFrames(float* frames, const uint32_t length) noexcept {
    if (offset_ < length) return 0;

    if (length != 0) {
        std::copy(buffer_.begin(), buffer_.begin() + length * channels_, frames);
        offset_ -= length;
        if (offset_ != 0) {
            std::move(buffer_.begin() + length * channels_,
                      buffer_.begin() + (length + offset_) * channels_,
                      buffer_.begin());
        }
    }

    return 1;
}

int AdaptiveBuffer::PushFrames(const float* frames, const uint32_t length) {
    if (length != 0) {
        if (offset_ + length > length_) {
            buffer_.resize((offset_ + length) * channels_);
            length_ = offset_ + length;
        }
        std::copy(frames, frames + length * channels_,
                  buffer_.begin() + offset_ * channels_);
        offset_ += length;
    }
    return 1;
}

void AdaptiveBuffer::ScaleFrames(const float scale) noexcept {
    for (uint32_t i = 0; i < offset_ * channels_; ++i) {
        buffer_[i] *= scale;
    }
}

int AdaptiveBuffer::PushZero(const uint32_t length) {
    if (offset_ + length > length_) {
        buffer_.resize((offset_ + length) * channels_);
        length_ = offset_ + length;
    }
    std::fill(buffer_.begin() + offset_ * channels_,
              buffer_.begin() + (offset_ + length) * channels_, 0.0f);
    offset_ += length;
    return 1;
}

void AdaptiveBuffer::FlushBuffer() noexcept {
    offset_ = 0;
    std::fill(buffer_.begin(), buffer_.end(), 0.0f);
}
