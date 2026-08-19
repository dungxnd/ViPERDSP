#include "CAllPassFilter.h"
#include <algorithm>

float CAllPassFilter::Process(const float sample) noexcept {
    const float out       = buffer_[buffer_index_];
    buffer_[buffer_index_] = sample + out * feedback_;
    if (++buffer_index_ >= buffer_size_) buffer_index_ = 0;
    return out - sample;
}

void CAllPassFilter::Mute() const noexcept {
    std::fill_n(buffer_, buffer_size_, 0.0f);
}

void CAllPassFilter::SetBuffer(float* const buffer, const uint32_t size) noexcept {
    buffer_       = buffer;
    buffer_size_  = size;
    buffer_index_ = 0;
}

void CAllPassFilter::SetFeedback(const float value) noexcept {
    feedback_ = value;
}

float CAllPassFilter::GetFeedback() const noexcept {
    return feedback_;
}
