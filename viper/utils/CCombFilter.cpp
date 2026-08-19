#include "CCombFilter.h"
#include <algorithm>

float CCombFilter::Process(const float sample) noexcept {
    const float out        = buffer_[buffer_index_];
    filter_store_          = out * damp2_ + filter_store_ * damp_;
    buffer_[buffer_index_] = sample + filter_store_ * feedback_;
    buffer_index_          = (buffer_index_ + 1) % buffer_size_;
    return out;
}

void CCombFilter::Mute() const noexcept {
    std::fill_n(buffer_, buffer_size_, 0.0f);
}

void CCombFilter::SetBuffer(float* const buffer, const uint32_t size) noexcept {
    buffer_       = buffer;
    buffer_size_  = size;
    buffer_index_ = 0;
}

void CCombFilter::SetDamp(const float value) noexcept {
    damp_  = value;
    damp2_ = 1.0f - value;
}

void CCombFilter::SetFeedback(const float value) noexcept {
    feedback_ = value;
}

float CCombFilter::GetDamp() const noexcept {
    return damp_;
}

float CCombFilter::GetFeedback() const noexcept {
    return feedback_;
}
