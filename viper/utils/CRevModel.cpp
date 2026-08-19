#include "CRevModel.h"
#include <algorithm>
#include <array>
#include <numeric>

namespace {

// Freeverb buffer sizes (samples at 44.1 kHz): 8 comb pairs + 4 allpass pairs.
// Layout: [comb_l0, comb_r0, ..., comb_l7, comb_r7, ap_l0, ap_r0, ..., ap_l3, ap_r3]
constexpr std::array<uint32_t, 24> kBufferSizes = {
    1116, 1139, 1188, 1211, 1277, 1300, 1356, 1379,
    1422, 1445, 1491, 1514, 1557, 1580, 1617, 1640,
     556,  579,  441,  464,  341,  364,  225,  248,
};

} // anonymous namespace

CRevModel::CRevModel() {
    // Allocate single contiguous pool for all delay lines.
    const uint32_t total = std::accumulate(kBufferSizes.begin(), kBufferSizes.end(), 0u);
    buffer_pool_ = std::make_unique<float[]>(total);

    // Slice pool into per-filter views.
    uint32_t offset = 0;
    for (uint32_t i = 0; i < 24; ++i) {
        buffers_[i] = buffer_pool_.get() + offset;
        offset     += kBufferSizes[i];
    }

    // Wire comb filters (interleaved L/R: index 2i = left, 2i+1 = right).
    for (uint32_t i = 0; i < kNumCombs; ++i) {
        comb_l_[i].SetBuffer(buffers_[i * 2],     kBufferSizes[i * 2]);
        comb_r_[i].SetBuffer(buffers_[i * 2 + 1], kBufferSizes[i * 2 + 1]);
    }

    // Wire allpass filters (start after 16 comb entries).
    constexpr uint32_t kApOffset = kNumCombs * 2;
    for (uint32_t i = 0; i < kNumAllPass; ++i) {
        allpass_l_[i].SetBuffer(buffers_[kApOffset + i * 2],     kBufferSizes[kApOffset + i * 2]);
        allpass_r_[i].SetBuffer(buffers_[kApOffset + i * 2 + 1], kBufferSizes[kApOffset + i * 2 + 1]);
        allpass_l_[i].SetFeedback(0.5f);
        allpass_r_[i].SetFeedback(0.5f);
    }

    SetWet(0.167f);
    SetRoomSize(0.5f);
    SetDry(0.25f);
    SetDamp(0.5f);
    SetWidth(1.0f);
    Reset();
}

void CRevModel::ProcessReplace(
    float* const buf_l, float* const buf_r, const uint32_t size
) noexcept {
    for (uint32_t idx = 0; idx < size * 2; idx += 2) {
        float out_l = 0.0f;
        float out_r = 0.0f;
        const float input = (buf_l[idx] + buf_r[idx]) * gain_;

        for (uint32_t i = 0; i < kNumCombs; ++i) {
            out_l += comb_l_[i].Process(input);
            out_r += comb_r_[i].Process(input);
        }
        for (uint32_t i = 0; i < kNumAllPass; ++i) {
            out_l = allpass_l_[i].Process(out_l);
            out_r = allpass_r_[i].Process(out_r);
        }

        buf_l[idx] = out_l * wet1_ + out_r * wet2_ + buf_l[idx] * dry_;
        buf_r[idx] = out_r * wet1_ + out_l * wet2_ + buf_r[idx] * dry_;
    }
}

void CRevModel::Mute() const noexcept {
    for (uint32_t i = 0; i < kNumCombs;   ++i) { comb_l_[i].Mute();    comb_r_[i].Mute(); }
    for (uint32_t i = 0; i < kNumAllPass; ++i) { allpass_l_[i].Mute(); allpass_r_[i].Mute(); }
}

void CRevModel::Reset() const noexcept {
    Mute();
}

void CRevModel::SetRoomSize(const float value) noexcept {
    room_size_ = value * 0.28f + 0.7f;
    UpdateCoeffs();
}

void CRevModel::SetDamp(const float value) noexcept {
    damp_ = value * 0.4f;
    UpdateCoeffs();
}

void CRevModel::SetWet(const float value) noexcept {
    wet_ = value * 3.0f;
    UpdateCoeffs();
}

void CRevModel::SetDry(const float value) noexcept {
    dry_ = value * 2.0f;
}

void CRevModel::SetWidth(const float value) noexcept {
    width_ = value;
    UpdateCoeffs();
}

float CRevModel::GetRoomSize() const noexcept { return (room_size_ - 0.7f) / 0.28f; }
float CRevModel::GetDamp()     const noexcept { return damp_ / 0.4f; }
float CRevModel::GetDry()      const noexcept { return dry_  / 2.0f; }
float CRevModel::GetWet()      const noexcept { return wet_  / 3.0f; }
float CRevModel::GetWidth()    const noexcept { return width_; }

void CRevModel::UpdateCoeffs() noexcept {
    wet1_ = wet_ * (width_ * 0.5f + 0.5f);
    wet2_ = wet_ * (1.0f - width_) * 0.5f;

    internal_room_size_ = room_size_;
    internal_damp_      = damp_;
    gain_               = 0.015f;

    for (uint32_t i = 0; i < kNumCombs; ++i) {
        comb_l_[i].SetFeedback(internal_room_size_);
        comb_l_[i].SetDamp(internal_damp_);
        comb_r_[i].SetFeedback(internal_room_size_);
        comb_r_[i].SetDamp(internal_damp_);
    }
}
