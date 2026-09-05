#include "VHE.h"
#include "../../include/log.h"
#include "VHE_L0.h"
#include "VHE_L1.h"
#include "VHE_L2.h"
#include "VHE_L3.h"
#include "VHE_L4.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <span>

// Per-level, per-rate HRIR descriptor.
// gain/gain_r = original_gain × dequant_scale (precomputed; L and R differ for L2–L4 at 48 kHz).
struct VheKernel {
    const int16_t* left;
    const int16_t* right;
    float          gain;    // left  channel: original_gain × dequant_scale
    float          gain_r;  // right channel: original_gain × dequant_scale
    uint32_t       size;
};

// Worst-case HRIR length across all levels/rates; bounds the Reset() stack buffers.
static constexpr uint32_t kVheMaxKernelSize = 4096u;

static void Dequantize(std::span<const int16_t> src, float scale,
                       std::span<float> dst) noexcept {
    for (uint32_t i = 0; i < static_cast<uint32_t>(src.size()); ++i) {
        dst[i] = static_cast<float>(src[i]) * scale;
    }
}

// kVheKernels[level][0=44100Hz, 1=48000Hz]
static const std::array<std::array<VheKernel, 2>, 5> kVheKernels{{
    {{ { kVheL0_44100_L, kVheL0_44100_R,
         2.94595f * kVheL0_44100_L_Scale, 2.94595f * kVheL0_44100_R_Scale, 4096u },
       { kVheL0_48000_L, kVheL0_48000_R,
         2.94595f * kVheL0_48000_L_Scale, 2.94595f * kVheL0_48000_R_Scale, 4096u } }},
    {{ { kVheL1_44100_L, kVheL1_44100_R,
         0.944061f * kVheL1_44100_L_Scale, 0.944061f * kVheL1_44100_R_Scale, 2047u },
       { kVheL1_48000_L, kVheL1_48000_R,
         0.944061f * kVheL1_48000_L_Scale, 0.944061f * kVheL1_48000_R_Scale, 2047u } }},
    {{ { kVheL2_44100_L, kVheL2_44100_R,
         1.544582f * kVheL2_44100_L_Scale, 1.544582f * kVheL2_44100_R_Scale, 4096u },
       { kVheL2_48000_L, kVheL2_48000_R,
         1.531516f * kVheL2_48000_L_Scale, 1.531516f * kVheL2_48000_R_Scale, 4096u } }},
    {{ { kVheL3_44100_L, kVheL3_44100_R,
         1.584257f * kVheL3_44100_L_Scale, 1.584257f * kVheL3_44100_R_Scale, 4096u },
       { kVheL3_48000_L, kVheL3_48000_R,
         1.567789f * kVheL3_48000_L_Scale, 1.567789f * kVheL3_48000_R_Scale, 4096u } }},
    {{ { kVheL4_44100_L, kVheL4_44100_R,
         1.466681f * kVheL4_44100_L_Scale, 1.466681f * kVheL4_44100_R_Scale, 4096u },
       { kVheL4_48000_L, kVheL4_48000_R,
         1.487227f * kVheL4_48000_L_Scale, 1.487227f * kVheL4_48000_R_Scale, 4096u } }},
}};

VHE::VHE() {
    Reset();
}

uint32_t VHE::Process(const float* source, float* dest, const uint32_t frame_size) {
    if (!enable_ || !conv_left_.InstanceUsable() || !conv_right_.InstanceUsable()
        || frame_size == 0) {
        if (source != dest && frame_size > 0) {
            std::copy_n(source, frame_size * 2u, dest);
        }
        return frame_size;
    }

    if (source != dest) {
        std::copy_n(source, frame_size * 2u, dest);
    }

    conv_left_.ProcessInterleaved(dest, dest, 0u, 2u, frame_size);
    conv_right_.ProcessInterleaved(dest, dest, 1u, 2u, frame_size);

    return frame_size;
}

void VHE::ProcessPlanar(std::span<float> L, std::span<float> R) noexcept {
    if (!enable_ || !conv_left_.InstanceUsable() || !conv_right_.InstanceUsable()
        || L.empty()) {
        return;  // pass-through: planar buffers contain the unmodified input
    }
    const auto frame_size = static_cast<uint32_t>(L.size());

    // PConvZeroLatency::Process() operates on contiguous non-interleaved buffers.
    conv_left_.Process(L.data(), L.data(), frame_size);
    conv_right_.Process(R.data(), R.data(), frame_size);
}

void VHE::Reset() {
    conv_left_.UnloadKernel();
    conv_right_.UnloadKernel();

    if (effect_level_ >= kVheKernels.size()) {
        VIPER_LOGD("VHE: Unsupported effect level %d", effect_level_);
        return;
    }

    int rate_idx = -1;
    if      (sampling_rate_ == 44100u) rate_idx = 0;
    else if (sampling_rate_ == 48000u) rate_idx = 1;
    else {
        VIPER_LOGD("VHE: Unsupported sampling rate %d", sampling_rate_);
        return;
    }

    const auto& k = kVheKernels[effect_level_][static_cast<std::size_t>(rate_idx)];

    // 2 × 16 KiB stack scratch; alignas(64) satisfies PFFFT's SIMD requirement.
    alignas(64) std::array<float, kVheMaxKernelSize> left_buf{};
    alignas(64) std::array<float, kVheMaxKernelSize> right_buf{};
    Dequantize(std::span<const int16_t>{k.left,  k.size}, k.gain,   left_buf);
    Dequantize(std::span<const int16_t>{k.right, k.size}, k.gain_r, right_buf);
    conv_left_.LoadKernel(left_buf.data(),  k.size);
    conv_right_.LoadKernel(right_buf.data(), k.size);
}

void VHE::SetConfig(const Config& config) noexcept {
    config_ = config;
    SetEnable(config.enable);
    SetEffectLevel(config.quality);
}

bool VHE::GetEnable() const noexcept {
    return enable_;
}

void VHE::SetEnable(const bool enable) noexcept {
    config_.enable = enable;
    if (enable_ != enable) {
        enable_ = enable;
        if (enable_) Reset();
    }
}

void VHE::SetEffectLevel(const uint32_t value) noexcept {
    if (effect_level_ != value && value < kVheKernels.size()) {
        effect_level_ = value;
        Reset();
    }
}

void VHE::SetSamplingRate(const uint32_t sampling_rate) noexcept {
    if (sampling_rate_ != sampling_rate) {
        sampling_rate_ = sampling_rate;
        Reset();
    }
}
