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

// Kernel descriptor: int16_t pointer pair + effective gain + tap count.
// gain already incorporates the per-array dequantisation scale so that
// LoadKernel receives the same effective float coefficients as before:
//   effective_gain = original_gain * array_scale
struct VheKernel {
    const int16_t *left;
    const int16_t *right;
    float          gain;        // original_gain * dequant_scale (per L/R)
    float          gain_r;      // right channel may differ (L2/L3/L4 48kHz)
    uint32_t       size;
};

// Maximum HRIR tap count across all levels / sample rates.
// Used to size the stack-allocated dequantisation buffers in Reset().
// (Checked against kVheKernels: all entries are 4096 or 2047.)
static constexpr uint32_t kVheMaxKernelSize = 4096u;

// Dequantize int16 kernel into a caller-supplied float buffer.
// n must be <= kVheMaxKernelSize.
static void Dequantize(const int16_t* src, uint32_t n, float scale,
                       float* dst) noexcept {
    for (uint32_t i = 0; i < n; ++i) {
        dst[i] = static_cast<float>(src[i]) * scale;
    }
}

// Indexed by [effect_level][0=44100, 1=48000].
// gain  = original_gain * left_array_Scale
// gain_r= original_gain * right_array_Scale
// When L and R use the same scale (L0, L1) gain_r == gain.
static const std::array<std::array<VheKernel, 2>, 5> kVheKernels{{
    // level 0  (L_Scale == R_Scale for both rates)
    {{ { kVheL0_44100_L, kVheL0_44100_R,
         2.94595f * kVheL0_44100_L_Scale, 2.94595f * kVheL0_44100_R_Scale, 4096u },
       { kVheL0_48000_L, kVheL0_48000_R,
         2.94595f * kVheL0_48000_L_Scale, 2.94595f * kVheL0_48000_R_Scale, 4096u } }},
    // level 1
    {{ { kVheL1_44100_L, kVheL1_44100_R,
         0.944061f * kVheL1_44100_L_Scale, 0.944061f * kVheL1_44100_R_Scale, 2047u },
       { kVheL1_48000_L, kVheL1_48000_R,
         0.944061f * kVheL1_48000_L_Scale, 0.944061f * kVheL1_48000_R_Scale, 2047u } }},
    // level 2
    {{ { kVheL2_44100_L, kVheL2_44100_R,
         1.544582f * kVheL2_44100_L_Scale, 1.544582f * kVheL2_44100_R_Scale, 4096u },
       { kVheL2_48000_L, kVheL2_48000_R,
         1.531516f * kVheL2_48000_L_Scale, 1.531516f * kVheL2_48000_R_Scale, 4096u } }},
    // level 3
    {{ { kVheL3_44100_L, kVheL3_44100_R,
         1.584257f * kVheL3_44100_L_Scale, 1.584257f * kVheL3_44100_R_Scale, 4096u },
       { kVheL3_48000_L, kVheL3_48000_R,
         1.567789f * kVheL3_48000_L_Scale, 1.567789f * kVheL3_48000_R_Scale, 4096u } }},
    // level 4
    {{ { kVheL4_44100_L, kVheL4_44100_R,
         1.466681f * kVheL4_44100_L_Scale, 1.466681f * kVheL4_44100_R_Scale, 4096u },
       { kVheL4_48000_L, kVheL4_48000_R,
         1.487227f * kVheL4_48000_L_Scale, 1.487227f * kVheL4_48000_R_Scale, 4096u } }},
}};

VHE::VHE() {
    Reset();
}

uint32_t VHE::Process(const float *source, float *dest, const uint32_t frame_size) {
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

    // Stack-allocated dequantisation buffers — no heap allocation on the
    // audio thread.  kVheMaxKernelSize = 4096 → 2 × 16 KB on the stack,
    // well within Android RT thread stack limits (typically ≥ 1 MB).
    alignas(64) float left_buf[kVheMaxKernelSize];
    alignas(64) float right_buf[kVheMaxKernelSize];
    Dequantize(k.left,  k.size, k.gain,   left_buf);
    Dequantize(k.right, k.size, k.gain_r, right_buf);
    conv_left_.LoadKernel(left_buf,  k.size);
    conv_right_.LoadKernel(right_buf, k.size);
}

bool VHE::GetEnable() const noexcept {
    return enable_;
}

void VHE::SetEnable(const bool enable) noexcept {
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
