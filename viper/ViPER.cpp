#include "ViPER.h"
#include "../include/ViPERParams.h"
#include "../include/log.h"
#include "constants.h"
#include "utils/Crc32.h"

#include <algorithm>
#include <cstring>

using namespace viper::params;

constexpr uint32_t kKernelChunkFloats = 2046;

ViPER::ViPER() :
    sampling_rate_(VIPER_DEFAULT_SAMPLING_RATE),
    process_frame_count_(0),
    frame_scale_(1.0f),
    left_pan_(1.0f),
    right_pan_(1.0f) {
    VIPER_LOGI("Welcome to ViPER FX");
    VIPER_LOGI("Current version is %s (%d)", VERSION_NAME, VERSION_CODE);

    pipeline_.SetSamplingRate(sampling_rate_);
    pipeline_.Reset();

    for (auto &software_limiter : software_limiters_) {
        software_limiter.SetSamplingRate(sampling_rate_);
        software_limiter.Reset();
    }
}

void ViPER::Process(std::vector<float> &buffer, const uint32_t size) {
    // planar_context_ is AudioProcessContext<4096>: a host block > 4096 frames
    // would overflow the planar left/right arrays.  Guard against both an
    // undersized buffer and an oversized block.
    if (size == 0 || size > 4096u || buffer.size() < size * 2u) return;

    // ── 1. Pending resets (atomic, RT-safe) ────────────────────────────────
    if (pending_effects_reset_.exchange(false, std::memory_order_acquire)) {
        pending_buffers_reset_.store(false, std::memory_order_relaxed);
        ResetAllEffects();
    } else if (pending_buffers_reset_.exchange(false, std::memory_order_acquire)) {
        ResetBuffers();
    }

    // ── 2. Lock-free parameter ingestion (zero mutex overhead) ────────────
    if (param_exchange_.HasPendingUpdate()) {
        ApplyParamsToEffects(param_exchange_.ReadLatest());
    }

    process_frame_count_ += size;

    float* const raw_io = buffer.data();

    // ── 3. Ingress: single SIMD deinterleave [L0 R0 L1 R1…] → planar ─────
    planar_context_.Deinterleave(raw_io, size);

    float* __restrict L = planar_context_.left.data();
    float* __restrict R = planar_context_.right.data();

    // ── 4. Compact active-stage dispatch — only enabled effects ──────────
    const std::span<float> L_span{L, size};
    const std::span<float> R_span{R, size};
    pipeline_.ProcessPlanar(L_span, R_span);

    // ── 5. Master bus: planar gain+pan (SIMD-friendly, no stride) ─────────
    if (frame_scale_ != 1.0f || left_pan_ < 1.0f || right_pan_ < 1.0f) {
        planar_context_.ApplyGainPan(frame_scale_, left_pan_, right_pan_);
    }

    // ── 6. True-peak limiter (planar domain: stride-1, 100% cache-local) ──
    // Stereo-linked: both channels share one peak detector + gain envelope so
    // a transient on one channel never shifts the stereo image.
    software_limiters_[0].ProcessBlockStereoLinked(L, R, size);

    // ── 7. Egress: single SIMD interleave back to caller's buffer ─────────
    planar_context_.Interleave(raw_io);
}

void ViPER::DispatchRawParam(
    const int param,
    int val1,
    const int val2,
    const int val3,
    const uint32_t arr_size,
    signed char *arr
) {
    const bool should_publish = viper::core::RawParamAdapter::Dispatch(
        param, val1, val2, val3, arr_size, arr,
        staged_params_,
        pipeline_.Get<Convolver>(),
        pipeline_.Get<ViPERDDC>(),
        [this]() { ResetAllEffects(); }
    );
    if (should_publish) {
        param_exchange_.Update(staged_params_);
    }
}

void ViPER::RequestEffectsReset() {
    pending_effects_reset_.store(true, std::memory_order_release);
}

void ViPER::ResetAllEffects() {
    pipeline_.SetSamplingRate(sampling_rate_);
    pipeline_.Reset();
    for (auto &software_limiter : software_limiters_) {
        software_limiter.SetSamplingRate(sampling_rate_);
        software_limiter.Reset();
    }
}

void ViPER::RequestBuffersReset() {
    pending_buffers_reset_.store(true, std::memory_order_release);
}

void ViPER::ResetBuffers() {
    pipeline_.Get<Reverberation>().Reset();
}

void ViPER::ApplyParams(const viper::ViPERParams &params) {
    staged_params_ = params;
    param_exchange_.Update(staged_params_);
}

void ViPER::ApplyParamsToEffects(const viper::ViPERParams &params) {
    if (!(params.master_limiter == last_applied_.master_limiter)) {
        ApplyMasterLimiter(params.master_limiter);
    }
    pipeline_.ApplyParams(params);
    last_applied_ = params;
}

void ViPER::ApplyMasterLimiter(const viper::MasterLimiterParams &p) {
    software_limiters_[0].SetGate(p.threshold);
    software_limiters_[1].SetGate(p.threshold);
    frame_scale_ = p.output_volume;
    if (p.channel_pan < 0.0f) {
        left_pan_ = 1.0f;
        right_pan_ = 1.0f + p.channel_pan;
    } else {
        left_pan_ = 1.0f - p.channel_pan;
        right_pan_ = 1.0f;
    }
    last_applied_.master_limiter = p;
}

template <typename Effect, typename Field>
static void ApplyEffectConfig(ViPER::Pipeline& pipeline, Field& field, const typename Effect::Config& p) {
    pipeline.Get<Effect>().SetConfig(p);
    pipeline.RebuildActiveTopology();
    field = p;
}

void ViPER::ApplyPlaybackGainControl(const viper::PlaybackGainControlParams &p) {
    ApplyEffectConfig<PlaybackGain>(pipeline_, last_applied_.playback_gain_control, p);
}

void ViPER::ApplyLufs(const viper::LufsParams &p) {
    ApplyEffectConfig<LUFSTargeting>(pipeline_, last_applied_.lufs, p);
}

void ViPER::ApplyFetCompressor(const viper::FetCompressorParams &p) {
    ApplyEffectConfig<FETCompressor>(pipeline_, last_applied_.fet_compressor, p);
}

void ViPER::ApplyBass(const viper::BassParams &p) {
    ApplyEffectConfig<ViPERBass>(pipeline_, last_applied_.bass, p);
}

void ViPER::ApplyBassMono(const viper::BassMonoParams &p) {
    ApplyEffectConfig<ViPERBassMono>(pipeline_, last_applied_.bass_mono, p);
}

void ViPER::ApplyPsychoacousticBass(const viper::PsychoacousticBassParams &p) {
    ApplyEffectConfig<PsychoacousticBass>(pipeline_, last_applied_.psychoacoustic_bass, p);
}

void ViPER::ApplySpectrumExtension(const viper::SpectrumExtensionParams &p) {
    ApplyEffectConfig<SpectrumExtend>(pipeline_, last_applied_.spectrum_extension, p);
}

void ViPER::ApplyEqualizer(const viper::EqualizerParams &p) {
    ApplyEffectConfig<IIRFilter>(pipeline_, last_applied_.equalizer, p);
}

void ViPER::ApplyConvolver(const viper::ConvolverParams &p) {
    ApplyEffectConfig<Convolver>(pipeline_, last_applied_.convolver, p);
}

void ViPER::ApplyDdc(const viper::DdcParams &p) {
    ApplyEffectConfig<ViPERDDC>(pipeline_, last_applied_.ddc, p);
}

void ViPER::ApplyFieldSurround(const viper::FieldSurroundParams &p) {
    ApplyEffectConfig<ColorfulMusic>(pipeline_, last_applied_.field_surround, p);
}

void ViPER::ApplyDiffSurround(const viper::DiffSurroundParams &p) {
    ApplyEffectConfig<DiffSurround>(pipeline_, last_applied_.diff_surround, p);
}

void ViPER::ApplyStereoImager(const viper::StereoImagerParams &p) {
    ApplyEffectConfig<StereoImager>(pipeline_, last_applied_.stereo_imager, p);
}

void ViPER::ApplyHeadphoneSurround(const viper::HeadphoneSurroundParams &p) {
    ApplyEffectConfig<VHE>(pipeline_, last_applied_.headphone_surround, p);
}

void ViPER::ApplyReverb(const viper::ReverbParams &p) {
    ApplyEffectConfig<Reverberation>(pipeline_, last_applied_.reverb, p);
}

void ViPER::ApplyDynamicSystem(const viper::DynamicSystemParams &p) {
    ApplyEffectConfig<DynamicSystem>(pipeline_, last_applied_.dynamic_system, p);
}

void ViPER::ApplyClarity(const viper::ClarityParams &p) {
    ApplyEffectConfig<ViPERClarity>(pipeline_, last_applied_.clarity, p);
}

void ViPER::ApplyCure(const viper::CureParams &p) {
    ApplyEffectConfig<Cure>(pipeline_, last_applied_.cure, p);
}

void ViPER::ApplyTubeSimulator(const viper::TubeSimulatorParams &p) {
    ApplyEffectConfig<TubeSimulator>(pipeline_, last_applied_.tube_simulator, p);
}

void ViPER::ApplyAnalogX(const viper::AnalogXParams &p) {
    ApplyEffectConfig<AnalogX>(pipeline_, last_applied_.analog_x, p);
}

void ViPER::ApplySpeakerCorrection(const viper::SpeakerCorrectionParams &p) {
    ApplyEffectConfig<SpeakerCorrection>(pipeline_, last_applied_.speaker_correction, p);
}

void ViPER::ApplyMultibandCompressor(const viper::MultibandCompressorParams &p) {
    ApplyEffectConfig<MultibandCompressor>(pipeline_, last_applied_.multiband_compressor, p);
}

void ViPER::ApplyDynamicEq(const viper::DynamicEqParams &p) {
    ApplyEffectConfig<DynamicEQ>(pipeline_, last_applied_.dynamic_eq, p);
}

std::optional<uint32_t> ViPER::LoadConvolverKernel(
    const float *samples,
    const uint32_t frame_count,
    const uint32_t channels,
    uint32_t kernel_id
) {
    if (samples == nullptr) return std::nullopt;
    if (channels < 1 || channels > 2) return std::nullopt;
    if (frame_count < 16) return std::nullopt;

    const uint32_t total_floats = frame_count * channels;
    if (total_floats == 0) return std::nullopt;

    auto& convolver = pipeline_.Get<Convolver>();
    convolver.PrepareKernelBuffer(total_floats, channels, false);

    uint32_t written = 0;
    while (written < total_floats) {
        const uint32_t remaining = total_floats - written;
        const uint32_t chunk =
            remaining < kKernelChunkFloats ? remaining : kKernelChunkFloats;
        convolver.SetKernelBuffer(samples + written, chunk);
        written += chunk;
    }

    const uint32_t crc =
        Crc32(reinterpret_cast<const uint8_t *>(samples), total_floats * sizeof(float));

    convolver.CommitKernelBuffer(total_floats, crc, kernel_id);

    if (convolver.GetKernelID() != kernel_id) {
        return std::nullopt;
    }
    return kernel_id;
}

void ViPER::UnloadConvolverKernel() {
    pipeline_.Get<Convolver>().PrepareKernelBuffer(0, 0, true);
}

void ViPER::LoadDdcCoefficients(
    const viper::BiquadSection *sections44100,
    const viper::BiquadSection *sections48000,
    const uint32_t section_count
) {
    auto& ddc = pipeline_.Get<ViPERDDC>();
    if (section_count == 0) {
        ddc.SetCoeffs(0, nullptr, nullptr);
        return;
    }

    static_assert(
        sizeof(viper::BiquadSection) == 5 * sizeof(float),
        "BiquadSection must be tightly packed for reinterpret_cast"
    );
    const uint32_t total_floats = section_count * 5;
    ddc.SetCoeffs(
        total_floats,
        reinterpret_cast<const float *>(sections44100),
        reinterpret_cast<const float *>(sections48000)
    );
}
