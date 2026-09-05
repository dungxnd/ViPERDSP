#pragma once

#include <array>
#include <atomic>
#include <optional>
#include <span>
#include <tuple>
#include <vector>

#include "ViPERParams.h"
#include "core/AudioContext.h"
#include "core/DspPipeline.h"
#include "core/ParamExchange.h"
#include "core/RawParamAdapter.h"
#include "effects/AnalogX.h"
#include "effects/ColorfulMusic.h"
#include "effects/Convolver.h"
#include "effects/Cure.h"
#include "effects/DiffSurround.h"
#include "effects/DynamicEQ.h"
#include "effects/DynamicSystem.h"
#include "effects/FETCompressor.h"
#include "effects/IIRFilter.h"
#include "effects/LUFSTargeting.h"
#include "effects/MultibandCompressor.h"
#include "effects/PlaybackGain.h"
#include "effects/PsychoacousticBass.h"
#include "effects/Reverberation.h"
#include "effects/SoftwareLimiter.h"
#include "effects/SpeakerCorrection.h"
#include "effects/SpectrumExtend.h"
#include "effects/StereoImager.h"
#include "effects/TubeSimulator.h"
#include "effects/VHE.h"
#include "effects/ViPERBass.h"
#include "effects/ViPERBassMono.h"
#include "effects/ViPERClarity.h"
#include "effects/ViPERDDC.h"

class ViPER {
public:
    using Pipeline = viper::core::DspPipeline<
        Convolver, VHE, ViPERDDC, SpectrumExtend, IIRFilter,
        DynamicEQ, ColorfulMusic, StereoImager, DiffSurround,
        PlaybackGain, MultibandCompressor, FETCompressor, DynamicSystem,
        TubeSimulator, PsychoacousticBass, ViPERBass, ViPERBassMono,
        ViPERClarity, Cure, AnalogX, Reverberation, SpeakerCorrection,
        LUFSTargeting
    >;

    ViPER();

    void Process(std::vector<float> &buffer, uint32_t size);
    void DispatchRawParam(
        int param, int val1, int val2, int val3, uint32_t arr_size, signed char *arr
    );

    void RequestEffectsReset();
    void ResetAllEffects();

    void RequestBuffersReset();
    void ResetBuffers();

    // Stage a new parameter snapshot for the RT thread to consume on the
    // next Process() call.  Safe to call from any non-RT thread.
    // Does NOT mutate effect state directly — the RT thread applies the
    // snapshot at the top of Process() via ApplyParamsToEffects().
    void ApplyParams(const viper::ViPERParams &params);
    void ApplyMasterLimiter(const viper::MasterLimiterParams &p);
    void ApplyPlaybackGainControl(const viper::PlaybackGainControlParams &p);
    void ApplyLufs(const viper::LufsParams &p);
    void ApplyFetCompressor(const viper::FetCompressorParams &p);
    void ApplyBass(const viper::BassParams &p);
    void ApplyBassMono(const viper::BassMonoParams &p);
    void ApplyPsychoacousticBass(const viper::PsychoacousticBassParams &p);
    void ApplySpectrumExtension(const viper::SpectrumExtensionParams &p);
    void ApplyEqualizer(const viper::EqualizerParams &p);
    void ApplyConvolver(const viper::ConvolverParams &p);
    void ApplyDdc(const viper::DdcParams &p);
    void ApplyFieldSurround(const viper::FieldSurroundParams &p);
    void ApplyDiffSurround(const viper::DiffSurroundParams &p);
    void ApplyStereoImager(const viper::StereoImagerParams &p);
    void ApplyHeadphoneSurround(const viper::HeadphoneSurroundParams &p);
    void ApplyReverb(const viper::ReverbParams &p);
    void ApplyDynamicSystem(const viper::DynamicSystemParams &p);
    void ApplyClarity(const viper::ClarityParams &p);
    void ApplyCure(const viper::CureParams &p);
    void ApplyTubeSimulator(const viper::TubeSimulatorParams &p);
    void ApplyAnalogX(const viper::AnalogXParams &p);
    void ApplySpeakerCorrection(const viper::SpeakerCorrectionParams &p);
    void ApplyMultibandCompressor(const viper::MultibandCompressorParams &p);
    void ApplyDynamicEq(const viper::DynamicEqParams &p);

    // Typed Convolver kernel loader.
    std::optional<uint32_t> LoadConvolverKernel(
        const float *samples, uint32_t frame_count, uint32_t channels, uint32_t kernel_id
    );

    // Typed wrapper for the kernel-unload path.
    void UnloadConvolverKernel();

    // Typed DDC coefficient loader.
    void LoadDdcCoefficients(
        const viper::BiquadSection *sections44100,
        const viper::BiquadSection *sections48000,
        uint32_t section_count
    );

    const viper::ViPERParams &CurrentParams() const { return last_applied_; }

    [[nodiscard]] uint32_t GetSamplingRate() const { return sampling_rate_; }
    [[nodiscard]] uint64_t GetProcessedFrames() const { return process_frame_count_; }
    [[nodiscard]] uint32_t GetConvolverKernelID() const {
        return pipeline_.Get<Convolver>().GetKernelID();
    }

    template <typename T> [[nodiscard]] T& GetEffect() noexcept { return pipeline_.Get<T>(); }
    template <typename T> [[nodiscard]] const T& GetEffect() const noexcept { return pipeline_.Get<T>(); }

    // Updating the sample rate requires rebuilding all IIR/filter coefficients.
    // RequestEffectsReset() schedules a full ResetAllEffects() on the next
    // Process() call (audio-thread-safe via atomic flag) so we never rebuild
    // coefficients mid-buffer from the IPC thread.
    void SetSamplingRate(const uint32_t rate) {
        if (sampling_rate_ != rate) {
            sampling_rate_ = rate;
            RequestEffectsReset();
        }
    }

    // ── Private helpers ────────────────────────────────────────────────────

    // Apply a parameter snapshot to all effect objects.
    // Called exclusively from the RT audio thread inside Process().
    void ApplyParamsToEffects(const viper::ViPERParams &params);

private:
    std::atomic<bool> pending_effects_reset_{false};
    std::atomic<bool> pending_buffers_reset_{false};

    uint32_t sampling_rate_;
    uint64_t process_frame_count_;

    float frame_scale_;
    float left_pan_;
    float right_pan_;

    // ── Control-plane / data-plane parameter exchange ──────────────────────
    // IPC thread writes via ApplyParams(); RT thread reads at top of Process().
    viper::core::DoubleBufferedState<viper::ViPERParams> param_exchange_;

    // Accumulates incremental DispatchRawParam() mutations so the RT thread
    // always sees a consistent full-state snapshot via param_exchange_.
    viper::ViPERParams staged_params_;

    // Cache-aligned planar scratch block for the end-to-end SoA pipeline.
    // ONE deinterleave at ingress, ONE interleave at egress; all effects run
    // in the planar domain between those two passes.
    viper::core::AudioProcessContext<4096> planar_context_;

    Pipeline pipeline_{};
    std::array<SoftwareLimiter, 2> software_limiters_{};

    // Tracks the last snapshot successfully applied to effects
    viper::ViPERParams last_applied_{};
};
