#pragma once

#include <array>
#include <concepts>
#include <cstdint>
#include <span>
#include <tuple>
#include <utility>

#include "AudioContext.h"
#include "ViPERParams.h"

namespace viper::core {

template <typename Config>
inline const Config& GetParamSubstruct(const viper::ViPERParams& p);

template <> inline const viper::ConvolverParams& GetParamSubstruct<viper::ConvolverParams>(const viper::ViPERParams& p) { return p.convolver; }
template <> inline const viper::HeadphoneSurroundParams& GetParamSubstruct<viper::HeadphoneSurroundParams>(const viper::ViPERParams& p) { return p.headphone_surround; }
template <> inline const viper::DdcParams& GetParamSubstruct<viper::DdcParams>(const viper::ViPERParams& p) { return p.ddc; }
template <> inline const viper::SpectrumExtensionParams& GetParamSubstruct<viper::SpectrumExtensionParams>(const viper::ViPERParams& p) { return p.spectrum_extension; }
template <> inline const viper::EqualizerParams& GetParamSubstruct<viper::EqualizerParams>(const viper::ViPERParams& p) { return p.equalizer; }
template <> inline const viper::DynamicEqParams& GetParamSubstruct<viper::DynamicEqParams>(const viper::ViPERParams& p) { return p.dynamic_eq; }
template <> inline const viper::FieldSurroundParams& GetParamSubstruct<viper::FieldSurroundParams>(const viper::ViPERParams& p) { return p.field_surround; }
template <> inline const viper::StereoImagerParams& GetParamSubstruct<viper::StereoImagerParams>(const viper::ViPERParams& p) { return p.stereo_imager; }
template <> inline const viper::DiffSurroundParams& GetParamSubstruct<viper::DiffSurroundParams>(const viper::ViPERParams& p) { return p.diff_surround; }
template <> inline const viper::PlaybackGainControlParams& GetParamSubstruct<viper::PlaybackGainControlParams>(const viper::ViPERParams& p) { return p.playback_gain_control; }
template <> inline const viper::MultibandCompressorParams& GetParamSubstruct<viper::MultibandCompressorParams>(const viper::ViPERParams& p) { return p.multiband_compressor; }
template <> inline const viper::FetCompressorParams& GetParamSubstruct<viper::FetCompressorParams>(const viper::ViPERParams& p) { return p.fet_compressor; }
template <> inline const viper::DynamicSystemParams& GetParamSubstruct<viper::DynamicSystemParams>(const viper::ViPERParams& p) { return p.dynamic_system; }
template <> inline const viper::TubeSimulatorParams& GetParamSubstruct<viper::TubeSimulatorParams>(const viper::ViPERParams& p) { return p.tube_simulator; }
template <> inline const viper::PsychoacousticBassParams& GetParamSubstruct<viper::PsychoacousticBassParams>(const viper::ViPERParams& p) { return p.psychoacoustic_bass; }
template <> inline const viper::BassParams& GetParamSubstruct<viper::BassParams>(const viper::ViPERParams& p) { return p.bass; }
template <> inline const viper::BassMonoParams& GetParamSubstruct<viper::BassMonoParams>(const viper::ViPERParams& p) { return p.bass_mono; }
template <> inline const viper::ClarityParams& GetParamSubstruct<viper::ClarityParams>(const viper::ViPERParams& p) { return p.clarity; }
template <> inline const viper::CureParams& GetParamSubstruct<viper::CureParams>(const viper::ViPERParams& p) { return p.cure; }
template <> inline const viper::AnalogXParams& GetParamSubstruct<viper::AnalogXParams>(const viper::ViPERParams& p) { return p.analog_x; }
template <> inline const viper::ReverbParams& GetParamSubstruct<viper::ReverbParams>(const viper::ViPERParams& p) { return p.reverb; }
template <> inline const viper::SpeakerCorrectionParams& GetParamSubstruct<viper::SpeakerCorrectionParams>(const viper::ViPERParams& p) { return p.speaker_correction; }
template <> inline const viper::LufsParams& GetParamSubstruct<viper::LufsParams>(const viper::ViPERParams& p) { return p.lufs; }

template <typename Effect>
inline void ApplyConfigIfChanged(Effect& effect, const viper::ViPERParams& params) noexcept {
    if constexpr (requires { typename Effect::Config; }) {
        using Config = typename Effect::Config;
        const Config& new_config = GetParamSubstruct<Config>(params);
        if (!(effect.GetConfig() == new_config)) {
            effect.SetConfig(new_config);
        }
    }
}

template <typename... Effects>
class DspPipeline {
public:
    static constexpr size_t kMaxStages = sizeof...(Effects);

    DspPipeline() {
        RebuildActiveTopology();
    }

    template <typename T>
    [[nodiscard]] T& Get() noexcept {
        return std::get<T>(effects_);
    }

    template <typename T>
    [[nodiscard]] const T& Get() const noexcept {
        return std::get<T>(effects_);
    }

    void SetSamplingRate(uint32_t sampling_rate) noexcept {
        std::apply([sampling_rate](auto&... e) {
            (e.SetSamplingRate(sampling_rate), ...);
        }, effects_);
        RebuildActiveTopology();
    }

    void Reset() noexcept {
        std::apply([](auto&... e) {
            (e.Reset(), ...);
        }, effects_);
        RebuildActiveTopology();
    }

    void ApplyParams(const viper::ViPERParams& params) noexcept {
        std::apply([&](auto&... e) {
            (ApplyConfigIfChanged(e, params), ...);
        }, effects_);
        RebuildActiveTopology();
    }

    void RebuildActiveTopology() noexcept {
        active_stage_count_ = 0u;
        std::apply([this](auto&... e) {
            (AddStageIfEnabled(e), ...);
        }, effects_);
    }

    void ProcessPlanar(std::span<float> L, std::span<float> R) noexcept {
        for (size_t i = 0u; i < active_stage_count_; ++i) {
            active_stages_[i].process_fn(active_stages_[i].instance, L, R);
        }
    }

    [[nodiscard]] size_t GetActiveStageCount() const noexcept {
        return active_stage_count_;
    }

private:
    struct PipelineStage {
        void* instance{nullptr};
        void (*process_fn)(void* instance, std::span<float> L, std::span<float> R) noexcept {nullptr};
    };

    template <typename T>
    static void InvokeStage(void* inst, std::span<float> L, std::span<float> R) noexcept {
        static_cast<T*>(inst)->ProcessPlanar(L, R);
    }

    template <typename T>
    void AddStageIfEnabled(T& effect) noexcept {
        if (effect.IsEnabled()) {
            active_stages_[active_stage_count_++] = PipelineStage{
                .instance   = &effect,
                .process_fn = &InvokeStage<T>
            };
        }
    }

    std::tuple<Effects...> effects_{};
    std::array<PipelineStage, kMaxStages> active_stages_{};
    size_t active_stage_count_{0u};
};

} // namespace viper::core
