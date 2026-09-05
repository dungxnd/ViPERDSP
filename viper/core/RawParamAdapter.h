#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

#include "ViPERParams.h"
#include "log.h"
#include "../effects/Convolver.h"
#include "../effects/ViPERDDC.h"

namespace viper::core {

enum class ParamType : uint8_t {
    Bool,
    Int32,
    UInt32,
    Int16,
    FloatDirect,
    FloatDiv100,
    FloatDiv10,
    FloatDivNeg10,
};

struct ParamBinding {
    int param_id;
    ParamType type;
    uint32_t offset;
};

#define VIPER_BIND(id, type, member) \
    ParamBinding{ id, ParamType::type, static_cast<uint32_t>(offsetof(viper::ViPERParams, member)) }

inline constexpr std::array kParamBindings = []() consteval {
    std::array bindings{
        VIPER_BIND(params::kParamMasterLimiterThreshold,        FloatDiv100, master_limiter.threshold),
        VIPER_BIND(params::kParamMasterLimiterOutputVolume,     FloatDiv100, master_limiter.output_volume),
        VIPER_BIND(params::kParamMasterLimiterChannelPan,        FloatDiv100, master_limiter.channel_pan),

        VIPER_BIND(params::kParamPlaybackGainControlEnable,         Bool,        playback_gain_control.enable),
        VIPER_BIND(params::kParamPlaybackGainControlStrength,       FloatDiv100, playback_gain_control.strength),
        VIPER_BIND(params::kParamPlaybackGainControlMaxGain,        FloatDiv100, playback_gain_control.max_gain),
        VIPER_BIND(params::kParamPlaybackGainControlOutputThreshold,FloatDiv100, playback_gain_control.output_threshold),

        VIPER_BIND(params::kParamLufsEnable,  Bool,          lufs.enable),
        VIPER_BIND(params::kParamLufsTarget,  FloatDivNeg10, lufs.target),
        VIPER_BIND(params::kParamLufsMaxGain, FloatDiv10,    lufs.max_gain),
        VIPER_BIND(params::kParamLufsSpeed,   Int32,         lufs.speed),

        VIPER_BIND(params::kParamFetCompressorEnable,      Bool,        fet_compressor.enable),
        VIPER_BIND(params::kParamFetCompressorThreshold,   FloatDiv100, fet_compressor.threshold),
        VIPER_BIND(params::kParamFetCompressorRatio,       FloatDiv100, fet_compressor.ratio),
        VIPER_BIND(params::kParamFetCompressorKnee,        FloatDiv100, fet_compressor.knee),
        VIPER_BIND(params::kParamFetCompressorKneeAuto,    Bool,        fet_compressor.knee_auto),
        VIPER_BIND(params::kParamFetCompressorGain,        FloatDiv100, fet_compressor.gain),
        VIPER_BIND(params::kParamFetCompressorGainAuto,    Bool,        fet_compressor.gain_auto),
        VIPER_BIND(params::kParamFetCompressorAttack,      FloatDiv100, fet_compressor.attack),
        VIPER_BIND(params::kParamFetCompressorAttackAuto,  Bool,        fet_compressor.attack_auto),
        VIPER_BIND(params::kParamFetCompressorRelease,     FloatDiv100, fet_compressor.release),
        VIPER_BIND(params::kParamFetCompressorReleaseAuto, Bool,        fet_compressor.release_auto),
        VIPER_BIND(params::kParamFetCompressorKneeMulti,   FloatDiv100, fet_compressor.knee_multi),
        VIPER_BIND(params::kParamFetCompressorMaxAttack,   FloatDiv100, fet_compressor.max_attack),
        VIPER_BIND(params::kParamFetCompressorMaxRelease,  FloatDiv100, fet_compressor.max_release),
        VIPER_BIND(params::kParamFetCompressorCrest,       FloatDiv100, fet_compressor.crest),
        VIPER_BIND(params::kParamFetCompressorAdapt,       FloatDiv100, fet_compressor.adapt),
        VIPER_BIND(params::kParamFetCompressorNoClip,      Bool,        fet_compressor.no_clip),

        VIPER_BIND(params::kParamBassEnable,    Bool,        bass.enable),
        VIPER_BIND(params::kParamBassMode,      Int32,       bass.mode),
        VIPER_BIND(params::kParamBassFrequency, UInt32,      bass.frequency),
        VIPER_BIND(params::kParamBassGain,      FloatDiv100, bass.gain),
        VIPER_BIND(params::kParamBassAntiPop,   Bool,        bass.anti_pop),

        VIPER_BIND(params::kParamBassMonoEnable,    Bool,        bass_mono.enable),
        VIPER_BIND(params::kParamBassMonoMode,      Int32,       bass_mono.mode),
        VIPER_BIND(params::kParamBassMonoFrequency, UInt32,      bass_mono.frequency),
        VIPER_BIND(params::kParamBassMonoGain,      FloatDiv100, bass_mono.gain),
        VIPER_BIND(params::kParamBassMonoAntiPop,   Bool,        bass_mono.anti_pop),

        VIPER_BIND(params::kParamPsychoacousticBassEnable,        Bool,   psychoacoustic_bass.enable),
        VIPER_BIND(params::kParamPsychoacousticBassCutoff,        UInt32, psychoacoustic_bass.cutoff),
        VIPER_BIND(params::kParamPsychoacousticBassIntensity,     UInt32, psychoacoustic_bass.intensity),
        VIPER_BIND(params::kParamPsychoacousticBassHarmonicOrder, UInt32, psychoacoustic_bass.harmonic_order),
        VIPER_BIND(params::kParamPsychoacousticBassOriginalLevel, UInt32, psychoacoustic_bass.original_level),

        VIPER_BIND(params::kParamSpectrumExtensionEnable,   Bool,        spectrum_extension.enable),
        VIPER_BIND(params::kParamSpectrumExtensionStrength, Int32,       spectrum_extension.strength),
        VIPER_BIND(params::kParamSpectrumExtensionExciter,  FloatDiv100, spectrum_extension.exciter),

        VIPER_BIND(params::kParamEqualizerEnable,    Bool,   equalizer.enable),
        VIPER_BIND(params::kParamEqualizerBandCount, UInt32, equalizer.band_count),

        VIPER_BIND(params::kParamConvolverEnable,       Bool,        convolver.enable),
        VIPER_BIND(params::kParamConvolverCrossChannel, FloatDiv100, convolver.cross_channel),

        VIPER_BIND(params::kParamDdcEnable, Bool, ddc.enable),

        VIPER_BIND(params::kParamFieldSurroundEnable,   Bool,        field_surround.enable),
        VIPER_BIND(params::kParamFieldSurroundWidening, FloatDiv100, field_surround.widening),
        VIPER_BIND(params::kParamFieldSurroundMidImage, FloatDiv100, field_surround.mid_image),
        VIPER_BIND(params::kParamFieldSurroundDepth,    Int16,       field_surround.depth),

        VIPER_BIND(params::kParamDiffSurroundEnable,    Bool,        diff_surround.enable),
        VIPER_BIND(params::kParamDiffSurroundDelay,     FloatDiv100, diff_surround.delay),
        VIPER_BIND(params::kParamDiffSurroundReverse,   Bool,        diff_surround.reverse),
        VIPER_BIND(params::kParamDiffSurroundWetDryMix, FloatDiv100, diff_surround.wet_dry_mix),
        VIPER_BIND(params::kParamDiffSurroundLpCutoff,  FloatDirect, diff_surround.lp_cutoff),

        VIPER_BIND(params::kParamStereoImagerEnable,        Bool,        stereo_imager.enable),
        VIPER_BIND(params::kParamStereoImagerLowWidth,      FloatDirect, stereo_imager.low_width),
        VIPER_BIND(params::kParamStereoImagerMidWidth,      FloatDirect, stereo_imager.mid_width),
        VIPER_BIND(params::kParamStereoImagerHighWidth,     FloatDirect, stereo_imager.high_width),
        VIPER_BIND(params::kParamStereoImagerLowCrossover,  FloatDirect, stereo_imager.low_crossover),
        VIPER_BIND(params::kParamStereoImagerHighCrossover, FloatDirect, stereo_imager.high_crossover),

        VIPER_BIND(params::kParamHeadphoneSurroundEnable,  Bool,  headphone_surround.enable),
        VIPER_BIND(params::kParamHeadphoneSurroundQuality, Int32, headphone_surround.quality),

        VIPER_BIND(params::kParamReverbEnable,   Bool,        reverb.enable),
        VIPER_BIND(params::kParamReverbRoomSize, FloatDiv100, reverb.room_size),
        VIPER_BIND(params::kParamReverbWidth,    FloatDiv100, reverb.width),
        VIPER_BIND(params::kParamReverbDamp,     FloatDiv100, reverb.damp),
        VIPER_BIND(params::kParamReverbWet,      FloatDiv100, reverb.wet),
        VIPER_BIND(params::kParamReverbDry,      FloatDiv100, reverb.dry),

        VIPER_BIND(params::kParamDynamicSystemEnable,       Bool,        dynamic_system.enable),
        VIPER_BIND(params::kParamDynamicSystemXLow,         Int32,       dynamic_system.x_coeff_low),
        VIPER_BIND(params::kParamDynamicSystemXHigh,        Int32,       dynamic_system.x_coeff_high),
        VIPER_BIND(params::kParamDynamicSystemYLow,         Int32,       dynamic_system.y_coeff_low),
        VIPER_BIND(params::kParamDynamicSystemYHigh,        Int32,       dynamic_system.y_coeff_high),
        VIPER_BIND(params::kParamDynamicSystemSideGainLow,  FloatDiv100, dynamic_system.side_gain_low),
        VIPER_BIND(params::kParamDynamicSystemSideGainHigh, FloatDiv100, dynamic_system.side_gain_high),
        VIPER_BIND(params::kParamDynamicSystemStrength,     FloatDiv100, dynamic_system.strength),

        VIPER_BIND(params::kParamClarityEnable, Bool,        clarity.enable),
        VIPER_BIND(params::kParamClarityMode,   Int32,       clarity.mode),
        VIPER_BIND(params::kParamClarityGain,   FloatDiv100, clarity.gain),

        VIPER_BIND(params::kParamCureEnable,           Bool,  cure.enable),
        VIPER_BIND(params::kParamCureCrossfeedPreset,  Int32, cure.crossfeed_preset),

        VIPER_BIND(params::kParamTubeSimulatorEnable,    Bool,        tube_simulator.enable),
        VIPER_BIND(params::kParamTubeSimulatorModel,     Int32,       tube_simulator.model),
        VIPER_BIND(params::kParamTubeSimulatorDrive,     FloatDiv100, tube_simulator.drive),
        VIPER_BIND(params::kParamTubeSimulatorMix,       FloatDiv100, tube_simulator.mix),
        VIPER_BIND(params::kParamTubeSimulatorHpfCutoff, FloatDirect, tube_simulator.hpf_cutoff),
        VIPER_BIND(params::kParamTubeSimulatorMode,      Int32,       tube_simulator.mode),

        VIPER_BIND(params::kParamAnalogXEnable, Bool,  analog_x.enable),
        VIPER_BIND(params::kParamAnalogXMode,   Int32, analog_x.mode),

        VIPER_BIND(params::kParamSpeakerCorrectionEnable, Bool, speaker_correction.enable),

        VIPER_BIND(params::kParamMultibandCompressorEnable,    Bool,   multiband_compressor.enable),
        VIPER_BIND(params::kParamMultibandCompressorBandCount, UInt32, multiband_compressor.band_count),

        VIPER_BIND(params::kParamDynamicEqEnable,    Bool,   dynamic_eq.enable),
        VIPER_BIND(params::kParamDynamicEqBandCount, UInt32, dynamic_eq.band_count),
    };
    std::sort(bindings.begin(), bindings.end(), [](const auto& a, const auto& b) {
        return a.param_id < b.param_id;
    });
    return bindings;
}();

#undef VIPER_BIND

class RawParamAdapter {
public:
    template <typename ResetFn>
    static bool Dispatch(
        int param, int val1, int val2, int val3,
        uint32_t arr_size, signed char* arr,
        viper::ViPERParams& staged,
        Convolver& convolver,
        ViPERDDC& ddc,
        ResetFn&& reset_effects
    ) {
        auto it = std::lower_bound(
            kParamBindings.begin(), kParamBindings.end(), param,
            [](const ParamBinding& b, int id) { return b.param_id < id; }
        );
        if (it != kParamBindings.end() && it->param_id == param) {
            char* target = reinterpret_cast<char*>(&staged) + it->offset;
            switch (it->type) {
                case ParamType::Bool:
                    *reinterpret_cast<bool*>(target) = (val1 != 0);
                    break;
                case ParamType::Int32:
                    *reinterpret_cast<int32_t*>(target) = val1;
                    break;
                case ParamType::UInt32:
                    *reinterpret_cast<uint32_t*>(target) = static_cast<uint32_t>(val1);
                    break;
                case ParamType::Int16:
                    *reinterpret_cast<int16_t*>(target) = static_cast<int16_t>(val1);
                    break;
                case ParamType::FloatDirect:
                    *reinterpret_cast<float*>(target) = static_cast<float>(val1);
                    break;
                case ParamType::FloatDiv100:
                    *reinterpret_cast<float*>(target) = static_cast<float>(val1) / 100.0f;
                    break;
                case ParamType::FloatDiv10:
                    *reinterpret_cast<float*>(target) = static_cast<float>(val1) / 10.0f;
                    break;
                case ParamType::FloatDivNeg10:
                    *reinterpret_cast<float*>(target) = static_cast<float>(val1) / -10.0f;
                    break;
            }
            return true;
        }

        return HandleSpecialParam(
            param, val1, val2, val3, arr_size, arr, staged, convolver, ddc, reset_effects
        );
    }

private:
    template <typename ResetFn>
    static bool HandleSpecialParam(
        int param, int val1, int val2, int val3,
        uint32_t arr_size, signed char* arr,
        viper::ViPERParams& staged,
        Convolver& convolver,
        ViPERDDC& ddc,
        ResetFn&& reset_effects
    ) {
        switch (param) {
            case params::kParamResetAllEffects:
                VIPER_LOGI("ResetAllEffects");
                reset_effects();
                return false;

            case params::kParamEqualizerBandLevel:
                VIPER_LOGI("EQ: band=%d level=%d", val1, val2);
                if (val1 >= 0 && val1 < 31) {
                    staged.equalizer.band_levels[val1] = static_cast<float>(val2) / 100.0f;
                    return true;
                }
                return false;

            case params::kParamEqualizerBandLevels: {
                const uint32_t float_count = arr_size / static_cast<uint32_t>(sizeof(float));
                VIPER_LOGI("EQ: bands_levels=%u floats", float_count);
                const auto* src = reinterpret_cast<const float*>(arr);
                if (src) {
                    const uint32_t n = std::min(float_count, static_cast<uint32_t>(staged.equalizer.band_levels.size()));
                    for (uint32_t i = 0; i < n; ++i) {
                        staged.equalizer.band_levels[i] = src[i];
                    }
                    return true;
                }
                return false;
            }

            case params::kParamConvolverSetKernel: {
                if (arr_size > 0 && arr != nullptr) {
                    char path[256] = {};
                    std::memcpy(path, arr, std::min<size_t>(arr_size, 255u));
                    VIPER_LOGI("Convolver: SetKernel path=%s", path);
                    convolver.SetKernel(path);
                }
                return false;
            }

            case params::kParamConvolverPrepareBuffer: {
                VIPER_LOGI("Convolver: PrepareBuffer buf_size=%d ch=%d reset=%d", val1, val2, val3);
                convolver.PrepareKernelBuffer(val1, val2, val3 != 0);
                return false;
            }

            case params::kParamConvolverSetBuffer: {
                VIPER_LOGI("Convolver: SetBuffer size=%u", arr_size);
                convolver.SetKernelBuffer(reinterpret_cast<float*>(arr), arr_size);
                return false;
            }

            case params::kParamConvolverCommitBuffer: {
                VIPER_LOGI("Convolver: CommitBuffer val1=%d val2=%d val3=%d", val1, val2, val3);
                if (val1 <= 2 && static_cast<uint32_t>(val1 * val2) == convolver.GetExpectedSize()) {
                    convolver.CommitKernelBuffer(static_cast<uint32_t>(val1 * val2), 0u, 1u);
                } else {
                    convolver.CommitKernelBuffer(static_cast<uint32_t>(val1), static_cast<uint32_t>(val2), static_cast<uint32_t>(val3));
                }
                return false;
            }

            case params::kParamDdcCoefficients: {
                VIPER_LOGI("DDC: SetCoeffs arr_size=%u", arr_size);
                if (arr && arr_size >= sizeof(float) * 2) {
                    const uint32_t total_floats = arr_size / static_cast<uint32_t>(sizeof(float));
                    const uint32_t half_floats  = total_floats / 2u;
                    const auto* const f_arr     = reinterpret_cast<const float*>(arr);
                    ddc.SetCoeffs(half_floats, f_arr, f_arr + half_floats);
                }
                return false;
            }

            case params::kParamMultibandCompressorCrossoverFrequency:
                if (val1 >= 0 && val1 < 5) {
                    staged.multiband_compressor.crossover_frequencies[val1] = static_cast<float>(val2);
                    return true;
                }
                return false;

            case params::kParamMultibandCompressorBandThreshold:
            case params::kParamMultibandCompressorBandRatio:
            case params::kParamMultibandCompressorBandKnee:
            case params::kParamMultibandCompressorBandKneeAuto:
            case params::kParamMultibandCompressorBandGain:
            case params::kParamMultibandCompressorBandGainAuto:
            case params::kParamMultibandCompressorBandAttack:
            case params::kParamMultibandCompressorBandAttackAuto:
            case params::kParamMultibandCompressorBandRelease:
            case params::kParamMultibandCompressorBandReleaseAuto:
            case params::kParamMultibandCompressorBandKneeMulti:
            case params::kParamMultibandCompressorBandMaxAttack:
            case params::kParamMultibandCompressorBandMaxRelease:
            case params::kParamMultibandCompressorBandCrest:
            case params::kParamMultibandCompressorBandAdapt:
            case params::kParamMultibandCompressorBandNoClip:
            case params::kParamMultibandCompressorBandEnable:
                return HandleMbCompBand(param, val1, val2, staged.multiband_compressor);

            case params::kParamDynamicEqBandFrequency:
            case params::kParamDynamicEqBandQ:
            case params::kParamDynamicEqBandGain:
            case params::kParamDynamicEqBandThreshold:
            case params::kParamDynamicEqBandAttack:
            case params::kParamDynamicEqBandRelease:
            case params::kParamDynamicEqBandFilterType:
                return HandleDynEqBand(param, val1, val2, staged.dynamic_eq);

            default:
                VIPER_LOGI("Unknown param: 0x%X val1=%d val2=%d", param, val1, val2);
                return false;
        }
    }

    static bool HandleMbCompBand(int param, int val1, int val2, viper::MultibandCompressorParams& mb) {
        if (val1 < 0 || val1 >= 5) return false;
        auto& b = mb.bands[val1];
        switch (param) {
            case params::kParamMultibandCompressorBandThreshold:    b.threshold = static_cast<float>(val2) / 100.0f; break;
            case params::kParamMultibandCompressorBandRatio:        b.ratio = static_cast<float>(val2) / 100.0f; break;
            case params::kParamMultibandCompressorBandKnee:         b.knee = static_cast<float>(val2) / 100.0f; break;
            case params::kParamMultibandCompressorBandKneeAuto:     b.knee_auto = (val2 != 0); break;
            case params::kParamMultibandCompressorBandGain:         b.gain = static_cast<float>(val2) / 100.0f; break;
            case params::kParamMultibandCompressorBandGainAuto:     b.gain_auto = (val2 != 0); break;
            case params::kParamMultibandCompressorBandAttack:       b.attack = static_cast<float>(val2) / 100.0f; break;
            case params::kParamMultibandCompressorBandAttackAuto:   b.attack_auto = (val2 != 0); break;
            case params::kParamMultibandCompressorBandRelease:      b.release = static_cast<float>(val2) / 100.0f; break;
            case params::kParamMultibandCompressorBandReleaseAuto:  b.release_auto = (val2 != 0); break;
            case params::kParamMultibandCompressorBandKneeMulti:    b.knee_multi = static_cast<float>(val2) / 100.0f; break;
            case params::kParamMultibandCompressorBandMaxAttack:    b.max_attack = static_cast<float>(val2) / 100.0f; break;
            case params::kParamMultibandCompressorBandMaxRelease:   b.max_release = static_cast<float>(val2) / 100.0f; break;
            case params::kParamMultibandCompressorBandCrest:        b.crest = static_cast<float>(val2) / 100.0f; break;
            case params::kParamMultibandCompressorBandAdapt:        b.adapt = static_cast<float>(val2) / 100.0f; break;
            case params::kParamMultibandCompressorBandNoClip:       b.no_clip = (val2 != 0); break;
            case params::kParamMultibandCompressorBandEnable:       b.enable = (val2 != 0); break;
            default: return false;
        }
        return true;
    }

    static bool HandleDynEqBand(int param, int val1, int val2, viper::DynamicEqParams& deq) {
        if (val1 < 0 || val1 >= 10) return false;
        auto& b = deq.bands[val1];
        switch (param) {
            case params::kParamDynamicEqBandFrequency:  b.frequency = static_cast<float>(val2); break;
            case params::kParamDynamicEqBandQ:          b.q = static_cast<float>(val2) / 100.0f; break;
            case params::kParamDynamicEqBandGain:       b.gain = static_cast<float>(val2) / 10.0f; break;
            case params::kParamDynamicEqBandThreshold:  b.threshold = static_cast<float>(val2) / 10.0f; break;
            case params::kParamDynamicEqBandAttack:     b.attack = static_cast<float>(val2); break;
            case params::kParamDynamicEqBandRelease:    b.release = static_cast<float>(val2); break;
            case params::kParamDynamicEqBandFilterType: b.filter_type = val2; break;
            default: return false;
        }
        return true;
    }
};

} // namespace viper::core
