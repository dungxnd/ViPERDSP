#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>

struct WavData {
    std::unique_ptr<float[]> samples;
    uint32_t frame_count = 0;
    uint32_t channels    = 0;
    uint32_t sample_rate = 0;
};

// Decode a raw PCM buffer into normalised floats.
// Returns nullptr on unrecognised format.
[[nodiscard]] inline std::unique_ptr<float[]> DecodeSamples(
    std::FILE* const fp,
    const bool is_float,
    const uint16_t bits_per_sample,
    const uint32_t total_samples
) {
    auto samples = std::make_unique<float[]>(total_samples);
    if (!samples) return nullptr;

    if (is_float && bits_per_sample == 32) {
        if (std::fread(samples.get(), sizeof(float), total_samples, fp) != total_samples) {
            return nullptr;
        }
        return samples;
    }

    if (!is_float && bits_per_sample == 16) {
        auto tmp = std::make_unique<int16_t[]>(total_samples);
        if (!tmp || std::fread(tmp.get(), sizeof(int16_t), total_samples, fp) != total_samples) {
            return nullptr;
        }
        for (uint32_t i = 0; i < total_samples; ++i) {
            samples[i] = static_cast<float>(tmp[i]) / 32768.0f;
        }
        return samples;
    }

    if (!is_float && bits_per_sample == 24) {
        auto tmp = std::make_unique<uint8_t[]>(total_samples * 3);
        if (!tmp || std::fread(tmp.get(), 3, total_samples, fp) != total_samples) {
            return nullptr;
        }
        for (uint32_t i = 0; i < total_samples; ++i) {
            int32_t val = tmp[i*3] << 8 | tmp[i*3+1] << 16 | tmp[i*3+2] << 24;
            val >>= 8;
            samples[i] = static_cast<float>(val) / 8388608.0f;
        }
        return samples;
    }

    if (!is_float && bits_per_sample == 32) {
        auto tmp = std::make_unique<int32_t[]>(total_samples);
        if (!tmp || std::fread(tmp.get(), sizeof(int32_t), total_samples, fp) != total_samples) {
            return nullptr;
        }
        for (uint32_t i = 0; i < total_samples; ++i) {
            samples[i] = static_cast<float>(tmp[i]) / 2147483648.0f;
        }
        return samples;
    }

    return nullptr;  // unsupported format
}

inline bool ReadWavFile(const char* const path, WavData& out) {
    if (!path || path[0] == '\0') return false;

    FILE* const fp = std::fopen(path, "rb");
    if (!fp) return false;

    out = {};

    uint8_t header[44];
    if (std::fread(header, 1, 44, fp) != 44) { std::fclose(fp); return false; }

    if (std::memcmp(header,      "RIFF", 4) != 0 ||
        std::memcmp(header +  8, "WAVE", 4) != 0 ||
        std::memcmp(header + 12, "fmt ", 4) != 0) {
        std::fclose(fp); return false;
    }

    uint32_t fmt_size;        std::memcpy(&fmt_size,        header + 16, 4);
    uint16_t audio_format;    std::memcpy(&audio_format,    header + 20, 2);
    uint16_t num_channels;    std::memcpy(&num_channels,    header + 22, 2);
    uint32_t sample_rate;     std::memcpy(&sample_rate,     header + 24, 4);
    uint16_t bits_per_sample; std::memcpy(&bits_per_sample, header + 34, 2);

    if ((audio_format != 1 && audio_format != 3) ||
        bits_per_sample == 0 || num_channels == 0) {
        std::fclose(fp); return false;
    }

    const bool     is_float         = audio_format == 3;
    const uint32_t bytes_per_sample = bits_per_sample / 8;

    std::fseek(fp, 12 + 8 + static_cast<int32_t>(fmt_size), SEEK_SET);

    uint8_t  chunk_header[8];
    uint32_t data_size = 0;
    while (std::fread(chunk_header, 1, 8, fp) == 8) {
        std::memcpy(&data_size, chunk_header + 4, 4);
        if (std::memcmp(chunk_header, "data", 4) == 0) break;
        std::fseek(fp, data_size, SEEK_CUR);
        data_size = 0;
    }

    if (data_size == 0) { std::fclose(fp); return false; }

    const uint32_t total_samples = data_size / bytes_per_sample;
    const uint32_t frame_count   = total_samples / num_channels;

    auto samples = DecodeSamples(fp, is_float, bits_per_sample, total_samples);
    std::fclose(fp);
    if (!samples) return false;

    out.samples     = std::move(samples);
    out.frame_count = frame_count;
    out.channels    = num_channels;
    out.sample_rate = sample_rate;
    return true;
}
