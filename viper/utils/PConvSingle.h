#pragma once

#include "AudioFIFO.h"
#include <cstddef>
#include <cstdint>
#include <vector>

using PFFFT_Setup = struct PFFFT_Setup;

class PConvSingle {
public:
    PConvSingle() noexcept = default;
    ~PConvSingle();

    // Non-copyable: owns FFT-aligned memory
    PConvSingle(const PConvSingle&)            = delete;
    PConvSingle& operator=(const PConvSingle&) = delete;
    PConvSingle(PConvSingle&&)                 = delete;
    PConvSingle& operator=(PConvSingle&&)      = delete;

    void Reset() noexcept;
    void ReleaseResources() noexcept;
    void UnloadKernel() noexcept;

    [[nodiscard]] uint32_t GetFFTSize()      const noexcept { return fft_size_;       }
    [[nodiscard]] uint32_t GetSegmentCount() const noexcept { return segment_count_;  }
    [[nodiscard]] uint32_t GetSegmentSize()  const noexcept { return segment_size_;   }
    [[nodiscard]] bool     InstanceUsable()  const noexcept { return instance_usable_;}

    uint32_t LoadKernel(const float* kernel, uint32_t kernel_size, uint32_t segment_size);
    uint32_t LoadKernel(const float* kernel, float gain, uint32_t kernel_size, uint32_t segment_size);

    /// Process n mono samples: reads from `input`, writes convolved result to `output`.
    /// input == output (in-place) is allowed.
    void Process(const float* input, float* output, uint32_t n) noexcept;

    /// Process a single channel from/to an interleaved stereo (or multi-ch) buffer.
    /// stride == 2 for standard stereo interleaved. In-place (input == output) allowed.
    void ProcessInterleaved(const float* input, float* output,
                            uint32_t channel, uint32_t stride, uint32_t n) noexcept;

private:
    void ProcessBlock() noexcept;

    bool     instance_usable_{false};
    uint32_t segment_size_{0};   // L
    uint32_t fft_size_{0};       // N = 2L
    uint32_t segment_count_{0};  // P
    uint32_t fdl_index_{0};      // circular head of the Frequency Delay Line

    PFFFT_Setup* fft_setup_{nullptr};

    // PFFFT aligned scratch (each fft_size_ floats)
    float* fft_in_{nullptr};
    float* fft_out_{nullptr};
    float* fft_work_{nullptr};
    float* accum_spectrum_{nullptr};

    // Previous L input samples for overlap-save construction
    std::vector<float> prev_input_;

    // IR partitions in frequency domain: [P][fft_size_]
    std::vector<float*> filter_segments_;

    // Frequency Delay Line: input spectra ring buffer [P][fft_size_]
    std::vector<float*> fdl_segments_;

    // Decoupling FIFOs — host chunk size n is fully decoupled from L
    AudioFIFO input_fifo_;
    AudioFIFO output_fifo_;
};
