#include "HiFi.h"

HiFi::HiFi() {
    for (auto& buf : buffers_) {
        buf = std::make_unique<WaveBuffer>(2, 0x800);
    }
    for (auto& fs : filters_) {
        fs.lowpass  = std::make_unique<IIR_NOrder_BW_LH>(1);
        fs.highpass = std::make_unique<IIR_NOrder_BW_LH>(3);
        fs.bandpass = std::make_unique<IIR_NOrder_BW_BP>(3);
    }
    Reset();
}

void HiFi::Process(float* samples, const uint32_t size) {
    if (size == 0) return;

    float* bp_buf = buffers_[0]->PushZerosGetBuffer(size);
    float* lp_buf = buffers_[1]->PushZerosGetBuffer(size);
    if (bp_buf == nullptr || lp_buf == nullptr) {
        Reset();
        return;
    }

    for (uint32_t i = 0; i < size * 2; ++i) {
        const uint32_t index = i % 2;
        const float out1 = FilterLH(filters_[index].lowpass.get(),  samples[i]);
        const float out2 = FilterLH(filters_[index].highpass.get(), samples[i]);
        const float out3 = FilterBP(*filters_[index].bandpass,      samples[i]);
        samples[i] = out2;
        lp_buf[i]  = out1;
        bp_buf[i]  = out3;
    }

    const float* bp_out = buffers_[0]->GetBuffer();
    const float* lp_out = buffers_[1]->GetBuffer();
    for (uint32_t i = 0; i < size * 2; ++i) {
        const float hp = samples[i] * gain_ * 1.2f;
        const float bp = bp_out[i] * gain_;
        samples[i] = hp + bp + lp_out[i];
    }
    buffers_[0]->PopSamples(size, false);
    buffers_[1]->PopSamples(size, false);
}

void HiFi::Reset() {
    for (auto& fs : filters_) {
        fs.lowpass->SetLPF(120.0f, sampling_rate_);
        fs.lowpass->Mute();
        fs.highpass->SetHPF(1200.0f, sampling_rate_);
        fs.highpass->Mute();
        fs.bandpass->SetBPF(120.0f, 1200.0f, sampling_rate_);
        fs.bandpass->Mute();
    }
    buffers_[0]->Reset();
    buffers_[0]->PushZeros(sampling_rate_ / 400);
    buffers_[1]->Reset();
    buffers_[1]->PushZeros(sampling_rate_ / 200);
}

void HiFi::SetClarity(const float value) noexcept {
    gain_ = value;
}

void HiFi::SetSamplingRate(const uint32_t sampling_rate) {
    sampling_rate_ = sampling_rate;
    Reset();
}
