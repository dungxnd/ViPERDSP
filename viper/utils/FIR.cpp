#include "FIR.h"
#include <algorithm>
#include <cstdint>

void FIR::FilterSamplesInterleaved(
    float* samples, const uint32_t size, const uint32_t channels
) noexcept {
    if (!has_coefficients_ || size == 0) return;

    for (uint32_t i = 0; i < size; ++i) {
        block_[i] = samples[i * channels];
    }

    if (block_length_ > size) {
        std::fill(block_.begin() + size, block_.begin() + block_length_, 0.0f);
    }

    std::copy(block_.begin(), block_.begin() + block_length_,
              offset_block_.begin() + coeffs_size_ - 1);

    for (uint32_t i = 0; i < block_length_; ++i) {
        float sample = 0.0f;
        for (uint32_t j = 0; j < coeffs_size_; ++j) {
            sample += coeffs_[j] * offset_block_[coeffs_size_ + i - j - 1];
        }
        if (i < size) {
            samples[i * channels] = sample;
        }
    }

    if (coeffs_size_ > 1) {
        const uint32_t carry_count = coeffs_size_ - 1;
        std::copy(block_.end() - carry_count, block_.end(), offset_block_.begin());
    }
}

void FIR::Reset() noexcept {
    if (coeffs_size_ + block_length_ > 0) {
        std::fill(offset_block_.begin(),
                  offset_block_.begin() + coeffs_size_ + block_length_ + 1, 0.0f);
    }
}

int FIR::LoadCoefficients(
    const float* coeffs, const uint32_t coeffs_size, const uint32_t block_length
) {
    if (coeffs == nullptr || coeffs_size == 0 || block_length == 0) return 0;

    offset_block_.assign(coeffs_size + block_length + 1, 0.0f);
    coeffs_.assign(coeffs, coeffs + coeffs_size);
    block_.assign(block_length, 0.0f);

    coeffs_size_  = coeffs_size;
    block_length_ = block_length;

    has_coefficients_ = true;
    return 1;
}

uint32_t FIR::GetBlockLength() const noexcept {
    return block_length_;
}
