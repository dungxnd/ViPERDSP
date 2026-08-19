#pragma once

#include <cstdint>
#include <vector>

class FIR {
public:
    FIR() noexcept = default;

    void FilterSamplesInterleaved(float* samples, uint32_t size, uint32_t channels) noexcept;
    void Reset() noexcept;

    [[nodiscard]] uint32_t GetBlockLength() const noexcept;

    int LoadCoefficients(const float* coeffs, uint32_t coeffs_size, uint32_t block_length);

private:
    bool     has_coefficients_{false};
    uint32_t coeffs_size_{0};
    uint32_t block_length_{0};

    std::vector<float> offset_block_;
    std::vector<float> coeffs_;
    std::vector<float> block_;
};
