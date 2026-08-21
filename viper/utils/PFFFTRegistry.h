#pragma once

#include "pffft.h"
#include <cstdint>
#include <mutex>
#include <unordered_map>

/// ---------------------------------------------------------------------------
/// PFFFTRegistry — reference-counted cache of PFFFT_Setup* objects.
///
/// Motivation: PFFFT_Setup allocations are expensive (twiddle-factor tables,
/// aligned scratch). Multiple stages of PConvNUPC / PConvZeroLatency that
/// share the same FFT size must NOT create duplicate setups.
///
/// Thread safety:
///   - Acquire() / Release() hold a mutex only during bookkeeping.
///   - Once a setup is acquired the pointer is read-only during Process(),
///     so the audio thread never contends the lock.
/// ---------------------------------------------------------------------------

class PFFFTRegistry {
public:
    // Returns the process-wide singleton.
    [[nodiscard]] static PFFFTRegistry& Instance() noexcept;

    // Acquire or reuse a PFFFT_Setup for `size` real samples.
    // Returns nullptr if PFFFT fails to allocate.
    [[nodiscard]] PFFFT_Setup* Acquire(uint32_t size) noexcept;

    // Decrement reference count; frees the setup when it reaches zero.
    void Release(PFFFT_Setup* setup) noexcept;

    // Non-copyable / non-movable singleton.
    PFFFTRegistry(const PFFFTRegistry&)            = delete;
    PFFFTRegistry& operator=(const PFFFTRegistry&) = delete;
    PFFFTRegistry(PFFFTRegistry&&)                 = delete;
    PFFFTRegistry& operator=(PFFFTRegistry&&)      = delete;

private:
    PFFFTRegistry() = default;
    ~PFFFTRegistry();  // Frees any leaked setups in debug scenarios.

    struct Entry {
        PFFFT_Setup* setup{nullptr};
        uint32_t     size{0};
        int          ref_count{0};
    };

    std::mutex                                      mutex_;
    // Keyed by raw pointer so Release() can find the entry in O(1).
    std::unordered_map<PFFFT_Setup*, Entry>         entries_;
    // Secondary index: size → pointer (latest created; reused on Acquire).
    std::unordered_map<uint32_t, PFFFT_Setup*>      by_size_;
};
