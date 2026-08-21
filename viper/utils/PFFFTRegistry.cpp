#include "PFFFTRegistry.h"

// ---------------------------------------------------------------------------
// Singleton
// ---------------------------------------------------------------------------

PFFFTRegistry& PFFFTRegistry::Instance() noexcept {
    static PFFFTRegistry instance;
    return instance;
}

PFFFTRegistry::~PFFFTRegistry() {
    // Belt-and-suspenders: release any setups that were never returned.
    for (auto& [ptr, entry] : entries_) {
        if (entry.setup) {
            pffft_destroy_setup(entry.setup);
        }
    }
}

// ---------------------------------------------------------------------------
// Acquire
// ---------------------------------------------------------------------------

PFFFT_Setup* PFFFTRegistry::Acquire(const uint32_t size) noexcept {
    std::lock_guard lock{mutex_};

    // Check if we already have a setup for this size.
    if (const auto it = by_size_.find(size); it != by_size_.end()) {
        PFFFT_Setup* ptr = it->second;
        ++entries_[ptr].ref_count;
        return ptr;
    }

    // Allocate a fresh setup.
    PFFFT_Setup* setup = pffft_new_setup(static_cast<int>(size), PFFFT_REAL);
    if (!setup) return nullptr;

    entries_.try_emplace(setup, Entry{.setup = setup, .size = size, .ref_count = 1});
    by_size_.try_emplace(size, setup);
    return setup;
}

// ---------------------------------------------------------------------------
// Release
// ---------------------------------------------------------------------------

void PFFFTRegistry::Release(PFFFT_Setup* const setup) noexcept {
    if (!setup) return;
    std::lock_guard lock{mutex_};

    const auto it = entries_.find(setup);
    if (it == entries_.end()) return;  // Unknown pointer — ignore.

    Entry& entry = it->second;
    if (--entry.ref_count > 0) return;  // Still referenced by other users.

    // Zero references: free the setup and remove from both maps.
    by_size_.erase(entry.size);
    pffft_destroy_setup(entry.setup);
    entries_.erase(it);
}
