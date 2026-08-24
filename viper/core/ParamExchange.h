#pragma once
// Control-plane / data-plane parameter exchange.
//
// IPC/Binder thread writes via Update().
// Real-time audio thread reads via HasPendingUpdate() + ReadLatest().
//
// Guarantee: ReadLatest() always returns the *most recently committed* state;
// intermediate writes that arrive before the RT thread consumes the previous
// update are safely overwritten ("latest wins" — acceptable for audio params).
//
// Memory ordering:
//   Update() — release-store on write_idx_ after the buffer write, so the RT
//               thread sees the complete struct when it acquires write_idx_.
//   ReadLatest() — acquire-load on write_idx_ mirrors the release above.

#include <array>
#include <atomic>
#include <cstdint>

namespace viper::core {

template <typename T>
class DoubleBufferedState {
public:
    // Writer thread (IPC / Binder) ─────────────────────────────────────────

    // Publish a new parameter snapshot.  The write goes into the buffer that
    // the RT thread is *not* currently reading, then the index is atomically
    // flipped with a release store so all struct writes are visible on acquire.
    void Update(const T &new_state) noexcept {
        const uint32_t current = write_idx_.load(std::memory_order_relaxed);
        const uint32_t next    = 1u - current;
        buffers_[next]         = new_state;
        // Release: guarantees buffers_[next] is fully written before index flip.
        write_idx_.store(next, std::memory_order_release);
        has_update_.store(true, std::memory_order_release);
    }

    // Reader thread (Real-Time Audio) ───────────────────────────────────────

    // Returns true if Update() has been called since the last ReadLatest().
    // Cheap acquire-load; call at the top of Process() to skip snapshot work
    // when nothing has changed.
    [[nodiscard]] bool HasPendingUpdate() const noexcept {
        return has_update_.load(std::memory_order_acquire);
    }

    // Consume the latest snapshot.  Must be called only from the RT thread,
    // and only after HasPendingUpdate() returns true.
    // Clears the pending flag and returns a const-ref to the active buffer.
    // The reference is valid until the next call to Update() from the writer.
    [[nodiscard]] const T &ReadLatest() noexcept {
        has_update_.store(false, std::memory_order_relaxed);
        // Acquire: pairs with the release in Update(), ensuring we see the
        // entire struct written into buffers_[active].
        const uint32_t active = write_idx_.load(std::memory_order_acquire);
        return buffers_[active];
    }

private:
    alignas(64) std::array<T, 2> buffers_{};
    std::atomic<uint32_t>        write_idx_{0u};
    std::atomic<bool>            has_update_{false};
};

} // namespace viper::core
