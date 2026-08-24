#pragma once
// Control-plane / data-plane parameter exchange.
//
// IPC/Binder thread writes via Update().
// Real-time audio thread reads via HasPendingUpdate() + ReadLatest().
//
// Uses a wait-free atomic triple buffer.  Three slots: one owned by the
// writer, one published (shared), one owned by the reader.  shared_idx_
// holds the index of the most recently published snapshot.  Update()
// exchanges write_idx_ with shared_idx_ to atomically publish; ReadLatest()
// exchanges read_idx_ with shared_idx_ to atomically consume.
//
// Guarantee: ReadLatest() always returns the *most recently committed* state.
// Intermediate writes that arrive before the RT thread consumes the previous
// update are safely discarded ("latest wins" — acceptable for audio params).
//
// Memory ordering:
//   Update()     — acq_rel on shared_idx_ exchange; ensures the buffer write
//                  is visible before the index swap and that we see the
//                  reader's previous exchange.
//   ReadLatest() — acq_rel on has_new_data_ exchange + shared_idx_ exchange.

#include <array>
#include <atomic>
#include <cstdint>

namespace viper::core {

template <typename T>
class TripleBufferedState {
public:
    TripleBufferedState() noexcept {
        shared_idx_.store(0u, std::memory_order_relaxed);
    }

    // Writer thread (IPC / Binder) — Lock-free, Wait-free
    void Update(const T& new_state) noexcept {
        buffers_[write_idx_] = new_state;
        // Atomically exchange our write slot with the shared slot so the
        // RT thread can fetch it.  acq_rel: acquire to see any prior reader
        // exchange; release so the buffer write is visible.
        write_idx_ = shared_idx_.exchange(write_idx_, std::memory_order_acq_rel);
        has_new_data_.store(true, std::memory_order_release);
    }

    // Reader thread (RT Audio) — Lock-free, Wait-free
    [[nodiscard]] bool HasPendingUpdate() const noexcept {
        return has_new_data_.load(std::memory_order_acquire);
    }

    [[nodiscard]] const T& ReadLatest() noexcept {
        if (has_new_data_.exchange(false, std::memory_order_acq_rel)) {
            // Atomically take the published slot; give our old read slot back.
            read_idx_ = shared_idx_.exchange(read_idx_, std::memory_order_acq_rel);
        }
        return buffers_[read_idx_];
    }

private:
    alignas(64) std::array<T, 3> buffers_{};

    // shared_idx_ is the "mailbox" between writer and reader threads.
    std::atomic<uint8_t> shared_idx_{0u};
    // has_new_data_ signals the RT thread that a new snapshot is available.
    std::atomic<bool> has_new_data_{false};

    // write_idx_ is accessed exclusively by the writer (IPC thread).
    // read_idx_  is accessed exclusively by the reader (RT audio thread).
    // Neither needs to be atomic.
    uint8_t write_idx_{1u};
    uint8_t read_idx_{2u};
};

// Backward-compat alias — ViPER.h declares DoubleBufferedState<ViPERParams>.
template <typename T>
using DoubleBufferedState = TripleBufferedState<T>;

} // namespace viper::core
