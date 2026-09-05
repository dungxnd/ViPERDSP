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
        shared_state_.store(0u, std::memory_order_relaxed);
    }

    // Writer thread (IPC / Binder) — Lock-free, Wait-free
    void Update(const T& new_state) noexcept {
        buffers_[write_idx_] = new_state;
        const uint8_t prev = shared_state_.exchange(
            static_cast<uint8_t>(write_idx_ | kDirtyFlag),
            std::memory_order_acq_rel
        );
        write_idx_ = prev & kIndexMask;
    }

    // Reader thread (RT Audio) — Lock-free, Wait-free
    [[nodiscard]] bool HasPendingUpdate() const noexcept {
        return (shared_state_.load(std::memory_order_acquire) & kDirtyFlag) != 0;
    }

    [[nodiscard]] const T& ReadLatest() noexcept {
        uint8_t curr = shared_state_.load(std::memory_order_acquire);
        while (curr & kDirtyFlag) {
            if (shared_state_.compare_exchange_weak(
                    curr,
                    read_idx_,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                read_idx_ = curr & kIndexMask;
                break;
            }
        }
        return buffers_[read_idx_];
    }

private:
    alignas(64) std::array<T, 3> buffers_{};

    static constexpr uint8_t kIndexMask = 0x03u;
    static constexpr uint8_t kDirtyFlag = 0x80u;

    // shared_state_ holds the published buffer index (bits 0-1) and dirty flag (bit 7)
    // packed together so publishing and consumption are single atomic operations.
    std::atomic<uint8_t> shared_state_{0u};

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
