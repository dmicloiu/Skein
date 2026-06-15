#include "flock/runtime/llm_client.h"

#include <atomic>
#include <cstdint>

namespace flock {

uint64_t NextGeneration() {
    // relaxed: we need uniqueness, not ordering (the value is published into the
    // sink state before any other thread observes it).
    static std::atomic<uint64_t> counter{1};
    return counter.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace flock
