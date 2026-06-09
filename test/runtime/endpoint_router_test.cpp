#include "flock/runtime/endpoint_router.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace flock {
namespace {

using Strategy = EndpointRouter::Strategy;

std::vector<std::string> Endpoints(size_t n) {
    std::vector<std::string> v;
    for (size_t i = 0; i < n; ++i) {
        v.push_back("http://host" + std::to_string(i) + ":8000/v1/chat/completions");
    }
    return v;
}

// 1. ParseStrategy/StrategyName round-trip; unknown string throws.
TEST(EndpointRouterTest, ParseAndNameRoundTrip) {
    const std::vector<std::pair<std::string, Strategy>> cases = {
        {"single", Strategy::Single},
        {"round_robin", Strategy::RoundRobin},
        {"sticky_by_prefix", Strategy::StickyByPrefix},
        {"least_loaded", Strategy::LeastLoaded},
    };
    for (const auto& [name, strat] : cases) {
        EXPECT_EQ(EndpointRouter::ParseStrategy(name), strat);
        EXPECT_EQ(std::string(EndpointRouter::StrategyName(strat)), name);
    }
    EXPECT_THROW(EndpointRouter::ParseStrategy("bogus"), std::runtime_error);
    EXPECT_THROW(EndpointRouter::ParseStrategy(""), std::runtime_error);
    EXPECT_THROW(EndpointRouter::ParseStrategy("RoundRobin"), std::runtime_error);
}

// 2. Single: every Choose returns index 0, varied keys.
TEST(EndpointRouterTest, SingleAlwaysIndexZero) {
    EndpointRouter r(Endpoints(3), Strategy::Single);
    const std::vector<std::string> keys = {"", "a", "long-prefix-key", "a", "zzz"};
    for (const auto& k : keys) {
        auto pick = r.Choose(k);
        EXPECT_EQ(pick.index, 0u);
        EXPECT_EQ(pick.url, "http://host0:8000/v1/chat/completions");
        r.OnComplete(pick.index);
    }
}

// 3. RoundRobin: cycles 0..n-1; even distribution; key ignored.
TEST(EndpointRouterTest, RoundRobinCyclesAndIgnoresKey) {
    const size_t n = 4;
    EndpointRouter r(Endpoints(n), Strategy::RoundRobin);

    // Identical keys still rotate sequentially.
    for (size_t round = 0; round < 3; ++round) {
        for (size_t i = 0; i < n; ++i) {
            auto pick = r.Choose("same-key");
            EXPECT_EQ(pick.index, i);
            r.OnComplete(pick.index);
        }
    }

    // Even distribution over many calls.
    std::vector<size_t> counts(n, 0);
    const size_t calls = 4000;
    for (size_t i = 0; i < calls; ++i) {
        auto pick = r.Choose("");
        counts[pick.index]++;
        r.OnComplete(pick.index);
    }
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(counts[i], calls / n);
    }
}

// 4. StickyByPrefix: same key -> same index (deterministic across re-construct);
//    different keys spread; empty key falls back to RR.
TEST(EndpointRouterTest, StickyByPrefixDeterministic) {
    const size_t n = 5;
    EndpointRouter r(Endpoints(n), Strategy::StickyByPrefix);

    auto first = r.Choose("user-42");
    r.OnComplete(first.index);
    for (size_t i = 0; i < 50; ++i) {
        auto pick = r.Choose("user-42");
        EXPECT_EQ(pick.index, first.index);
        r.OnComplete(pick.index);
    }

    // Determinism across a freshly re-constructed router (same endpoint count).
    EndpointRouter r2(Endpoints(n), Strategy::StickyByPrefix);
    auto pick2 = r2.Choose("user-42");
    EXPECT_EQ(pick2.index, first.index);
    r2.OnComplete(pick2.index);

    // Different keys can land on different endpoints (not all identical).
    std::set<size_t> seen;
    for (int i = 0; i < 200; ++i) {
        auto pick = r.Choose("key-" + std::to_string(i));
        seen.insert(pick.index);
        r.OnComplete(pick.index);
    }
    EXPECT_GT(seen.size(), 1u);

    // Empty key -> RoundRobin fallback: rotates 0,1,2,...
    for (size_t i = 0; i < n; ++i) {
        auto pick = r.Choose("");
        EXPECT_EQ(pick.index, i);
        r.OnComplete(pick.index);
    }
}

// 5. LeastLoaded: picks least-loaded; ties resolve to lowest index.
TEST(EndpointRouterTest, LeastLoadedPicksMinAndTieToLowest) {
    const size_t n = 3;
    EndpointRouter r(Endpoints(n), Strategy::LeastLoaded);

    // All zero -> lowest index wins.
    auto p0 = r.Choose("");  // loads: [1,0,0]
    EXPECT_EQ(p0.index, 0u);
    auto p1 = r.Choose("");  // loads: [1,1,0]
    EXPECT_EQ(p1.index, 1u);
    auto p2 = r.Choose("");  // loads: [1,1,1]
    EXPECT_EQ(p2.index, 2u);
    auto p3 = r.Choose("");  // tie at 1 each -> lowest index 0 -> [2,1,1]
    EXPECT_EQ(p3.index, 0u);

    // Skew load heavily onto index 0, then 1; index 2 must be picked next.
    EndpointRouter s(Endpoints(n), Strategy::LeastLoaded);
    for (int i = 0; i < 5; ++i) s.Choose("");  // [1],[1,1],[1,1,1],[2,1,1],[2,2,1]
    // Current loads: [2,2,1] -> least is index 2.
    auto next = s.Choose("");
    EXPECT_EQ(next.index, 2u);
}

// 6. Load accounting: Choose then OnComplete returns to 0; out-of-range no-op.
TEST(EndpointRouterTest, LoadAccounting) {
    EndpointRouter r(Endpoints(2), Strategy::RoundRobin);

    auto pick = r.Choose("");
    EXPECT_EQ(r.InFlight(pick.index), 1u);
    r.OnComplete(pick.index);
    EXPECT_EQ(r.InFlight(pick.index), 0u);

    // Out-of-range OnComplete and InFlight are safe.
    r.OnComplete(999);            // no crash, no underflow
    EXPECT_EQ(r.InFlight(999), 0u);

    // OnComplete below zero stays at zero (extra decrement is a no-op).
    r.OnComplete(0);
    EXPECT_EQ(r.InFlight(0), 0u);
}

// 7. Empty endpoints: Choose throws.
TEST(EndpointRouterTest, EmptyEndpointsThrows) {
    EndpointRouter r({}, Strategy::RoundRobin);
    EXPECT_EQ(r.EndpointCount(), 0u);
    EXPECT_THROW(r.Choose("k"), std::runtime_error);

    // After populating, Choose works again.
    r.SetEndpoints(Endpoints(2));
    EXPECT_EQ(r.EndpointCount(), 2u);
    auto pick = r.Choose("k");
    EXPECT_LT(pick.index, 2u);
    r.OnComplete(pick.index);

    // Emptying again restores the throw.
    r.SetEndpoints({});
    EXPECT_THROW(r.Choose("k"), std::runtime_error);
}

// 8. TSan concurrency: many threads Choose+OnComplete while another thread
//    reconfigures endpoints/strategy. No data race, no crash, counters bounded.
TEST(EndpointRouterTest, ConcurrentChooseCompleteAndReconfigure) {
    EndpointRouter r(Endpoints(4), Strategy::LeastLoaded);

    std::atomic<bool> stop{false};
    const int n_workers = 6;
    std::vector<std::thread> workers;

    for (int t = 0; t < n_workers; ++t) {
        workers.emplace_back([&r, &stop, t] {
            int i = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                try {
                    auto pick = r.Choose("worker-" + std::to_string(t) + "-" +
                                         std::to_string(i++));
                    // The chosen index is valid at pick time; OnComplete on a
                    // now-stale index after a concurrent shrink is a safe no-op.
                    r.OnComplete(pick.index);
                } catch (const std::runtime_error&) {
                    // Endpoint set momentarily empty during reconfigure; retry.
                }
            }
        });
    }

    // Reconfigurer: churns endpoint count and strategy at query-boundary cadence.
    std::thread reconfig([&r, &stop] {
        const Strategy strategies[] = {Strategy::Single, Strategy::RoundRobin,
                                       Strategy::StickyByPrefix,
                                       Strategy::LeastLoaded};
        int i = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            r.SetEndpoints(Endpoints(1 + (i % 5)));
            r.SetStrategy(strategies[i % 4]);
            i++;
            std::this_thread::yield();
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    stop.store(true, std::memory_order_relaxed);
    for (auto& w : workers) w.join();
    reconfig.join();

    // Counters must remain bounded by the live endpoint count and consistent.
    const size_t count = r.EndpointCount();
    for (size_t i = 0; i < count; ++i) {
        // No underflow wraparound to a huge value.
        EXPECT_LT(r.InFlight(i), static_cast<size_t>(1) << 40);
    }
}

}  // namespace
}  // namespace flock
