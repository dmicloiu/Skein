#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace flock {

// Pure endpoint-selection policy for a pool of vLLM endpoints.
//
// One process-lifetime instance, owned by ExtensionState and reconfigured by
// the SET-variable layer. It holds a mutable set of endpoint URLs, a routing
// strategy, and a per-endpoint in-flight counter, and answers exactly one
// question: given a prefix key, which endpoint should this request go to?
//
// It performs NO HTTP and holds NO reference to AsyncLLMClient. In-flight
// accounting is cooperative: the caller increments via Choose() and MUST
// decrement via OnComplete() exactly once per Choose() -- including on the
// cancellation path. Admission/backpressure, coalescing, retries and metrics
// polling are all out of scope; they live in the operator layer.
class EndpointRouter {
public:
    enum class Strategy { Single, RoundRobin, StickyByPrefix, LeastLoaded };

    // "single"|"round_robin"|"sticky_by_prefix"|"least_loaded"; throws
    // std::runtime_error on an unknown string.
    static Strategy ParseStrategy(const std::string& s);
    static const char* StrategyName(Strategy s);

    EndpointRouter(std::vector<std::string> endpoints, Strategy strategy);
    ~EndpointRouter();

    EndpointRouter(const EndpointRouter&) = delete;
    EndpointRouter& operator=(const EndpointRouter&) = delete;
    EndpointRouter(EndpointRouter&&) = delete;
    EndpointRouter& operator=(EndpointRouter&&) = delete;

    struct Pick {
        size_t index;     // endpoint index, valid until the next SetEndpoints
        std::string url;  // BY VALUE -- stays valid even if SetEndpoints
                          // replaces the set after this returns
    };

    // Select an endpoint AND increment its in-flight counter, atomically.
    // prefix_key: used ONLY by StickyByPrefix (same key -> same endpoint, for
    //             vLLM prefix-cache reuse). Empty key -> RoundRobin fallback.
    //             Ignored by all other strategies.
    // Throws std::runtime_error if the endpoint set is empty.
    Pick Choose(const std::string& prefix_key);

    // Decrement the in-flight counter for a previously-chosen endpoint.
    // The caller MUST call this exactly once per Choose -- including on the
    // cancellation path (operator decrements for rows it drops). Out-of-range
    // index (after a SetEndpoints shrink) is a safe no-op.
    void OnComplete(size_t index);

    // Runtime reconfiguration (the SET layer calls these). Thread-safe.
    // Expected to be called at query boundaries; remains memory-safe if called
    // concurrently with Choose/OnComplete. Resets in-flight counters.
    void SetEndpoints(std::vector<std::string> endpoints);
    void SetStrategy(Strategy s);

    // Introspection for tests / future metrics.
    size_t EndpointCount() const;
    size_t InFlight(size_t index) const;  // 0 if index out of range

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace flock
