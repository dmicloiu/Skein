#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace flock {

// The non-blocking LLM transport contract.
//
// Submit is called roughly once per coalesced batch (off the row hot path), so
// the virtual dispatch cost here is irrelevant.
//
// AsyncLLMClient is the one production implementation; the interface is NOT
// added for hypothetical alternate transports.

// Result of one HTTP request.
struct LLMResponse {
    bool ok = false;          // false on HTTP non-2xx, timeout, or transport error
    int http_status = 0;      // HTTP status code, 0 if no response
    std::string body;         // raw response body (caller parses JSON)
    std::string error;        // human-readable error if !ok
    int64_t latency_us = 0;   // submit -> complete wall time
};

// Opaque per-submission handle.
struct LLMRequestHandle {
    uint64_t id = 0;
};

// Completion callback. Invoked at most once on the client's IO thread. Must be
// cheap and thread-safe. Skipped iff the generation was cancelled or the client
// is being destroyed.
using LLMOnDone = std::function<void(LLMResponse)>;

class ILLMClient {
public:
    virtual ~ILLMClient() = default;

    // Non-blocking, thread-safe. Returns immediately.
    //
    // endpoint    : full URL e.g. "http://localhost:8000/v1/completions"
    // payload     : JSON body. Copied internally; caller may free after return.
    // generation  : opaque tag for cancellation correlation
    // request_id  : stamped as X-Request-ID (for vLLM /metrics correlation)
    // on_done     : invoked on the IO thread; see LLMOnDone.
    virtual LLMRequestHandle Submit(const std::string& endpoint, const std::string& payload,
                                    uint64_t generation, const std::string& request_id,
                                    LLMOnDone on_done) = 0;

    // Mark a generation as dead. Any in-flight or queued request with this
    // generation has its callback dropped (not invoked). Thread-safe, idempotent.
    virtual void CancelByGeneration(uint64_t generation) = 0;
};

}  // namespace flock
