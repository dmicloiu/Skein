#pragma once

#include "flock/runtime/llm_client.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace flock {

// Non-blocking HTTP client for vLLM's OpenAI-compatible API.
//
// One process-lifetime instance, owned by ExtensionState. Internally drives a
// single libcurl multi handle from a dedicated IO thread, with a CURLSH-shared
// connection + DNS pool. Callers Submit() pre-built JSON payloads and receive
// completions via callbacks invoked on the IO thread.
//
// The client does NOT enforce in-flight caps; that is the caller's problem.
// Retries, backoff, batching and routing are also out of scope.
class AsyncLLMClient : public ILLMClient {
public:
    using Response = LLMResponse;
    using RequestHandle = LLMRequestHandle;
    using OnDone = LLMOnDone;

    struct Options {
        // CURLOPT_TIMEOUT_MS per request. Default 60s, intended to catch a
        // stuck vLLM. Workloads with long prompts + long outputs (e.g. the
        // sembench vLLM performance analysis cell at ~1357-prompt-token /
        // 64-output-token) can exceed this and need to bump it.
        int request_timeout_ms = 60000;
    };

    AsyncLLMClient();                       // default Options
    explicit AsyncLLMClient(Options opts);
    ~AsyncLLMClient();

    AsyncLLMClient(const AsyncLLMClient&) = delete;
    AsyncLLMClient& operator=(const AsyncLLMClient&) = delete;
    AsyncLLMClient(AsyncLLMClient&&) = delete;
    AsyncLLMClient& operator=(AsyncLLMClient&&) = delete;

    // Non-blocking. Thread-safe. Returns immediately.
    //
    // endpoint    : full URL e.g. "http://localhost:8000/v1/chat/completions"
    // payload     : JSON body. Copied internally; caller may free after return.
    // generation  : opaque tag for cancellation correlation
    // request_id  : stamped as X-Request-ID header (for vLLM /metrics correlation)
    // on_done     : invoked on the IO thread. Must be cheap and thread-safe.
    //               Skipped iff the generation was cancelled or the client is
    //               being destroyed.
    RequestHandle Submit(
        const std::string& endpoint,
        const std::string& payload,
        uint64_t generation,
        const std::string& request_id,
        OnDone on_done) override;

    // Drains as the interface specifies: drops this gen's queued/in-flight
    // callbacks, blocks until any mid-flight one returns. Only the canceller
    // blocks; the IO thread / data path never does. Curl resources are still
    // freed for dropped requests.
    void CancelByGeneration(uint64_t generation) override;

    // Optional: remove a generation from the dead set, allowing future
    // submissions on that generation to fire callbacks again. Bounded-growth
    // hygiene only; not required for correctness.
    void ForgetGeneration(uint64_t generation);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace flock
