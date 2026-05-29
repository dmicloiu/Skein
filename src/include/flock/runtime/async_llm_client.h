#pragma once

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
class AsyncLLMClient {
public:
    struct Response {
        bool ok = false;          // false on HTTP non-2xx, timeout, or curl error
        int http_status = 0;      // HTTP status code, 0 if no response
        std::string body;         // raw response body (caller parses JSON)
        std::string error;        // human-readable error if !ok
        int64_t latency_us = 0;   // submit -> complete wall time
    };

    struct RequestHandle {
        uint64_t id = 0;
    };

    using OnDone = std::function<void(Response)>;

    struct Options {
        // CURLOPT_TIMEOUT_MS per request. Default 60s, intended to catch a
        // stuck vLLM. Workloads with long prompts + long outputs (e.g. the
        // E1 standalone study at 1357-prompt-token / 64-output-token) can
        // exceed this and need to bump it.
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
        OnDone on_done);

    // Mark a generation as dead. Any in-flight or queued request with this
    // generation has its callback dropped (not invoked). Curl resources are
    // still cleaned up. Thread-safe. Idempotent.
    void CancelByGeneration(uint64_t generation);

    // Optional: remove a generation from the dead set, allowing future
    // submissions on that generation to fire callbacks again. Bounded-growth
    // hygiene only; not required for correctness.
    void ForgetGeneration(uint64_t generation);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace flock
