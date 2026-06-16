#include "flock/runtime/async_llm_client.h"

#include <curl/curl.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace flock {
namespace {

// curl_global_init is required exactly once per process before any other curl
// call. Wrap in a function-local static so initialization is thread-safe and
// runs the first time any AsyncLLMClient is constructed.
struct CurlGlobalGuard {
    CurlGlobalGuard() { curl_global_init(CURL_GLOBAL_ALL); }
    ~CurlGlobalGuard() { curl_global_cleanup(); }
};

void EnsureCurlGlobalInit() {
    static CurlGlobalGuard g;
    (void)g;
}

int64_t NowMicros() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

}  // namespace

struct AsyncLLMClient::Impl {
    Options opts;

    struct Pending {
        std::string endpoint;
        std::string payload;
        uint64_t generation = 0;
        std::string request_id;
        OnDone on_done;
        uint64_t handle_id = 0;
        int64_t submit_micros = 0;
    };

    struct InFlight {
        CURL* easy = nullptr;
        std::string write_buffer;
        struct curl_slist* headers = nullptr;
        uint64_t generation = 0;
        std::string request_id;
        OnDone on_done;
        uint64_t handle_id = 0;
        int64_t submit_micros = 0;
    };

    // ---- shared state ----
    std::mutex mu;                                // protects inbox, dead_generations, in_progress
    std::deque<Pending> inbox;
    std::unordered_set<uint64_t> dead_generations;
    // gen -> live callbacks past the dead-check. CancelByGeneration waits on `cv`
    // for the cancelled gen to drain to 0 (single IO thread => 0 or 1 in practice).
    std::unordered_map<uint64_t, size_t> in_progress;
    std::condition_variable cv;
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> next_id{1};

    // ---- IO-thread-only state ----
    CURLM* multi = nullptr;
    CURLSH* share = nullptr;
    std::unordered_map<CURL*, std::unique_ptr<InFlight>> in_flight;

    // CURLSH lock callbacks. curl asserts on null callbacks even when only one
    // thread owns the easy handles, so provide real (lightweight) mutexes per
    // data class. Indexed by curl_lock_data which is small (<= CURL_LOCK_DATA_LAST).
    std::mutex share_locks[CURL_LOCK_DATA_LAST];

    std::thread io_thread;

    static void ShareLockFn(CURL*, curl_lock_data data, curl_lock_access, void* userp) {
        auto* self = static_cast<Impl*>(userp);
        if (data >= 0 && data < CURL_LOCK_DATA_LAST) {
            self->share_locks[data].lock();
        }
    }
    static void ShareUnlockFn(CURL*, curl_lock_data data, void* userp) {
        auto* self = static_cast<Impl*>(userp);
        if (data >= 0 && data < CURL_LOCK_DATA_LAST) {
            self->share_locks[data].unlock();
        }
    }

    static size_t WriteCallback(char* ptr, size_t size, size_t nmemb, void* userdata) {
        auto* buf = static_cast<std::string*>(userdata);
        const size_t n = size * nmemb;
        buf->append(ptr, n);
        return n;
    }

    explicit Impl(Options o) : opts(o) {
        EnsureCurlGlobalInit();
        multi = curl_multi_init();
        share = curl_share_init();
        curl_share_setopt(share, CURLSHOPT_SHARE, CURL_LOCK_DATA_CONNECT);
        curl_share_setopt(share, CURLSHOPT_SHARE, CURL_LOCK_DATA_DNS);
        curl_share_setopt(share, CURLSHOPT_LOCKFUNC, &ShareLockFn);
        curl_share_setopt(share, CURLSHOPT_UNLOCKFUNC, &ShareUnlockFn);
        curl_share_setopt(share, CURLSHOPT_USERDATA, this);

        io_thread = std::thread([this] { IoLoop(); });
    }

    ~Impl() {
        stop.store(true, std::memory_order_release);
        if (multi) curl_multi_wakeup(multi);
        if (io_thread.joinable()) io_thread.join();

        // IO thread has exited. No further concurrent access; safe to touch
        // inbox and in_flight without holding the mutex.
        inbox.clear();

        for (auto& kv : in_flight) {
            CURL* easy = kv.first;
            InFlight* in = kv.second.get();
            curl_multi_remove_handle(multi, easy);
            curl_easy_cleanup(easy);
            if (in->headers) curl_slist_free_all(in->headers);
        }
        in_flight.clear();

        if (share) curl_share_cleanup(share);
        if (multi) curl_multi_cleanup(multi);
    }

    bool IsDeadLocked(uint64_t generation) {
        return dead_generations.count(generation) > 0;
    }

    bool IsDead(uint64_t generation) {
        std::lock_guard<std::mutex> lock(mu);
        return IsDeadLocked(generation);
    }

    // Every callback invocation funnels through here. The dead-check and the
    // in_progress increment are one atomic step under `mu`. This closes the
    // cancel race -> then `cb` runs outside the lock. The RAII guard decrements
    // and notifies the drain even if `cb` throws.
    void InvokeIfAlive(uint64_t gen, OnDone& cb, Response resp) {
        if (!cb) return;
        {
            std::lock_guard<std::mutex> lock(mu);
            if (stop.load(std::memory_order_acquire) || IsDeadLocked(gen)) {
                return;
            }
            ++in_progress[gen];
        }
        struct DrainGuard {
            Impl* self;
            uint64_t gen;
            ~DrainGuard() {
                std::lock_guard<std::mutex> lock(self->mu);
                auto it = self->in_progress.find(gen);
                if (it != self->in_progress.end() && --it->second == 0) {
                    self->in_progress.erase(it);
                    self->cv.notify_all();
                }
            }
        } guard{this, gen};
        cb(std::move(resp));
    }

    RequestHandle Submit(const std::string& endpoint,
                         const std::string& payload,
                         uint64_t generation,
                         const std::string& request_id,
                         OnDone on_done) {
        Pending p;
        p.endpoint = endpoint;
        p.payload = payload;
        p.generation = generation;
        p.request_id = request_id;
        p.on_done = std::move(on_done);
        p.handle_id = next_id.fetch_add(1, std::memory_order_relaxed);
        p.submit_micros = NowMicros();

        const uint64_t id = p.handle_id;
        {
            std::lock_guard<std::mutex> lock(mu);
            inbox.push_back(std::move(p));
        }
        // Do not hold the mutex across the wakeup.
        curl_multi_wakeup(multi);
        return RequestHandle{id};
    }

    void CancelByGeneration(uint64_t generation) {
        std::unique_lock<std::mutex> lock(mu);
        dead_generations.insert(generation);
        // Drain in-flight callbacks for this gen (contract in the header).
        // Do NOT ForgetGeneration here -- see ForgetGeneration.
        cv.wait(lock, [&] {
            auto it = in_progress.find(generation);
            return it == in_progress.end() || it->second == 0;
        });
    }

    // SAFETY: only call when no request for `generation` can still complete.
    // After CancelByGeneration, requests may still be on the wire; dropping the
    // gen lets such a late completion run its callback against a freed operator
    // (UAF). Safe only once abort-in-flight (W4 (D)) clears outstanding handles.
    void ForgetGeneration(uint64_t generation) {
        std::lock_guard<std::mutex> lock(mu);
        dead_generations.erase(generation);
    }

    void IoLoop() {
        while (!stop.load(std::memory_order_acquire)) {
            // 1) Drain inbox under the lock and process locally.
            std::deque<Pending> drained;
            {
                std::lock_guard<std::mutex> lock(mu);
                drained.swap(inbox);
            }

            for (auto& p : drained) {
                if (stop.load(std::memory_order_acquire) || IsDead(p.generation)) {
                    // Early drop: never submitted, callback never fires.
                    continue;
                }
                StartTransfer(std::move(p));
            }

            // 2) Drive in-flight transfers.
            int still_running = 0;
            curl_multi_perform(multi, &still_running);

            // 3) Wait for activity (or wakeup from a Submit / destructor / cancel).
            int numfds = 0;
            curl_multi_poll(multi, nullptr, 0, /*timeout_ms=*/100, &numfds);

            // 4) Reap completed transfers.
            int msgs_in_queue = 0;
            CURLMsg* msg;
            while ((msg = curl_multi_info_read(multi, &msgs_in_queue)) != nullptr) {
                if (msg->msg == CURLMSG_DONE) {
                    FinishTransfer(msg->easy_handle, msg->data.result);
                }
            }
        }
    }

    void StartTransfer(Pending p) {
        auto in = std::make_unique<InFlight>();
        in->easy = curl_easy_init();
        if (!in->easy) {
            // Synthesize a failure response and invoke on_done if alive.
            Response r;
            r.ok = false;
            r.http_status = 0;
            r.error = "curl_easy_init failed";
            r.latency_us = NowMicros() - p.submit_micros;
            InvokeIfAlive(p.generation, p.on_done, std::move(r));
            return;
        }
        in->generation = p.generation;
        in->request_id = std::move(p.request_id);
        in->on_done = std::move(p.on_done);
        in->handle_id = p.handle_id;
        in->submit_micros = p.submit_micros;

        curl_easy_setopt(in->easy, CURLOPT_URL, p.endpoint.c_str());
        curl_easy_setopt(in->easy, CURLOPT_POST, 1L);
        // COPYPOSTFIELDS copies the payload into libcurl-owned memory, so
        // payload's lifetime ends here. Pair with POSTFIELDSIZE_LARGE so the
        // copy doesn't fall through to strlen() (and to support non-text bodies).
        curl_easy_setopt(in->easy, CURLOPT_POSTFIELDSIZE_LARGE,
                         static_cast<curl_off_t>(p.payload.size()));
        curl_easy_setopt(in->easy, CURLOPT_COPYPOSTFIELDS, p.payload.data());
        curl_easy_setopt(in->easy, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(in->easy, CURLOPT_TIMEOUT_MS,
                         static_cast<long>(opts.request_timeout_ms));
        curl_easy_setopt(in->easy, CURLOPT_WRITEFUNCTION, &WriteCallback);
        curl_easy_setopt(in->easy, CURLOPT_WRITEDATA, &in->write_buffer);
        curl_easy_setopt(in->easy, CURLOPT_SHARE, share);

        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        headers = curl_slist_append(headers, "Expect:");  // suppress 100-continue
        if (!in->request_id.empty()) {
            std::string h = "X-Request-ID: " + in->request_id;
            headers = curl_slist_append(headers, h.c_str());
        }
        in->headers = headers;
        curl_easy_setopt(in->easy, CURLOPT_HTTPHEADER, headers);

        CURLMcode rc = curl_multi_add_handle(multi, in->easy);
        if (rc != CURLM_OK) {
            Response r;
            r.ok = false;
            r.http_status = 0;
            r.error = std::string("curl_multi_add_handle: ") + curl_multi_strerror(rc);
            r.latency_us = NowMicros() - in->submit_micros;
            OnDone cb = std::move(in->on_done);
            uint64_t gen = in->generation;
            if (in->headers) curl_slist_free_all(in->headers);
            curl_easy_cleanup(in->easy);
            InvokeIfAlive(gen, cb, std::move(r));
            return;
        }
        CURL* key = in->easy;
        in_flight[key] = std::move(in);
    }

    void FinishTransfer(CURL* easy, CURLcode result) {
        auto it = in_flight.find(easy);
        if (it == in_flight.end()) return;
        std::unique_ptr<InFlight> in = std::move(it->second);
        in_flight.erase(it);

        Response r;
        r.latency_us = NowMicros() - in->submit_micros;
        if (result == CURLE_OK) {
            long status = 0;
            curl_easy_getinfo(easy, CURLINFO_RESPONSE_CODE, &status);
            r.http_status = static_cast<int>(status);
            r.body = std::move(in->write_buffer);
            r.ok = (status >= 200 && status < 300);
            if (!r.ok) {
                r.error = "http status " + std::to_string(status);
            }
        } else {
            r.ok = false;
            r.http_status = 0;
            r.error = curl_easy_strerror(result);
            r.body = std::move(in->write_buffer);
        }

        // Order matters: remove from multi BEFORE easy_cleanup.
        curl_multi_remove_handle(multi, easy);
        curl_easy_cleanup(easy);
        if (in->headers) curl_slist_free_all(in->headers);

        // Drops if cancelled/destroyed; else runs under the drain count.
        InvokeIfAlive(in->generation, in->on_done, std::move(r));
    }
};

AsyncLLMClient::AsyncLLMClient() : impl_(std::make_unique<Impl>(Options{})) {}
AsyncLLMClient::AsyncLLMClient(Options opts) : impl_(std::make_unique<Impl>(opts)) {}
AsyncLLMClient::~AsyncLLMClient() = default;

AsyncLLMClient::RequestHandle AsyncLLMClient::Submit(
    const std::string& endpoint,
    const std::string& payload,
    uint64_t generation,
    const std::string& request_id,
    OnDone on_done) {
    return impl_->Submit(endpoint, payload, generation, request_id, std::move(on_done));
}

void AsyncLLMClient::CancelByGeneration(uint64_t generation) {
    impl_->CancelByGeneration(generation);
}

void AsyncLLMClient::ForgetGeneration(uint64_t generation) {
    impl_->ForgetGeneration(generation);
}

}  // namespace flock
