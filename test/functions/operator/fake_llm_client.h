#pragma once

#include "flock/runtime/llm_client.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace flock {

// Deterministic, in-process stand-in for AsyncLLMClient, used by the threading
// test. It does NO I/O: a small worker pool sleeps a configurable delay, then
// synthesizes a /v1/completions body with ONE completion whose text is the
// batch's boolean_array under the "items" key (flock's batch response shape):
//     {"choices":[{"index":0,"text":"{\"items\":[true,true,...]}"}]}
// It IGNORES the request schema; the verdict count is taken from the prompt,
// which the test's per-batch render emits as one newline-terminated line per row
// (so partial tail batches size correctly). Options::verdict_count_delta injects
// a count != row-count to exercise the engine's fail-loud length check.
class FakeLLMClient : public ILLMClient {
public:
    struct Options {
        std::chrono::milliseconds delay = std::chrono::milliseconds(10);   // base completion delay
        std::chrono::milliseconds jitter = std::chrono::milliseconds(0);   // uniform extra delay in [0, jitter]
        bool inject_error = false;             // synthesize ok=false (fail-loud test)
        int verdict_count_delta = 0;           // emit (rows + delta) items (fail-loud length test)
        int num_workers = 4;
        uint32_t seed = 0xF10C5EEDu;           // deterministic jitter RNG seed
    };

    FakeLLMClient() : FakeLLMClient(Options()) {}

    explicit FakeLLMClient(Options opts) : opts_(opts), rng_(opts.seed) {
        const int n = opts_.num_workers < 1 ? 1 : opts_.num_workers;
        for (int i = 0; i < n; ++i) {
            workers_.emplace_back([this] { WorkerLoop(); });
        }
    }

    ~FakeLLMClient() override {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
        }
        queue_cv_.notify_all();
        for (auto& t : workers_) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

    LLMRequestHandle Submit(const std::string& endpoint, const std::string& payload, uint64_t generation,
                            const std::string& request_id, LLMOnDone on_done) override {
        (void)endpoint;
        (void)request_id;
        Job job;
        job.generation = generation;
        job.batch_size = CountRows(payload);
        job.extra_delay_us = JitterMicros();
        job.on_done = std::move(on_done);
        const uint64_t id = next_id_.fetch_add(1, std::memory_order_relaxed);
        job.id = id;
        {
            std::lock_guard<std::mutex> lock(mu_);
            queue_.push_back(std::move(job));
        }
        queue_cv_.notify_one();
        return LLMRequestHandle{id};
    }

    void CancelByGeneration(uint64_t generation) override {
        std::unique_lock<std::mutex> lock(mu_);
        dead_.insert(generation);
        // Wait for any callback of this generation already past the dead-check
        // to finish, so the caller (a sink-state destructor) can free safely.
        drain_cv_.wait(lock, [&] { return in_progress_[generation] == 0; });
    }

    // -- test introspection --
    size_t FiredCount() const { return fired_count_.load(std::memory_order_acquire); }

private:
    struct Job {
        uint64_t id = 0;
        uint64_t generation = 0;
        size_t batch_size = 0;
        int64_t extra_delay_us = 0;
        LLMOnDone on_done;
    };

    // Row count = newline-terminated lines in the single multi-row prompt string
    // (the test's render emits one line per row). The schema is ignored.
    static size_t CountRows(const std::string& payload) {
        auto parsed = nlohmann::json::parse(payload, nullptr, /*allow_exceptions=*/false);
        if (parsed.is_discarded()) {
            return 0;
        }
        auto it = parsed.find("prompt");
        if (it == parsed.end() || !it->is_string()) {
            return 0;
        }
        const auto& prompt = it->get_ref<const std::string&>();
        return static_cast<size_t>(std::count(prompt.begin(), prompt.end(), '\n'));
    }

    int64_t JitterMicros() {
        if (opts_.jitter.count() <= 0) {
            return 0;
        }
        std::lock_guard<std::mutex> lock(rng_mu_);
        std::uniform_int_distribution<int64_t> dist(0, opts_.jitter.count() * 1000);
        return dist(rng_);
    }

    LLMResponse Synthesize(const Job& job) const {
        LLMResponse r;
        r.latency_us = static_cast<int64_t>(opts_.delay.count()) * 1000;
        if (opts_.inject_error) {
            r.ok = false;
            r.http_status = 500;
            r.error = "injected error";
            r.body = R"({"error":"injected"})";
            return r;
        }
        // ONE completion for the whole batch. Its text is a JSON object holding
        // a boolean_array of per-row verdicts under "items". verdict_count_delta
        // lets a test produce a count != rows to trip the fail-loud length check.
        const int64_t count = std::max<int64_t>(
                0, static_cast<int64_t>(job.batch_size) + opts_.verdict_count_delta);
        auto items = nlohmann::json::array();
        for (int64_t i = 0; i < count; ++i) {
            items.push_back(true);
        }
        nlohmann::json completion;
        completion["items"] = std::move(items);
        nlohmann::json choice;
        choice["index"] = 0;
        choice["text"] = completion.dump();  // completion text is the JSON-encoded items object
        nlohmann::json body;
        body["choices"] = nlohmann::json::array({std::move(choice)});
        r.ok = true;
        r.http_status = 200;
        r.body = body.dump();
        return r;
    }

    void WorkerLoop() {
        for (;;) {
            Job job;
            {
                std::unique_lock<std::mutex> lock(mu_);
                queue_cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
                if (stop_) {
                    return;  // drop any remaining jobs on shutdown
                }
                job = std::move(queue_.front());
                queue_.pop_front();
                // Early skip: if already cancelled, don't even simulate latency.
                if (dead_.find(job.generation) != dead_.end()) {
                    continue;
                }
            }
            // Simulate latency OUTSIDE the lock.
            std::this_thread::sleep_for(opts_.delay + std::chrono::microseconds(job.extra_delay_us));

            bool fire = false;
            {
                std::lock_guard<std::mutex> lock(mu_);
                if (!stop_ && dead_.find(job.generation) == dead_.end()) {
                    in_progress_[job.generation] += 1;
                    fire = true;
                }
            }
            if (!fire) {
                continue;
            }
            // on_done runs WITHOUT the lock -> real concurrency vs the engine.
            fired_count_.fetch_add(1, std::memory_order_relaxed);
            if (job.on_done) {
                job.on_done(Synthesize(job));
            }
            {
                std::lock_guard<std::mutex> lock(mu_);
                in_progress_[job.generation] -= 1;
            }
            drain_cv_.notify_all();
        }
    }

    Options opts_;
    std::mutex mu_;
    std::condition_variable queue_cv_;
    std::condition_variable drain_cv_;
    std::deque<Job> queue_;
    std::unordered_set<uint64_t> dead_;
    std::unordered_map<uint64_t, int> in_progress_;
    bool stop_ = false;
    std::atomic<uint64_t> next_id_{1};
    std::atomic<size_t> fired_count_{0};
    std::vector<std::thread> workers_;
    std::mutex rng_mu_;
    std::mt19937 rng_;
};

}  // namespace flock