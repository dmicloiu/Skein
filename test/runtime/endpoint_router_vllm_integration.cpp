// EndpointRouter integration test: drive a pool of live vLLM endpoints through
// the real EndpointRouter and measure the system-level effect of each routing
// strategy (no DuckDB, no mock). This is the glue the operator layer will later
// own; here it lets us validate the router against real GPUs.
//
//   for each request:
//       pick = router.Choose(prefix_key)   // selects endpoint + bumps in-flight
//       client.Submit(pick.url, payload, ...)
//       on completion: router.OnComplete(pick.index)
//
// It records, for the main phase: per-endpoint request counts (the routing
// decision), aggregate throughput, and the full latency distribution. Output is
// a single JSON blob (--result-out).
//
// Pair this with a /metrics snapshot before+after each run (the slurm driver
// does that with curl) to capture vLLM's prefix-cache hit rate per endpoint;
// that is the signal that makes sticky_by_prefix's win visible.
//
// Usage:
//   flock_endpoint_router_vllm_integration
//       --endpoints http://127.0.0.1:8000/v1/completions,http://127.0.0.1:8001/v1/completions
//       --strategy round_robin
//       --payload-file /path/payloads.jsonl [--keys-file /path/keys.txt]
//       --inflight 128 --total 2048 --warmup 128
//       --rows-per-request 32 --timeout-ms 120000
//       --result-out /path/result.json

#include "flock/runtime/async_llm_client.h"
#include "flock/runtime/endpoint_router.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Args {
    std::string endpoints_csv;  // comma-separated endpoint URLs
    std::string strategy = "round_robin";
    std::string model = "Qwen/Qwen2.5-7B-Instruct";
    int inflight = 128;
    int total = 2048;
    int warmup = 0;
    int max_tokens = 32;
    int rows_per_request = 1;
    int timeout_ms = 120000;
    std::string payload_file;
    std::string keys_file;
    std::string result_out;
};

void Usage(const char* prog) {
    std::fprintf(stderr,
                 "Usage: %s --endpoints URL1,URL2,... --strategy NAME\n"
                 "          [--model NAME] [--inflight N] [--total N] [--warmup N]\n"
                 "          [--max-tokens N] [--rows-per-request N] [--timeout-ms N]\n"
                 "          [--payload-file PATH] [--keys-file PATH]\n"
                 "          [--result-out PATH]\n"
                 "  strategy: single|round_robin|sticky_by_prefix|least_loaded\n",
                 prog);
}

std::vector<std::string> SplitCsv(const std::string& s) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= s.size()) {
        size_t comma = s.find(',', start);
        if (comma == std::string::npos) {
            out.push_back(s.substr(start));
            break;
        }
        out.push_back(s.substr(start, comma - start));
        start = comma + 1;
    }
    return out;
}

bool ParseArgs(int argc, char** argv, Args* a) {
    for (int i = 1; i < argc; ++i) {
        const char* k = argv[i];
        auto need = [&](const char* name) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (!std::strcmp(k, "--endpoints")) a->endpoints_csv = need("--endpoints");
        else if (!std::strcmp(k, "--strategy")) a->strategy = need("--strategy");
        else if (!std::strcmp(k, "--model")) a->model = need("--model");
        else if (!std::strcmp(k, "--inflight")) a->inflight = std::atoi(need("--inflight"));
        else if (!std::strcmp(k, "--total")) a->total = std::atoi(need("--total"));
        else if (!std::strcmp(k, "--warmup")) a->warmup = std::atoi(need("--warmup"));
        else if (!std::strcmp(k, "--max-tokens")) a->max_tokens = std::atoi(need("--max-tokens"));
        else if (!std::strcmp(k, "--rows-per-request")) a->rows_per_request = std::atoi(need("--rows-per-request"));
        else if (!std::strcmp(k, "--timeout-ms")) a->timeout_ms = std::atoi(need("--timeout-ms"));
        else if (!std::strcmp(k, "--payload-file")) a->payload_file = need("--payload-file");
        else if (!std::strcmp(k, "--keys-file")) a->keys_file = need("--keys-file");
        else if (!std::strcmp(k, "--result-out")) a->result_out = need("--result-out");
        else if (!std::strcmp(k, "-h") || !std::strcmp(k, "--help")) {
            Usage(argv[0]);
            return false;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", k);
            Usage(argv[0]);
            return false;
        }
    }
    if (a->endpoints_csv.empty()) {
        std::fprintf(stderr, "--endpoints is required\n");
        return false;
    }
    if (a->inflight <= 0 || a->total <= 0) {
        std::fprintf(stderr, "inflight and total must be positive\n");
        return false;
    }
    return true;
}

std::string BuildSyntheticPayload(const Args& a) {
    std::string p;
    p += "{\"model\":\"";
    p += a.model;
    p += "\",\"prompt\":\"Reply with the single word YES.\"";
    p += ",\"max_tokens\":";
    p += std::to_string(a.max_tokens);
    p += ",\"temperature\":0}";
    return p;
}

std::vector<std::string> LoadLines(const std::string& path, const char* what) {
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "cannot open %s file: %s\n", what, path.c_str());
        std::exit(1);
    }
    std::vector<std::string> out;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) out.push_back(std::move(line));
    }
    if (out.empty()) {
        std::fprintf(stderr, "%s file is empty: %s\n", what, path.c_str());
        std::exit(1);
    }
    return out;
}

struct Stats {
    std::atomic<int> outstanding{0};
    std::atomic<int> completed{0};
    std::atomic<int> ok{0};
    std::atomic<int> err{0};
    std::atomic<int64_t> sum_latency_us{0};
    std::atomic<int64_t> max_latency_us{0};

    std::mutex lat_mu;
    std::vector<int64_t> latencies_us;  // main-phase only
};

// Linear-interpolated percentile, matching numpy's default. q in [0, 1].
double Percentile(std::vector<int64_t>& xs, double q) {
    if (xs.empty()) return 0.0;
    std::sort(xs.begin(), xs.end());
    if (xs.size() == 1) return static_cast<double>(xs.front());
    double pos = q * (xs.size() - 1);
    size_t lo = static_cast<size_t>(pos);
    size_t hi = std::min(lo + 1, xs.size() - 1);
    double frac = pos - lo;
    return xs[lo] + frac * (xs[hi] - xs[lo]);
}

// Drives `count` requests through the router. The submit loop runs on this
// thread, so per-endpoint submit counts need no lock. on_done runs on the
// AsyncLLMClient IO thread: it decrements the router's in-flight counter (so
// least_loaded sees real load) and records latency.
void RunPhase(flock::AsyncLLMClient& client,
              flock::EndpointRouter& router,
              const Args& args,
              const std::vector<std::string>& payloads,
              const std::vector<std::string>& keys,
              const std::string& rid_prefix,
              int count,
              Stats& stats,
              std::vector<long>& submit_counts,
              bool record_latency,
              const char* phase_name) {
    std::mutex slot_mu;
    std::condition_variable slot_cv;

    const int start_completed = stats.completed.load();
    auto t0 = std::chrono::steady_clock::now();
    auto last_report = t0;
    int last_completed = start_completed;

    for (int i = 0; i < count; ++i) {
        {
            std::unique_lock<std::mutex> lock(slot_mu);
            slot_cv.wait(lock, [&] {
                return stats.outstanding.load(std::memory_order_relaxed) < args.inflight;
            });
        }

        const std::string& key =
            keys.empty() ? std::string() : keys[i % keys.size()];
        flock::EndpointRouter::Pick pick = router.Choose(key);
        submit_counts[pick.index]++;
        const size_t idx = pick.index;

        auto on_done = [&, idx](flock::AsyncLLMClient::Response r) {
            router.OnComplete(idx);
            if (r.ok) stats.ok.fetch_add(1, std::memory_order_relaxed);
            else stats.err.fetch_add(1, std::memory_order_relaxed);
            stats.sum_latency_us.fetch_add(r.latency_us, std::memory_order_relaxed);
            int64_t prev = stats.max_latency_us.load(std::memory_order_relaxed);
            while (r.latency_us > prev &&
                   !stats.max_latency_us.compare_exchange_weak(prev, r.latency_us,
                                                              std::memory_order_relaxed)) {}
            if (record_latency) {
                std::lock_guard<std::mutex> lock(stats.lat_mu);
                stats.latencies_us.push_back(r.latency_us);
            }
            stats.outstanding.fetch_sub(1, std::memory_order_relaxed);
            stats.completed.fetch_add(1, std::memory_order_relaxed);
            { std::lock_guard<std::mutex> lock(slot_mu); }
            slot_cv.notify_all();
        };

        stats.outstanding.fetch_add(1, std::memory_order_relaxed);
        const std::string& payload = payloads[i % payloads.size()];
        client.Submit(pick.url, payload, /*generation=*/1,
                      rid_prefix + std::to_string(i), on_done);

        auto now = std::chrono::steady_clock::now();
        if (now - last_report >= std::chrono::seconds(2)) {
            int c = stats.completed.load(std::memory_order_relaxed);
            double dt = std::chrono::duration<double>(now - last_report).count();
            double rps = (c - last_completed) / dt;
            std::fprintf(stderr,
                         "  [%s] submitted=%d completed=%d inflight=%d rate=%.1f req/s\n",
                         phase_name, i + 1, c - start_completed,
                         stats.outstanding.load(), rps);
            last_completed = c;
            last_report = now;
        }
    }

    {
        std::unique_lock<std::mutex> lock(slot_mu);
        slot_cv.wait(lock, [&] {
            return stats.completed.load(std::memory_order_relaxed) >=
                   start_completed + count;
        });
    }
    auto t1 = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();
    std::fprintf(stderr, "  [%s] done: %d req in %.2fs = %.2f req/s\n",
                 phase_name, count, dt, count / dt);
}

void WriteResultJson(const Args& args,
                     const std::vector<std::string>& endpoints,
                     const std::vector<long>& counts,
                     double elapsed_s, int ok, int err,
                     std::vector<int64_t>& latencies_us) {
    if (args.result_out.empty()) return;
    std::ofstream out(args.result_out);
    if (!out) {
        std::fprintf(stderr, "cannot write result file: %s\n", args.result_out.c_str());
        return;
    }
    const double req_s = args.total / elapsed_s;
    const double rows_s = static_cast<double>(ok) * args.rows_per_request / elapsed_s;

    out << "{\n";
    out << "  \"strategy\": \"" << args.strategy << "\",\n";
    out << "  \"model\": \"" << args.model << "\",\n";
    out << "  \"inflight\": " << args.inflight << ",\n";
    out << "  \"rows_per_request\": " << args.rows_per_request << ",\n";
    out << "  \"total\": " << args.total << ",\n";
    out << "  \"ok\": " << ok << ",\n";
    out << "  \"err\": " << err << ",\n";
    out << "  \"elapsed_s\": " << elapsed_s << ",\n";
    out << "  \"throughput_requests_per_s\": " << req_s << ",\n";
    out << "  \"throughput_rows_per_s\": " << rows_s << ",\n";

    out << "  \"endpoints\": [";
    for (size_t i = 0; i < endpoints.size(); ++i) {
        out << (i ? ", " : "") << "\"" << endpoints[i] << "\"";
    }
    out << "],\n";

    out << "  \"per_endpoint_count\": [";
    for (size_t i = 0; i < counts.size(); ++i) {
        out << (i ? ", " : "") << counts[i];
    }
    out << "],\n";

    double sum = 0.0;
    for (int64_t v : latencies_us) sum += v;
    double mean_ms = latencies_us.empty() ? 0.0 : sum / latencies_us.size() / 1000.0;
    out << "  \"latency_ms\": {";
    out << "\"mean\": " << mean_ms;
    out << ", \"p50\": " << Percentile(latencies_us, 0.50) / 1000.0;
    out << ", \"p95\": " << Percentile(latencies_us, 0.95) / 1000.0;
    out << ", \"p99\": " << Percentile(latencies_us, 0.99) / 1000.0;
    out << ", \"max\": " << Percentile(latencies_us, 1.0) / 1000.0;
    out << "},\n";

    // Full latency array (ms) so the local plotter can build CDFs.
    out << "  \"latencies_ms\": [";
    for (size_t i = 0; i < latencies_us.size(); ++i) {
        out << (i ? ", " : "") << latencies_us[i] / 1000.0;
    }
    out << "]\n";
    out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!ParseArgs(argc, argv, &args)) return 2;

    std::vector<std::string> endpoints = SplitCsv(args.endpoints_csv);
    if (endpoints.empty()) {
        std::fprintf(stderr, "no endpoints parsed from --endpoints\n");
        return 2;
    }

    flock::EndpointRouter::Strategy strat;
    try {
        strat = flock::EndpointRouter::ParseStrategy(args.strategy);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 2;
    }
    flock::EndpointRouter router(endpoints, strat);

    std::vector<std::string> payloads;
    if (!args.payload_file.empty()) {
        payloads = LoadLines(args.payload_file, "payload");
        std::fprintf(stderr, "loaded %zu payloads from %s\n",
                     payloads.size(), args.payload_file.c_str());
    } else {
        payloads.push_back(BuildSyntheticPayload(args));
    }

    std::vector<std::string> keys;
    if (!args.keys_file.empty()) {
        keys = LoadLines(args.keys_file, "keys");
        std::fprintf(stderr, "loaded %zu prefix keys from %s\n",
                     keys.size(), args.keys_file.c_str());
        if (keys.size() != payloads.size()) {
            std::fprintf(stderr,
                         "WARNING: keys (%zu) != payloads (%zu); they are cycled "
                         "independently so alignment requires equal counts\n",
                         keys.size(), payloads.size());
        }
    }

    flock::AsyncLLMClient::Options opts;
    opts.request_timeout_ms = args.timeout_ms;
    flock::AsyncLLMClient client(opts);

    std::fprintf(stderr,
                 "strategy=%s endpoints=%zu inflight=%d warmup=%d total=%d "
                 "timeout_ms=%d rows_per_request=%d payloads=%zu keys=%zu\n",
                 args.strategy.c_str(), endpoints.size(), args.inflight,
                 args.warmup, args.total, args.timeout_ms,
                 args.rows_per_request, payloads.size(), keys.size());

    std::vector<long> submit_counts(endpoints.size(), 0);

    if (args.warmup > 0) {
        Stats warm;
        RunPhase(client, router, args, payloads, keys, "warm-", args.warmup,
                 warm, submit_counts, /*record_latency=*/false, "warmup");
    }

    // Reset counts so the reported distribution reflects the main phase only.
    std::fill(submit_counts.begin(), submit_counts.end(), 0);

    Stats stats;
    auto t0 = std::chrono::steady_clock::now();
    RunPhase(client, router, args, payloads, keys, "main-", args.total,
             stats, submit_counts, /*record_latency=*/true, "main");
    auto t1 = std::chrono::steady_clock::now();

    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    int ok = stats.ok.load();
    int err = stats.err.load();

    // Human-readable line (stderr) + structured JSON (file).
    std::fprintf(stderr,
                 "MAIN strategy=%s total=%d ok=%d err=%d elapsed_s=%.3f "
                 "throughput_rows_per_s=%.4f\n",
                 args.strategy.c_str(), args.total, ok, err, elapsed_s,
                 static_cast<double>(ok) * args.rows_per_request / elapsed_s);
    std::fprintf(stderr, "per_endpoint_count:");
    for (size_t i = 0; i < submit_counts.size(); ++i) {
        std::fprintf(stderr, " ep%zu=%ld", i, submit_counts[i]);
    }
    std::fprintf(stderr, "\n");

    WriteResultJson(args, endpoints, submit_counts, elapsed_s, ok, err,
                    stats.latencies_us);

    return err == 0 ? 0 : 1;
}