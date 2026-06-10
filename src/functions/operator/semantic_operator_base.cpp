#include "flock/functions/operator/semantic_operator_base.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace flock {

using duckdb::DataChunk;
using duckdb::idx_t;
using duckdb::InterruptState;
using duckdb::SinkFinalizeType;
using duckdb::SinkResultType;
using duckdb::SourceResultType;

namespace {

// Parse an OpenAI-style completions response: {"choices":[ ... ]}.
//
// Fail-loud policy: a failed HTTP request, unparseable body, missing choices
// array, or a choices/rows length mismatch all throw.
nlohmann::json ParseChoices(const CompletedBatch& batch) {
    if (!batch.response.ok) {
        throw std::runtime_error("flock semantic operator: LLM request failed (http " +
                                 std::to_string(batch.response.http_status) + "): " + batch.response.error);
    }
    nlohmann::json parsed = nlohmann::json::parse(batch.response.body, /*cb=*/nullptr, /*allow_exceptions=*/false);
    if (parsed.is_discarded()) {
        throw std::runtime_error("flock semantic operator: response body is not valid JSON");
    }
    auto it = parsed.find("choices");
    if (it == parsed.end() || !it->is_array()) {
        throw std::runtime_error("flock semantic operator: response missing a 'choices' array");
    }
    if (it->size() != batch.row_ids.size()) {
        throw std::runtime_error("flock semantic operator: choices/rows length mismatch (" +
                                 std::to_string(it->size()) + " choices vs " +
                                 std::to_string(batch.row_ids.size()) + " rows)");
    }
    return std::move(*it);
}

}  // namespace

SemGlobalSinkState::SemGlobalSinkState(std::shared_ptr<ILLMClient> client,
                                       std::shared_ptr<EndpointRouter> router, SemanticParams cfg)
    : client(std::move(client)), router(std::move(router)), cfg(std::move(cfg)) {
}

SemGlobalSinkState::~SemGlobalSinkState() {
    // UAF guard on teardown / cancellation. Bump the generation FIRST so any
    // late completion that races the destructor no-ops in OnBatchComplete, then
    // tell the client to drop every outstanding callback for the prior
    // generation. The client serializes the drop against in-progress callbacks,
    // so once CancelByGeneration returns, no callback referencing `this` can
    // still be running -- and member teardown that follows is safe.
    const uint64_t old = generation.fetch_add(1, std::memory_order_acq_rel);
    if (client) {
        client->CancelByGeneration(old);
    }
}

void SemGlobalSinkState::NoteInFlight(size_t now_in_flight) {
    size_t prev = max_in_flight_seen.load(std::memory_order_relaxed);
    while (now_in_flight > prev &&
           !max_in_flight_seen.compare_exchange_weak(prev, now_in_flight, std::memory_order_relaxed)) {
        // prev reloaded by compare_exchange_weak on failure; retry.
    }
}

std::string SemGlobalSinkState::BuildPayload(const std::vector<PendingRequest>& batch) const {
    // N prompts in one /v1/completions
    // body via prompt:[array]; the response returns a choices:[array] aligned by
    // index. [TO DO - add functionality] Model name + response_format schema wiring.
    nlohmann::json body;
    auto prompts = nlohmann::json::array();
    for (const auto& p : batch) {
        prompts.push_back(p.prompt);
    }
    body["prompt"] = std::move(prompts);
    body["max_tokens"] = cfg.max_output_tokens;
    return body.dump();
}

void SemGlobalSinkState::SubmitBatch(std::vector<PendingRequest> batch) {
    if (batch.empty()) {
        return;
    }
    // OUTSIDE the lock: routing + payload build + submit.
    // Sticky routing key: the first row's rendered prompt (its cacheable head).
    const std::string& prefix_key = batch.front().prompt;
    EndpointRouter::Pick pick = router->Choose(prefix_key);
    std::string payload = BuildPayload(batch);
    const uint64_t gen = generation.load(std::memory_order_acquire);

    // Carry the per-row data the completion needs, preserving submission order:
    //   row_ids[i] <-> rows[i] <-> prompt[i] <-> choices[i].
    std::vector<uint64_t> row_ids;
    std::vector<RowData> rows;
    row_ids.reserve(batch.size());
    rows.reserve(batch.size());
    for (auto& p : batch) {
        row_ids.push_back(p.row_id);
        rows.push_back(std::move(p.row));
    }
    const size_t endpoint_index = pick.index;
    const std::string request_id = "sem-batch-" + std::to_string(row_ids.front());

    // Move the carry-over into the callback; it fires at most once on the IO
    // thread. The captured `gen` is the generation guard's left-hand side.
    client->Submit(pick.url, payload, gen, request_id,
                   [this, gen, endpoint_index, ids = std::move(row_ids),
                    rws = std::move(rows)](LLMResponse resp) mutable {
                       OnBatchComplete(gen, endpoint_index, std::move(ids), std::move(rws), std::move(resp));
                   });
}

SinkResultType SemGlobalSinkState::MergeAndCoalesce(SemLocalSinkState& local, idx_t resume_idx,
                                                    InterruptState& interrupt) {
    std::vector<std::vector<PendingRequest>> to_submit;
    bool must_block = false;
    {
        auto guard = Lock();
        // Force-merge the per-thread staging buffer into the global queue.
        for (auto& p : local.pending) {
            pending.push_back(std::move(p));
        }
        local.pending.clear();

        // Coalesce as many full batches as the cap allows, in submission order.
        // The read-check-of-in_flight and the increment are BOTH under the lock,
        // so the cap is authoritative - - -> in_flight can never exceed the cap.
        while (pending.size() >= cfg.coalesce_size) {
            if (in_flight.load(std::memory_order_relaxed) < cfg.in_flight_cap) {
                std::vector<PendingRequest> batch;
                batch.reserve(static_cast<size_t>(cfg.coalesce_size));
                for (uint64_t k = 0; k < cfg.coalesce_size; ++k) {
                    batch.push_back(std::move(pending[k]));
                }
                pending.erase(pending.begin(),
                              pending.begin() + static_cast<std::ptrdiff_t>(cfg.coalesce_size));
                const size_t now = in_flight.fetch_add(1, std::memory_order_relaxed) + 1;
                NoteInFlight(now);
                to_submit.push_back(std::move(batch));
            } else {
                // Cap saturated AND a full batch is ready -> block THIS task
                // (per-batch, never per-row). Resume mid-chunk on wakeup.
                local.next_row_idx = resume_idx;
                blocked_sinks.push_back(interrupt);
                blocked_at_least_once.store(true, std::memory_order_relaxed);
                must_block = true;
                break;
            }
        }
    }
    // Submit OUTSIDE the lock.
    for (auto& batch : to_submit) {
        SubmitBatch(std::move(batch));
    }
    return must_block ? SinkResultType::BLOCKED : SinkResultType::NEED_MORE_INPUT;
}

SinkResultType SemGlobalSinkState::SinkChunk(SemLocalSinkState& local, DataChunk& chunk,
                                             InterruptState& interrupt) {
    const idx_t n = chunk.size();
    const idx_t cols = chunk.ColumnCount();
    // [TO DO - add functionality] run the residual ExpressionExecutor here -> survivors. Now, every row
    // is a survivor.
    for (idx_t i = local.next_row_idx; i < n; ++i) {
        // Full-row COPY: GetValue returns an owning Value, so the snapshot
        // outlives this (soon-recycled) chunk across the async round-trip.
        RowData row;
        row.values.reserve(cols);
        for (idx_t c = 0; c < cols; ++c) {
            row.values.push_back(chunk.GetValue(c, i));
        }
        PendingRequest req;
        req.row_id = next_row_id.fetch_add(1, std::memory_order_relaxed);  // query-global, lock-free
        req.prompt = render_prompt(row);
        req.row = std::move(row);
        local.pending.push_back(std::move(req));

        if (local.pending.size() >= LOCAL_MERGE_THRESHOLD) {
            // Resume AFTER this row -> it is already staged locally / merged.
            if (MergeAndCoalesce(local, i + 1, interrupt) == SinkResultType::BLOCKED) {
                return SinkResultType::BLOCKED;
            }
        }
    }
    // Chunk fully consumed: flush the local remainder and coalesce.
    if (MergeAndCoalesce(local, n, interrupt) == SinkResultType::BLOCKED) {
        return SinkResultType::BLOCKED;
    }
    local.next_row_idx = 0;
    return SinkResultType::NEED_MORE_INPUT;
}

void SemGlobalSinkState::MergeLocal(SemLocalSinkState& local) {
    if (local.pending.empty()) {
        return;
    }
    auto guard = Lock();
    for (auto& p : local.pending) {
        pending.push_back(std::move(p));
    }
    local.pending.clear();
}

SinkFinalizeType SemGlobalSinkState::FinalizeFlush(InterruptState& interrupt) {
    std::vector<std::vector<PendingRequest>> to_submit;
    bool must_block = false;
    {
        auto guard = Lock();
        // Submit ALL remaining rows -- including a trailing partial batch
        // (< coalesce_size) respecting the cap.
        // [TO DO] coalesce_max_age_ms is honored opportunistically here
        // and in SinkChunk; a dedicated timer thread that flushes aged partial
        // batches during idle / trickle input is deferred.
        while (!pending.empty()) {
            if (in_flight.load(std::memory_order_relaxed) < cfg.in_flight_cap) {
                const size_t take = std::min<size_t>(static_cast<size_t>(cfg.coalesce_size), pending.size());
                std::vector<PendingRequest> batch;
                batch.reserve(take);
                for (size_t k = 0; k < take; ++k) {
                    batch.push_back(std::move(pending[k]));
                }
                pending.erase(pending.begin(), pending.begin() + static_cast<std::ptrdiff_t>(take));
                const size_t now = in_flight.fetch_add(1, std::memory_order_relaxed) + 1;
                NoteInFlight(now);
                to_submit.push_back(std::move(batch));
            } else {
                // Cap saturated, leftover remains -> block; resume on wakeup.
                blocked_sinks.push_back(interrupt);
                blocked_at_least_once.store(true, std::memory_order_relaxed);
                must_block = true;
                break;
            }
        }
        if (!must_block) {
            // Everything submitted; no more input will arrive. Outstanding
            // batches are already counted in in_flight, so Drain still waits.
            input_exhausted = true;
        }
    }
    for (auto& batch : to_submit) {
        SubmitBatch(std::move(batch));
    }
    return must_block ? SinkFinalizeType::BLOCKED : SinkFinalizeType::READY;
}

void SemGlobalSinkState::OnBatchComplete(uint64_t submit_generation, size_t endpoint_index,
                                         std::vector<uint64_t> row_ids, std::vector<RowData> rows,
                                         LLMResponse response) {
    // Generation guard: the sole in-engine protection against use-after-free.
    // If the query was cancelled/destroyed, the captured generation no longer
    // matches the live one and we MUST NOT touch `this` past this point. (The
    // client also drops callbacks for cancelled generations; this handles a
    // completion that raced the bump before the client's drop took effect.)
    if (submit_generation != generation.load(std::memory_order_acquire)) {
        return;
    }
    std::vector<InterruptState> to_wake;
    {
        auto guard = Lock();
        CompletedBatch cb;
        cb.row_ids = std::move(row_ids);
        cb.rows = std::move(rows);
        cb.response = std::move(response);
        cb.endpoint_index = endpoint_index;
        completed.push(std::move(cb));
        in_flight.fetch_sub(1, std::memory_order_relaxed);
        router->OnComplete(endpoint_index);
        // Splice out blocked tasks; one completion wakes BOTH a cap-blocked
        // Sink and an empty-blocked Drain, each re-checks and re-blocks if it
        // still cannot make progress.
        for (auto& is : blocked_sinks) {
            to_wake.push_back(is);
        }
        blocked_sinks.clear();
        for (auto& is : blocked_sources) {
            to_wake.push_back(is);
        }
        blocked_sources.clear();
    }
    // Fire callbacks OUTSIDE the lock.
    for (auto& is : to_wake) {
        is.Callback();
    }
}

SourceResultType SemGlobalSinkState::Drain(DataChunk& out, const ParseFn& parse_fn,
                                           InterruptState& interrupt) {
    CompletedBatch batch;
    {
        auto guard = Lock();
        if (completed.empty()) {
            if (in_flight.load(std::memory_order_relaxed) > 0) {
                // Work outstanding: block until a completion wakes us.
                blocked_sources.push_back(interrupt);
                return SourceResultType::BLOCKED;
            }
            // Nothing queued and nothing in flight. In the source phase the sink
            // has finalized (input_exhausted), so there is no more output.
            return SourceResultType::FINISHED;
        }
        batch = std::move(completed.front());
        completed.pop();
    }
    // Parse + emit OUTSIDE the lock. One batch per call; coalesce_size is
    // bounded well under STANDARD_VECTOR_SIZE, so a batch always fits in `out`.
    nlohmann::json choices = ParseChoices(batch);  // fail-loud on any anomaly
    for (size_t i = 0; i < batch.rows.size(); ++i) {
        parse_fn(choices[i], batch.rows[i], out);
    }
    return SourceResultType::HAVE_MORE_OUTPUT;
}

}  // namespace flock