#include "flock/functions/operator/semantic_operator_base.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
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

// FILTER output-schema mode from FLOCK_SEM_SCHEMA, read once. Default kBool keeps
// the committed behaviour byte-identical.
SemSchema SemSchemaModeFromEnv() {
    static const SemSchema mode = [] {
        const char* p = std::getenv("FLOCK_SEM_SCHEMA");
        const std::string s = p ? p : "";
        if (s == "id") {
            return SemSchema::kId;
        }
        if (s == "id_reason") {
            return SemSchema::kIdReason;
        }
        return SemSchema::kBool;
    }();
    return mode;
}

// kIdReason rationale budget from FLOCK_SEM_REASON_WORDS (default 12), read once.
int SemReasonWords() {
    static const int words = [] {
        const char* p = std::getenv("FLOCK_SEM_REASON_WORDS");
        if (p && *p) {
            const int v = std::atoi(p);
            if (v > 0) {
                return v;
            }
        }
        return 12;
    }();
    return words;
}

// Check if prompt slim-ing is on.
bool SemPromptSlim() {
    static const bool slim = [] {
        const char* p = std::getenv("FLOCK_SEM_PROMPT");
        return p && std::string(p) == "slim";
    }();
    return slim;
}

namespace {

// Parse one /v1/chat/completions response into the batch's per-row response array.
//
// The batch was sent as ONE multi-row chat prompt, so the response carries ONE
// completion whose message.content is a JSON object holding one element per row
// under the "items" key. This is flock's shared batch envelope (BatchAndComplete
// returns the same {"items":[...]} for FILTER, COMPLETE, ...; see
// scalar/llm_filter/implementation.cpp's CollectCompletions()[0]["items"]), so
// the operator and the scalar agree. Element TYPE (bool, string, struct, ...) is
// the operator's concern -> this function only realigns the array with its rows.
//
// Degraded-response parity with the scalar's BatchAndComplete: a short items
// array is padded with null and a long one truncated, so the result aligns by
// row (ParseVerdict maps null -> pass) instead of failing the query.
//
// Fail-loud policy: a failed HTTP request or an unparseable body/content/missing
// choices/message/items field still throws.
nlohmann::json ParseItems(const CompletedBatch& batch) {
    if (!batch.response.ok) {
        throw std::runtime_error("flock semantic operator: LLM request failed (http " +
                                 std::to_string(batch.response.http_status) + "): " + batch.response.error);
    }
    nlohmann::json parsed = nlohmann::json::parse(batch.response.body, /*cb=*/nullptr, /*allow_exceptions=*/false);
    if (parsed.is_discarded()) {
        std::fprintf(stderr, "[flock debug] operator: response body not valid JSON (len=%zu):\n%.2000s\n",
                     batch.response.body.size(), batch.response.body.c_str());
        throw std::runtime_error("flock semantic operator: response body is not valid JSON");
    }
    auto cit = parsed.find("choices");
    if (cit == parsed.end() || !cit->is_array() || cit->empty()) {
        throw std::runtime_error("flock semantic operator: response missing a non-empty 'choices' array");
    }
    // ONE chat completion for the whole batch: choices[0].message.content holds
    // the {"items":[...]} envelope (chat/completions, for parity with llm_filter).
    const auto& choice = (*cit)[0];
    auto mit = choice.find("message");
    if (mit == choice.end() || !mit->is_object()) {
        throw std::runtime_error("flock semantic operator: chat choice missing a 'message' object");
    }
    auto ctit = mit->find("content");
    if (ctit == mit->end() || !ctit->is_string()) {
        throw std::runtime_error("flock semantic operator: chat message missing a 'content' string");
    }
    const std::string completion_text = ctit->get<std::string>();
    nlohmann::json completion =
            nlohmann::json::parse(completion_text, /*cb=*/nullptr, /*allow_exceptions=*/false);
    if (completion.is_discarded()) {
        std::fprintf(stderr, "[flock debug] operator: message content not valid JSON (len=%zu):\n%.2000s\n",
                     completion_text.size(), completion_text.c_str());
        throw std::runtime_error("flock semantic operator: message content is not valid JSON");
    }
    auto iit = completion.find("items");
    if (iit == completion.end() || !iit->is_array()) {
        throw std::runtime_error("flock semantic operator: completion missing an 'items' array");
    }
    const size_t want = batch.row_ids.size();
    // id / id_reason (Rec-1 ablation): each element is an object {id[,reason],
    // verdict}. Realign to positional order BY the batch-local id (1..R) and
    // reduce to the bare verdict, so ParseVerdict / ParseAndEmit / DumpVerdicts
    // and the diagnostic all stay identical to the bool path. A missing id stays
    // null -> pass (parity); extras / out-of-range ids are ignored.
    if (SemSchemaModeFromEnv() != SemSchema::kBool) {
        nlohmann::json verdicts = nlohmann::json::array();
        for (size_t i = 0; i < want; ++i) {
            verdicts.push_back(nullptr);
        }
        for (const auto& obj: *iit) {
            if (!obj.is_object()) {
                continue;
            }
            const auto id_it = obj.find("id");
            const auto v_it = obj.find("verdict");
            if (id_it == obj.end() || !id_it->is_number_integer() || v_it == obj.end()) {
                continue;
            }
            const long id = id_it->get<long>();
            if (id >= 1 && static_cast<size_t>(id) <= want) {
                verdicts[static_cast<size_t>(id) - 1] = *v_it;
            }
        }
        return verdicts;
    }
    // bool: pad short with null (-> pass via ParseVerdict), truncate long. Under
    // guided decoding the array is already exactly batch-sized, so this only
    // matters for degraded responses.
    nlohmann::json items = std::move(*iit);
    while (items.size() < want) {
        items.push_back(nullptr);
    }
    if (items.size() > want) {
        items.erase(items.begin() + static_cast<long>(want), items.end());
    }
    return items;
}

// Optional per-row verdict dump for prompt-degradation diagnostics
// (env FLOCK_VERDICT_DUMP=<path>). One JSONL line per row: the query-global
// row_id (== table scan order at threads=1), the row's 0-based position within
// its batch, and the raw model verdict element. Inert unless the env var is set;
// buffered (no per-batch flush) and, by protocol, only ever enabled on an
// untimed pass, so it cannot perturb a measured run. Serialises the raw element
// generically, so it is not filter-specific.
void DumpVerdicts(const CompletedBatch& batch, const nlohmann::json& items) {
    static std::FILE* out = [] {
        const char* p = std::getenv("FLOCK_VERDICT_DUMP");
        return (p && *p) ? std::fopen(p, "wb") : nullptr;
    }();
    if (!out) {
        return;
    }
    static std::mutex mu;
    std::lock_guard<std::mutex> guard(mu);
    for (size_t i = 0; i < batch.rows.size(); ++i) {
        std::fprintf(out, "{\"id\":%llu,\"pos\":%zu,\"v\":%s}\n",
                     static_cast<unsigned long long>(batch.row_ids[i]), i, items[i].dump().c_str());
    }
}

}  // namespace

SemGlobalSinkState::SemGlobalSinkState(std::shared_ptr<ILLMClient> client,
                                       std::shared_ptr<EndpointRouter> router, SemanticParams cfg)
    : generation(NextGeneration()), client(std::move(client)), router(std::move(router)),
      cfg(std::move(cfg)) {
}

SemGlobalSinkState::~SemGlobalSinkState() {
    // First statement, and WITHOUT the sink lock: the drain blocks until the IO
    // thread's in-flight callback returns, so members are still alive when it
    // finishes + that callback wants the lock, so holding it would deadlock.
    if (client) {
        client->CancelByGeneration(generation.load(std::memory_order_acquire));
    }
}

void SemGlobalSinkState::NoteInFlight(size_t now_in_flight) {
    size_t prev = max_in_flight_seen.load(std::memory_order_relaxed);
    while (now_in_flight > prev &&
           !max_in_flight_seen.compare_exchange_weak(prev, now_in_flight, std::memory_order_relaxed)) {
        // prev reloaded by compare_exchange_weak on failure; retry.
    }
}

std::string SemGlobalSinkState::BuildPayload(const std::string& prompt, size_t batch_rows) const {
    // ONE multi-row prompt -> ONE /v1/chat/completions request (flock's
    // BatchAndComplete strategy). The single completion returns an "items" array
    // of batch_rows elements, realigned with the rows in ParseItems.
    nlohmann::json body;
    // Chat-completions for parity with the scalar llm_filter path: same single
    // user message with a text content part, so vLLM applies the model's chat
    // template (+ default system prompt) identically for both arms. (Mirrors
    // flock's openai adapter message shape.)
    body["messages"] = nlohmann::json::array(
            {{{"role", "user"}, {"content", nlohmann::json::array({{{"type", "text"}, {"text", prompt}}})}}});
    if (!served_model.empty()) {
        body["model"] = served_model;
    }
    body["max_tokens"] = cfg.max_output_tokens;
    // Deterministic, fair A/B: greedy decoding. Explicitly overrides the model's
    // generation_config.json default (temperature=0.7) so the operator arm is
    // reproducible and matches the scalar arm (which pins temperature via
    // model_parameters). temperature=0 -> greedy, so top_p/top_k are no-ops.
    body["temperature"] = 0.0;
    if (cfg.response_format == "json_schema" && response_schema) {
        nlohmann::json rf = response_schema(batch_rows);
        if (!rf.is_null()) {
            body["response_format"] = std::move(rf);
        }
    }
    return body.dump();
}

void SemGlobalSinkState::SubmitBatch(std::vector<PendingRequest> batch) {
    if (batch.empty()) {
        return;
    }
    // OUTSIDE the lock: gather the carry-over, render, route, build, submit.
    // Preserve submission order: row_ids[i] <-> rows[i] <-> items[i].
    std::vector<uint64_t> row_ids;
    std::vector<RowData> rows;
    row_ids.reserve(batch.size());
    rows.reserve(batch.size());
    for (auto& p : batch) {
        row_ids.push_back(p.row_id);
        rows.push_back(std::move(p.row));
    }
    // Pack the whole batch into ONE multi-row prompt (render hook is read-only
    // after setup, so it is lock-free here).
    const std::string prompt = render_prompt(rows);
    // Routing key for StickyByPrefix. Prefer the operator-supplied sticky_key (the
    // query-stable template head). Fall back to a bounded leading slice of the
    // prompt, which stays inside flock's shared instruction/template head while
    // excluding the variable per-row tail; keep that slice short so it never
    // reaches the tail and scatters a query's batches across endpoints.
    const std::string key = sticky_key.empty() ? prompt.substr(0, kStickyPrefixBytes) : sticky_key;
    EndpointRouter::Pick pick = router->Choose(key);
    std::string payload = BuildPayload(prompt, rows.size());
    const uint64_t gen = generation.load(std::memory_order_acquire);
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
        while (pending.size() >= cfg.batch_size) {
            if (in_flight.load(std::memory_order_relaxed) < cfg.in_flight_cap) {
                std::vector<PendingRequest> batch;
                batch.reserve(static_cast<size_t>(cfg.batch_size));
                for (uint64_t k = 0; k < cfg.batch_size; ++k) {
                    batch.push_back(std::move(pending.front()));
                    pending.pop_front();  // O(1) on deque
                }
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
    const idx_t cols = chunk.ColumnCount();
    // Residual pre-filter: drop rows failing the non-LLM conjuncts before the LLM
    // round-trip. Computed ONLY on a fresh chunk (next_row_idx == 0) and cached in
    // `local`; a BLOCKED resume reuses the cached selection. No executor -> identity.
    if (local.next_row_idx == 0) {
        local.survivor_count = local.residual_executor
                                       ? local.residual_executor->SelectExpression(chunk, local.residual_sel)
                                       : chunk.size();
    }
    const idx_t n = local.survivor_count;
    for (idx_t i = local.next_row_idx; i < n; ++i) {
        const idx_t src = local.residual_executor ? local.residual_sel.get_index(i) : i;
        // Full-row COPY: GetValue returns an owning Value, so the snapshot
        // outlives this (soon-recycled) chunk across the async round-trip.
        RowData row;
        row.values.reserve(cols);
        for (idx_t c = 0; c < cols; ++c) {
            row.values.push_back(chunk.GetValue(c, src));
        }
        // Capture the row only; the prompt is rendered per BATCH by the
        // coalescer (SubmitBatch), not per row.
        PendingRequest req;
        req.row_id = next_row_id.fetch_add(1, std::memory_order_relaxed);  // query-global, lock-free
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
    std::vector<InterruptState> to_wake;
    bool must_block = false;
    {
        auto guard = Lock();
        // Submit ALL remaining rows -- including a trailing partial batch
        // (< batch_size) respecting the cap.
        // [TO DO] coalesce_max_age_ms is honored opportunistically here
        // and in SinkChunk; a dedicated timer thread that flushes aged partial
        // batches during idle / trickle input is deferred.
        while (!pending.empty()) {
            if (in_flight.load(std::memory_order_relaxed) < cfg.in_flight_cap) {
                const size_t take = std::min<size_t>(static_cast<size_t>(cfg.batch_size), pending.size());
                std::vector<PendingRequest> batch;
                batch.reserve(take);
                for (size_t k = 0; k < take; ++k) {
                    batch.push_back(std::move(pending.front()));
                    pending.pop_front();  // O(1) front pop
                }
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
            for (auto& is : blocked_sources) {
                to_wake.push_back(is);
            }
            blocked_sources.clear();
        }
    }
    for (auto& batch : to_submit) {
        SubmitBatch(std::move(batch));
    }
    for (auto& is : to_wake) {
        is.Callback();
    }
    return must_block ? SinkFinalizeType::BLOCKED : SinkFinalizeType::READY;
}

void SemGlobalSinkState::OnBatchComplete(uint64_t submit_generation, size_t endpoint_index,
                                         std::vector<uint64_t> row_ids, std::vector<RowData> rows,
                                         LLMResponse response) {
    // Defense-in-depth, not the UAF guard (the client's drain is authoritative
    // and per-query generations never collide); a cheap no-op for a cancelled gen.
    // TODO: this skips router->OnComplete below (leaks the endpoint counter)
    // fix if live-query row-level cancellation is added; harmless on teardown.
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
            // Terminate ONLY once the sink is finalized AND nothing is in flight:
            if (in_flight.load(std::memory_order_relaxed) > 0 || !input_exhausted) {
                blocked_sources.push_back(interrupt);
                return SourceResultType::BLOCKED;
            }
            return SourceResultType::FINISHED;
        }
        batch = std::move(completed.front());
        completed.pop();
    }
    // Parse + emit OUTSIDE the lock. One batch per call; batch_size is bounded
    // well under STANDARD_VECTOR_SIZE, so a batch always fits in `out`.
    nlohmann::json items = ParseItems(batch);  // fail-loud on any anomaly
    for (size_t i = 0; i < batch.rows.size(); ++i) {
        parse_fn(items[i], batch.rows[i], out);
    }
    DumpVerdicts(batch, items);  // no-op unless FLOCK_VERDICT_DUMP is set (diagnostics)
    return SourceResultType::HAVE_MORE_OUTPUT;
}

// =============================================================================
// SemanticOperatorBase -- thin PhysicalOperator adapters over the engine.
// Each method Casts DuckDB's state to the engine state and forwards; no logic
// lives here. The two pure virtuals (RenderPrompt / ParseAndEmit) are the only
// per-operator behavior, bound into the engine via the render hook + ParseFn.
// =============================================================================

SemanticOperatorBase::SemanticOperatorBase(duckdb::PhysicalPlan& physical_plan,
                                           duckdb::PhysicalOperatorType type,
                                           duckdb::vector<duckdb::LogicalType> types,
                                           duckdb::idx_t estimated_cardinality,
                                           std::shared_ptr<ILLMClient> client,
                                           std::shared_ptr<EndpointRouter> router, SemanticParams cfg)
    : duckdb::PhysicalOperator(physical_plan, type, std::move(types), estimated_cardinality),
      client(std::move(client)), router(std::move(router)), cfg(std::move(cfg)) {
}

nlohmann::json SemanticOperatorBase::BuildResponseFormat(size_t /*batch_rows*/) const {
    // Default: no guided schema. Concrete operators override to supply one.
    return nullptr;
}

nlohmann::json SemanticOperatorBase::ItemsResponseFormat(const std::string& name, nlohmann::json element,
                                                         size_t batch_rows) {
    nlohmann::json items_schema = {{"type", "array"},
                                   {"minItems", batch_rows},
                                   {"maxItems", batch_rows},
                                   {"items", std::move(element)}};
    return {{"type", "json_schema"},
            {"json_schema",
             {{"name", name},
              {"schema",
               {{"type", "object"},
                {"properties", {{"items", std::move(items_schema)}}},
                {"required", nlohmann::json::array({"items"})},
                {"additionalProperties", false}}}}}};
}

duckdb::unique_ptr<SemGlobalSinkState> SemanticOperatorBase::CreateGlobalSinkState() const {
    auto state = duckdb::make_uniq<SemGlobalSinkState>(client, router, cfg);
    // Bind the engine's render + schema hooks to this operator. `this` outlives the
    // sink state (the operator owns sink_state), and both hooks are set once before
    // any Sink call, then only read.
    state->render_prompt = [this](const std::vector<RowData>& batch) { return RenderPrompt(batch); };
    state->response_schema = [this](size_t batch_rows) { return BuildResponseFormat(batch_rows); };
    return state;
}

duckdb::unique_ptr<duckdb::GlobalSinkState> SemanticOperatorBase::GetGlobalSinkState(
    duckdb::ClientContext& /*context*/) const {
    return CreateGlobalSinkState();
}

duckdb::unique_ptr<duckdb::LocalSinkState> SemanticOperatorBase::GetLocalSinkState(
    duckdb::ExecutionContext& /*context*/) const {
    return duckdb::make_uniq<SemLocalSinkState>();
}

duckdb::SinkResultType SemanticOperatorBase::Sink(duckdb::ExecutionContext& /*context*/, DataChunk& chunk,
                                                  duckdb::OperatorSinkInput& input) const {
    return input.global_state.Cast<SemGlobalSinkState>().SinkChunk(
        input.local_state.Cast<SemLocalSinkState>(), chunk, input.interrupt_state);
}

duckdb::SinkCombineResultType SemanticOperatorBase::Combine(duckdb::ExecutionContext& /*context*/,
                                                            duckdb::OperatorSinkCombineInput& input) const {
    input.global_state.Cast<SemGlobalSinkState>().MergeLocal(input.local_state.Cast<SemLocalSinkState>());
    return duckdb::SinkCombineResultType::FINISHED;
}

SinkFinalizeType SemanticOperatorBase::Finalize(duckdb::Pipeline& /*pipeline*/, duckdb::Event& /*event*/,
                                                duckdb::ClientContext& /*context*/,
                                                duckdb::OperatorSinkFinalizeInput& input) const {
    return input.global_state.Cast<SemGlobalSinkState>().FinalizeFlush(input.interrupt_state);
}

duckdb::unique_ptr<duckdb::GlobalSourceState> SemanticOperatorBase::GetGlobalSourceState(
    duckdb::ClientContext& /*context*/) const {
    return duckdb::make_uniq<SemGlobalSourceState>(cfg.in_flight_cap);
}

duckdb::unique_ptr<duckdb::LocalSourceState> SemanticOperatorBase::GetLocalSourceState(
    duckdb::ExecutionContext& /*context*/, duckdb::GlobalSourceState& /*gstate*/) const {
    return duckdb::make_uniq<SemLocalSourceState>();
}

SourceResultType SemanticOperatorBase::GetDataInternal(duckdb::ExecutionContext& /*context*/, DataChunk& chunk,
                                                       duckdb::OperatorSourceInput& input) const {
    // The confirmed bridge: the source phase reads the SINK state.
    auto& gss = sink_state->Cast<SemGlobalSinkState>();
    auto parse = [this](const nlohmann::json& element, const RowData& row, DataChunk& out) {
        ParseAndEmit(element, row, out);
    };
    return gss.Drain(chunk, parse, input.interrupt_state);
}

}  // namespace flock