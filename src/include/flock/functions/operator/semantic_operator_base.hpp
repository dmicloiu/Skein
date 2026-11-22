#pragma once

#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/selection_vector.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/physical_operator_states.hpp"
#include "duckdb/parallel/interrupt.hpp"
#include "flock/functions/operator/proxy_stage.hpp"
#include "flock/runtime/endpoint_router.h"
#include "flock/runtime/llm_client.h"
#include "flock/runtime/pending_request.h"
#include "flock/runtime/semantic_settings.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace flock {

// =============================================================================
// Async Tier-3 semantic-operator dispatch engine.
//
// THE ONE CLAIM THIS CODE MAKES TRUE: morsel/thread count must not govern GPU
// concurrency. The async submission and not the thread count is what
// decouples GPU concurrency from DuckDB's parallelism.
//
// All shared mutable state and the dispatch state machine live in
// SemGlobalSinkState. The PhysicalOperator is a thin adapter that
// Casts DuckDB's state objects and forwards here.
// =============================================================================

// PROMPT SLIMMING -> default includes the original flock meta-prompt
bool SemPromptSlim();

// URL for an image context value as sent in chat payloads: http(s):// and data:
// URLs pass through, a local path becomes a base64 data URL (MIME type from the
// column's type or the file extension). Throws if the file cannot be read.
std::string ImageDataUrl(const nlohmann::json& column, const std::string& path);

// A coalesced batch that has returned from the LLM, awaiting parse + emit by
// the source phase. Carries everything Drain needs to realign the response
// items with their originating rows.
struct CompletedBatch {
    std::vector<uint64_t> row_ids;  // submission order
    std::vector<RowData> rows;      // rows[i] <-> row_ids[i] <-> items[i]
    LLMResponse response;           // raw HTTP result; parsed once, in Drain
    size_t endpoint_index = 0;      // router slot (already released on completion)
    bool proxy = false;
};

// Per-thread sink state. Touched by exactly one DuckDB thread, so it is
// lock-free by construction.
class SemLocalSinkState : public duckdb::LocalSinkState {
public:
    // Per-thread staging buffer. Flushed into the global `pending` once it
    // reaches LOCAL_MERGE_THRESHOLD, or when a chunk is fully consumed.
    std::vector<PendingRequest> pending;

    // Resume cursor INTO THE CURRENT CHUNK.
    duckdb::idx_t next_row_idx = 0;

    // Residual-predicate pre-filter. Null when there are no non-LLM conjuncts, in
    // which case every row survives. SinkChunk evaluates it once per fresh chunk and
    // caches the survivor selection + count so a BLOCKED resume reuses them.
    duckdb::unique_ptr<duckdb::ExpressionExecutor> residual_executor;
    duckdb::SelectionVector residual_sel;
    duckdb::idx_t survivor_count = 0;
};

// Global sink state + the dispatch state machine. One instance per query.
class SemGlobalSinkState : public duckdb::GlobalSinkState {
public:
    // DuckDB 2.0 removed StateWithBlockableTasks::Lock() (the base now exposes an
    // annotated `lock`). Keep the scoped-guard idiom over that same mutex.
    duckdb::annotated_unique_lock<duckdb::annotated_mutex> Lock() const {
        return duckdb::annotated_unique_lock<duckdb::annotated_mutex>(lock);
    }

    // Renders ONE multi-row prompt that packs the whole batch (flock's
    // BatchAndComplete strategy). batch_size rows -> one prompt -> one request.
    using RenderFn = std::function<std::string(const std::vector<RowData>& batch)>;
    using MediaFn = std::function<nlohmann::json(const std::vector<RowData>& batch)>;
    // Parses one element of the response's boolean_array and appends its output
    // row to `out`. items[i] <-> rows[i].
    using ParseFn = std::function<void(const nlohmann::json& element, const RowData&, duckdb::DataChunk& out)>;

    // No ClientContext
    SemGlobalSinkState(std::shared_ptr<ILLMClient> client, std::shared_ptr<EndpointRouter> router,
                       SemanticParams cfg);
    // Drain-cancels this query's callbacks via the client before members die --
    // the UAF guard on teardown / cancellation.
    ~SemGlobalSinkState() override;

    // Set ONCE before any Sink call (by the operator or directly by the test);
    // read-only and lock-free thereafter, so concurrent batch renders share it
    // safely. Invoked per batch by SubmitBatch (the coalescer), OUTSIDE the lock;
    // the analogous parse hook is a Drain parameter because parsing runs in the
    // source phase.
    RenderFn render_prompt;
    // Image columns for the same rows, in prompt column and row order.
    // Empty by default for text-only operators and direct engine tests.
    MediaFn render_media;

    // Served-model id stamped into body["model"]. Set ONCE before any Sink (like
    // render_prompt); empty -> omitted from the body.
    std::string served_model;
    // Optional score-only first stage for filters. Empty disables the cascade.
    std::string proxy_served_model;
    RenderFn render_proxy_prompt;
    // Optional buffered embedding proxy (kinds "embedding" / "lotus"). Set ONCE
    // before any Sink; null -> rows stream to the main model as before. When set,
    // Sink only buffers; FinalizeFlush plans (ProxyStage::Plan) and the policy
    // then decides which units reach the main model. Never set together with
    // proxy_served_model (the LLM-scored cascade).
    std::shared_ptr<ProxyStage> proxy_stage;
    // Main-model element -> boolean verdict, fed to the proxy policy. Required
    // when proxy_stage is set (the filter binds PhysicalSemFilter::ParseVerdict).
    std::function<bool(const nlohmann::json&)> verdict_of;
    // Routing key for StickyByPrefix: a per-query-stable string so a query's batches
    // co-locate on one endpoint for vLLM prefix-cache reuse. Set ONCE before any
    // Sink; empty -> SubmitBatch falls back to a leading slice of the prompt.
    std::string sticky_key;

    // Per-operator guided-decoding schema. Set ONCE before any Sink
    // (like render_prompt), bound by CreateGlobalSinkState to the operator's
    // BuildResponseFormat. Returns the JSON assigned to body["response_format"]
    // for a batch of `batch_rows` rows, or null to send no guided schema. Null /
    // unset -> BuildPayload emits no response_format (free_form). This is where
    // filter (boolean element) and extract (string element) diverge; the engine
    // stays element-type agnostic.
    using SchemaFn = std::function<nlohmann::json(size_t batch_rows)>;
    SchemaFn response_schema;

    // LIMIT early stop. Set ONCE before any Sink. limit_rows > 0 means a LIMIT
    // above this operator needs only that many output rows (limit + offset), and
    // limit_emits tells whether a response element yields an output row. Sink
    // then counts the output rows of the batches that already came back; once
    // there are enough, it stops reading input (FINISHED) and Finalize drops the
    // rows not yet sent. Batches in flight still complete and are emitted, so
    // the operator may return more rows than the LIMIT; the LIMIT trims them.
    // 0 (default) -> off. Never set together with a proxy.
    uint64_t limit_rows = 0;
    std::function<bool(const nlohmann::json&)> limit_emits;

    // --- engine entry points (operator forwards here; the test calls directly) ---

    // Sink one chunk. Returns NEED_MORE_INPUT when the chunk is consumed, or
    // BLOCKED when the cap is saturated with a full batch read. The caller
    // awaits a completion and retries the SAME chunk.
    duckdb::SinkResultType SinkChunk(SemLocalSinkState& local, duckdb::DataChunk& chunk,
                                     duckdb::InterruptState& interrupt);
    // Combine: force-merge a finished thread's leftover staging buffer.
    void MergeLocal(SemLocalSinkState& local);
    // Finalize: submit everything left (including a trailing partial batch),
    // respecting the cap. May return BLOCKED; on resume, submit the rest. Sets
    // input_exhausted and returns READY once `pending` is empty.
    duckdb::SinkFinalizeType FinalizeFlush(duckdb::InterruptState& interrupt);
    // Source phase: pop one completed batch, parse its body once, emit its rows.
    duckdb::SourceResultType Drain(duckdb::DataChunk& out, const ParseFn& parse_fn,
                                   duckdb::InterruptState& interrupt);

    // --- monitoring helpers for testing ---
    size_t MaxInFlightSeen() const { return max_in_flight_seen.load(std::memory_order_acquire); }
    bool BlockedAtLeastOnce() const { return blocked_at_least_once.load(std::memory_order_acquire); }
    size_t InFlight() const { return in_flight.load(std::memory_order_acquire); }
    uint64_t Generation() const { return generation.load(std::memory_order_acquire); }
    bool LimitReachedFlag() const { return limit_reached.load(std::memory_order_acquire); }
    uint64_t LimitRowsCounted() const { return limit_counted_rows.load(std::memory_order_acquire); }
    const SemanticParams& Config() const { return cfg; }

    duckdb::idx_t MaxThreads(duckdb::idx_t /*source_max_threads*/) override {
        return static_cast<duckdb::idx_t>(cfg.in_flight_cap);
    }

private:
    // Completion callback body. Runs on the ILLMClient IO thread; keep cheap.
    void OnBatchComplete(uint64_t submit_generation, size_t endpoint_index, std::vector<uint64_t> row_ids,
                         std::vector<RowData> rows, LLMResponse response, bool proxy);
    // Merge local.pending into the global buffer, then coalesce + submit as many
    // full batches as the cap allows. Returns BLOCKED (and records resume_idx)
    // iff a full batch is ready but the cap is saturated.
    duckdb::SinkResultType MergeAndCoalesce(SemLocalSinkState& local, duckdb::idx_t resume_idx,
                                            duckdb::InterruptState& interrupt);
    // Build the /v1/completions JSON body for a batch: ONE multi-row prompt
    std::string BuildPayload(const std::string& prompt, const nlohmann::json& image_columns,
                             size_t batch_rows, bool proxy) const;
    // Render the batch's single prompt, choose an endpoint, build the payload,
    // and Submit (all OUTSIDE the lock).
    void SubmitBatch(std::vector<PendingRequest> batch, bool proxy);
    std::vector<std::vector<PendingRequest>> TakeReadyMainBatches();
    // Embedding proxy: queue a policy step (sends -> pending_main, resolutions ->
    // pending_emit). Caller holds Lock().
    void ApplyProxyStepLocked(const ProxyStep& step);
    // Embedding proxy: move up to one vector of decided rows into `out`.
    // Returns true iff it emitted anything or rows remain queued.
    bool EmitDecided(duckdb::DataChunk& out, const ParseFn& parse_fn);
    // Monotonically record the in-flight peak (test instrumentation).
    void NoteInFlight(size_t now_in_flight);
    // LIMIT early stop: count the output rows of completed batches not yet
    // counted and return true once limit_rows is reached (always false when
    // limit_rows == 0). Called from Sink / Finalize only, never from the source.
    bool LimitReached();

    static constexpr size_t LOCAL_MERGE_THRESHOLD = 8;
    // Bytes of the rendered prompt used as the StickyByPrefix routing key. Kept
    // short on purpose: it must stay inside the shared template head so a query's
    // batches hash identically; a longer slice would reach the per-row tail and
    // scatter them. See SubmitBatch.
    static constexpr size_t kStickyPrefixBytes = 256;

    // --- guarded by THE one lock: the inherited StateWithBlockableTasks::Lock()
    //     Critical sections move buffers and adjust counters ONLY,
    //     router->Choose, client->Submit, or Callback(). ---
    std::deque<PendingRequest> pending;   // global FIFO
    std::deque<PendingRequest> pending_main;
    // Drained (pop_front) by the source phase. A deque so the LIMIT counter can
    // read completed batches in place during the sink phase: push_back keeps
    // references to existing elements valid, and nothing pops before the source.
    std::deque<CompletedBatch> completed;
    size_t limit_scanned = 0;  // completed[0, limit_scanned) already counted
    bool input_exhausted = false;
    // Embedding proxy state. proxy_rows / unit_rows / row_unit_by_id are written
    // once in FinalizeFlush (before input_exhausted) and read-only afterwards.
    bool proxy_planned = false;
    std::vector<PendingRequest> proxy_rows;
    std::vector<std::vector<size_t>> unit_rows;           // unit -> row positions
    std::unordered_map<uint64_t, size_t> row_unit_by_id;  // row_id -> unit
    std::deque<std::pair<size_t, bool>> pending_emit;     // (row position, verdict), guarded by Lock()
    std::unique_ptr<ProxyPolicy> policy;                  // guarded by policy_mu
    std::mutex policy_mu;
    std::atomic<bool> proxy_stats_printed{false};
    // Our OWN blocked-task lists (not StateWithBlockableTasks::blocked_tasks):
    // we splice them out under the lock and fire Callback() AFTER releasing, so
    // no foreign synchronization primitive is ever held under our lock.
    std::vector<duckdb::InterruptState> blocked_sinks;
    std::vector<duckdb::InterruptState> blocked_sources;

    // --- atomics: lock-free hint reads + a cross-thread id source ---
    std::atomic<size_t> in_flight{0};        // outstanding HTTP batches
    // Unique per query (NextGeneration() at construction, never bumped): tags
    // this query's Submits and is the gen drain-cancelled on teardown.
    std::atomic<uint64_t> generation;
    std::atomic<uint64_t> next_row_id{0};    // monotonic, query-global row ids
    std::atomic<size_t> max_in_flight_seen{0};
    std::atomic<bool> blocked_at_least_once{false};
    std::atomic<uint64_t> limit_counted_rows{0};  // output rows seen by LimitReached
    std::atomic<bool> limit_reached{false};
    std::atomic<uint64_t> rows_dropped{0};        // rows never sent (LIMIT early stop)

    // dependencies
    std::shared_ptr<ILLMClient> client;
    std::shared_ptr<EndpointRouter> router;
    std::shared_ptr<EndpointRouter> proxy_router;
    SemanticParams cfg;
};

// Source-side states. The source phase reads the SINK state (the completed
// queue) via the operator's sink_state; these carry only the thread-count cap
// and a per-thread placeholder.
class SemGlobalSourceState : public duckdb::GlobalSourceState {
public:
    explicit SemGlobalSourceState(uint64_t max_threads) : max_threads(max_threads) {}
    duckdb::idx_t MaxThreads() override { return static_cast<duckdb::idx_t>(max_threads); }

private:
    uint64_t max_threads;
};

class SemLocalSourceState : public duckdb::LocalSourceState {};

// Thin PhysicalOperator adapter over the dispatch engine.
//
// A pipeline-breaker that is BOTH sink and source. Every interface method just
// Casts DuckDB's state objects to the engine states and forwards; the real
// logic lives in SemGlobalSinkState. Concrete operators (W4: PhysicalSemFilter,
// ...) inherit this and implement the two hooks.
class SemanticOperatorBase : public duckdb::PhysicalOperator {
public:
    SemanticOperatorBase(duckdb::PhysicalPlan& physical_plan, duckdb::PhysicalOperatorType type,
                         duckdb::vector<duckdb::LogicalType> types, duckdb::idx_t estimated_cardinality,
                         std::shared_ptr<ILLMClient> client, std::shared_ptr<EndpointRouter> router,
                         SemanticParams cfg);

    bool IsSink() const override { return true; }
    bool IsSource() const override { return true; }
    bool ParallelSink() const override { return true; }
    bool ParallelSource() const override { return true; }

    // Sink interface (Cast DuckDB state -> engine, forward).
    duckdb::unique_ptr<duckdb::GlobalSinkState> GetGlobalSinkState(duckdb::ClientContext& context) const override;
    duckdb::unique_ptr<duckdb::LocalSinkState> GetLocalSinkState(duckdb::ExecutionContext& context) const override;
    duckdb::SinkResultType Sink(duckdb::ExecutionContext& context, duckdb::DataChunk& chunk,
                                duckdb::OperatorSinkInput& input) const override;
    duckdb::SinkCombineResultType Combine(duckdb::ExecutionContext& context,
                                          duckdb::OperatorSinkCombineInput& input) const override;
    duckdb::SinkFinalizeType Finalize(duckdb::Pipeline& pipeline, duckdb::Event& event,
                                      duckdb::ClientContext& context,
                                      duckdb::OperatorSinkFinalizeInput& input) const override;

    // Source interface.
    duckdb::unique_ptr<duckdb::GlobalSourceState> GetGlobalSourceState(duckdb::ClientContext& context) const override;
    duckdb::unique_ptr<duckdb::LocalSourceState> GetLocalSourceState(
        duckdb::ExecutionContext& context, duckdb::GlobalSourceState& gstate) const override;

    // The two operator-specific hooks W4 subclasses implement. RenderPrompt
    // packs a whole batch into one multi-row prompt (W4: flock's tuple
    // formatter); ParseAndEmit consumes one response element per row.
    virtual std::string RenderPrompt(const std::vector<RowData>& batch) const = 0;
    virtual nlohmann::json RenderMedia(const std::vector<RowData>& batch) const;
    virtual void ParseAndEmit(const nlohmann::json& element, const RowData& row, duckdb::DataChunk& out) const = 0;

    // Guided-decoding schema hook. Returns the value for body["response_format"]
    // for a batch of `batch_rows` rows, or null (the default) to send no schema.
    // Concrete operators override it with their per-row element schema; the base
    // stays agnostic to bool vs string. Bound into the engine via response_schema.
    virtual nlohmann::json BuildResponseFormat(size_t batch_rows) const;

    // Wrap a per-row `element` schema in flock's shared {"items":[...]} envelope of
    // exactly `batch_rows` elements, under the json_schema `name`. Both operators'
    // guided schemas share this shape; only the element schema (+ name) differ.
    static nlohmann::json ItemsResponseFormat(const std::string& name, nlohmann::json element, size_t batch_rows);

    // The context-free core of GetGlobalSinkState: build the engine sink state
    // and bind its render hook to this operator's RenderPrompt. Public so tests
    // can construct the wired state without a ClientContext.
    duckdb::unique_ptr<SemGlobalSinkState> CreateGlobalSinkState() const;

protected:
    // Source reads the sink state via op.sink_state->Cast<SemGlobalSinkState>().
    duckdb::SourceResultType GetDataInternal(duckdb::ExecutionContext& context, duckdb::DataChunk& chunk,
                                             duckdb::OperatorSourceInput& input) const override;

    std::shared_ptr<ILLMClient> client;
    std::shared_ptr<EndpointRouter> router;
    SemanticParams cfg;
};

}  // namespace flock
