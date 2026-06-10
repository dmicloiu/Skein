#pragma once

#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/physical_operator_states.hpp"
#include "duckdb/parallel/interrupt.hpp"
#include "flock/runtime/endpoint_router.h"
#include "flock/runtime/llm_client.h"
#include "flock/runtime/pending_request.h"
#include "flock/runtime/semantic_settings.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <deque>
#include <functional>
#include <memory>
#include <queue>
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

// A coalesced batch that has returned from the LLM, awaiting parse + emit by
// the source phase. Carries everything Drain needs to realign the response
// choices with their originating rows.
struct CompletedBatch {
    std::vector<uint64_t> row_ids;  // submission order
    std::vector<RowData> rows;      // rows[i] <-> row_ids[i] <-> choices[i]
    LLMResponse response;           // raw HTTP result; parsed once, in Drain
    size_t endpoint_index = 0;      // router slot (already released on completion)
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

    // [TO DO - future addition] the residual-predicate ExpressionExecutor lives here
};

// Global sink state + the dispatch state machine. One instance per query.
class SemGlobalSinkState : public duckdb::GlobalSinkState {
public:
    // Renders the prompt for one row.
    using RenderFn = std::function<std::string(const RowData&)>;
    // Parses one response choice and appends its output row to `out`.
    using ParseFn = std::function<void(const nlohmann::json& choice, const RowData&, duckdb::DataChunk& out)>;

    // No ClientContext
    SemGlobalSinkState(std::shared_ptr<ILLMClient> client, std::shared_ptr<EndpointRouter> router,
                       SemanticParams cfg);
    // Bumps the generation and cancels outstanding callbacks. This is the guard
    // against use-after-free on query teardown / cancellation.
    ~SemGlobalSinkState() override;

    // Set ONCE before any Sink call (by the operator or directly by the test);
    // read-only and lock-free thereafter, so concurrent SinkChunk calls share
    // it safely. SinkChunk needs it during the sink phase; the analogous parse
    // hook is a Drain parameter because parsing runs in the source phase.
    RenderFn render_prompt;

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
    const SemanticParams& Config() const { return cfg; }

    duckdb::idx_t MaxThreads(duckdb::idx_t /*source_max_threads*/) override {
        return static_cast<duckdb::idx_t>(cfg.in_flight_cap);
    }

private:
    // Completion callback body. Runs on the ILLMClient IO thread; keep cheap.
    void OnBatchComplete(uint64_t submit_generation, size_t endpoint_index, std::vector<uint64_t> row_ids,
                         std::vector<RowData> rows, LLMResponse response);
    // Merge local.pending into the global buffer, then coalesce + submit as many
    // full batches as the cap allows. Returns BLOCKED (and records resume_idx)
    // iff a full batch is ready but the cap is saturated.
    duckdb::SinkResultType MergeAndCoalesce(SemLocalSinkState& local, duckdb::idx_t resume_idx,
                                            duckdb::InterruptState& interrupt);
    // Build the /v1/completions JSON body for a batch (prompt: [array]). Pure;
    // safe to call outside the lock.
    std::string BuildPayload(const std::vector<PendingRequest>& batch) const;
    // Choose an endpoint, build the payload, and Submit (all OUTSIDE the lock).
    void SubmitBatch(std::vector<PendingRequest> batch);
    // Monotonically record the in-flight peak (test instrumentation).
    void NoteInFlight(size_t now_in_flight);

    static constexpr size_t LOCAL_MERGE_THRESHOLD = 8;

    // --- guarded by THE one lock: the inherited StateWithBlockableTasks::Lock()
    //     Critical sections move buffers and adjust counters ONLY,
    //     router->Choose, client->Submit, or Callback(). ---
    std::deque<PendingRequest> pending;   // global FIFO
    std::queue<CompletedBatch> completed; // drained by the source phase
    bool input_exhausted = false;
    // Our OWN blocked-task lists (not StateWithBlockableTasks::blocked_tasks):
    // we splice them out under the lock and fire Callback() AFTER releasing, so
    // no foreign synchronization primitive is ever held under our lock.
    std::vector<duckdb::InterruptState> blocked_sinks;
    std::vector<duckdb::InterruptState> blocked_sources;

    // --- atomics: lock-free hint reads + a cross-thread id source ---
    std::atomic<size_t> in_flight{0};        // outstanding HTTP batches
    std::atomic<uint64_t> generation{1};     // bumped on destroy/cancel; callback no-ops on mismatch
    std::atomic<uint64_t> next_row_id{0};    // monotonic, query-global row ids
    std::atomic<size_t> max_in_flight_seen{0};
    std::atomic<bool> blocked_at_least_once{false};

    // dependencies
    std::shared_ptr<ILLMClient> client;
    std::shared_ptr<EndpointRouter> router;
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

    // The two operator-specific hooks W4 subclasses implement.
    virtual std::string RenderPrompt(const RowData& row) const = 0;
    virtual void ParseAndEmit(const nlohmann::json& choice, const RowData& row, duckdb::DataChunk& out) const = 0;

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