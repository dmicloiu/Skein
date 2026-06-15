// Scheduler-free threading test for the async semantic-operator dispatch engine.
//
// It drives SemGlobalSinkState directly with a FakeLLMClient -- no DuckDB
// executor -- so the block -> wakeup ordering, the cap, cancellation, and the
// blocking finalize are all reproducible (and TSan-clean). InterruptState is
// used in BLOCKING mode via a fresh one-shot InterruptDoneSignalState per block
// attempt: the worker holds the shared_ptr alive across Await() while the engine
// keeps only a weak_ptr, and the signal latches `done`, so a Signal() that lands
// before Await() is not a lost wakeup.

#include "flock/functions/operator/semantic_operator_base.hpp"
#include "flock/runtime/endpoint_router.h"

#include "fake_llm_client.h"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/helper.hpp"  // duckdb::make_shared_ptr
#include "duckdb/common/types.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/execution/physical_plan_generator.hpp" 
#include "duckdb/parallel/interrupt.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>

namespace flock {
namespace {

using duckdb::DataChunk;
using duckdb::idx_t;
using duckdb::InterruptDoneSignalState;
using duckdb::InterruptState;
using duckdb::LogicalType;
using duckdb::SinkFinalizeType;
using duckdb::SinkResultType;
using duckdb::SourceResultType;
using duckdb::Value;

SemanticParams MakeParams(uint64_t cap, uint64_t coalesce) {
    SemanticParams p;
    p.in_flight_cap = cap;
    p.coalesce_size = coalesce;
    p.coalesce_max_age_ms = 500;
    p.max_output_tokens = 16;
    p.response_format = "json_schema";
    return p;
}

// Test render hook: a stub prompt derived from the row's first value.
SemGlobalSinkState::RenderFn StubRender() {
    return [](const RowData& row) {
        return std::string("p:") + (row.values.empty() ? std::string() : row.values[0].ToString());
    };
}

// Test parse hook (W3 pass-through): append the original row's values to `out`.
SemGlobalSinkState::ParseFn PassThroughParse() {
    return [](const nlohmann::json& choice, const RowData& row, DataChunk& out) {
        (void)choice;
        const idx_t idx = out.size();
        for (idx_t c = 0; c < row.values.size(); ++c) {
            out.SetValue(c, idx, row.values[c]);
        }
        out.SetCardinality(idx + 1);
    };
}

// Build chunks of `chunk_size` BIGINT rows holding values [0, total).
std::vector<std::unique_ptr<DataChunk>> BuildChunks(int64_t total, idx_t chunk_size,
                                                    const duckdb::vector<LogicalType>& types) {
    std::vector<std::unique_ptr<DataChunk>> chunks;
    auto& alloc = duckdb::Allocator::DefaultAllocator();
    int64_t v = 0;
    while (v < total) {
        auto chunk = std::make_unique<DataChunk>();
        chunk->Initialize(alloc, types);
        idx_t n = 0;
        while (n < chunk_size && v < total) {
            chunk->SetValue(0, n, Value::BIGINT(v));
            ++n;
            ++v;
        }
        chunk->SetCardinality(n);
        chunks.push_back(std::move(chunk));
    }
    return chunks;
}

// SINK worker: retries the SAME chunk on BLOCKED (engine resumes mid-chunk via
// next_row_idx); advances to the next chunk on NEED_MORE_INPUT.
void RunSinkWorker(SemGlobalSinkState& g, SemLocalSinkState& local, const std::vector<DataChunk*>& my_chunks,
                   std::atomic<size_t>& blocked_count) {
    for (DataChunk* chunk : my_chunks) {
        for (;;) {
            auto signal = duckdb::make_shared_ptr<InterruptDoneSignalState>();
            InterruptState interrupt(signal);  // weak_ptr -> BLOCKING mode
            auto res = g.SinkChunk(local, *chunk, interrupt);
            if (res == SinkResultType::BLOCKED) {
                blocked_count.fetch_add(1, std::memory_order_relaxed);
                signal->Await();  // woken by an IO-thread Callback()
                continue;         // retry the same chunk
            }
            break;  // NEED_MORE_INPUT -> next chunk
        }
    }
}

// FINALIZE (single thread, after all sink workers join + per-local MergeLocal).
void RunFinalize(SemGlobalSinkState& g, std::atomic<size_t>& finalize_blocked) {
    for (;;) {
        auto signal = duckdb::make_shared_ptr<InterruptDoneSignalState>();
        InterruptState interrupt(signal);
        if (g.FinalizeFlush(interrupt) != SinkFinalizeType::BLOCKED) {
            break;  // READY
        }
        finalize_blocked.fetch_add(1, std::memory_order_relaxed);
        signal->Await();
    }
}

// DRAIN worker (source phase): collect emitted column-0 values for no-loss.
void RunDrainWorker(SemGlobalSinkState& g, const duckdb::vector<LogicalType>& types,
                    const SemGlobalSinkState::ParseFn& parse, std::vector<int64_t>& sink_out,
                    std::mutex& out_mtx) {
    auto& alloc = duckdb::Allocator::DefaultAllocator();
    for (;;) {
        DataChunk out;
        out.Initialize(alloc, types);
        auto signal = duckdb::make_shared_ptr<InterruptDoneSignalState>();
        InterruptState interrupt(signal);
        auto res = g.Drain(out, parse, interrupt);
        if (res == SourceResultType::BLOCKED) {
            signal->Await();
            continue;
        }
        if (res == SourceResultType::HAVE_MORE_OUTPUT) {
            std::lock_guard<std::mutex> l(out_mtx);
            for (idx_t i = 0; i < out.size(); ++i) {
                sink_out.push_back(out.GetValue(0, i).GetValue<int64_t>());
            }
            continue;
        }
        break;  // FINISHED
    }
}

struct PipelineResult {
    std::vector<int64_t> out_values;
    size_t max_in_flight = 0;
    size_t sink_blocked = 0;
    size_t finalize_blocked = 0;
    uint64_t cap = 0;
};

// Full sink -> combine -> finalize -> source run across N threads.
PipelineResult RunPipeline(int N, const SemanticParams& cfg, const FakeLLMClient::Options& fake_opts,
                           int64_t total, idx_t chunk_size) {
    const duckdb::vector<LogicalType> types{LogicalType::BIGINT};
    auto fake = std::make_shared<FakeLLMClient>(fake_opts);
    auto router = std::make_shared<EndpointRouter>(std::vector<std::string>{"http://localhost:8000/v1"},
                                                   EndpointRouter::Strategy::RoundRobin);
    auto g = std::make_unique<SemGlobalSinkState>(fake, router, cfg);
    g->render_prompt = StubRender();

    auto chunks = BuildChunks(total, chunk_size, types);
    std::vector<std::vector<DataChunk*>> partitions(N);
    for (size_t i = 0; i < chunks.size(); ++i) {
        partitions[i % N].push_back(chunks[i].get());
    }

    // 1. partition across N locals + N threads -> RunSinkWorker; join.
    std::vector<std::unique_ptr<SemLocalSinkState>> locals;
    for (int t = 0; t < N; ++t) {
        locals.push_back(std::make_unique<SemLocalSinkState>());
    }
    std::atomic<size_t> blocked{0};
    {
        std::vector<std::thread> sink_threads;
        for (int t = 0; t < N; ++t) {
            sink_threads.emplace_back(RunSinkWorker, std::ref(*g), std::ref(*locals[t]),
                                      std::cref(partitions[t]), std::ref(blocked));
        }
        for (auto& th : sink_threads) {
            th.join();
        }
    }

    // 2. Combine each local.
    for (auto& l : locals) {
        g->MergeLocal(*l);
    }

    // 3. Finalize (sets input_exhausted).
    std::atomic<size_t> fin_blocked{0};
    RunFinalize(*g, fin_blocked);

    // 4. N drain threads -> RunDrainWorker; join.
    std::vector<int64_t> out_values;
    std::mutex out_mtx;
    const auto parse = PassThroughParse();
    {
        std::vector<std::thread> drain_threads;
        for (int t = 0; t < N; ++t) {
            drain_threads.emplace_back(RunDrainWorker, std::ref(*g), std::cref(types), std::cref(parse),
                                       std::ref(out_values), std::ref(out_mtx));
        }
        for (auto& th : drain_threads) {
            th.join();
        }
    }

    PipelineResult r;
    r.out_values = std::move(out_values);
    r.max_in_flight = g->MaxInFlightSeen();
    r.sink_blocked = blocked.load();
    r.finalize_blocked = fin_blocked.load();
    r.cap = cfg.in_flight_cap;
    return r;
}

// Assertions 1-3 (no loss/dup, cap respected, BLOCKED+wakeup) for a given N.
void RunMainPipelineChecks(int N) {
    const auto cfg = MakeParams(/*cap=*/4, /*coalesce=*/4);
    FakeLLMClient::Options fopts;
    fopts.delay = std::chrono::milliseconds(8);
    fopts.num_workers = 4;
    const int64_t total = 200;

    auto r = RunPipeline(N, cfg, fopts, total, /*chunk_size=*/8);

    // No loss / no dup: emitted multiset == input multiset.
    std::vector<int64_t> got = r.out_values;
    std::sort(got.begin(), got.end());
    std::vector<int64_t> expected(static_cast<size_t>(total));
    std::iota(expected.begin(), expected.end(), 0);
    ASSERT_EQ(got.size(), static_cast<size_t>(total)) << "row count mismatch at N=" << N;
    EXPECT_EQ(got, expected) << "rows lost/duplicated/corrupted at N=" << N;

    // Cap respected.
    EXPECT_LE(r.max_in_flight, r.cap) << "in_flight exceeded cap at N=" << N;
    EXPECT_GT(r.max_in_flight, 0u);

    // BLOCKED + wakeup exercised and the run terminated (no deadlock).
    EXPECT_GT(r.sink_blocked, 0u) << "Sink never returned BLOCKED at N=" << N;
}

TEST(SemanticOperatorBase, Pipeline_NoLoss_Cap_Block_N4) {
    RunMainPipelineChecks(4);
}

TEST(SemanticOperatorBase, Pipeline_NoLoss_Cap_Block_N16) {
    RunMainPipelineChecks(16);
}

// Tail batch (count not divisible by coalesce_size) is flushed + emitted, and a
// cap-saturated finalize returns BLOCKED then completes.
TEST(SemanticOperatorBase, TailFlush_And_BlockingFinalize) {
    const auto cfg = MakeParams(/*cap=*/1, /*coalesce=*/4);
    FakeLLMClient::Options fopts;
    fopts.delay = std::chrono::milliseconds(80);  // keep the in-flight batch live across finalize entry
    fopts.num_workers = 2;
    const int64_t total = 10;  // 2 full batches of 4 + a trailing partial of 2

    auto r = RunPipeline(/*N=*/1, cfg, fopts, total, /*chunk_size=*/16);

    std::vector<int64_t> got = r.out_values;
    std::sort(got.begin(), got.end());
    std::vector<int64_t> expected(static_cast<size_t>(total));
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(got, expected) << "tail batch lost";
    EXPECT_LE(r.max_in_flight, 1u);
    EXPECT_GT(r.finalize_blocked, 0u) << "finalize never blocked on a saturated cap";
}

// Destroying the sink state with batches in flight must not UAF or hang: the
// destructor bumps the generation + cancels, and no completion fires afterward.
void RunCancellationCheck(int N) {
    const auto cfg = MakeParams(/*cap=*/64, /*coalesce=*/4);  // big cap -> sink never blocks
    FakeLLMClient::Options fopts;
    fopts.delay = std::chrono::milliseconds(100);  // callbacks stay pending past the cancel
    fopts.num_workers = 4;
    auto fake = std::make_shared<FakeLLMClient>(fopts);
    auto router = std::make_shared<EndpointRouter>(std::vector<std::string>{"http://localhost:8000/v1"},
                                                   EndpointRouter::Strategy::RoundRobin);
    auto g = std::make_unique<SemGlobalSinkState>(fake, router, cfg);
    g->render_prompt = StubRender();

    const duckdb::vector<LogicalType> types{LogicalType::BIGINT};
    auto chunks = BuildChunks(80, 8, types);
    std::vector<std::vector<DataChunk*>> partitions(N);
    for (size_t i = 0; i < chunks.size(); ++i) {
        partitions[i % N].push_back(chunks[i].get());
    }
    std::vector<std::unique_ptr<SemLocalSinkState>> locals;
    for (int t = 0; t < N; ++t) {
        locals.push_back(std::make_unique<SemLocalSinkState>());
    }
    std::atomic<size_t> blocked{0};
    {
        std::vector<std::thread> ts;
        for (int t = 0; t < N; ++t) {
            ts.emplace_back(RunSinkWorker, std::ref(*g), std::ref(*locals[t]), std::cref(partitions[t]),
                            std::ref(blocked));
        }
        for (auto& th : ts) {
            th.join();
        }
    }
    EXPECT_EQ(blocked.load(), 0u) << "big cap should not have blocked the sink (N=" << N << ")";

    // Batches are now in flight (sleeping in the fake). Tear down on a side
    // thread; ~SemGlobalSinkState bumps the generation + CancelByGeneration.
    std::thread killer([&] { g.reset(); });
    const auto start = std::chrono::steady_clock::now();
    killer.join();
    EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(2)) << "teardown hung (N=" << N << ")";

    // Let still-sleeping fake workers wake; they must skip (generation is dead).
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_EQ(fake->FiredCount(), 0u) << "a completion fired after cancellation (N=" << N << ")";
}

TEST(SemanticOperatorBase, Cancellation_NoUAF_NoHang_N4) {
    RunCancellationCheck(4);
}

TEST(SemanticOperatorBase, Cancellation_NoUAF_NoHang_N16) {
    RunCancellationCheck(16);
}

// Fail-loud: a failed LLM response makes Drain throw rather than skip rows.
TEST(SemanticOperatorBase, FailLoud_OnFailedResponse) {
    const auto cfg = MakeParams(/*cap=*/4, /*coalesce=*/4);
    FakeLLMClient::Options fopts;
    fopts.delay = std::chrono::milliseconds(2);
    fopts.inject_error = true;
    fopts.num_workers = 2;
    auto fake = std::make_shared<FakeLLMClient>(fopts);
    auto router = std::make_shared<EndpointRouter>(std::vector<std::string>{"http://localhost:8000/v1"},
                                                   EndpointRouter::Strategy::RoundRobin);
    auto g = std::make_unique<SemGlobalSinkState>(fake, router, cfg);
    g->render_prompt = StubRender();

    const duckdb::vector<LogicalType> types{LogicalType::BIGINT};
    auto chunks = BuildChunks(4, 8, types);  // exactly one batch
    auto local = std::make_unique<SemLocalSinkState>();
    std::atomic<size_t> blocked{0};
    RunSinkWorker(*g, *local, {chunks[0].get()}, blocked);
    g->MergeLocal(*local);
    std::atomic<size_t> fin_blocked{0};
    RunFinalize(*g, fin_blocked);

    const auto parse = PassThroughParse();
    auto& alloc = duckdb::Allocator::DefaultAllocator();
    bool threw = false;
    for (int attempt = 0; attempt < 2000 && !threw; ++attempt) {
        DataChunk out;
        out.Initialize(alloc, types);
        auto signal = duckdb::make_shared_ptr<InterruptDoneSignalState>();
        InterruptState interrupt(signal);
        try {
            auto res = g->Drain(out, parse, interrupt);
            if (res == SourceResultType::BLOCKED) {
                signal->Await();
                continue;
            }
            if (res == SourceResultType::FINISHED) {
                break;
            }
        } catch (const std::exception&) {
            threw = true;
        }
    }
    EXPECT_TRUE(threw) << "Drain did not fail loud on a failed response";
}

// -- Operator-level smoke test -----------------------------------

// Test-only concrete operator: pass-through hooks with call counters, so the
// smoke test can confirm RenderPrompt/ParseAndEmit are actually driven through
// the engine via the operator's wiring.
class TestSemanticOperator : public SemanticOperatorBase {
public:
    TestSemanticOperator(duckdb::PhysicalPlan& plan, duckdb::vector<LogicalType> types,
                         std::shared_ptr<ILLMClient> client, std::shared_ptr<EndpointRouter> router,
                         SemanticParams cfg)
        : SemanticOperatorBase(plan, duckdb::PhysicalOperatorType::EXTENSION, std::move(types),
                               /*estimated_cardinality=*/0, std::move(client), std::move(router),
                               std::move(cfg)) {}

    std::string RenderPrompt(const RowData& row) const override {
        render_calls.fetch_add(1, std::memory_order_relaxed);
        return std::string("op:") + (row.values.empty() ? std::string() : row.values[0].ToString());
    }
    void ParseAndEmit(const nlohmann::json& choice, const RowData& row, DataChunk& out) const override {
        (void)choice;
        const idx_t idx = out.size();
        for (idx_t c = 0; c < row.values.size(); ++c) {
            out.SetValue(c, idx, row.values[c]);
        }
        out.SetCardinality(idx + 1);
        emit_calls.fetch_add(1, std::memory_order_relaxed);
    }

    mutable std::atomic<size_t> render_calls{0};
    mutable std::atomic<size_t> emit_calls{0};
};

// Smokes the operator wrappers: flags, MaxThreads == cap, the source-reads-sink
// bridge, and a full Sink -> Finalize -> Drain round-trip driven through the
// operator's own RenderPrompt (auto-bound) + ParseAndEmit. The thin
// Sink/Combine/Finalize/GetDataInternal adapters (Cast + forward) carry no logic
// beyond the engine paths exercised here and in the threading tests; they get a
// real ExecutionContext/Pipeline only under W4's planner.
TEST(SemanticOperatorBase, Operator_FlagsBridgeAndHooks) {
    const auto cfg = MakeParams(/*cap=*/8, /*coalesce=*/4);
    auto fake = std::make_shared<FakeLLMClient>();
    auto router = std::make_shared<EndpointRouter>(std::vector<std::string>{"http://localhost:8000/v1"},
                                                   EndpointRouter::Strategy::RoundRobin);
    duckdb::PhysicalPlan plan(duckdb::Allocator::DefaultAllocator());
    const duckdb::vector<LogicalType> types{LogicalType::BIGINT};
    auto& op = plan.Make<TestSemanticOperator>(types, fake, router, cfg).Cast<TestSemanticOperator>();

    // Pipeline-breaker flags.
    EXPECT_TRUE(op.IsSink());
    EXPECT_TRUE(op.IsSource());
    EXPECT_TRUE(op.ParallelSink());
    EXPECT_TRUE(op.ParallelSource());

    // MaxThreads == in_flight_cap on the source state (decoupling claim).
    SemGlobalSourceState source_state(cfg.in_flight_cap);
    EXPECT_EQ(source_state.MaxThreads(), cfg.in_flight_cap);

    // Build the wired sink state through the operator; confirm the bridge
    // resolves to it and reports the same cap.
    op.sink_state = op.CreateGlobalSinkState();
    auto& gss = op.sink_state->Cast<SemGlobalSinkState>();
    EXPECT_EQ(gss.MaxThreads(1), cfg.in_flight_cap);

    // Sink -> Finalize -> Drain through the operator's hooks.
    const int64_t total = 20;  // 5 full batches of 4, no partial
    auto chunks = BuildChunks(total, /*chunk_size=*/8, types);
    auto local = std::make_unique<SemLocalSinkState>();
    std::atomic<size_t> blocked{0};
    for (auto& c : chunks) {
        RunSinkWorker(gss, *local, {c.get()}, blocked);
    }
    gss.MergeLocal(*local);
    std::atomic<size_t> fin_blocked{0};
    RunFinalize(gss, fin_blocked);

    std::vector<int64_t> out_values;
    std::mutex out_mtx;
    const SemGlobalSinkState::ParseFn parse = [&op](const nlohmann::json& choice, const RowData& row,
                                                    DataChunk& out) { op.ParseAndEmit(choice, row, out); };
    RunDrainWorker(gss, types, parse, out_values, out_mtx);

    std::sort(out_values.begin(), out_values.end());
    std::vector<int64_t> expected(static_cast<size_t>(total));
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(out_values, expected) << "operator round-trip lost/corrupted rows";
    EXPECT_EQ(op.render_calls.load(), static_cast<size_t>(total)) << "RenderPrompt not driven per row";
    EXPECT_EQ(op.emit_calls.load(), static_cast<size_t>(total)) << "ParseAndEmit not driven per row";
}

}  // namespace
}  // namespace flock
