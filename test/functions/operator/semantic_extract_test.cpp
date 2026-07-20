// Component tests for the async semantic extract operator: prompt parity with the
// scalar llm_complete, ParseAndEmit projection slotting (string column at the
// llm_complete position + passthrough columns), LogicalSemExtract bindings vs
// LogicalProjection, CreatePlan config resolution, and a full operator round-trip
// that materializes the projection at Sink and emits every row.

#include "flock/functions/input_parser.hpp"
#include "flock/functions/operator/semantic_extract.hpp"
#include "flock/functions/operator/semantic_operator_common.hpp"
#include "flock/model_manager/model.hpp"
#include "flock/prompt_manager/prompt_manager.hpp"
#include "flock/runtime/endpoint_router.h"
#include "flock/runtime/llm_client.h"

#include "duckdb.hpp"
#include "duckdb/common/allocator.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/execution/execution_context.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parallel/interrupt.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/planner.hpp"
#include "flock/core/config.hpp"

#include "../../unit/functions/mock_provider.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <utility>
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

duckdb::DuckDB* g_db = nullptr;

SemanticParams MakeParams(uint64_t cap = 64, uint64_t batch_size = 64) {
    SemanticParams p;
    p.in_flight_cap = cap;
    p.batch_size = batch_size;
    p.coalesce_max_age_ms = 500;
    p.max_output_tokens = 16;
    p.response_format = "json_schema";
    return p;
}

std::shared_ptr<EndpointRouter> MakeRouter() {
    return std::make_shared<EndpointRouter>(std::vector<std::string>{"http://localhost:8000/v1"},
                                            EndpointRouter::Strategy::RoundRobin);
}

// Deterministic in-process client. Fires on_done inline (single-threaded tests),
// synthesizing flock's batch envelope: one completion whose message.content is
// {"items":[string,...]}. Each row's completion is prefix + its first <column>
// value, so it aligns with rows regardless of how the engine batches them.
class ScriptedFakeClient : public ILLMClient {
public:
    explicit ScriptedFakeClient(std::string prefix) : prefix_(std::move(prefix)) {}

    LLMRequestHandle Submit(const std::string& /*endpoint*/, const std::string& payload, uint64_t /*generation*/,
                            const std::string& /*request_id*/, LLMOnDone on_done) override {
        LLMResponse r = Synthesize(payload);
        if (on_done) {
            on_done(std::move(r));
        }
        return LLMRequestHandle{id_.fetch_add(1, std::memory_order_relaxed) + 1};
    }
    void CancelByGeneration(uint64_t /*generation*/) override {}

private:
    LLMResponse Synthesize(const std::string& payload) const {
        auto parsed = nlohmann::json::parse(payload, nullptr, /*allow_exceptions=*/false);
        std::string prompt;
        if (!parsed.is_discarded() && parsed.contains("messages") && parsed["messages"].is_array() &&
            !parsed["messages"].empty()) {
            const auto& content = parsed["messages"][0].value("content", nlohmann::json());
            if (content.is_string()) {
                prompt = content.get<std::string>();
            } else if (content.is_array()) {
                for (const auto& part : content) {
                    if (part.contains("text") && part["text"].is_string()) {
                        prompt += part["text"].get<std::string>();
                    }
                }
            }
        }
        // One completion per <row> block, keyed on the row's first <column> value.
        auto items = nlohmann::json::array();
        const std::string row_open = "<row>";
        const std::string row_close = "</row>";
        const std::string col_open = "<column>";
        const std::string col_close = "</column>";
        for (size_t rp = 0; (rp = prompt.find(row_open, rp)) != std::string::npos;) {
            const size_t row_end = prompt.find(row_close, rp);
            if (row_end == std::string::npos) {
                break;
            }
            std::string value;
            const size_t cp = prompt.find(col_open, rp);
            if (cp != std::string::npos && cp < row_end) {
                const size_t start = cp + col_open.size();
                const size_t end = prompt.find(col_close, start);
                if (end != std::string::npos && end <= row_end) {
                    value = prompt.substr(start, end - start);
                }
            }
            items.push_back(prefix_ + value);
            rp = row_end + row_close.size();
        }
        nlohmann::json completion;
        completion["items"] = std::move(items);
        nlohmann::json choice;
        choice["index"] = 0;
        choice["message"]["content"] = completion.dump();
        nlohmann::json body;
        body["choices"] = nlohmann::json::array({std::move(choice)});
        LLMResponse r;
        r.ok = true;
        r.http_status = 200;
        r.body = body.dump();
        return r;
    }

    std::string prefix_;
    mutable std::atomic<uint64_t> id_{0};
};

// Build chunks of single-column BIGINT rows holding [0, total).
std::vector<std::unique_ptr<DataChunk>> BuildBigintChunks(int64_t total, idx_t chunk_size) {
    const duckdb::vector<LogicalType> types{LogicalType::BIGINT};
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

// A logical child with caller-set bindings/types, to compare LogicalSemExtract and
// LogicalProjection binding/typing in isolation.
class StubChild : public duckdb::LogicalOperator {
public:
    StubChild(duckdb::vector<duckdb::ColumnBinding> bindings, duckdb::vector<LogicalType> child_types)
        : duckdb::LogicalOperator(duckdb::LogicalOperatorType::LOGICAL_DUMMY_SCAN),
          stub_bindings(std::move(bindings)), fixed_types(std::move(child_types)) {
        types = fixed_types;
    }
    duckdb::vector<duckdb::ColumnBinding> GetColumnBindings() override { return stub_bindings; }

protected:
    void ResolveTypes() override { types = fixed_types; }

private:
    duckdb::vector<duckdb::ColumnBinding> stub_bindings;
    duckdb::vector<LogicalType> fixed_types;
};

// Construct a bare operator for the unit hooks (RenderPrompt / ParseAndEmit /
// BuildResponseFormat): no capture executor (those hooks read only the resolved
// config + the captured RowData).
PhysicalSemExtract& MakeExtractOp(duckdb::PhysicalPlan& plan, duckdb::vector<LogicalType> types,
                                  idx_t llm_call_index, std::vector<SemContextColumn> context_columns,
                                  const std::string& prompt = "p", const std::string& tuple_format = "XML") {
    return plan
            .Make<PhysicalSemExtract>(std::move(types), 0, std::shared_ptr<ILLMClient>(), MakeRouter(),
                                      MakeParams(), duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{},
                                      duckdb::vector<LogicalType>{}, llm_call_index, std::move(context_columns),
                                      prompt, tuple_format, std::string(), std::string())
            .Cast<PhysicalSemExtract>();
}

// ===========================================================================
// 1. Prompt parity: RenderPrompt == the scalar llm_complete prompt for a batch.
// ===========================================================================
TEST(SemExtract, Prompt_ParityWithScalar) {
    const std::string tmpl = "Classify the symptoms.";
    const std::string tuple_format = "XML";
    const std::vector<std::string> symptoms = {"fever and cough", "rash on arm", "headache"};

    auto cc_elem_type = LogicalType::STRUCT({{"data", LogicalType::VARCHAR}, {"name", LogicalType::VARCHAR}});
    auto prompt_type =
            LogicalType::STRUCT({{"prompt", LogicalType::VARCHAR}, {"context_columns", LogicalType::LIST(cc_elem_type)}});
    duckdb::Vector prompt_vec(prompt_type, symptoms.size());
    for (idx_t i = 0; i < symptoms.size(); ++i) {
        auto element = Value::STRUCT(cc_elem_type, {Value(symptoms[i]), Value("symptoms")});
        auto list = Value::LIST(cc_elem_type, {element});
        prompt_vec.SetValue(i, Value::STRUCT(prompt_type, {Value(tmpl), list}));
    }
    auto ref_columns = CastVectorOfStructsToJson(prompt_vec, static_cast<int>(symptoms.size()))["context_columns"];
    auto ref_prompt = std::get<0>(
            PromptManager::Render(tmpl, ref_columns, ScalarFunctionType::COMPLETE, tuple_format));

    // Operator: one captured context column at index 0 with name "symptoms".
    std::vector<SemContextColumn> ccs;
    SemContextColumn cc;
    cc.data_index = 0;
    cc.data_type = LogicalType::VARCHAR;
    cc.metadata["name"] = "symptoms";
    ccs.push_back(std::move(cc));

    duckdb::PhysicalPlan plan(duckdb::Allocator::DefaultAllocator());
    auto& op = MakeExtractOp(plan, {LogicalType::JSON()}, /*llm_call_index=*/0, std::move(ccs), tmpl, tuple_format);

    std::vector<RowData> batch;
    for (const auto& s : symptoms) {
        RowData row;
        row.values.push_back(Value(s));  // captured context data lives at index 0
        batch.push_back(std::move(row));
    }
    EXPECT_EQ(op.RenderPrompt(batch), ref_prompt);
}

// ===========================================================================
// 1b. Slim extract prompt (pure function; the operator's env gate mirrors the
//     filter's and SemPromptSlim is a read-once static, so the wording is
//     asserted on RenderSlimSemanticPrompt directly).
// ===========================================================================
TEST(SemExtract, SlimPromptWording) {
    const std::string tmpl = "Classify the symptoms.";
    nlohmann::json col;
    col["name"] = "symptoms";
    col["data"] = std::vector<std::string>{"fever and cough"};
    auto columns = nlohmann::json::array({col});

    // Single-row form: task framing + the original slim tuple block.
    const auto prompt = RenderSlimSemanticPrompt(SlimKind::kExtract, tmpl, columns, "XML");
    EXPECT_NE(prompt.find("produce the output the task requests"), std::string::npos);
    EXPECT_NE(prompt.find("Task: " + tmpl), std::string::npos);
    EXPECT_NE(prompt.find(PromptManager::ConstructInputTuples(columns, "XML")), std::string::npos);
    EXPECT_NE(prompt.find("one string per row, in row order"), std::string::npos);
    EXPECT_EQ(prompt.find("Criterion"), std::string::npos);

    // Batched form: string count/index contract + chunked row-major rows.
    columns[0]["data"].push_back("rash on arm");
    const auto batched = RenderSlimSemanticPrompt(SlimKind::kExtract, tmpl, columns, "XML");
    EXPECT_NE(batched.find("The table has 2 rows. Return exactly 2 strings; the i-th string answers row i."),
              std::string::npos);
    EXPECT_NE(batched.find("### Rows 1-2"), std::string::npos);
    EXPECT_NE(batched.find("{\"id\": 2, \"symptoms\": \"rash on arm\"}"), std::string::npos);
}

// ===========================================================================
// 2. ParseAndEmit: string completion slotted at the llm position + passthrough.
// ===========================================================================
TEST(SemExtract, ParseAndEmit_ProjectsRowWithStringColumn) {
    auto& alloc = duckdb::Allocator::DefaultAllocator();
    duckdb::PhysicalPlan plan(alloc);

    // Output [BIGINT id, JSON dx, VARCHAR note]; llm at position 1. Captured row =
    // the two siblings in output order: [id, note].
    {
        auto& op = MakeExtractOp(plan, {LogicalType::BIGINT, LogicalType::JSON(), LogicalType::VARCHAR},
                                 /*llm_call_index=*/1, {});
        RowData row;
        row.values = {Value::BIGINT(7), Value("mynote")};
        DataChunk out;
        out.Initialize(alloc, op.GetTypes());
        op.ParseAndEmit(nlohmann::json("VARICOSE VEINS"), row, out);
        op.ParseAndEmit(nlohmann::json("MALARIA"), row, out);  // every row emitted
        ASSERT_EQ(out.size(), 2u);
        EXPECT_EQ(out.GetValue(0, 0).GetValue<int64_t>(), 7);
        EXPECT_EQ(out.GetValue(1, 0).ToString(), "VARICOSE VEINS");  // arbitrary text into JSON, no validation
        EXPECT_EQ(out.GetValue(2, 0).GetValue<std::string>(), "mynote");
        EXPECT_EQ(out.GetValue(1, 1).ToString(), "MALARIA");
    }
    // llm at position 0.
    {
        auto& op = MakeExtractOp(plan, {LogicalType::JSON(), LogicalType::BIGINT}, /*llm_call_index=*/0, {});
        RowData row;
        row.values = {Value::BIGINT(9)};
        DataChunk out;
        out.Initialize(alloc, op.GetTypes());
        op.ParseAndEmit(nlohmann::json("ACNE"), row, out);
        ASSERT_EQ(out.size(), 1u);
        EXPECT_EQ(out.GetValue(0, 0).ToString(), "ACNE");
        EXPECT_EQ(out.GetValue(1, 0).GetValue<int64_t>(), 9);
    }
    // Non-string element (a padded null or an object) is dumped, mirroring scalar.
    EXPECT_EQ(PhysicalSemExtract::ParseCompletion(nlohmann::json("hi")), "hi");
    EXPECT_EQ(PhysicalSemExtract::ParseCompletion(nlohmann::json()), "null");
    EXPECT_EQ(PhysicalSemExtract::ParseCompletion(nlohmann::json{{"k", "v"}}), "{\"k\":\"v\"}");
}

// ===========================================================================
// 3. Bindings: LogicalSemExtract mirrors LogicalProjection's bindings + types.
// ===========================================================================
TEST(SemExtract, Bindings_MirrorLogicalProjection) {
    duckdb::vector<duckdb::ColumnBinding> bindings{{1, 0}, {1, 1}};
    duckdb::vector<LogicalType> child_types{LogicalType::BIGINT, LogicalType::VARCHAR};
    const idx_t table_index = 5;

    auto make_select_list = [] {
        duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs;
        exprs.push_back(duckdb::make_uniq<duckdb::BoundConstantExpression>(Value::BIGINT(1)));
        exprs.push_back(duckdb::make_uniq<duckdb::BoundConstantExpression>(Value("x")));
        return exprs;
    };

    duckdb::LogicalProjection lp(table_index, make_select_list());
    lp.children.push_back(duckdb::make_uniq<StubChild>(bindings, child_types));
    lp.ResolveOperatorTypes();

    LogicalSemExtract se(duckdb::make_uniq<StubChild>(bindings, child_types), table_index, make_select_list(),
                         /*llm_call_index=*/0);
    se.ResolveOperatorTypes();

    EXPECT_EQ(se.GetColumnBindings(), lp.GetColumnBindings());
    EXPECT_EQ(se.types, lp.types);
    EXPECT_EQ(se.GetTableIndex(), lp.GetTableIndex());
}

// ===========================================================================
// 4. Engine integration: every row emitted with the extracted string + passthrough.
// ===========================================================================

// Drive Sink (through the operator's pre-projection override) -> finalize -> drain,
// returning the emitted (id, dx) pairs. Needs a real ClientContext for the capture
// ExpressionExecutor, so it opens an in-memory DuckDB.
std::vector<std::pair<int64_t, std::string>> DriveExtract(PhysicalSemExtract& op, duckdb::ClientContext& ctx,
                                                          const std::vector<DataChunk*>& chunks) {
    duckdb::ThreadContext thread(ctx);
    duckdb::ExecutionContext econtext(ctx, thread, nullptr);
    auto gstate = op.GetGlobalSinkState(ctx);
    auto& gss = gstate->Cast<SemGlobalSinkState>();
    auto lstate = op.GetLocalSinkState(econtext);

    for (auto* chunk : chunks) {
        for (;;) {
            auto signal = duckdb::make_shared_ptr<InterruptDoneSignalState>();
            InterruptState interrupt(signal);
            duckdb::OperatorSinkInput sink_input{*gstate, *lstate, interrupt};
            if (op.Sink(econtext, *chunk, sink_input) != SinkResultType::BLOCKED) {
                break;
            }
            signal->Await();
        }
    }
    gss.MergeLocal(lstate->Cast<SemLocalSinkState>());
    for (;;) {
        auto signal = duckdb::make_shared_ptr<InterruptDoneSignalState>();
        InterruptState interrupt(signal);
        if (gss.FinalizeFlush(interrupt) != SinkFinalizeType::BLOCKED) {
            break;
        }
        signal->Await();
    }

    std::vector<std::pair<int64_t, std::string>> rows;
    auto& alloc = duckdb::Allocator::DefaultAllocator();
    const auto parse = [&op](const nlohmann::json& element, const RowData& row, DataChunk& out) {
        op.ParseAndEmit(element, row, out);
    };
    for (;;) {
        DataChunk out;
        out.Initialize(alloc, op.GetTypes());
        auto signal = duckdb::make_shared_ptr<InterruptDoneSignalState>();
        InterruptState interrupt(signal);
        auto res = gss.Drain(out, parse, interrupt);
        if (res == SourceResultType::BLOCKED) {
            signal->Await();
            continue;
        }
        if (res == SourceResultType::HAVE_MORE_OUTPUT) {
            for (idx_t i = 0; i < out.size(); ++i) {
                rows.emplace_back(out.GetValue(0, i).GetValue<int64_t>(), out.GetValue(1, i).ToString());
            }
            continue;
        }
        break;
    }
    return rows;
}

TEST(SemExtract, Engine_EmitsExtractedColumn) {
    duckdb::Connection con(*g_db);
    auto& ctx = *con.context;

    const int64_t total = 20;
    auto fake = std::make_shared<ScriptedFakeClient>("dx:");

    // Output [BIGINT id (passthrough), JSON dx]; llm at position 1. Child chunk is a
    // single BIGINT column, referenced BOTH as the passthrough sibling and as the
    // context data for the prompt.
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> capture_exprs;
    capture_exprs.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(LogicalType::BIGINT, 0));  // sibling id
    capture_exprs.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(LogicalType::BIGINT, 0));  // context
    duckdb::vector<LogicalType> capture_types{LogicalType::BIGINT, LogicalType::BIGINT};

    std::vector<SemContextColumn> ccs;
    SemContextColumn cc;
    cc.data_index = 1;  // context data is the 2nd captured column
    cc.data_type = LogicalType::BIGINT;
    ccs.push_back(std::move(cc));

    duckdb::PhysicalPlan plan(duckdb::Allocator::DefaultAllocator());
    auto& op = plan.Make<PhysicalSemExtract>(
                       duckdb::vector<LogicalType>{LogicalType::BIGINT, LogicalType::JSON()}, 0, fake,
                       MakeRouter(), MakeParams(/*cap=*/64, /*batch_size=*/8), std::move(capture_exprs),
                       std::move(capture_types), /*llm_call_index=*/1, std::move(ccs), "p", "XML",
                       std::string(), std::string())
                       .Cast<PhysicalSemExtract>();

    auto chunks = BuildBigintChunks(total, /*chunk_size=*/8);
    std::vector<DataChunk*> chunk_ptrs;
    for (auto& c : chunks) {
        chunk_ptrs.push_back(c.get());
    }
    auto rows = DriveExtract(op, ctx, chunk_ptrs);

    ASSERT_EQ(rows.size(), static_cast<size_t>(total));
    std::sort(rows.begin(), rows.end());
    for (int64_t v = 0; v < total; ++v) {
        EXPECT_EQ(rows[v].first, v);
        EXPECT_EQ(rows[v].second, "dx:" + std::to_string(v));  // completion = prefix + the row's value
    }
}

// Concurrent Sink: N threads each drive op.Sink with their OWN
// SemExtractLocalSinkState (own capture executor + capture chunk) against the
// shared global sink state. Exercises the per-thread pre-projection path under
// parallelism; this catches logic races in the capture
// path (lost/duplicated/corrupted rows) that a single-threaded drive would miss.
TEST(SemExtract, Engine_ConcurrentSink_NoLossNoCorruption) {
    duckdb::Connection con(*g_db);
    auto& ctx = *con.context;

    const int N = 4;
    const int64_t total = 200;
    auto fake = std::make_shared<ScriptedFakeClient>("dx:");

    // Same shape as Engine_EmitsExtractedColumn: output [BIGINT id, JSON dx], llm
    // at position 1; the single BIGINT child column is both the sibling and the
    // context data.
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> capture_exprs;
    capture_exprs.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(LogicalType::BIGINT, 0));
    capture_exprs.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(LogicalType::BIGINT, 0));
    duckdb::vector<LogicalType> capture_types{LogicalType::BIGINT, LogicalType::BIGINT};
    std::vector<SemContextColumn> ccs;
    SemContextColumn cc;
    cc.data_index = 1;
    cc.data_type = LogicalType::BIGINT;
    ccs.push_back(std::move(cc));

    duckdb::PhysicalPlan plan(duckdb::Allocator::DefaultAllocator());
    auto& op = plan.Make<PhysicalSemExtract>(
                       duckdb::vector<LogicalType>{LogicalType::BIGINT, LogicalType::JSON()}, 0, fake,
                       MakeRouter(), MakeParams(/*cap=*/4, /*batch_size=*/4), std::move(capture_exprs),
                       std::move(capture_types), /*llm_call_index=*/1, std::move(ccs), "p", "XML",
                       std::string(), std::string())
                       .Cast<PhysicalSemExtract>();

    auto chunks = BuildBigintChunks(total, /*chunk_size=*/8);
    std::vector<std::vector<DataChunk*>> partitions(N);
    for (size_t i = 0; i < chunks.size(); ++i) {
        partitions[i % N].push_back(chunks[i].get());
    }

    auto gstate = op.GetGlobalSinkState(ctx);
    auto& gss = gstate->Cast<SemGlobalSinkState>();
    // Per-thread state built SEQUENTIALLY (executor construction reads ctx); the
    // threads then only USE their own local (thread-local by construction).
    std::vector<std::unique_ptr<duckdb::ThreadContext>> thread_ctxs;
    std::vector<std::unique_ptr<duckdb::ExecutionContext>> econtexts;
    std::vector<duckdb::unique_ptr<duckdb::LocalSinkState>> locals;
    for (int t = 0; t < N; ++t) {
        thread_ctxs.push_back(std::make_unique<duckdb::ThreadContext>(ctx));
        econtexts.push_back(std::make_unique<duckdb::ExecutionContext>(ctx, *thread_ctxs[t], nullptr));
        locals.push_back(op.GetLocalSinkState(*econtexts[t]));
    }

    auto worker = [&](int t) {
        for (DataChunk* chunk : partitions[t]) {
            for (;;) {
                auto signal = duckdb::make_shared_ptr<InterruptDoneSignalState>();
                InterruptState interrupt(signal);
                duckdb::OperatorSinkInput sink_input{*gstate, *locals[t], interrupt};
                if (op.Sink(*econtexts[t], *chunk, sink_input) != SinkResultType::BLOCKED) {
                    break;
                }
                signal->Await();
            }
        }
    };
    {
        std::vector<std::thread> ts;
        for (int t = 0; t < N; ++t) {
            ts.emplace_back(worker, t);
        }
        for (auto& th : ts) {
            th.join();
        }
    }

    for (int t = 0; t < N; ++t) {
        gss.MergeLocal(locals[t]->Cast<SemLocalSinkState>());
    }
    for (;;) {
        auto signal = duckdb::make_shared_ptr<InterruptDoneSignalState>();
        InterruptState interrupt(signal);
        if (gss.FinalizeFlush(interrupt) != SinkFinalizeType::BLOCKED) {
            break;
        }
        signal->Await();
    }

    std::vector<std::pair<int64_t, std::string>> rows;
    auto& alloc = duckdb::Allocator::DefaultAllocator();
    const auto parse = [&op](const nlohmann::json& element, const RowData& row, DataChunk& out) {
        op.ParseAndEmit(element, row, out);
    };
    for (;;) {
        DataChunk out;
        out.Initialize(alloc, op.GetTypes());
        auto signal = duckdb::make_shared_ptr<InterruptDoneSignalState>();
        InterruptState interrupt(signal);
        auto res = gss.Drain(out, parse, interrupt);
        if (res == SourceResultType::BLOCKED) {
            signal->Await();
            continue;
        }
        if (res == SourceResultType::HAVE_MORE_OUTPUT) {
            for (idx_t i = 0; i < out.size(); ++i) {
                rows.emplace_back(out.GetValue(0, i).GetValue<int64_t>(), out.GetValue(1, i).ToString());
            }
            continue;
        }
        break;
    }

    // No loss / dup / corruption: exactly `total` rows, each id once, dx aligned.
    ASSERT_EQ(rows.size(), static_cast<size_t>(total));
    std::sort(rows.begin(), rows.end());
    for (int64_t v = 0; v < total; ++v) {
        EXPECT_EQ(rows[v].first, v);
        EXPECT_EQ(rows[v].second, "dx:" + std::to_string(v));
    }
}

// ===========================================================================
// 5. CreatePlan: a bound llm_complete projection -> PhysicalSemExtract w/ config.
// ===========================================================================
TEST(SemExtract, CreatePlan_ResolvesConfig) {
    duckdb::Connection con(*g_db);
    con.Query("CREATE SECRET (TYPE OPENAI, API_KEY 'test-key');");
    Model::SetMockProvider(std::make_shared<MockProvider>(ModelDetails{}));

    con.Query("CREATE TABLE patients(id INTEGER, symptoms VARCHAR);");
    con.Query("INSERT INTO patients VALUES (1, 'fever'), (2, 'rash');");

    const std::string query =
            "SELECT id, llm_complete({'model_name': 'gpt-4o'}, "
            "{'prompt': 'Diagnose', 'context_columns': [{'data': symptoms}]}) AS dx FROM patients;";

    // Swap the LogicalProjection carrying the top-level llm_complete for a
    // LogicalSemExtract (a stand-in for the rewrite).
    std::function<duckdb::unique_ptr<duckdb::LogicalOperator>(duckdb::unique_ptr<duckdb::LogicalOperator>)> swap =
            [&](duckdb::unique_ptr<duckdb::LogicalOperator> node) -> duckdb::unique_ptr<duckdb::LogicalOperator> {
        for (auto& child : node->children) {
            child = swap(std::move(child));
        }
        if (node->type == duckdb::LogicalOperatorType::LOGICAL_PROJECTION) {
            auto& proj = node->Cast<duckdb::LogicalProjection>();
            idx_t llm_idx = 0;
            bool found_llm = false;
            for (idx_t i = 0; i < proj.expressions.size(); ++i) {
                if (proj.expressions[i]->expression_class == duckdb::ExpressionClass::BOUND_FUNCTION &&
                    proj.expressions[i]->Cast<duckdb::BoundFunctionExpression>().function.name == "llm_complete") {
                    llm_idx = i;
                    found_llm = true;
                }
            }
            if (found_llm) {
                return duckdb::make_uniq<LogicalSemExtract>(std::move(proj.children[0]), proj.table_index,
                                                            std::move(proj.expressions), llm_idx);
            }
        }
        return node;
    };

    auto statements = con.context->ParseStatements(query);
    ASSERT_FALSE(statements.empty());

    PhysicalSemExtract* found = nullptr;
    con.context->RunFunctionInTransaction([&]() {
        duckdb::Planner planner(*con.context);
        planner.CreatePlan(std::move(statements[0]));
        auto logical = swap(std::move(planner.plan));

        duckdb::PhysicalPlanGenerator generator(*con.context);
        auto physical = generator.Plan(std::move(logical));
        std::function<void(duckdb::PhysicalOperator&)> walk = [&](duckdb::PhysicalOperator& op) {
            if (auto* sem = dynamic_cast<PhysicalSemExtract*>(&op)) {
                found = sem;
            }
            for (auto& child : op.children) {
                walk(child.get());
            }
        };
        walk(physical->Root());

        ASSERT_NE(found, nullptr);
        EXPECT_EQ(found->PromptTemplate(), "Diagnose");
        EXPECT_FALSE(found->ServedModel().empty());
        EXPECT_EQ(found->LlmCallIndex(), 1u);  // SELECT id, llm_complete(...) -> llm at position 1
        ASSERT_EQ(found->ContextColumns().size(), 1u);
        EXPECT_FALSE(found->StickyKey().empty());

        auto expected = ResolveSemanticParams(*con.context, "gpt-4o");
        EXPECT_EQ(found->Config().batch_size, expected.batch_size);
        EXPECT_EQ(found->Config().in_flight_cap, expected.in_flight_cap);
        EXPECT_EQ(found->Config().max_output_tokens, expected.max_output_tokens);
        EXPECT_EQ(found->Config().response_format, expected.response_format);
    });

    Model::ResetMockProvider();
    con.Query("DROP TABLE patients;");
}

}  // namespace
}  // namespace flock

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    duckdb::DuckDB db("flock_sem_extract_test.db");
    flock::g_db = &db;
    flock::Config::GetConnection(&*db.instance);
    int rc = RUN_ALL_TESTS();
    flock::g_db = nullptr;
    return rc;
}
