// Component tests for the async semantic filter operator: prompt parity with the
// scalar llm_filter, ParseAndEmit verdict/invert/projection, LogicalSemFilter
// bindings vs LogicalFilter, CreatePlan config resolution, and a full operator
// round-trip that filters rows through the dispatch engine.

#include "flock/functions/input_parser.hpp"
#include "flock/functions/operator/semantic_filter.hpp"
#include "flock/model_manager/model.hpp"
#include "flock/prompt_manager/prompt_manager.hpp"
#include "flock/runtime/endpoint_router.h"
#include "flock/runtime/llm_client.h"

#include "duckdb.hpp"
#include "duckdb/common/allocator.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parallel/interrupt.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/planner.hpp"
#include "flock/core/config.hpp"

#include "../../unit/functions/mock_provider.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <functional>
#include <memory>
#include <string>
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
// synthesizing flock's batch envelope: one completion whose message.content is {"items":[...]}.
// Verdicts are derived per <column>VALUE</column> in the rendered prompt, so they
// align with the rows regardless of how the engine batches them.
class ScriptedFakeClient : public ILLMClient {
public:
    explicit ScriptedFakeClient(std::function<bool(const std::string&)> verdict)
        : verdict_(std::move(verdict)) {}

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
        // The operator posts CHAT: the multi-row prompt is in messages[0].content
        // (an array of {type:text,text:...} parts, or a string).
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
        // One verdict per <row> block (skip the <header> row), keyed on the row's
        // first <column> value -> verdicts align with rows regardless of batching.
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
            bool verdict_value = false;
            const size_t cp = prompt.find(col_open, rp);
            if (cp != std::string::npos && cp < row_end) {
                const size_t start = cp + col_open.size();
                const size_t end = prompt.find(col_close, start);
                if (end != std::string::npos && end <= row_end) {
                    verdict_value = verdict_(prompt.substr(start, end - start));
                }
            }
            items.push_back(verdict_value);
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

    std::function<bool(const std::string&)> verdict_;
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

// Single-threaded sink -> finalize -> drain through the operator's engine + hooks.
// Returns the emitted column-0 values (BIGINT).
std::vector<int64_t> DriveFilter(PhysicalSemFilter& op, const std::vector<DataChunk*>& chunks) {
    auto gss = op.CreateGlobalSinkState();
    SemLocalSinkState local;
    for (auto* chunk : chunks) {
        for (;;) {
            auto signal = duckdb::make_shared_ptr<InterruptDoneSignalState>();
            InterruptState interrupt(signal);
            if (gss->SinkChunk(local, *chunk, interrupt) != SinkResultType::BLOCKED) {
                break;
            }
            signal->Await();
        }
    }
    gss->MergeLocal(local);
    for (;;) {
        auto signal = duckdb::make_shared_ptr<InterruptDoneSignalState>();
        InterruptState interrupt(signal);
        if (gss->FinalizeFlush(interrupt) != SinkFinalizeType::BLOCKED) {
            break;
        }
        signal->Await();
    }
    std::vector<int64_t> out_values;
    auto& alloc = duckdb::Allocator::DefaultAllocator();
    const auto parse = [&op](const nlohmann::json& element, const RowData& row, DataChunk& out) {
        op.ParseAndEmit(element, row, out);
    };
    for (;;) {
        DataChunk out;
        out.Initialize(alloc, op.GetTypes());
        auto signal = duckdb::make_shared_ptr<InterruptDoneSignalState>();
        InterruptState interrupt(signal);
        auto res = gss->Drain(out, parse, interrupt);
        if (res == SourceResultType::BLOCKED) {
            signal->Await();
            continue;
        }
        if (res == SourceResultType::HAVE_MORE_OUTPUT) {
            for (idx_t i = 0; i < out.size(); ++i) {
                out_values.push_back(out.GetValue(0, i).GetValue<int64_t>());
            }
            continue;
        }
        break;
    }
    return out_values;
}

// A logical child with caller-set bindings/types, to compare LogicalSemFilter and
// LogicalFilter binding/typing in isolation.
class StubChild : public duckdb::LogicalOperator {
public:
    StubChild(duckdb::vector<duckdb::ColumnBinding> bindings, duckdb::vector<LogicalType> child_types)
        : duckdb::LogicalOperator(duckdb::LogicalOperatorType::LOGICAL_DUMMY_SCAN),
          stub_bindings(std::move(bindings)), fixed_types(std::move(child_types)) {
        types = fixed_types;
    }
    duckdb::vector<duckdb::ColumnBinding> GetColumnBindings() override { return stub_bindings; }

protected:
    // ResolveOperatorTypes clears types before this runs, so repopulate them.
    void ResolveTypes() override { types = fixed_types; }

private:
    duckdb::vector<duckdb::ColumnBinding> stub_bindings;
    duckdb::vector<LogicalType> fixed_types;
};

// ===========================================================================
// 1. Prompt parity: RenderPrompt == the scalar's prompt for the same batch.
// ===========================================================================
TEST(SemFilter, Prompt_ParityWithScalar) {
    const std::string tmpl = "Is this review positive?";
    const std::string tuple_format = "XML";
    const std::vector<std::string> reviews = {"Great product!", "Terrible quality", "It's okay"};

    // Reference: build the prompt-struct vector the binder would hand the scalar,
    // run it through the real CastVectorOfStructsToJson + PromptManager::Render.
    auto cc_elem_type = LogicalType::STRUCT({{"data", LogicalType::VARCHAR}, {"name", LogicalType::VARCHAR}});
    auto prompt_type =
            LogicalType::STRUCT({{"prompt", LogicalType::VARCHAR}, {"context_columns", LogicalType::LIST(cc_elem_type)}});
    duckdb::Vector prompt_vec(prompt_type, reviews.size());
    for (idx_t i = 0; i < reviews.size(); ++i) {
        auto element = Value::STRUCT(cc_elem_type, {Value(reviews[i]), Value("review")});
        auto list = Value::LIST(cc_elem_type, {element});
        prompt_vec.SetValue(i, Value::STRUCT(prompt_type, {Value(tmpl), list}));
    }
    auto ref_columns = CastVectorOfStructsToJson(prompt_vec, static_cast<int>(reviews.size()))["context_columns"];
    auto ref_prompt = std::get<0>(
            PromptManager::Render(tmpl, ref_columns, ScalarFunctionType::FILTER, tuple_format));

    // Operator: one context column at index 0 with name "review".
    std::vector<SemContextColumn> ccs;
    SemContextColumn cc;
    cc.data_index = 0;
    cc.metadata["name"] = "review";
    ccs.push_back(std::move(cc));

    duckdb::PhysicalPlan plan(duckdb::Allocator::DefaultAllocator());
    auto& op = plan.Make<PhysicalSemFilter>(duckdb::vector<LogicalType>{LogicalType::VARCHAR}, 0,
                                            std::shared_ptr<ILLMClient>(), MakeRouter(), MakeParams(), nullptr,
                                            false, duckdb::vector<idx_t>{}, ccs, tmpl, tuple_format,
                                            std::string(), std::string())
                       .Cast<PhysicalSemFilter>();

    std::vector<RowData> batch;
    for (const auto& review : reviews) {
        RowData row;
        row.values.push_back(Value(review));
        batch.push_back(std::move(row));
    }
    EXPECT_EQ(op.RenderPrompt(batch), ref_prompt);
}

// ===========================================================================
// 1b. Slim prompt + FLOCK_SEM_VARIANTS rendering (pure function; the env gates
//     SemPromptSlim/SemVariantsFromEnv are read-once statics, so tests target
//     ParseSemVariants / RenderSlimSemanticPrompt directly).
// ===========================================================================

nlohmann::json SlimTestColumns(const std::vector<std::string>& reviews) {
    nlohmann::json col;
    col["name"] = "review";
    col["data"] = reviews;
    return nlohmann::json::array({col});
}

TEST(SemFilterSlim, ParseVariants) {
    const SemVariants none = ParseSemVariants("");
    EXPECT_FALSE(none.Any());

    const SemVariants two = ParseSemVariants("rowmajor,sandwich");
    EXPECT_TRUE(two.rowmajor);
    EXPECT_TRUE(two.sandwich);
    EXPECT_FALSE(two.symmetric);

    // chunk headers only make sense over row-major rows.
    const SemVariants chunked = ParseSemVariants("chunk");
    EXPECT_TRUE(chunked.chunk);
    EXPECT_TRUE(chunked.rowmajor);

    EXPECT_THROW(ParseSemVariants("rowmajor,typo"), std::runtime_error);
}

// Variant-free slim must stay byte-identical to the committed slim prompt, so
// new sweeps remain comparable with the existing slim results.
TEST(SemFilterSlim, BaselineByteParity) {
    const std::string tmpl = "Is this review positive?";
    const auto columns = SlimTestColumns({"Great product!", "Terrible quality"});
    const std::string expected =
            "For each row in the table below, decide whether it satisfies the criterion, "
            "judging every row independently on its own merits.\n"
            "Criterion: " + tmpl + "\n\n"
            + PromptManager::ConstructInputTuples(columns, "XML")
            + "\n\nReturn a JSON object {\"items\": [...]} with one boolean per row, in row order.";
    EXPECT_EQ(RenderSlimSemanticPrompt(SlimKind::kFilter, tmpl, columns, "XML", SemVariants{},
                                       /*state_output_shape=*/true),
              expected);
    // id/id_reason modes suppress the tail (they append their own output block).
    const auto no_tail = RenderSlimSemanticPrompt(SlimKind::kFilter, tmpl, columns, "XML",
                                                  SemVariants{}, /*state_output_shape=*/false);
    EXPECT_EQ(no_tail.find("Return a JSON object"), std::string::npos);
}

TEST(SemFilterSlim, VariantsCompose) {
    const std::string tmpl = "Is this review positive?";
    const auto columns = SlimTestColumns({"Great product!", "Terrible quality", "It's okay"});
    const auto v = ParseSemVariants("rowmajor,sandwich,symmetric,count,example,chunk");
    const auto prompt = RenderSlimSemanticPrompt(SlimKind::kFilter, tmpl, columns, "JSON", v,
                                                 /*state_output_shape=*/true);

    // P3: symmetric head replaces the default head.
    EXPECT_NE(prompt.find("output true if it satisfies the criterion and false if it does not"),
              std::string::npos);
    EXPECT_EQ(prompt.find("decide whether it satisfies"), std::string::npos);
    // P5: worked example precedes the real rows.
    const auto example_pos = prompt.find("Example with a different criterion");
    const auto first_row_pos = prompt.find("{\"id\": 1, \"review\": \"Great product!\"}");
    ASSERT_NE(example_pos, std::string::npos);
    ASSERT_NE(first_row_pos, std::string::npos);
    EXPECT_LT(example_pos, first_row_pos);
    // P4: explicit count line replaces the legacy one.
    EXPECT_NE(prompt.find("The table has 3 rows. Return exactly 3 booleans"), std::string::npos);
    EXPECT_EQ(prompt.find("Number of Tuples to Generate"), std::string::npos);
    // P1: row-major lines, one per row, id first (no columnar JSON dump).
    EXPECT_NE(prompt.find("{\"id\": 3, \"review\": \"It's okay\"}"), std::string::npos);
    // P6: chunk header (all 3 rows fit the first 8-row group).
    EXPECT_NE(prompt.find("### Rows 1-3"), std::string::npos);
    // P2: criterion restated after the rows.
    const auto reminder_pos = prompt.find("Reminder of the criterion: " + tmpl);
    ASSERT_NE(reminder_pos, std::string::npos);
    EXPECT_GT(reminder_pos, first_row_pos);
    // Tail still stated once.
    EXPECT_NE(prompt.find("one boolean per row, in row order"), std::string::npos);
}

TEST(SemFilterSlim, RowMajorEscapesAndChunks) {
    const std::string tmpl = "criterion";
    // 9 rows -> two chunk groups; a row with quote + newline must be JSON-escaped.
    std::vector<std::string> reviews(9, "plain");
    reviews[1] = "say \"hi\"\nnewline";
    const auto columns = SlimTestColumns(reviews);
    const auto v = ParseSemVariants("chunk");
    const auto prompt = RenderSlimSemanticPrompt(SlimKind::kFilter, tmpl, columns, "JSON", v,
                                                 /*state_output_shape=*/true);
    EXPECT_NE(prompt.find("### Rows 1-8"), std::string::npos);
    EXPECT_NE(prompt.find("### Rows 9-9"), std::string::npos);
    EXPECT_NE(prompt.find("{\"id\": 2, \"review\": \"say \\\"hi\\\"\\nnewline\"}"), std::string::npos);
    // count variant off -> legacy count line retained.
    EXPECT_NE(prompt.find("- The Number of Tuples to Generate Responses for: 9"), std::string::npos);
}

// ===========================================================================
// 2. ParseAndEmit: verdict, invert, projection, free_form fallback.
// ===========================================================================
PhysicalSemFilter& MakeParseOp(duckdb::PhysicalPlan& plan, duckdb::vector<LogicalType> out_types, bool invert,
                               duckdb::vector<idx_t> projection) {
    return plan
            .Make<PhysicalSemFilter>(std::move(out_types), 0, std::shared_ptr<ILLMClient>(), MakeRouter(),
                                     MakeParams(), nullptr, invert, std::move(projection),
                                     std::vector<SemContextColumn>{}, "p", "XML", std::string(), std::string())
            .Cast<PhysicalSemFilter>();
}

TEST(SemFilter, ParseAndEmit_VerdictInvertProjection) {
    auto& alloc = duckdb::Allocator::DefaultAllocator();
    duckdb::PhysicalPlan plan(alloc);

    RowData row;
    row.values = {Value::BIGINT(7), Value("hello")};

    // identity projection, no invert: true emits, false skips.
    {
        auto& op = MakeParseOp(plan, {LogicalType::BIGINT, LogicalType::VARCHAR}, /*invert=*/false, {});
        DataChunk out;
        out.Initialize(alloc, op.GetTypes());
        op.ParseAndEmit(nlohmann::json(true), row, out);
        op.ParseAndEmit(nlohmann::json(false), row, out);
        ASSERT_EQ(out.size(), 1u);
        EXPECT_EQ(out.GetValue(0, 0).GetValue<int64_t>(), 7);
        EXPECT_EQ(out.GetValue(1, 0).GetValue<std::string>(), "hello");
    }
    // invert: false emits, true skips.
    {
        auto& op = MakeParseOp(plan, {LogicalType::BIGINT, LogicalType::VARCHAR}, /*invert=*/true, {});
        DataChunk out;
        out.Initialize(alloc, op.GetTypes());
        op.ParseAndEmit(nlohmann::json(true), row, out);
        op.ParseAndEmit(nlohmann::json(false), row, out);
        ASSERT_EQ(out.size(), 1u);
        EXPECT_EQ(out.GetValue(0, 0).GetValue<int64_t>(), 7);
    }
    // projection to column 1 only.
    {
        auto& op = MakeParseOp(plan, {LogicalType::VARCHAR}, /*invert=*/false, duckdb::vector<idx_t>{1});
        DataChunk out;
        out.Initialize(alloc, op.GetTypes());
        op.ParseAndEmit(nlohmann::json(true), row, out);
        ASSERT_EQ(out.size(), 1u);
        EXPECT_EQ(out.GetValue(0, 0).GetValue<std::string>(), "hello");
    }
    // free_form fallback: parse a verdict from free text.
    {
        auto& op = MakeParseOp(plan, {LogicalType::BIGINT, LogicalType::VARCHAR}, /*invert=*/false, {});
        DataChunk out;
        out.Initialize(alloc, op.GetTypes());
        op.ParseAndEmit(nlohmann::json("true"), row, out);
        op.ParseAndEmit(nlohmann::json("false"), row, out);
        op.ParseAndEmit(nlohmann::json("Yes, definitely"), row, out);
        ASSERT_EQ(out.size(), 2u);
    }
    EXPECT_TRUE(PhysicalSemFilter::ParseVerdict(nlohmann::json(true)));
    EXPECT_FALSE(PhysicalSemFilter::ParseVerdict(nlohmann::json(false)));
    EXPECT_TRUE(PhysicalSemFilter::ParseVerdict(nlohmann::json("TRUE")));
    EXPECT_FALSE(PhysicalSemFilter::ParseVerdict(nlohmann::json("no")));
}

// ===========================================================================
// 3. Bindings: LogicalSemFilter mirrors LogicalFilter's bindings + types.
// ===========================================================================
TEST(SemFilter, Bindings_MirrorLogicalFilter) {
    duckdb::vector<duckdb::ColumnBinding> bindings{{1, 0}, {1, 1}, {1, 2}};
    duckdb::vector<LogicalType> child_types{LogicalType::BIGINT, LogicalType::VARCHAR, LogicalType::DOUBLE};
    duckdb::vector<idx_t> projection_map{2, 0};

    StubChild filter_child(bindings, child_types);
    duckdb::LogicalFilter lf(duckdb::make_uniq<duckdb::BoundConstantExpression>(Value::BOOLEAN(true)));
    lf.children.push_back(duckdb::make_uniq<StubChild>(bindings, child_types));
    lf.projection_map = projection_map;
    lf.ResolveOperatorTypes();

    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> residuals;
    LogicalSemFilter sf(duckdb::make_uniq<StubChild>(bindings, child_types),
                        duckdb::make_uniq<duckdb::BoundConstantExpression>(Value::BOOLEAN(true)),
                        std::move(residuals), projection_map, /*invert=*/false);
    sf.ResolveOperatorTypes();

    EXPECT_EQ(sf.GetColumnBindings(), lf.GetColumnBindings());
    EXPECT_EQ(sf.types, lf.types);
    EXPECT_TRUE(sf.HasProjectionMap());
    EXPECT_EQ(sf.llm_call_index, 0u);
}

// ===========================================================================
// 4. Engine integration: rows survive iff verdict XOR invert, projected.
// ===========================================================================
TEST(SemFilter, Engine_FiltersRows) {
    const int64_t total = 20;
    // Verdict true iff the row value is even.
    auto verdict = [](const std::string& value) {
        try {
            return std::stoll(value) % 2 == 0;
        } catch (...) {
            return false;
        }
    };
    auto fake = std::make_shared<ScriptedFakeClient>(verdict);

    std::vector<SemContextColumn> ccs;
    SemContextColumn cc;
    cc.data_index = 0;
    ccs.push_back(std::move(cc));

    duckdb::PhysicalPlan plan(duckdb::Allocator::DefaultAllocator());
    auto& op = plan.Make<PhysicalSemFilter>(duckdb::vector<LogicalType>{LogicalType::BIGINT}, 0, fake,
                                            MakeRouter(), MakeParams(/*cap=*/64, /*batch_size=*/8), nullptr,
                                            /*invert=*/false, duckdb::vector<idx_t>{}, ccs, "p", "XML",
                                            std::string(), std::string())
                       .Cast<PhysicalSemFilter>();

    auto chunks = BuildBigintChunks(total, /*chunk_size=*/8);
    std::vector<DataChunk*> chunk_ptrs;
    for (auto& c : chunks) {
        chunk_ptrs.push_back(c.get());
    }
    auto survivors = DriveFilter(op, chunk_ptrs);
    std::sort(survivors.begin(), survivors.end());
    std::vector<int64_t> expected;
    for (int64_t v = 0; v < total; v += 2) {
        expected.push_back(v);
    }
    EXPECT_EQ(survivors, expected);
}

TEST(SemFilter, Engine_InvertFlipsVerdict) {
    const int64_t total = 10;
    auto fake = std::make_shared<ScriptedFakeClient>(
            [](const std::string& value) { return std::stoll(value) % 2 == 0; });
    std::vector<SemContextColumn> ccs;
    SemContextColumn cc;
    cc.data_index = 0;
    ccs.push_back(std::move(cc));
    duckdb::PhysicalPlan plan(duckdb::Allocator::DefaultAllocator());
    auto& op = plan.Make<PhysicalSemFilter>(duckdb::vector<LogicalType>{LogicalType::BIGINT}, 0, fake,
                                            MakeRouter(), MakeParams(64, 16), nullptr, /*invert=*/true,
                                            duckdb::vector<idx_t>{}, ccs, "p", "XML", std::string(),
                                            std::string())
                       .Cast<PhysicalSemFilter>();
    auto chunks = BuildBigintChunks(total, 16);
    auto survivors = DriveFilter(op, {chunks[0].get()});
    std::sort(survivors.begin(), survivors.end());
    std::vector<int64_t> expected;  // odd rows survive under invert
    for (int64_t v = 1; v < total; v += 2) {
        expected.push_back(v);
    }
    EXPECT_EQ(survivors, expected);
}

// ===========================================================================
// 5. CreatePlan: a bound llm_filter -> PhysicalSemFilter with resolved config.
// ===========================================================================
TEST(SemFilter, CreatePlan_ResolvesConfig) {
    duckdb::Connection con(*g_db);
    con.Query("CREATE SECRET (TYPE OPENAI, API_KEY 'test-key');");
    Model::SetMockProvider(std::make_shared<MockProvider>(ModelDetails{}));

    con.Query("CREATE TABLE reviews(review VARCHAR, score INT);");
    con.Query("INSERT INTO reviews VALUES ('good', 5), ('bad', 1);");

    const std::string query =
            "SELECT review FROM reviews "
            "WHERE llm_filter({'model_name': 'gpt-4o'}, "
            "{'prompt': 'Is this positive?', 'context_columns': [{'data': review}]});";

    // Swap the LogicalFilter for a LogicalSemFilter (a stand-in for the rewrite).
    std::function<duckdb::unique_ptr<duckdb::LogicalOperator>(duckdb::unique_ptr<duckdb::LogicalOperator>)> swap =
            [&](duckdb::unique_ptr<duckdb::LogicalOperator> node) -> duckdb::unique_ptr<duckdb::LogicalOperator> {
        if (node->type == duckdb::LogicalOperatorType::LOGICAL_FILTER) {
            auto& filter = node->Cast<duckdb::LogicalFilter>();
            auto child = std::move(filter.children[0]);
            // WHERE llm_filter(...) wraps the VARCHAR result in a cast to BOOLEAN;
            // peel it as the real optimizer rewrite does to reach the bare call.
            auto llm_call = std::move(filter.expressions[0]);
            while (llm_call->expression_class == duckdb::ExpressionClass::BOUND_CAST) {
                llm_call = std::move(llm_call->Cast<duckdb::BoundCastExpression>().child);
            }
            duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> residuals;
            for (idx_t i = 1; i < filter.expressions.size(); ++i) {
                residuals.push_back(std::move(filter.expressions[i]));
            }
            return duckdb::make_uniq<LogicalSemFilter>(std::move(child), std::move(llm_call),
                                                       std::move(residuals), filter.projection_map,
                                                       /*invert=*/false);
        }
        for (auto& child : node->children) {
            child = swap(std::move(child));
        }
        return node;
    };

    // Parse outside the transaction lambda: ParseStatements takes the context lock,
    // which RunFunctionInTransaction also holds (re-locking would self-deadlock).
    auto statements = con.context->ParseStatements(query);
    ASSERT_FALSE(statements.empty());

    PhysicalSemFilter* found = nullptr;
    con.context->RunFunctionInTransaction([&]() {
        // Plan WITHOUT the optimizer (which would push the filter into the scan),
        // so a LogicalFilter survives for the rewrite stand-in to swap.
        duckdb::Planner planner(*con.context);
        planner.CreatePlan(std::move(statements[0]));
        auto logical = swap(std::move(planner.plan));

        duckdb::PhysicalPlanGenerator generator(*con.context);
        auto physical = generator.Plan(std::move(logical));
        std::function<void(duckdb::PhysicalOperator&)> walk = [&](duckdb::PhysicalOperator& op) {
            if (auto* sem = dynamic_cast<PhysicalSemFilter*>(&op)) {
                found = sem;
            }
            for (auto& child : op.children) {
                walk(child.get());
            }
        };
        walk(physical->Root());

        ASSERT_NE(found, nullptr);
        EXPECT_EQ(found->PromptTemplate(), "Is this positive?");
        EXPECT_FALSE(found->ServedModel().empty());
        ASSERT_EQ(found->ContextColumns().size(), 1u);
        EXPECT_EQ(found->ContextColumns()[0].data_index, 0u);
        EXPECT_FALSE(found->StickyKey().empty());

        auto expected = ResolveSemanticParams(*con.context, "gpt-4o");
        EXPECT_EQ(found->Config().batch_size, expected.batch_size);
        EXPECT_EQ(found->Config().in_flight_cap, expected.in_flight_cap);
        EXPECT_EQ(found->Config().max_output_tokens, expected.max_output_tokens);
        EXPECT_EQ(found->Config().response_format, expected.response_format);
    });

    Model::ResetMockProvider();
    con.Query("DROP TABLE reviews;");
}

}  // namespace
}  // namespace flock

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    duckdb::DuckDB db("flock_sem_filter_test.db");
    flock::g_db = &db;
    flock::Config::GetConnection(&*db.instance);
    int rc = RUN_ALL_TESTS();
    flock::g_db = nullptr;
    return rc;
}
