// Component tests for the async semantic filter operator: prompt parity with the
// scalar llm_filter, ParseAndEmit verdict/invert/projection, LogicalSemFilter
// bindings vs LogicalFilter, CreatePlan config resolution, and a full operator
// round-trip that filters rows through the dispatch engine.

#include "flock/functions/input_parser.hpp"
#include "flock/functions/operator/semantic_filter.hpp"
#include "flock/functions/operator/embedding_proxy.hpp"
#include "flock/model_manager/model.hpp"
#include "flock/prompt_manager/prompt_manager.hpp"
#include "flock/runtime/endpoint_router.h"
#include "flock/runtime/llm_client.h"

#include "duckdb.hpp"
#include "duckdb/parser/parser.hpp"
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
#include <fstream>
#include <filesystem>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
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
        {
            std::lock_guard<std::mutex> lock(payload_mutex_);
            last_payload_ = payload;
        }
        LLMResponse r = Synthesize(payload);
        if (on_done) {
            on_done(std::move(r));
        }
        return LLMRequestHandle{id_.fetch_add(1, std::memory_order_relaxed) + 1};
    }
    void CancelByGeneration(uint64_t /*generation*/) override {}

    nlohmann::json LastPayload() const {
        std::lock_guard<std::mutex> lock(payload_mutex_);
        return nlohmann::json::parse(last_payload_);
    }
    uint64_t Submitted() const { return id_.load(std::memory_order_relaxed); }

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
    mutable std::mutex payload_mutex_;
    std::string last_payload_;
};

class ProxyFakeClient : public ILLMClient {
public:
    LLMRequestHandle Submit(const std::string& endpoint, const std::string& payload, uint64_t,
                            const std::string&, LLMOnDone on_done) override {
        const auto body = nlohmann::json::parse(payload);
        const bool proxy = body.at("model") == "cheap";
        if (proxy) proxy_endpoints.push_back(endpoint);
        else main_endpoints.push_back(endpoint);
        const std::string prompt = body.at("messages")[0].at("content")[0].at("text");
        std::istringstream rows(prompt);
        std::string value;
        nlohmann::json items = nlohmann::json::array();
        while (std::getline(rows, value, ',')) {
            if (value.empty()) continue;
            const int row = std::stoi(value);
            items.push_back(proxy ? nlohmann::json(row <= 1 ? 0.01 : row == 2 ? 0.5 : 0.99)
                                  : nlohmann::json(row == 2));
            if (!proxy) main_rows.push_back(row);
        }
        if (proxy) {
            EXPECT_EQ(body.at("response_format").at("json_schema").at("name"), "proxy_scores");
        }
        LLMResponse response;
        response.ok = true;
        response.http_status = 200;
        response.body = nlohmann::json{{"choices", nlohmann::json::array(
                {{{"message", {{"content", nlohmann::json{{"items", items}}.dump()}}}}})}}.dump();
        on_done(std::move(response));
        return {1};
    }
    void CancelByGeneration(uint64_t) override {}
    std::vector<int> main_rows;
    std::vector<std::string> proxy_endpoints;
    std::vector<std::string> main_endpoints;
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
// Returns the emitted column-0 values (BIGINT). Like DuckDB, stops feeding input
// once the sink returns FINISHED. `setup` may adjust the engine state first.
std::vector<int64_t> DriveFilter(PhysicalSemFilter& op, const std::vector<DataChunk*>& chunks,
                                 const std::function<void(SemGlobalSinkState&)>& setup = nullptr,
                                 size_t* chunks_sunk = nullptr) {
    auto gss = op.CreateGlobalSinkState();
    if (setup) {
        setup(*gss);
    }
    SemLocalSinkState local;
    bool finished = false;
    size_t sunk = 0;
    for (auto* chunk : chunks) {
        if (finished) {
            break;
        }
        for (;;) {
            auto signal = duckdb::make_shared_ptr<InterruptDoneSignalState>();
            InterruptState interrupt(signal);
            const auto res = gss->SinkChunk(local, *chunk, interrupt);
            if (res == SinkResultType::FINISHED) {
                finished = true;
                break;
            }
            if (res != SinkResultType::BLOCKED) {
                ++sunk;
                break;
            }
            signal->Await();
        }
    }
    if (chunks_sunk) {
        *chunks_sunk = sunk;
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

TEST(SemFilter, ProxyCascadeRoutesOnlyUncertainRowsToMainModel) {
    auto fake = std::make_shared<ProxyFakeClient>();
    auto cfg = MakeParams(1, 3);
    cfg.filter_proxy_model = "cheap";
    cfg.filter_proxy_endpoint = "http://localhost:8001/v1/chat/completions";
    cfg.filter_proxy_negative_threshold = 0.05;
    cfg.filter_proxy_positive_threshold = 0.95;
    SemGlobalSinkState state(fake, MakeRouter(), cfg);
    state.served_model = "main";
    state.proxy_served_model = "cheap";
    state.render_prompt = [](const std::vector<RowData>& rows) {
        std::string text;
        for (const auto& row : rows) {
            if (!text.empty()) text += ',';
            text += std::to_string(row.values[0].GetValue<int64_t>());
        }
        return text;
    };
    state.render_proxy_prompt = state.render_prompt;
    state.response_schema = [](size_t n) {
        return SemanticOperatorBase::ItemsResponseFormat("filter_results", {{"type", "boolean"}}, n);
    };

    auto chunks = BuildBigintChunks(4, 4); // 0 and 1 share the negative score.
    SemLocalSinkState local;
    auto signal = duckdb::make_shared_ptr<InterruptDoneSignalState>();
    InterruptState interrupt(signal);
    EXPECT_EQ(state.SinkChunk(local, *chunks[0], interrupt), SinkResultType::NEED_MORE_INPUT);
    EXPECT_EQ(state.FinalizeFlush(interrupt), SinkFinalizeType::READY);

    std::vector<int64_t> result;
    for (;;) {
        DataChunk out;
        out.Initialize(duckdb::Allocator::DefaultAllocator(), {LogicalType::BIGINT});
        const auto parse = [](const nlohmann::json& verdict, const RowData& row, DataChunk& chunk) {
            if (verdict.get<bool>()) {
                chunk.SetValue(0, chunk.size(), row.values[0]);
                chunk.SetCardinality(chunk.size() + 1);
            }
        };
        const auto status = state.Drain(out, parse, interrupt);
        if (status == SourceResultType::FINISHED) break;
        ASSERT_NE(status, SourceResultType::BLOCKED);
        for (idx_t i = 0; i < out.size(); ++i) result.push_back(out.GetValue(0, i).GetValue<int64_t>());
    }
    std::sort(result.begin(), result.end());
    EXPECT_EQ(result, (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(fake->main_rows, (std::vector<int>{2}));
    EXPECT_EQ(fake->proxy_endpoints, (std::vector<std::string>{
            "http://localhost:8001/v1/chat/completions", "http://localhost:8001/v1/chat/completions"}));
    EXPECT_EQ(fake->main_endpoints, (std::vector<std::string>{"http://localhost:8000/v1"}));
}

// ---- Embedding proxy: engine wiring (scripted stage + policy) ----------------

// Policy over 4 units: asks unit 0, auto-accepts unit 3; once unit 0 is
// answered asks unit 1; unit 2 is never asked (rejected).
class ScriptedPolicy : public ProxyPolicy {
public:
    ProxyStep Start() override {
        ProxyStep step;
        step.send = {0};
        step.resolve = {{3, true}};
        return step;
    }
    ProxyStep OnVerdict(size_t unit, bool verdict) override {
        ProxyStep step;
        step.resolve = {{unit, verdict}};
        if (unit == 0) step.send = {1};
        verdicts.emplace_back(unit, verdict);
        return step;
    }
    std::string Stats() const override { return "{}"; }
    std::vector<std::pair<size_t, bool>> verdicts;
};

// Rows i and i+1 (i even) form one unit: unit = value / 2.
class ScriptedStage : public ProxyStage {
public:
    explicit ScriptedStage(bool exact) : exact_(exact) {}
    ProxyPlan Plan(const std::vector<PendingRequest>& rows, ILLMClient&, uint64_t) override {
        ProxyPlan plan;
        planned_rows = rows.size();
        if (exact_) {
            plan.info = "exact (scripted)";
            return plan;
        }
        for (const auto& r : rows) plan.row_unit.push_back(static_cast<size_t>(r.row.values[0].GetValue<int64_t>() / 2));
        plan.num_units = 4;
        auto policy = std::make_unique<ScriptedPolicy>();
        policy_view = policy.get();
        plan.policy = std::move(policy);
        return plan;
    }
    size_t planned_rows = 0;
    ScriptedPolicy* policy_view = nullptr;

private:
    bool exact_;
};

// Main model: TRUE only for value 0. Records the values it was asked about.
class ValueFakeClient : public ILLMClient {
public:
    LLMRequestHandle Submit(const std::string&, const std::string& payload, uint64_t, const std::string&,
                            LLMOnDone on_done) override {
        const auto body = nlohmann::json::parse(payload);
        const std::string prompt = body.at("messages")[0].at("content")[0].at("text");
        std::istringstream rows(prompt);
        std::string value;
        nlohmann::json items = nlohmann::json::array();
        while (std::getline(rows, value, ',')) {
            if (value.empty()) continue;
            asked.push_back(std::stoi(value));
            items.push_back(std::stoi(value) == 0);
        }
        LLMResponse response;
        response.ok = true;
        response.http_status = 200;
        response.body = nlohmann::json{{"choices", nlohmann::json::array(
                {{{"message", {{"content", nlohmann::json{{"items", items}}.dump()}}}}})}}.dump();
        on_done(std::move(response));
        return {1};
    }
    void CancelByGeneration(uint64_t) override {}
    std::vector<int> asked;
};

std::vector<int64_t> RunProxyEngine(const std::shared_ptr<ProxyStage>& stage, const std::shared_ptr<ILLMClient>& client,
                                    int64_t rows) {
    SemGlobalSinkState state(client, MakeRouter(), MakeParams(2, 1));
    state.served_model = "main";
    state.proxy_stage = stage;
    state.verdict_of = [](const nlohmann::json& e) { return PhysicalSemFilter::ParseVerdict(e); };
    state.render_prompt = [](const std::vector<RowData>& batch) {
        std::string text;
        for (const auto& row : batch) text += std::to_string(row.values[0].GetValue<int64_t>()) + ",";
        return text;
    };
    state.response_schema = [](size_t n) {
        return SemanticOperatorBase::ItemsResponseFormat("filter_results", {{"type", "boolean"}}, n);
    };
    auto chunks = BuildBigintChunks(rows, 3);
    SemLocalSinkState local;
    auto signal = duckdb::make_shared_ptr<InterruptDoneSignalState>();
    InterruptState interrupt(signal);
    for (auto& chunk : chunks) {
        EXPECT_EQ(state.SinkChunk(local, *chunk, interrupt), SinkResultType::NEED_MORE_INPUT);
    }
    state.MergeLocal(local);
    // The exact fallback flushes through the ordinary path, which returns
    // BLOCKED at the in-flight cap; retry like DuckDB does after the wakeup.
    for (;;) {
        auto finalize_signal = duckdb::make_shared_ptr<InterruptDoneSignalState>();
        InterruptState finalize_interrupt(finalize_signal);
        if (state.FinalizeFlush(finalize_interrupt) != SinkFinalizeType::BLOCKED) {
            break;
        }
        finalize_signal->Await();
    }
    std::vector<int64_t> result;
    for (int guard = 0; guard < 1000; ++guard) {
        DataChunk out;
        out.Initialize(duckdb::Allocator::DefaultAllocator(), {LogicalType::BIGINT});
        const auto parse = [](const nlohmann::json& verdict, const RowData& row, DataChunk& chunk) {
            if (PhysicalSemFilter::ParseVerdict(verdict)) {
                chunk.SetValue(0, chunk.size(), row.values[0]);
                chunk.SetCardinality(chunk.size() + 1);
            }
        };
        const auto status = state.Drain(out, parse, interrupt);
        for (idx_t i = 0; i < out.size(); ++i) result.push_back(out.GetValue(0, i).GetValue<int64_t>());
        if (status == SourceResultType::FINISHED) break;
        EXPECT_NE(status, SourceResultType::BLOCKED);
        if (status == SourceResultType::BLOCKED) break;
    }
    std::sort(result.begin(), result.end());
    return result;
}

TEST(SemFilter, EmbeddingProxyEngineFollowsPolicy) {
    auto stage = std::make_shared<ScriptedStage>(false);
    auto client = std::make_shared<ValueFakeClient>();
    const auto result = RunProxyEngine(stage, client, 8);
    EXPECT_EQ(stage->planned_rows, 8u);  // everything buffered before planning
    // One representative row per asked unit: unit 0 (row 0), then unit 1 (row 2).
    EXPECT_EQ(client->asked, (std::vector<int>{0, 2}));
    // Unit 0 TRUE -> rows 0, 1; unit 1 FALSE; unit 2 rejected; unit 3 auto-accepted -> rows 6, 7.
    EXPECT_EQ(result, (std::vector<int64_t>{0, 1, 6, 7}));
    ASSERT_NE(stage->policy_view, nullptr);
    EXPECT_EQ(stage->policy_view->verdicts.size(), 2u);
}

TEST(SemFilter, EmbeddingProxyExactFallbackAsksEveryRow) {
    auto stage = std::make_shared<ScriptedStage>(true);
    auto client = std::make_shared<ValueFakeClient>();
    const auto result = RunProxyEngine(stage, client, 8);
    std::sort(client->asked.begin(), client->asked.end());
    EXPECT_EQ(client->asked, (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}));
    EXPECT_EQ(result, (std::vector<int64_t>{0}));
}

TEST(SemFilter, EmbeddingProxyEmitsMoreThanOneVector) {
    // Auto-accepting a unit with > STANDARD_VECTOR_SIZE rows must span Drain calls.
    class BigStage : public ProxyStage {
    public:
        ProxyPlan Plan(const std::vector<PendingRequest>& rows, ILLMClient&, uint64_t) override {
            class AcceptAll : public ProxyPolicy {
                ProxyStep Start() override { return {{}, {{0, true}}}; }
                ProxyStep OnVerdict(size_t, bool) override { return {}; }
                std::string Stats() const override { return "{}"; }
            };
            ProxyPlan plan;
            plan.row_unit.assign(rows.size(), 0);
            plan.num_units = 1;
            plan.policy = std::make_unique<AcceptAll>();
            return plan;
        }
    };
    auto client = std::make_shared<ValueFakeClient>();
    const int64_t n = STANDARD_VECTOR_SIZE * 2 + 5;
    const auto result = RunProxyEngine(std::make_shared<BigStage>(), client, n);
    EXPECT_EQ(result.size(), static_cast<size_t>(n));
    EXPECT_TRUE(client->asked.empty());
}

// ---- Embedding proxy: EmbeddingProxyStage against a fake /v1/embeddings ------

// Texts and images get fixed 3-d vectors; image files hold one letter, so the
// data URL ends with its base64 ("A" -> "QQ==").
class EmbeddingFakeClient : public ILLMClient {
public:
    explicit EmbeddingFakeClient(bool fail = false) : fail_(fail) {}
    LLMRequestHandle Submit(const std::string& endpoint, const std::string& payload, uint64_t, const std::string&,
                            LLMOnDone on_done) override {
        endpoints.push_back(endpoint);
        const auto body = nlohmann::json::parse(payload);
        EXPECT_EQ(body.at("model"), "clip");
        nlohmann::json data = nlohmann::json::array();
        if (body.contains("input")) {
            size_t i = 0;
            for (const auto& text : body.at("input")) {
                texts.push_back(text.get<std::string>());
                data.push_back({{"index", i++}, {"embedding", TextVector(text.get<std::string>())}});
            }
        } else {
            const std::string url = body.at("messages")[0].at("content")[0].at("image_url").at("url");
            EXPECT_EQ(url.rfind("data:image/png;base64,", 0), 0u);
            data.push_back({{"index", 0}, {"embedding", ImageVector(url)}});
        }
        LLMResponse response;
        response.ok = !fail_;
        response.http_status = fail_ ? 500 : 200;
        response.body = nlohmann::json{{"data", data}}.dump();
        on_done(std::move(response));
        return {1};
    }
    void CancelByGeneration(uint64_t) override {}
    std::vector<std::string> endpoints;
    std::vector<std::string> texts;

private:
    static std::vector<float> TextVector(const std::string& t) {
        if (t == "logo of x") return {1.f, 0.f, 0.f};
        if (t == "logo of y") return {0.f, 1.f, 0.f};
        return {0.f, 0.f, 1.f};
    }
    static std::vector<float> ImageVector(const std::string& url) {
        if (url.size() >= 4 && url.compare(url.size() - 4, 4, "QQ==") == 0) return {1.f, 0.2f, 0.f};  // A
        if (url.size() >= 4 && url.compare(url.size() - 4, 4, "Qg==") == 0) return {0.2f, 1.f, 0.f};  // B
        return {0.5f, 0.5f, 0.1f};                                                                  // C
    }
    bool fail_;
};

struct EmbeddingFixture {
    std::vector<PendingRequest> rows;
    std::vector<std::string> paths;
    EmbeddingFixture() {
        const auto dir = std::filesystem::temp_directory_path() / "flock_embedding_proxy_test";
        std::filesystem::create_directories(dir);
        for (const char* letter : {"A", "B", "C"}) {
            const auto path = (dir / (std::string(letter) + ".png")).string();
            std::ofstream(path, std::ios::binary) << letter;
            paths.push_back(path);
        }
        uint64_t id = 0;
        for (const char* airline : {"x", "y"}) {
            for (const auto& path : paths) {
                PendingRequest req;
                req.row_id = id++;
                req.row.values = {Value(airline), Value(path)};
                rows.push_back(std::move(req));
            }
        }
    }
    EmbeddingProxyConfig Config(const std::string& kind = "embedding") const {
        EmbeddingProxyConfig cfg;
        cfg.kind = kind;
        cfg.endpoint = "http://localhost:8001/v1/embeddings";
        cfg.model = "clip";
        cfg.text_template = "logo of {}";
        cfg.image_index = 1;
        cfg.image_column = {{"name", "logo"}, {"type", "image"}};
        cfg.text_indices = {0};
        cfg.min_rows = 1;
        cfg.topk.k_min = 1;
        cfg.topk.gap = 0;
        cfg.topk.audit_near = 0.0;
        cfg.topk.audit_far = 0.0;
        return cfg;
    }
};

TEST(SemFilter, EmbeddingProxyStageRanksPerGroup) {
    EmbeddingFixture fx;
    EmbeddingProxyStage stage(fx.Config());
    EmbeddingFakeClient client;
    auto plan = stage.Plan(fx.rows, client, 1);
    ASSERT_NE(plan.policy, nullptr) << plan.info;
    EXPECT_EQ(plan.num_units, 6u);
    EXPECT_EQ(client.texts, (std::vector<std::string>{"logo of x", "logo of y"}));
    EXPECT_EQ(client.endpoints.size(), 1u + 3u);  // one text batch + one request per distinct image
    std::set<uint64_t> asked;
    for (size_t unit : plan.policy->Start().send) {
        for (size_t i = 0; i < plan.row_unit.size(); ++i) {
            if (plan.row_unit[i] == unit) asked.insert(fx.rows[i].row_id);
        }
    }
    // Top-1 per airline: x -> A (row 0), y -> B (row 4).
    EXPECT_EQ(asked, (std::set<uint64_t>{0, 4}));
}

TEST(SemFilter, EmbeddingProxyStageDedupsIdenticalRows) {
    EmbeddingFixture fx;
    auto rows = fx.rows;
    for (const auto& r : fx.rows) {
        PendingRequest dup;
        dup.row_id = r.row_id + 100;
        dup.row = r.row;
        rows.push_back(std::move(dup));
    }
    EmbeddingProxyStage stage(fx.Config());
    EmbeddingFakeClient client;
    auto plan = stage.Plan(rows, client, 1);
    ASSERT_NE(plan.policy, nullptr);
    EXPECT_EQ(plan.num_units, 6u);
    for (size_t i = 0; i < fx.rows.size(); ++i) EXPECT_EQ(plan.row_unit[i], plan.row_unit[i + fx.rows.size()]);
}

TEST(SemFilter, EmbeddingProxyStageFallsBackToExact) {
    EmbeddingFixture fx;
    auto cfg = fx.Config();
    cfg.min_rows = 100;
    EmbeddingFakeClient client;
    auto plan = EmbeddingProxyStage(cfg).Plan(fx.rows, client, 1);
    EXPECT_EQ(plan.policy, nullptr);
    EXPECT_NE(plan.info.find("min_rows"), std::string::npos);
    EXPECT_TRUE(client.endpoints.empty());

    EmbeddingFakeClient failing(true);
    plan = EmbeddingProxyStage(fx.Config()).Plan(fx.rows, failing, 1);
    EXPECT_EQ(plan.policy, nullptr);
    EXPECT_NE(plan.info.find("embedding stage failed"), std::string::npos);
}

TEST(SemFilter, EmbeddingProxyStageLotusKind) {
    EmbeddingFixture fx;
    EmbeddingFakeClient client;
    auto plan = EmbeddingProxyStage(fx.Config("lotus")).Plan(fx.rows, client, 1);
    ASSERT_NE(plan.policy, nullptr);
    EXPECT_NE(plan.policy->Stats().find("lotus_cascade"), std::string::npos);
}

TEST(SemFilter, EmbeddingProxyRenderText) {
    EXPECT_EQ(EmbeddingProxyStage::RenderText("the logo of {}", {"Delta"}, "p"), "the logo of Delta");
    EXPECT_EQ(EmbeddingProxyStage::RenderText("", {"Delta", "US"}, "p"), "Delta, US");
    EXPECT_EQ(EmbeddingProxyStage::RenderText("logo", {"Delta"}, "p"), "logo Delta");
    EXPECT_EQ(EmbeddingProxyStage::RenderText("", {}, "a zebra?"), "a zebra?");
    EXPECT_EQ(EmbeddingProxyStage::RenderText("a zebra", {}, "p"), "a zebra");
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
// 1b. Slim prompt rendering (pure function; the SemPromptSlim env gate is a
//     read-once static, so tests target RenderSlimSemanticPrompt directly).
//     Both forms are byte-pinned against the arms measured in the prompt-
//     variant study (analysis/prompt_engineering.md Q4).
// ===========================================================================

nlohmann::json SlimTestColumns(const std::vector<std::string>& reviews) {
    nlohmann::json col;
    col["name"] = "review";
    col["data"] = reviews;
    return nlohmann::json::array({col});
}

// A single-row batch must stay byte-identical to the original slim prompt
TEST(SemFilterSlim, SingleRowByteParity) {
    const std::string tmpl = "Is this review positive?";
    const auto columns = SlimTestColumns({"Great product!"});
    const std::string expected =
            "For each row in the table below, decide whether it satisfies the criterion, "
            "judging every row independently on its own merits.\n"
            "Criterion: " + tmpl + "\n\n"
            + PromptManager::ConstructInputTuples(columns, "XML")
            + "\n\nReturn a JSON object {\"items\": [...]} with one boolean per row, in row order.";
    EXPECT_EQ(RenderSlimSemanticPrompt(SlimKind::kFilter, tmpl, columns, "XML"), expected);
}

// A multi-row batch must stay byte-identical to the measured count,chunk arm
TEST(SemFilterSlim, BatchedByteParity) {
    const std::string tmpl = "Is this review positive?";
    const auto columns = SlimTestColumns({"Great product!", "Terrible quality", "It's okay"});
    const std::string expected =
            "For each row in the table below, decide whether it satisfies the criterion, "
            "judging every row independently on its own merits.\n"
            "Criterion: " + tmpl + "\n\n"
            "The table has 3 rows. Return exactly 3 booleans; the i-th boolean answers row i.\n\n"
            "### Rows 1-3\n"
            "{\"id\": 1, \"review\": \"Great product!\"}\n"
            "{\"id\": 2, \"review\": \"Terrible quality\"}\n"
            "{\"id\": 3, \"review\": \"It's okay\"}\n"
            "\n\nReturn a JSON object {\"items\": [...]} with one boolean per row, in row order.";
    EXPECT_EQ(RenderSlimSemanticPrompt(SlimKind::kFilter, tmpl, columns, "JSON"), expected);
}

TEST(SemFilterSlim, BatchedEscapesAndChunks) {
    const std::string tmpl = "criterion";
    // 9 rows -> two chunk groups; a row with quote + newline must be JSON-escaped.
    std::vector<std::string> reviews(9, "plain");
    reviews[1] = "say \"hi\"\nnewline";
    const auto columns = SlimTestColumns(reviews);
    const auto prompt = RenderSlimSemanticPrompt(SlimKind::kFilter, tmpl, columns, "JSON");
    EXPECT_NE(prompt.find("### Rows 1-8"), std::string::npos);
    EXPECT_NE(prompt.find("### Rows 9-9"), std::string::npos);
    EXPECT_NE(prompt.find("{\"id\": 2, \"review\": \"say \\\"hi\\\"\\nnewline\"}"), std::string::npos);
    EXPECT_NE(prompt.find("The table has 9 rows. Return exactly 9 booleans"), std::string::npos);
    // The legacy columnar count line belongs to the single-row form only.
    EXPECT_EQ(prompt.find("Number of Tuples to Generate"), std::string::npos);
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
    duckdb::vector<duckdb::ColumnBinding> bindings{duckdb::ColumnBinding(duckdb::TableIndex(1), duckdb::ProjectionIndex(0)), duckdb::ColumnBinding(duckdb::TableIndex(1), duckdb::ProjectionIndex(1)), duckdb::ColumnBinding(duckdb::TableIndex(1), duckdb::ProjectionIndex(2))};
    duckdb::vector<LogicalType> child_types{LogicalType::BIGINT, LogicalType::VARCHAR, LogicalType::DOUBLE};
    duckdb::vector<duckdb::ProjectionIndex> projection_map{duckdb::ProjectionIndex(2), duckdb::ProjectionIndex(0)};

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

TEST(SemFilter, Engine_AttachesImagesAtOriginalRowIndices) {
    const auto image_path = std::filesystem::temp_directory_path() /
                            ("skein-test-image-" + std::to_string(std::random_device{}()) + ".jpg");
    {
        std::ofstream image_file(image_path, std::ios::binary);
        image_file.write("\xff\xd8\xff", 3);
    }
    auto fake = std::make_shared<ScriptedFakeClient>([](const std::string&) { return true; });
    SemContextColumn text;
    text.data_index = 0;
    text.metadata["name"] = "id";
    SemContextColumn image;
    image.data_index = 1;
    image.metadata["name"] = "photo";
    image.metadata["type"] = "image";
    image.metadata["detail"] = "high";

    duckdb::PhysicalPlan plan(duckdb::Allocator::DefaultAllocator());
    auto& op = plan.Make<PhysicalSemFilter>(duckdb::vector<LogicalType>{LogicalType::BIGINT}, 0, fake,
                                            MakeRouter(), MakeParams(4, 2), nullptr, false,
                                            duckdb::vector<idx_t>{0},
                                            std::vector<SemContextColumn>{text, image}, "inspect photo", "XML",
                                            std::string(), std::string()).Cast<PhysicalSemFilter>();
    DataChunk input;
    input.Initialize(duckdb::Allocator::DefaultAllocator(),
                     duckdb::vector<LogicalType>{LogicalType::BIGINT, LogicalType::VARCHAR});
    input.SetValue(0, 0, Value::BIGINT(1));
    input.SetValue(1, 0, Value());
    input.SetValue(0, 1, Value::BIGINT(2));
    input.SetValue(1, 1, Value(image_path.string()));
    input.SetCardinality(2);

    EXPECT_EQ(DriveFilter(op, {&input}), (std::vector<int64_t>{1, 2}));
    const auto content = fake->LastPayload()["messages"][0]["content"];
    ASSERT_EQ(content.size(), 3u);
    EXPECT_EQ(content[1]["text"], "Attachment photo, row 2:");
    EXPECT_EQ(content[2]["image_url"]["url"], "data:image/jpeg;base64,/9j/");
    EXPECT_EQ(content[2]["image_url"]["detail"], "high");
    EXPECT_EQ(content[0]["text"].get<std::string>().find("{{TUPLES}}"), std::string::npos);
    std::filesystem::remove(image_path);
}

TEST(SemFilter, ImageOnlyPromptHasRowCount) {
    nlohmann::json columns = nlohmann::json::array(
            {{{"name", "photo"}, {"type", "image"}, {"data", {"a.jpg", "b.jpg"}}}});
    const auto rendered = PromptManager::Render("Describe each photo", columns,
                                                ScalarFunctionType::FILTER, "XML");
    const auto& prompt = std::get<0>(rendered);
    EXPECT_NE(prompt.find("The Number of Tuples to Generate Responses for: 2"), std::string::npos);
    EXPECT_EQ(prompt.find("{{TUPLES}}"), std::string::npos);
    EXPECT_EQ(std::get<1>(rendered)["image"].size(), 1u);
}

// Feed an actual SQL Cartesian product through the deterministic filter engine.
// Pair IDs encode left*10+right; duplicate left rows must remain duplicates.
TEST(SemFilter, Engine_UnkeyedPairsPreserveBagSemantics) {
    duckdb::Connection con(*g_db);
    auto pairs = con.Query("SELECT l.v * 10 + r.v AS pair_id "
                           "FROM (VALUES (0), (0), (1)) l(v) "
                           "CROSS JOIN (VALUES (2), (3)) r(v)");
    ASSERT_FALSE(pairs->HasError()) << pairs->GetError();
    ASSERT_EQ(pairs->RowCount(), 6u);

    DataChunk input;
    input.Initialize(duckdb::Allocator::DefaultAllocator(), duckdb::vector<LogicalType>{LogicalType::BIGINT});
    for (idx_t i = 0; i < pairs->RowCount(); ++i) {
        input.SetValue(0, i, Value::BIGINT(pairs->Collection().GetValue(0, i).GetValue<int32_t>()));
    }
    input.SetCardinality(pairs->RowCount());

    auto fake = std::make_shared<ScriptedFakeClient>(
            [](const std::string& pair_id) { return std::stoll(pair_id) % 2 == 0; });
    SemContextColumn cc;
    cc.data_index = 0;
    duckdb::PhysicalPlan plan(duckdb::Allocator::DefaultAllocator());
    auto& op = plan.Make<PhysicalSemFilter>(duckdb::vector<LogicalType>{LogicalType::BIGINT}, 6, fake,
                                            MakeRouter(), MakeParams(/*cap=*/8, /*batch_size=*/1), nullptr,
                                            /*invert=*/false, duckdb::vector<idx_t>{},
                                            std::vector<SemContextColumn>{cc}, "keep even pair ids", "XML",
                                            std::string(), std::string()).Cast<PhysicalSemFilter>();
    auto actual = DriveFilter(op, {&input});
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual, (std::vector<int64_t>{2, 2, 12}));
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
// LIMIT early stop.
// ===========================================================================
std::vector<SemContextColumn> OneContextColumn() {
    SemContextColumn cc;
    cc.data_index = 0;
    return {cc};
}

// The fake client completes inline, so after each chunk every sent row has come
// back. All rows pass, chunks of 8, batch 1: after chunk 1 there are 8 rows
// (< 10), after chunk 2 16 (>= 10) -> chunk 3 returns FINISHED; 16 requests, not 200.
TEST(SemFilter, Engine_LimitStopsEarly) {
    auto fake = std::make_shared<ScriptedFakeClient>([](const std::string&) { return true; });
    duckdb::PhysicalPlan plan(duckdb::Allocator::DefaultAllocator());
    auto& op = plan.Make<PhysicalSemFilter>(duckdb::vector<LogicalType>{LogicalType::BIGINT}, 0, fake,
                                            MakeRouter(), MakeParams(/*cap=*/64, /*batch_size=*/1), nullptr,
                                            /*invert=*/false, duckdb::vector<idx_t>{}, OneContextColumn(), "p",
                                            "XML", std::string(), std::string())
                       .Cast<PhysicalSemFilter>();
    auto chunks = BuildBigintChunks(200, 8);
    std::vector<DataChunk*> ptrs;
    for (auto& c : chunks) ptrs.push_back(c.get());
    size_t sunk = 0;
    auto out = DriveFilter(
            op, ptrs,
            [&](SemGlobalSinkState& g) {
                g.limit_rows = 10;
                g.limit_emits = [](const nlohmann::json& e) { return PhysicalSemFilter::ParseVerdict(e); };
            },
            &sunk);
    EXPECT_EQ(sunk, 2u);
    EXPECT_EQ(fake->Submitted(), 16u);
    EXPECT_EQ(out.size(), 16u);  // >= LIMIT; the LIMIT operator trims the rest
}

// Cap saturated (the GPU case): cap 4 blocks the sink mid-chunk, and the resume
// after each completion re-checks the count. Chunk 1: 4 sent, block, 4 more
// (8 rows). Chunk 2: 4 sent (12 rows), block; the resume sees 12 >= 10 ->
// FINISHED in the middle of chunk 2; its other 4 rows are never sent.
TEST(SemFilter, Engine_LimitStopsOnBlockedResume) {
    auto fake = std::make_shared<ScriptedFakeClient>([](const std::string&) { return true; });
    duckdb::PhysicalPlan plan(duckdb::Allocator::DefaultAllocator());
    auto& op = plan.Make<PhysicalSemFilter>(duckdb::vector<LogicalType>{LogicalType::BIGINT}, 0, fake,
                                            MakeRouter(), MakeParams(/*cap=*/4, /*batch_size=*/1), nullptr,
                                            false, duckdb::vector<idx_t>{}, OneContextColumn(), "p", "XML",
                                            std::string(), std::string())
                       .Cast<PhysicalSemFilter>();
    auto chunks = BuildBigintChunks(200, 8);
    std::vector<DataChunk*> ptrs;
    for (auto& c : chunks) ptrs.push_back(c.get());
    size_t sunk = 0;
    auto out = DriveFilter(
            op, ptrs,
            [&](SemGlobalSinkState& g) {
                g.limit_rows = 10;
                g.limit_emits = [](const nlohmann::json& e) { return PhysicalSemFilter::ParseVerdict(e); };
            },
            &sunk);
    EXPECT_EQ(sunk, 1u);
    EXPECT_EQ(fake->Submitted(), 12u);
    EXPECT_EQ(out.size(), 12u);
}

// Rows that fail the verdict do not count: only even values pass, so 10 output
// rows need 20 input rows (chunks 1-3 = 24 rows, 12 passing) -> chunk 4 stops.
TEST(SemFilter, Engine_LimitCountsOnlyEmittedRows) {
    auto fake = std::make_shared<ScriptedFakeClient>(
            [](const std::string& v) { return std::stoll(v) % 2 == 0; });
    duckdb::PhysicalPlan plan(duckdb::Allocator::DefaultAllocator());
    auto& op = plan.Make<PhysicalSemFilter>(duckdb::vector<LogicalType>{LogicalType::BIGINT}, 0, fake,
                                            MakeRouter(), MakeParams(64, 2), nullptr, /*invert=*/true,
                                            duckdb::vector<idx_t>{}, OneContextColumn(), "p", "XML",
                                            std::string(), std::string())
                       .Cast<PhysicalSemFilter>();
    auto chunks = BuildBigintChunks(200, 8);
    std::vector<DataChunk*> ptrs;
    for (auto& c : chunks) ptrs.push_back(c.get());
    size_t sunk = 0;
    auto out = DriveFilter(
            op, ptrs,
            [&](SemGlobalSinkState& g) {
                g.limit_rows = 10;
                // invert: an element yields a row iff its verdict is FALSE (odd values)
                g.limit_emits = [](const nlohmann::json& e) { return !PhysicalSemFilter::ParseVerdict(e); };
            },
            &sunk);
    EXPECT_EQ(sunk, 3u);
    EXPECT_EQ(fake->Submitted(), 12u);  // 24 rows / batch 2
    ASSERT_EQ(out.size(), 12u);
    for (auto v : out) EXPECT_EQ(v % 2, 1);
}

// Not enough matching rows: no early stop, every row is sent (same as no LIMIT).
TEST(SemFilter, Engine_LimitNotReachedSendsEverything) {
    auto fake = std::make_shared<ScriptedFakeClient>([](const std::string& v) { return std::stoll(v) < 3; });
    duckdb::PhysicalPlan plan(duckdb::Allocator::DefaultAllocator());
    auto& op = plan.Make<PhysicalSemFilter>(duckdb::vector<LogicalType>{LogicalType::BIGINT}, 0, fake,
                                            MakeRouter(), MakeParams(4, 1), nullptr, false,
                                            duckdb::vector<idx_t>{}, OneContextColumn(), "p", "XML",
                                            std::string(), std::string())
                       .Cast<PhysicalSemFilter>();
    auto chunks = BuildBigintChunks(40, 8);
    std::vector<DataChunk*> ptrs;
    for (auto& c : chunks) ptrs.push_back(c.get());
    auto out = DriveFilter(op, ptrs, [&](SemGlobalSinkState& g) {
        g.limit_rows = 5;
        g.limit_emits = [](const nlohmann::json& e) { return PhysicalSemFilter::ParseVerdict(e); };
    });
    EXPECT_EQ(fake->Submitted(), 40u);
    std::sort(out.begin(), out.end());
    EXPECT_EQ(out, (std::vector<int64_t>{0, 1, 2}));
}

// Wiring: the operator's hint reaches the engine, and a proxy disables it.
TEST(SemFilter, LimitHintWiring) {
    duckdb::Connection con(*g_db);
    auto fake = std::make_shared<ScriptedFakeClient>([](const std::string&) { return true; });
    duckdb::PhysicalPlan plan(duckdb::Allocator::DefaultAllocator());
    auto& plain = plan.Make<PhysicalSemFilter>(duckdb::vector<LogicalType>{LogicalType::BIGINT}, 0, fake,
                                               MakeRouter(), MakeParams(4, 1), nullptr, /*invert=*/true,
                                               duckdb::vector<idx_t>{}, OneContextColumn(), "p", "XML",
                                               std::string(), std::string())
                          .Cast<PhysicalSemFilter>();
    plain.SetLimitHint(7);
    auto g1 = plain.GetGlobalSinkState(*con.context);
    auto& s1 = g1->Cast<SemGlobalSinkState>();
    EXPECT_EQ(s1.limit_rows, 7u);
    ASSERT_TRUE(static_cast<bool>(s1.limit_emits));
    EXPECT_TRUE(s1.limit_emits(nlohmann::json(false)));  // invert: FALSE verdict -> row
    EXPECT_FALSE(s1.limit_emits(nlohmann::json(true)));

    auto& proxied = plan.Make<PhysicalSemFilter>(duckdb::vector<LogicalType>{LogicalType::BIGINT}, 0, fake,
                                                 MakeRouter(), MakeParams(4, 1), nullptr, false,
                                                 duckdb::vector<idx_t>{}, OneContextColumn(), "p", "XML",
                                                 std::string("main"), std::string(), std::string("proxy"))
                            .Cast<PhysicalSemFilter>();
    proxied.SetLimitHint(7);
    auto g2 = proxied.GetGlobalSinkState(*con.context);
    EXPECT_EQ(g2->Cast<SemGlobalSinkState>().limit_rows, 0u);
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
            while (duckdb::BoundCastExpression::IsCast(*llm_call)) {
                auto inner = std::move(
                        duckdb::BoundCastExpression::ChildMutable(llm_call->Cast<duckdb::BoundFunctionExpression>()));
                llm_call = std::move(inner);
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
    duckdb::Parser parser(*con.context);
    parser.ParseQuery(query);
    auto statements = std::move(parser.statements);
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
