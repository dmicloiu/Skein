// End-to-end correctness + A/B-parity test for the llm_complete -> PhysicalSemExtract
// chain, driven against a real in-memory DuckDB with flock loaded. No GPU / no vLLM:
// a content-deterministic mock HTTP server stands in for the endpoint.
//
// Two transports, one mock. The SAME query produces the SAME extracted column on
// both code paths because the mock derives each row's completion from a marker
// embedded in the row text ([[DX=<value>]]), independent of prompt formatting or
// batch grouping. Both arms speak chat-completions ("messages" + choices[0].
// message.content wrapping a stringified {"items":[string,...]}):
//   * operator path (semantic_rewrite_enabled=true): PhysicalSemExtract -> the
//     ExtensionState AsyncLLMClient -> POST semantic_endpoints.
//   * scalar path (semantic_rewrite_enabled=false): the VOLATILE llm_complete scalar
//     -> OpenAIProvider -> POST the openai secret's base_url + /chat/completions.
//
// The operator emits in completion order, so results are compared as a set keyed by id.

#include "flock/core/config.hpp"
#include "flock/functions/operator/semantic_extract.hpp"

#include "duckdb.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/planner/logical_operator.hpp"

#include "mock_vllm_server.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace flock {
namespace {

duckdb::DuckDB* g_db = nullptr;

// Walk the text left-to-right, emitting one completion per [[DX=<value>]] marker in
// the order it appears (== row order in the rendered batch prompt). The count equals
// the batch's row count (one marker per row), which is what both paths expect back.
std::vector<std::string> ScanCompletions(const std::string& text) {
    static const std::string kOpen = "[[DX=";
    static const std::string kClose = "]]";
    std::vector<std::string> out;
    size_t pos = 0;
    while ((pos = text.find(kOpen, pos)) != std::string::npos) {
        const size_t start = pos + kOpen.size();
        const size_t end = text.find(kClose, start);
        if (end == std::string::npos) {
            break;
        }
        out.push_back(text.substr(start, end - start));
        pos = end + kClose.size();
    }
    return out;
}

MockVLLMServer::Response MarkerHandler(const MockVLLMServer::Request& req) {
    MockVLLMServer::Response resp;
    resp.status = 200;
    resp.headers["Content-Type"] = "application/json";

    const nlohmann::json body = nlohmann::json::parse(req.body, /*cb=*/nullptr, /*allow_exceptions=*/false);

    // Both arms post chat ("messages"); scan every text part for the row markers.
    bool is_chat = false;
    std::string scan_text;
    if (!body.is_discarded() && body.contains("messages") && body["messages"].is_array()) {
        is_chat = true;
        for (const auto& msg : body["messages"]) {
            if (!msg.contains("content")) {
                continue;
            }
            const auto& content = msg["content"];
            if (content.is_string()) {
                scan_text += content.get<std::string>();
            } else if (content.is_array()) {
                for (const auto& part : content) {
                    if (part.contains("text") && part["text"].is_string()) {
                        scan_text += part["text"].get<std::string>();
                    }
                }
            }
        }
    } else {
        scan_text = req.body;
    }

    nlohmann::json items = nlohmann::json::array();
    for (const auto& dx : ScanCompletions(scan_text)) {
        items.push_back(dx);
    }
    nlohmann::json inner;
    inner["items"] = std::move(items);
    const std::string content = inner.dump();  // the stringified {"items":[...]}

    nlohmann::json choice;
    choice["index"] = 0;
    choice["finish_reason"] = "stop";
    if (is_chat) {
        choice["message"] = {{"role", "assistant"}, {"content", content}};
    } else {
        choice["text"] = content;
    }
    nlohmann::json out;
    out["choices"] = nlohmann::json::array({std::move(choice)});
    out["usage"] = {{"prompt_tokens", 1}, {"completion_tokens", 1}, {"total_tokens", 2}};
    resp.body = out.dump();
    return resp;
}

void CollectSemExtracts(duckdb::LogicalOperator& op, std::vector<LogicalSemExtract*>& out) {
    if (auto* se = dynamic_cast<LogicalSemExtract*>(&op)) {
        out.push_back(se);
    }
    for (auto& child : op.children) {
        CollectSemExtracts(*child, out);
    }
}

// True iff the POST-OPTIMIZER plan (incl. flock's OptimizerExtension) contains a
// LogicalSemExtract -- i.e. the rewrite engaged. Honors semantic_rewrite_enabled.
bool PlanHasSemExtract(duckdb::Connection& con, const std::string& query) {
    auto plan = con.ExtractPlan(query);
    std::vector<LogicalSemExtract*> extracts;
    CollectSemExtracts(*plan, extracts);
    return !extracts.empty();
}

// Execute and return the (id, dx) pairs sorted by id. dx is the JSON column read as
// its raw string (the extracted disease).
std::vector<std::pair<long long, std::string>> RunPairs(duckdb::Connection& con, const std::string& query) {
    auto res = con.Query(query);
    EXPECT_FALSE(res->HasError()) << query << " -> " << res->GetError();
    std::vector<std::pair<long long, std::string>> pairs;
    if (res->HasError()) {
        return pairs;
    }
    for (duckdb::idx_t r = 0; r < res->RowCount(); ++r) {
        pairs.emplace_back(std::stoll(res->GetValue(0, r).ToString()), res->GetValue(1, r).ToString());
    }
    std::sort(pairs.begin(), pairs.end());
    return pairs;
}

void ConfigureSession(duckdb::Connection& con, bool rewrite_on) {
    con.Query(std::string("SET semantic_rewrite_enabled=") + (rewrite_on ? "true" : "false") + ";");
    // Small values so the run exercises multi-batch coalescing + the in-flight cap
    // (6 rows / batch 4 -> a full batch of 4 and a partial of 2).
    con.Query("SET semantic_batch_size=4;");
    con.Query("SET semantic_in_flight_cap=2;");
}

std::string Query() {
    return "SELECT id, llm_complete({'model_name': 'gpt-4o'}, "
           "{'prompt': 'Classify the symptoms to a disease.', "
           "'context_columns': [{'data': symptoms}]}) AS dx FROM cases";
}

// Expected extracted column: the marker value embedded in each row's symptoms.
const std::vector<std::pair<long long, std::string>> kExpected = {
        {1, "ACNE"}, {2, "MALARIA"}, {3, "TYPHOID"}, {4, "MIGRAINE"}, {5, "ALLERGY"}, {6, "DENGUE"}};

// ---------------------------------------------------------------------------
// 1. Operator engaged (rewrite on) is correct.
// ---------------------------------------------------------------------------
TEST(SemExtractE2E, OperatorEngagedReturnsExtractedColumn) {
    duckdb::Connection con(*g_db);
    ConfigureSession(con, /*rewrite_on=*/true);
    EXPECT_TRUE(PlanHasSemExtract(con, Query()));
    EXPECT_EQ(RunPairs(con, Query()), kExpected);
}

// ---------------------------------------------------------------------------
// 2. Scalar (rewrite off) is correct and the operator is absent.
// ---------------------------------------------------------------------------
TEST(SemExtractE2E, ScalarReturnsExtractedColumn) {
    duckdb::Connection con(*g_db);
    ConfigureSession(con, /*rewrite_on=*/false);
    EXPECT_FALSE(PlanHasSemExtract(con, Query()));
    EXPECT_EQ(RunPairs(con, Query()), kExpected);
}

// ---------------------------------------------------------------------------
// 2b. Production shape: 2000 rows at batch_size=8 / cap 128.
//
// The 6-row cases above never leave one DataChunk and never fill the in-flight
// window. Cross-system Q103 diverges from the model verdicts only at R>=4 (R=1/2
// agree), so the suspect is scale: multi-batch coalescing, a full in-flight
// window, and rows spanning DuckDB's 2048-row chunk boundary. Each row carries a
// unique marker, so any pairing slip between a row and its completion shows up as
// a mismatched (id, dx) pair rather than as an aggregate quality drop.
// ---------------------------------------------------------------------------
TEST(SemExtractE2E, ProductionShapePairingHolds) {
    duckdb::Connection con(*g_db);
    con.Query("SET semantic_rewrite_enabled=true;");
    con.Query("SET semantic_batch_size=8;");
    con.Query("SET semantic_in_flight_cap=128;");

    const std::string q =
            "SELECT id, llm_complete({'model_name': 'gpt-4o'}, "
            "{'prompt': 'Classify the symptoms to a disease.', "
            "'context_columns': [{'data': symptoms}]}) AS dx FROM wide_cases";
    ASSERT_TRUE(PlanHasSemExtract(con, q));

    const auto pairs = RunPairs(con, q);
    ASSERT_EQ(pairs.size(), 2000u);
    std::vector<std::pair<long long, std::string>> expected;
    expected.reserve(2000);
    for (long long i = 1; i <= 2000; ++i) {
        expected.emplace_back(i, "D" + std::to_string(i));
    }
    // Report the first few slips rather than a 2000-pair diff.
    size_t mismatches = 0;
    for (size_t i = 0; i < pairs.size(); ++i) {
        if (pairs[i] != expected[i]) {
            if (++mismatches <= 5) {
                ADD_FAILURE() << "row " << pairs[i].first << ": got dx=" << pairs[i].second << ", want "
                              << expected[i].second;
            }
        }
    }
    EXPECT_EQ(mismatches, 0u) << mismatches << " of 2000 rows carry another row's completion";
}

// ---------------------------------------------------------------------------
// 3. A/B parity: operator and scalar return identical columns (== expected).
// ---------------------------------------------------------------------------
TEST(SemExtractE2E, ABParityIdenticalResults) {
    duckdb::Connection on(*g_db);
    ConfigureSession(on, /*rewrite_on=*/true);
    duckdb::Connection off(*g_db);
    ConfigureSession(off, /*rewrite_on=*/false);

    const auto with_operator = RunPairs(on, Query());
    const auto with_scalar = RunPairs(off, Query());
    EXPECT_EQ(with_operator, with_scalar);
    EXPECT_EQ(with_operator, kExpected);
}

}  // namespace
}  // namespace flock

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    flock::MockVLLMServer server;
    server.Start(flock::MarkerHandler);

    int rc = 0;
    {
        // The DuckDB instance owns the shared AsyncLLMClient (via ExtensionState); its
        // CURL keep-alive connections to the mock stay open until the client dies, so
        // the DB must be torn down BEFORE server.Stop() (else Stop()'s join hangs on
        // handler threads still reading live sockets). Same order as the filter e2e.
        duckdb::DuckDB db(nullptr);  // in-memory; flock auto-loads on construction
        flock::g_db = &db;
        flock::Config::GetConnection(&*db.instance);

        duckdb::Connection setup(db);
        // Scalar path: the openai secret's base_url -> the mock (handler appends
        // /chat/completions).
        setup.Query("CREATE SECRET (TYPE OPENAI, API_KEY 'test-key', BASE_URL '" + server.Url("/v1") + "');");
        // Operator path: the shared EndpointRouter posts here.
        setup.Query("SET semantic_endpoints='" + server.Url("/v1/completions") + "';");

        setup.Query("CREATE TABLE cases(id INTEGER, symptoms VARCHAR);");
        setup.Query(
                "INSERT INTO cases VALUES "
                "(1, 'patient reports [[DX=ACNE]] and skin issues'), "
                "(2, 'patient reports [[DX=MALARIA]] and fever'), "
                "(3, 'patient reports [[DX=TYPHOID]] and fatigue'), "
                "(4, 'patient reports [[DX=MIGRAINE]] and headache'), "
                "(5, 'patient reports [[DX=ALLERGY]] and sneezing'), "
                "(6, 'patient reports [[DX=DENGUE]] and joint pain');");

        // Production-shape table: 2000 rows, each with a unique marker, so a
        // row->completion slip is visible per row (see ProductionShapePairingHolds).
        setup.Query("CREATE TABLE wide_cases AS SELECT i AS id, "
                    "'patient reports [[DX=D' || i::VARCHAR || ']] and symptoms' AS symptoms "
                    "FROM range(1, 2001) t(i);");

        rc = RUN_ALL_TESTS();
        flock::g_db = nullptr;
    }  // db (and the AsyncLLMClient it owns) destroyed here -> mock sockets closed

    server.Stop();
    return rc;
}
