// End-to-end correctness + A/B-parity test for the llm_filter -> PhysicalSemFilter
// chain, driven against a real in-memory DuckDB with flock loaded. No GPU / no
// vLLM: a content-deterministic mock HTTP server stands in for the endpoint, so
// the whole thing is committable and CI-able.
//
// Two transports, one mock. The SAME query produces the SAME per-row verdicts on
// both code paths because the mock decides each verdict from a marker embedded in
// the row text, independent of prompt formatting or batch grouping:
//   * operator path (semantic_rewrite_enabled=true): PhysicalSemFilter -> the
//     ExtensionState AsyncLLMClient -> POST semantic_endpoints (/v1/completions).
//     Body carries a single multi-row "prompt"; response is choices[0].text.
//   * scalar path (semantic_rewrite_enabled=false): the VOLATILE llm_filter scalar
//     -> OpenAIProvider -> POST the openai secret's base_url + /chat/completions.
//     Body carries "messages"; response is choices[0].message.content.
// Both responses wrap a stringified {"items":[bool,...]} (flock's shared batch
// envelope), so a single marker scan serves both shapes.
//
// Markers are the distinctive tokens __KEEPROW__ / __DROPROW__ (never produced by
// a prompt template or JSON escaping), scanned left-to-right == row order.

#include "flock/core/config.hpp"
#include "flock/functions/operator/semantic_filter.hpp"

#include "duckdb.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/planner/logical_operator.hpp"

#include "mock_vllm_server.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace flock {
namespace {

duckdb::DuckDB* g_db = nullptr;

// ---------------------------------------------------------------------------
// Mock endpoint: content-deterministic verdicts from in-text markers.
// ---------------------------------------------------------------------------

// Walk the text left-to-right, emitting one verdict per marker in the order it
// appears (== row order in the rendered batch prompt): __KEEPROW__ -> true,
// __DROPROW__ -> false. The count equals the batch's row count (one marker per
// row), which is exactly what both paths expect back.
std::vector<bool> ScanVerdicts(const std::string& text) {
    static const std::string kKeep = "__KEEPROW__";
    static const std::string kDrop = "__DROPROW__";
    std::vector<bool> out;
    size_t pos = 0;
    while (pos <= text.size()) {
        const size_t k = text.find(kKeep, pos);
        const size_t d = text.find(kDrop, pos);
        if (k == std::string::npos && d == std::string::npos) {
            break;
        }
        if (d == std::string::npos || (k != std::string::npos && k < d)) {
            out.push_back(true);
            pos = k + kKeep.size();
        } else {
            out.push_back(false);
            pos = d + kDrop.size();
        }
    }
    return out;
}

MockVLLMServer::Response MarkerHandler(const MockVLLMServer::Request& req) {
    MockVLLMServer::Response resp;
    resp.status = 200;
    resp.headers["Content-Type"] = "application/json";

    const nlohmann::json body = nlohmann::json::parse(req.body, /*cb=*/nullptr, /*allow_exceptions=*/false);

    // Pick the text to scan and the response shape from the request body:
    // "messages" => chat/completions (scalar path); "prompt" => completions
    // (operator path). Fall back to scanning the raw body if neither is present.
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
    } else if (!body.is_discarded() && body.contains("prompt") && body["prompt"].is_string()) {
        scan_text = body["prompt"].get<std::string>();
    } else {
        scan_text = req.body;
    }

    nlohmann::json items = nlohmann::json::array();
    for (const bool v : ScanVerdicts(scan_text)) {
        items.push_back(v);
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

// ---------------------------------------------------------------------------
// Plan / result helpers.
// ---------------------------------------------------------------------------

void CollectSemFilters(duckdb::LogicalOperator& op, std::vector<LogicalSemFilter*>& out) {
    if (auto* sf = dynamic_cast<LogicalSemFilter*>(&op)) {
        out.push_back(sf);
    }
    for (auto& child : op.children) {
        CollectSemFilters(*child, out);
    }
}

// True iff the POST-OPTIMIZER plan (incl. flock's OptimizerExtension) contains a
// LogicalSemFilter -- i.e. the rewrite engaged. Honors this session's
// semantic_rewrite_enabled.
bool PlanHasSemFilter(duckdb::Connection& con, const std::string& query) {
    auto plan = con.ExtractPlan(query);
    std::vector<LogicalSemFilter*> filters;
    CollectSemFilters(*plan, filters);
    return !filters.empty();
}

// Execute and return the (sorted) `id` column. The operator emits in completion
// order, so callers compare as a set.
std::vector<long long> RunIds(duckdb::Connection& con, const std::string& query) {
    auto res = con.Query(query);
    EXPECT_FALSE(res->HasError()) << query << " -> " << res->GetError();
    std::vector<long long> ids;
    if (res->HasError()) {
        return ids;
    }
    for (duckdb::idx_t r = 0; r < res->RowCount(); ++r) {
        ids.push_back(std::stoll(res->GetValue(0, r).ToString()));
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

void ConfigureSession(duckdb::Connection& con, bool rewrite_on) {
    con.Query(std::string("SET semantic_rewrite_enabled=") + (rewrite_on ? "true" : "false") + ";");
    // Small values so the run exercises real multi-batch coalescing + the in-flight
    // cap (6 rows / batch 4 -> a full batch of 4 and a partial of 2).
    con.Query("SET semantic_batch_size=4;");
    con.Query("SET semantic_in_flight_cap=2;");
}

// Neutral prompt -- intentionally free of the KEEP/DROP markers so only the row
// text drives the verdicts.
const char* kPrompt = "Classify each provided review for relevance.";

std::string QueryMultiCol() {
    return "SELECT id FROM reviews WHERE llm_filter({'model_name': 'gpt-4o'}, "
           "{'prompt': '" + std::string(kPrompt) +
           "', 'context_columns': [{'data': review}, {'data': note}]})";
}

std::string QuerySingleCol() {
    return "SELECT id FROM reviews WHERE llm_filter({'model_name': 'gpt-4o'}, "
           "{'prompt': '" + std::string(kPrompt) + "', 'context_columns': [{'data': review}]})";
}

std::string QueryOrFallback() {
    return "SELECT id FROM reviews WHERE x > 5 OR llm_filter({'model_name': 'gpt-4o'}, "
           "{'prompt': '" + std::string(kPrompt) + "', 'context_columns': [{'data': review}]})";
}

// __KEEPROW__ rows -> kept; data designed so KEEP set and the OR fallback set are
// both known up front.
const std::vector<long long> kKeepSet = {1, 3, 5};
const std::vector<long long> kOrSet = {1, 3, 4, 5, 6};  // (x>5: 4,6) UNION (kept: 1,3,5)

// ---------------------------------------------------------------------------
// 1. Operator engaged (rewrite on) is correct.
// ---------------------------------------------------------------------------
TEST(SemFilterE2E, OperatorEngagedReturnsKeepSet) {
    duckdb::Connection con(*g_db);
    ConfigureSession(con, /*rewrite_on=*/true);
    EXPECT_TRUE(PlanHasSemFilter(con, QueryMultiCol()));
    EXPECT_EQ(RunIds(con, QueryMultiCol()), kKeepSet);
}

// ---------------------------------------------------------------------------
// 2. Scalar (rewrite off) is correct and the operator is absent.
// ---------------------------------------------------------------------------
TEST(SemFilterE2E, ScalarReturnsKeepSet) {
    duckdb::Connection con(*g_db);
    ConfigureSession(con, /*rewrite_on=*/false);
    EXPECT_FALSE(PlanHasSemFilter(con, QueryMultiCol()));
    EXPECT_EQ(RunIds(con, QueryMultiCol()), kKeepSet);
}

// ---------------------------------------------------------------------------
// 3. A/B parity: operator and scalar return identical sets (== expected KEEP).
// ---------------------------------------------------------------------------
TEST(SemFilterE2E, ABParityIdenticalResults) {
    duckdb::Connection on(*g_db);
    ConfigureSession(on, /*rewrite_on=*/true);
    duckdb::Connection off(*g_db);
    ConfigureSession(off, /*rewrite_on=*/false);

    const auto with_operator = RunIds(on, QueryMultiCol());
    const auto with_scalar = RunIds(off, QueryMultiCol());
    EXPECT_EQ(with_operator, with_scalar);
    EXPECT_EQ(with_operator, kKeepSet);
}

// ---------------------------------------------------------------------------
// 4. VOLATILE single-column engagement: a single-column WHERE llm_filter still
//    leaves a LogicalFilter for the rewrite (the VOLATILE / no-pushdown fix), so
//    the operator engages end-to-end and returns the KEEP set.
// ---------------------------------------------------------------------------
TEST(SemFilterE2E, VolatileSingleColumnEngages) {
    duckdb::Connection con(*g_db);
    ConfigureSession(con, /*rewrite_on=*/true);
    EXPECT_TRUE(PlanHasSemFilter(con, QuerySingleCol()));
    EXPECT_EQ(RunIds(con, QuerySingleCol()), kKeepSet);
}

// ---------------------------------------------------------------------------
// 5. Fallback stays scalar: a top-level OR is not a rewritable conjunct, so even
//    with the rewrite on the plan has no LogicalSemFilter and the (correct) rows
//    come from the scalar path.
// ---------------------------------------------------------------------------
TEST(SemFilterE2E, FallbackOrStaysScalar) {
    duckdb::Connection con(*g_db);
    ConfigureSession(con, /*rewrite_on=*/true);
    EXPECT_FALSE(PlanHasSemFilter(con, QueryOrFallback()));
    EXPECT_EQ(RunIds(con, QueryOrFallback()), kOrSet);
}

}  // namespace
}  // namespace flock

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    flock::MockVLLMServer server;
    server.Start(flock::MarkerHandler);

    int rc = 0;
    {
        // The DuckDB instance owns the shared AsyncLLMClient (via ExtensionState),
        // whose CURL keep-alive connections to the mock stay open until the client
        // is destroyed. So the DB must be torn down BEFORE server.Stop(); else
        // the mock's handler threads are still blocked reading those live sockets
        // and Stop()'s thread join hangs. (Same client-before-server order as
        // async_llm_client_test.)
        duckdb::DuckDB db(nullptr);  // in-memory; flock auto-loads on construction
        flock::g_db = &db;
        flock::Config::GetConnection(&*db.instance);

        duckdb::Connection setup(db);
        // Scalar path: the default openai secret's base_url -> the mock; the
        // handler appends /chat/completions.
        setup.Query("CREATE SECRET (TYPE OPENAI, API_KEY 'test-key', BASE_URL '" + server.Url("/v1") + "');");
        // Operator path: the shared EndpointRouter posts here (GLOBAL; rebuilds the
        // ExtensionState router via the SET callback).
        setup.Query("SET semantic_endpoints='" + server.Url("/v1/completions") + "';");

        setup.Query("CREATE TABLE reviews(id INTEGER, review VARCHAR, note VARCHAR, x INTEGER);");
        setup.Query(
                "INSERT INTO reviews VALUES "
                "(1, 'movie __KEEPROW__ alpha', 'na', 1), "
                "(2, 'movie __DROPROW__ bravo', 'nb', 2), "
                "(3, 'movie __KEEPROW__ charlie', 'nc', 3), "
                "(4, 'movie __DROPROW__ delta', 'nd', 8), "
                "(5, 'movie __KEEPROW__ echo', 'ne', 4), "
                "(6, 'movie __DROPROW__ foxtrot', 'nf', 9);");

        rc = RUN_ALL_TESTS();
        flock::g_db = nullptr;
    }  // db (and the AsyncLLMClient it owns) destroyed here -> mock sockets closed

    server.Stop();
    return rc;
}
