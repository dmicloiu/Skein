#include "flock/runtime/semantic_settings.h"

#include "duckdb.hpp"
#include "duckdb/common/vector_size.hpp"
#include "flock/custom_parser/query/model_parser.hpp"
#include "flock/runtime/endpoint_router.h"
#include "flock/runtime/extension_state.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>

namespace flock {

// Provided by semantic_test_main.cpp.
duckdb::DuckDB& TestDB();

namespace {

// Run SQL and fail the calling test if it errors. Returns the result so callers
// can inspect rows.
std::unique_ptr<duckdb::MaterializedQueryResult> RunSQL(duckdb::Connection& con, const std::string& sql) {
    auto result = con.Query(sql);
    EXPECT_FALSE(result->HasError()) << sql << " -> " << result->GetError();
    return result;
}

int64_t ReadInt(duckdb::ClientContext& ctx, const char* name) {
    duckdb::Value v;
    auto found = ctx.TryGetCurrentSetting(name, v);  // operator bool is non-const
    EXPECT_TRUE(static_cast<bool>(found)) << name;
    return v.GetValue<int64_t>();
}

}  // namespace

/**************************************************
 *                 SET surface                    *
 **************************************************/

// SET semantic_endpoints splits on ',' and mirrors into the live router.
TEST(SemanticSettingsTest, EndpointsSplitIntoRouter) {
    duckdb::Connection con(TestDB());
    RunSQL(con, "SET semantic_endpoints='a,b,c'");
    EXPECT_EQ(ExtensionState::Get(*con.context).router->EndpointCount(), 3u);

    // Whitespace around entries is trimmed.
    RunSQL(con, "SET semantic_endpoints=' x , y '");
    EXPECT_EQ(ExtensionState::Get(*con.context).router->EndpointCount(), 2u);
}

// SET semantic_routing='single' makes the router pick index 0 for any key.
TEST(SemanticSettingsTest, RoutingSingleAlwaysIndexZero) {
    duckdb::Connection con(TestDB());
    RunSQL(con, "SET semantic_endpoints='a,b,c'");
    RunSQL(con, "SET semantic_routing='single'");
    auto& router = *ExtensionState::Get(*con.context).router;
    for (const auto& key : {"", "k", "another-key"}) {
        auto pick = router.Choose(key);
        EXPECT_EQ(pick.index, 0u);
        router.OnComplete(pick.index);
    }
}

// An unknown routing strategy fails the SET (ParseStrategy throws).
TEST(SemanticSettingsTest, RoutingBogusErrors) {
    duckdb::Connection con(TestDB());
    auto result = con.Query("SET semantic_routing='bogus'");
    EXPECT_TRUE(result->HasError());
}

// An invalid response format fails the SET.
TEST(SemanticSettingsTest, ResponseFormatInvalidErrors) {
    duckdb::Connection con(TestDB());
    EXPECT_FALSE(con.Query("SET semantic_response_format='json_schema'")->HasError());
    EXPECT_FALSE(con.Query("SET semantic_response_format='free_form'")->HasError());
    EXPECT_TRUE(con.Query("SET semantic_response_format='xml'")->HasError());
}

// The calibration knobs round-trip through TryGetCurrentSetting.
TEST(SemanticSettingsTest, CalibrationKnobsReadBack) {
    duckdb::Connection con(TestDB());
    RunSQL(con, "SET semantic_in_flight_cap=64");
    RunSQL(con, "SET semantic_batch_size=16");
    RunSQL(con, "SET semantic_coalesce_max_age_ms=999");
    RunSQL(con, "SET semantic_max_output_tokens=8");
    EXPECT_EQ(ReadInt(*con.context, semantic_option::kInFlightCap), 64);
    EXPECT_EQ(ReadInt(*con.context, semantic_option::kBatchSize), 16);
    EXPECT_EQ(ReadInt(*con.context, semantic_option::kCoalesceMaxAgeMs), 999);
    EXPECT_EQ(ReadInt(*con.context, semantic_option::kMaxOutputTokens), 8);
}

// The relocated guardrail (cap>128 && batch_size>64) warns but does NOT error:
// the high SETs succeed, and ResolveSemanticParams returns normally (the warning
// is emitted to stderr) on the resolved high values.
TEST(SemanticSettingsTest, GuardrailWarnsButSucceeds) {
    duckdb::Connection con(TestDB());
    EXPECT_FALSE(con.Query("SET semantic_in_flight_cap=256")->HasError());
    EXPECT_FALSE(con.Query("SET semantic_batch_size=128")->HasError());
    EXPECT_EQ(ReadInt(*con.context, semantic_option::kInFlightCap), 256);
    EXPECT_EQ(ReadInt(*con.context, semantic_option::kBatchSize), 128);

    SemanticParams params;
    EXPECT_NO_THROW(params = ResolveSemanticParams(*con.context, "no_such_model_guardrail"));
    EXPECT_EQ(params.in_flight_cap, 256u);
    EXPECT_EQ(params.batch_size, 128u);
}

// ResolveSemanticParams is the authoritative range chokepoint: it throws on a
// batch_size outside [1, STANDARD_VECTOR_SIZE] or an in_flight_cap < 1 (both
// would hang/overflow the engine), reached via the SET surface here.
TEST(SemanticSettingsTest, ResolveRejectsOutOfRange) {
    duckdb::Connection con(TestDB());

    RunSQL(con, "SET semantic_batch_size=0");
    EXPECT_THROW(ResolveSemanticParams(*con.context, "no_such_model_bs0"), std::exception);

    RunSQL(con, "SET semantic_batch_size=" + std::to_string(STANDARD_VECTOR_SIZE + 1));
    EXPECT_THROW(ResolveSemanticParams(*con.context, "no_such_model_bsbig"), std::exception);

    RunSQL(con, "SET semantic_batch_size=32");  // back in range
    RunSQL(con, "SET semantic_in_flight_cap=0");
    EXPECT_THROW(ResolveSemanticParams(*con.context, "no_such_model_cap0"), std::exception);
}

// In-range SET values and the bare defaults (32 / 128) resolve without throwing.
TEST(SemanticSettingsTest, ResolveAcceptsInRangeAndDefaults) {
    duckdb::Connection con(TestDB());
    RunSQL(con, "SET semantic_batch_size=" + std::to_string(STANDARD_VECTOR_SIZE));  // boundary, valid
    RunSQL(con, "SET semantic_in_flight_cap=1");
    EXPECT_NO_THROW(ResolveSemanticParams(*con.context, "no_such_model_inrange"));

    // Fresh session -> pure defaults, which are in range.
    duckdb::Connection fresh(TestDB());
    EXPECT_NO_THROW(ResolveSemanticParams(*fresh.context, "no_such_model_defaults"));
}

/**************************************************
 *        CREATE/UPDATE MODEL parser surface      *
 **************************************************/

// The five semantic keys parse and land in model_args (create path).
TEST(SemanticModelParserTest, CreateAcceptsSemanticKeys) {
    std::unique_ptr<QueryStatement> statement;
    ModelParser parser;
    EXPECT_NO_THROW(parser.Parse(
            "CREATE MODEL ('m', 'model', 'openai', {\"in_flight_cap\": 8, \"batch_size\": 4, "
            "\"coalesce_max_age_ms\": 250, \"max_output_tokens\": 32, \"response_format\": \"free_form\"})",
            statement));
    auto* create = dynamic_cast<CreateModelStatement*>(statement.get());
    ASSERT_NE(create, nullptr);
    EXPECT_EQ(create->model_args["in_flight_cap"], 8);
    EXPECT_EQ(create->model_args["batch_size"], 4);
    EXPECT_EQ(create->model_args["coalesce_max_age_ms"], 250);
    EXPECT_EQ(create->model_args["max_output_tokens"], 32);
    EXPECT_EQ(create->model_args["response_format"], "free_form");
}

// The update path accepts them too.
TEST(SemanticModelParserTest, UpdateAcceptsSemanticKeys) {
    std::unique_ptr<QueryStatement> statement;
    ModelParser parser;
    EXPECT_NO_THROW(parser.Parse(
            "UPDATE MODEL ('m', 'model', 'openai', {\"in_flight_cap\": 2, \"response_format\": \"json_schema\"})",
            statement));
    auto* update = dynamic_cast<UpdateModelStatement*>(statement.get());
    ASSERT_NE(update, nullptr);
    EXPECT_EQ(update->new_model_args["in_flight_cap"], 2);
    EXPECT_EQ(update->new_model_args["response_format"], "json_schema");
}

// Unknown keys are still rejected.
TEST(SemanticModelParserTest, UnknownKeyRejected) {
    std::unique_ptr<QueryStatement> statement;
    ModelParser parser;
    EXPECT_THROW(parser.Parse("CREATE MODEL ('m', 'model', 'openai', {\"bogus_key\": 1})", statement),
                 std::runtime_error);
}

// Type/value mismatches throw: non-positive integer, non-integer, bad enum.
TEST(SemanticModelParserTest, BadTypesRejected) {
    ModelParser parser;
    std::unique_ptr<QueryStatement> statement;
    EXPECT_THROW(parser.Parse("CREATE MODEL ('m', 'model', 'openai', {\"in_flight_cap\": 0})", statement),
                 std::runtime_error);
    EXPECT_THROW(parser.Parse("CREATE MODEL ('m', 'model', 'openai', {\"batch_size\": \"x\"})", statement),
                 std::runtime_error);
    EXPECT_THROW(parser.Parse("CREATE MODEL ('m', 'model', 'openai', {\"max_output_tokens\": -4})", statement),
                 std::runtime_error);
    EXPECT_THROW(parser.Parse("CREATE MODEL ('m', 'model', 'openai', {\"response_format\": \"yaml\"})", statement),
                 std::runtime_error);
}

// batch_size must be a positive integer at parse time on BOTH paths, and the
// removed coalesce_size key is now rejected as unknown on UPDATE (matching CREATE).
TEST(SemanticModelParserTest, BatchSizeAndCoalesceSizeRejected) {
    ModelParser parser;
    std::unique_ptr<QueryStatement> statement;
    EXPECT_THROW(parser.Parse("CREATE MODEL ('m', 'model', 'openai', {\"batch_size\": 0})", statement),
                 std::runtime_error);
    EXPECT_THROW(parser.Parse("UPDATE MODEL ('m', 'model', 'openai', {\"batch_size\": 0})", statement),
                 std::runtime_error);
    EXPECT_THROW(parser.Parse("CREATE MODEL ('m', 'model', 'openai', {\"coalesce_size\": 16})", statement),
                 std::runtime_error);
    EXPECT_THROW(parser.Parse("UPDATE MODEL ('m', 'model', 'openai', {\"coalesce_size\": 16})", statement),
                 std::runtime_error);
}

/**************************************************
 *   CREATE MODEL persistence + ResolveSemanticParams
 **************************************************/

// A created model persists its semantic keys and they show up in GET MODEL.
TEST(SemanticModelTest, CreatePersistsAndGetModelShows) {
    duckdb::Connection con(TestDB());
    con.Query("DELETE MODEL 'semantic_persist_test'");  // best-effort cleanup
    RunSQL(con,
        "CREATE MODEL ('semantic_persist_test', 'some-model', 'openai', "
        "{\"in_flight_cap\": 11, \"response_format\": \"free_form\"})");

    auto result = RunSQL(con, "GET MODEL 'semantic_persist_test'");
    ASSERT_GE(result->RowCount(), 1u);
    const std::string dump = result->ToString();
    EXPECT_NE(dump.find("in_flight_cap"), std::string::npos);
    EXPECT_NE(dump.find("free_form"), std::string::npos);

    con.Query("DELETE MODEL 'semantic_persist_test'");
}

// Precedence: model_args override wins over SET; SET wins where the model omits;
// SemanticDefaults when neither sets a field. batch_size now follows the same
// two-check chain as in_flight_cap (model_args -> SET -> default).
TEST(SemanticModelTest, ResolvePrecedence) {
    duckdb::Connection con(TestDB());
    con.Query("DELETE MODEL 'semantic_resolve_test'");
    RunSQL(con, "CREATE MODEL ('semantic_resolve_test', 'some-model', 'openai', {\"in_flight_cap\": 7})");

    // Session SET that the model does NOT override.
    RunSQL(con, "SET semantic_batch_size=50");

    auto params = ResolveSemanticParams(*con.context, "semantic_resolve_test");
    EXPECT_EQ(params.in_flight_cap, 7u);  // model_args override
    EXPECT_EQ(params.batch_size, 50u);    // SET value (model omits it)

    // model_args wins over the SET when the model DOES pin batch_size.
    con.Query("DELETE MODEL 'semantic_resolve_batch'");
    RunSQL(con, "CREATE MODEL ('semantic_resolve_batch', 'some-model', 'openai', {\"batch_size\": 16})");
    auto pinned = ResolveSemanticParams(*con.context, "semantic_resolve_batch");
    EXPECT_EQ(pinned.batch_size, 16u);  // model_args override beats SET=50

    // A model with no overrides at all -> every field is the SET-or-default.
    duckdb::Connection fresh(TestDB());  // fresh session: no SESSION SETs applied
    auto defaults = ResolveSemanticParams(*fresh.context, "no_such_model_xyz");
    EXPECT_EQ(defaults.in_flight_cap, static_cast<uint64_t>(SemanticDefaults::kInFlightCap));
    EXPECT_EQ(defaults.batch_size, static_cast<uint64_t>(SemanticDefaults::kBatchSize));
    EXPECT_EQ(defaults.max_output_tokens, static_cast<uint64_t>(SemanticDefaults::kMaxOutputTokens));
    EXPECT_EQ(defaults.response_format, std::string(SemanticDefaults::kResponseFormat));

    con.Query("DELETE MODEL 'semantic_resolve_test'");
    con.Query("DELETE MODEL 'semantic_resolve_batch'");
}

}  // namespace flock