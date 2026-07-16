// Regression tests for the llm_filter -> LogicalSemFilter optimizer rewrite.
//
// Each test obtains the POST-OPTIMIZER logical plan via Connection::ExtractPlan
// (which runs DuckDB's optimizers AND the registered OptimizerExtension), then
// walks it. We never execute a query (only inspect the plan), so no vLLM is
// needed; a mock provider + secret are enough for llm_filter to bind.
//
// LogicalSemFilter nodes are identified by dynamic_cast (GetExtensionName is
// intentionally left throwing on the operator, so it cannot be used here).
//
// Pushdown note: a single-column llm_filter over a base table is absorbed into
// the LogicalGet as a generic ExpressionFilter, leaving no LogicalFilter for the
// rewrite to see. To keep a LogicalFilter alive deterministically, every test
// llm_filter references TWO columns (multi-column expressions are not pushed into
// the scan), and residuals that must survive use two-column predicates likewise.

#include "flock/core/config.hpp"
#include "flock/functions/operator/semantic_extract.hpp"
#include "flock/functions/operator/semantic_filter.hpp"
#include "flock/model_manager/model.hpp"

#include "duckdb.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"

#include "../../unit/functions/mock_provider.hpp"

#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace flock {
namespace {

using duckdb::LogicalOperator;
using duckdb::LogicalOperatorType;

duckdb::DuckDB* g_db = nullptr;

// A multi-column llm_filter call (references two columns -> survives filter
// pushdown so a LogicalFilter is left for the rewrite). `prompt` distinguishes
// otherwise-identical calls (so two of them are not collapsed by CSE).
std::string LlmFilter(const std::string& c1, const std::string& c2, const std::string& prompt = "p") {
    return "llm_filter({'model_name': 'gpt-4o'}, {'prompt': '" + prompt +
           "', 'context_columns': [{'data': " + c1 + "}, {'data': " + c2 + "}]})";
}

// A single-column llm_filter call (references one column). Because llm_filter is
// VOLATILE, filter pushdown leaves it as a surviving LogicalFilter instead of
// folding it into the scan, so the rewrite still engages.
std::string LlmFilter1(const std::string& c, const std::string& prompt = "p") {
    return "llm_filter({'model_name': 'gpt-4o'}, {'prompt': '" + prompt +
           "', 'context_columns': [{'data': " + c + "}]})";
}

// Collect every LogicalSemFilter in the tree (pre-order).
void CollectSemFilters(LogicalOperator& op, std::vector<LogicalSemFilter*>& out) {
    if (auto* sf = dynamic_cast<LogicalSemFilter*>(&op)) {
        out.push_back(sf);
    }
    for (auto& child : op.children) {
        CollectSemFilters(*child, out);
    }
}

// Count plain LogicalFilter nodes (a surviving, non-rewritten filter).
size_t CountLogicalFilters(LogicalOperator& op) {
    size_t n = (op.type == LogicalOperatorType::LOGICAL_FILTER) ? 1u : 0u;
    for (auto& child : op.children) {
        n += CountLogicalFilters(*child);
    }
    return n;
}

// Residual conjuncts of a sem-filter node = all expressions except the llm_call.
size_t ResidualCount(const LogicalSemFilter& sf) { return sf.expressions.size() - 1; }

std::string PlanToString(LogicalOperator& op, int depth = 0) {
    std::string line(static_cast<size_t>(depth) * 2, ' ');
    line += op.GetName();
    if (auto* sf = dynamic_cast<LogicalSemFilter*>(&op)) {
        line += "  <SEMFILTER invert=" + std::string(sf->invert ? "true" : "false") +
                " residuals=" + std::to_string(ResidualCount(*sf)) +
                " projmap=" + std::to_string(sf->projection_map.size()) + ">";
    } else if (op.type == LogicalOperatorType::LOGICAL_FILTER) {
        line += "  <FILTER conjuncts=" + std::to_string(op.expressions.size()) + ">";
    }
    line += "\n";
    for (auto& child : op.children) {
        line += PlanToString(*child, depth + 1);
    }
    return line;
}

// Open a fresh session and return its post-optimizer plan. `flag_off` sets
// semantic_rewrite_enabled=false on this session before planning.
std::unique_ptr<LogicalOperator> Plan(const std::string& query, bool flag_off = false) {
    duckdb::Connection con(*g_db);
    if (flag_off) {
        con.Query("SET semantic_rewrite_enabled=false;");
    }
    return con.ExtractPlan(query);
}

std::vector<LogicalSemFilter*> SemFiltersIn(LogicalOperator& plan) {
    std::vector<LogicalSemFilter*> v;
    CollectSemFilters(plan, v);
    return v;
}

// A single-column llm_complete projection expression. `prompt` distinguishes
// otherwise-identical calls so two of them are not collapsed by CSE.
std::string LlmComplete(const std::string& c, const std::string& prompt = "p") {
    return "llm_complete({'model_name': 'gpt-4o'}, {'prompt': '" + prompt +
           "', 'context_columns': [{'data': " + c + "}]})";
}

void CollectSemExtracts(LogicalOperator& op, std::vector<LogicalSemExtract*>& out) {
    if (auto* se = dynamic_cast<LogicalSemExtract*>(&op)) {
        out.push_back(se);
    }
    for (auto& child : op.children) {
        CollectSemExtracts(*child, out);
    }
}

std::vector<LogicalSemExtract*> SemExtractsIn(LogicalOperator& plan) {
    std::vector<LogicalSemExtract*> v;
    CollectSemExtracts(plan, v);
    return v;
}

// ===========================================================================
// Debug aid: print every query's post-optimizer plan once (always passes).
// ===========================================================================
TEST(OptimizerRewriteDump, AllPlans) {
    const std::vector<std::pair<std::string, std::string>> queries = {
            {"VOLATILE single-col WHERE", "SELECT a FROM docs WHERE " + LlmFilter1("a")},
            {"VOLATILE mixed (x>5 AND llm)", "SELECT a FROM docs WHERE x > 5 AND " + LlmFilter1("a")},
            {"#1 where llm_filter", "SELECT a FROM docs WHERE " + LlmFilter("a", "b")},
            {"#2 residual AND", "SELECT a FROM docs WHERE x <> y AND " + LlmFilter("a", "b")},
            {"#3 NOT", "SELECT a FROM docs WHERE NOT " + LlmFilter("a", "b")},
            {"#4 two calls", "SELECT a FROM docs WHERE " + LlmFilter("a", "b", "p1") + " AND " + LlmFilter("b", "a", "p2")},
            {"#5 OR", "SELECT a FROM docs WHERE x > 5 OR " + LlmFilter("a", "b")},
            {"#6 CASE", "SELECT CASE WHEN " + LlmFilter("a", "b") + " THEN 1 ELSE 0 END AS c FROM docs"},
            {"#7 HAVING", "SELECT a FROM docs GROUP BY a HAVING llm_filter({'model_name':'gpt-4o'}, {'prompt':'p', 'context_columns':[{'data': a}, {'data': CAST(COUNT(*) AS VARCHAR)}]})"},
            {"#8 subquery IN", "SELECT a FROM docs WHERE x IN (SELECT x FROM docs d2 WHERE " + LlmFilter("d2.a", "d2.b") + ")"},
            {"#9 projection", "SELECT " + LlmFilter("a", "b") + " AS keep FROM docs"},
            {"#10 CONCAT cmp", "SELECT a FROM docs WHERE CONCAT(" + LlmFilter("a", "b") + ", 'x') = 'truex'"},
            {"#11 JOIN ON", "SELECT d1.a FROM docs d1 JOIN docs d2 ON llm_filter({'model_name':'gpt-4o'}, {'prompt':'p', 'context_columns':[{'data': d1.a}, {'data': d2.a}]})"},
    };
    for (const auto& [label, sql] : queries) {
        std::cout << "==== " << label << " ====\n" << sql << "\n";
        try {
            auto plan = Plan(sql);
            std::cout << PlanToString(*plan);
        } catch (const std::exception& e) {
            std::cout << "  !! ExtractPlan threw: " << e.what() << "\n";
        }
        std::cout << std::endl;
    }
}

// ===========================================================================
// Handled patterns: #1-#4 + #8.
// ===========================================================================
TEST(OptimizerRewrite, Case1_PlainFilter) {
    auto plan = Plan("SELECT a FROM docs WHERE " + LlmFilter("a", "b"));
    auto sf = SemFiltersIn(*plan);
    ASSERT_EQ(sf.size(), 1u);
    EXPECT_FALSE(sf[0]->invert);
    EXPECT_EQ(ResidualCount(*sf[0]), 0u);
}

TEST(OptimizerRewrite, Case2_ResidualPartitioned) {
    auto plan = Plan("SELECT a FROM docs WHERE x <> y AND " + LlmFilter("a", "b"));
    auto sf = SemFiltersIn(*plan);
    ASSERT_EQ(sf.size(), 1u);
    EXPECT_FALSE(sf[0]->invert);
    EXPECT_EQ(ResidualCount(*sf[0]), 1u);  // x<>y stays as a residual conjunct
}

TEST(OptimizerRewrite, Case3_NotInverts) {
    auto plan = Plan("SELECT a FROM docs WHERE NOT " + LlmFilter("a", "b"));
    auto sf = SemFiltersIn(*plan);
    ASSERT_EQ(sf.size(), 1u);
    EXPECT_TRUE(sf[0]->invert);
    EXPECT_EQ(ResidualCount(*sf[0]), 0u);
}

TEST(OptimizerRewrite, Case4_TwoCallsStack) {
    auto plan =
            Plan("SELECT a FROM docs WHERE " + LlmFilter("a", "b", "p1") + " AND " + LlmFilter("b", "a", "p2"));
    auto sf = SemFiltersIn(*plan);
    ASSERT_EQ(sf.size(), 2u);
    // Stacked: neither node carries a residual, and only the TOP node (collected
    // first, pre-order) carries the original filter's projection_map.
    EXPECT_EQ(ResidualCount(*sf[0]), 0u);
    EXPECT_EQ(ResidualCount(*sf[1]), 0u);
    EXPECT_FALSE(sf[0]->projection_map.empty());  // top node
    EXPECT_TRUE(sf[1]->projection_map.empty());   // bottom node passes all columns
}

TEST(OptimizerRewrite, Case8_SubqueryInnerFilter) {
    auto plan = Plan("SELECT a FROM docs WHERE x IN (SELECT x FROM docs d2 WHERE " + LlmFilter("d2.a", "d2.b") + ")");
    auto sf = SemFiltersIn(*plan);
    EXPECT_GE(sf.size(), 1u);
}

// ===========================================================================
// VOLATILE: a single-column WHERE llm_filter survives pushdown and rewrites.
// (Before marking llm_filter VOLATILE, this predicate was folded into the scan
// as a generic ExpressionFilter, leaving no LogicalFilter and thus no rewrite.)
// ===========================================================================
TEST(OptimizerRewrite, Volatile_SingleColumnRewrites) {
    auto plan = Plan("SELECT a FROM docs WHERE " + LlmFilter1("a"));
    auto sf = SemFiltersIn(*plan);
    ASSERT_EQ(sf.size(), 1u);
    EXPECT_FALSE(sf[0]->invert);
    EXPECT_EQ(ResidualCount(*sf[0]), 0u);
}

// A pushable relational conjunct goes to the scan; only the llm_filter survives,
// so the rewritten node carries NO residual (contrast Case2, where a two-column
// residual could not be pushed and stayed on the node).
TEST(OptimizerRewrite, Volatile_MixedPushesRelationalKeepsLlm) {
    auto plan = Plan("SELECT a FROM docs WHERE x > 5 AND " + LlmFilter1("a"));
    auto sf = SemFiltersIn(*plan);
    ASSERT_EQ(sf.size(), 1u);
    EXPECT_EQ(ResidualCount(*sf[0]), 0u);  // x>5 was pushed into the scan, not kept
}

// Baseline (flag off): the single-column filter still SURVIVES pushdown (VOLATILE),
// but is left as a plain LogicalFilter running the scalar llm_filter (no rewrite).
TEST(OptimizerRewrite, Volatile_SingleColumnFlagOffStaysScalar) {
    auto plan = Plan("SELECT a FROM docs WHERE " + LlmFilter1("a"), /*flag_off=*/true);
    EXPECT_TRUE(SemFiltersIn(*plan).empty());
    EXPECT_GE(CountLogicalFilters(*plan), 1u);  // survived pushdown, just not rewritten
}

// ===========================================================================
// Fallback patterns: #5, #6, #7, #9, #10, #11 (no rewrite).
// ===========================================================================
TEST(OptimizerRewrite, Case5_OrIsResidual) {
    auto plan = Plan("SELECT a FROM docs WHERE x > 5 OR " + LlmFilter("a", "b"));
    EXPECT_TRUE(SemFiltersIn(*plan).empty());
}

TEST(OptimizerRewrite, Case6_CaseProjection) {
    auto plan = Plan("SELECT CASE WHEN " + LlmFilter("a", "b") + " THEN 1 ELSE 0 END AS c FROM docs");
    EXPECT_TRUE(SemFiltersIn(*plan).empty());
}

TEST(OptimizerRewrite, Case7_HavingAboveAggregate) {
    auto plan = Plan(
            "SELECT a FROM docs GROUP BY a HAVING llm_filter({'model_name':'gpt-4o'}, "
            "{'prompt':'p', 'context_columns':[{'data': a}, {'data': CAST(COUNT(*) AS VARCHAR)}]})");
    EXPECT_TRUE(SemFiltersIn(*plan).empty());
}

TEST(OptimizerRewrite, Case9_ProjectionValue) {
    auto plan = Plan("SELECT " + LlmFilter("a", "b") + " AS keep FROM docs");
    EXPECT_TRUE(SemFiltersIn(*plan).empty());
}

TEST(OptimizerRewrite, Case10_ComparisonRootIsResidual) {
    auto plan = Plan("SELECT a FROM docs WHERE CONCAT(" + LlmFilter("a", "b") + ", 'x') = 'truex'");
    EXPECT_TRUE(SemFiltersIn(*plan).empty());
}

TEST(OptimizerRewrite, Case11_JoinCondition) {
    auto plan = Plan(
            "SELECT d1.a FROM docs d1 JOIN docs d2 ON llm_filter({'model_name':'gpt-4o'}, "
            "{'prompt':'p', 'context_columns':[{'data': d1.a}, {'data': d2.a}]})");
    EXPECT_TRUE(SemFiltersIn(*plan).empty());
}

// ===========================================================================
// Flag off: even the canonical #1 is left untouched (A/B baseline).
// ===========================================================================
TEST(OptimizerRewrite, FlagOff_NoRewrite) {
    auto plan = Plan("SELECT a FROM docs WHERE " + LlmFilter("a", "b"), /*flag_off=*/true);
    EXPECT_TRUE(SemFiltersIn(*plan).empty());
}

// ===========================================================================
// sem_extract: a projected llm_complete rewrites; fallbacks + flag-off are no-ops.
// ===========================================================================

// Handled: SELECT cols..., llm_complete(...) AS alias -> one LogicalSemExtract.
TEST(OptimizerRewrite, Extract_ProjectionRewrites) {
    auto plan = Plan("SELECT a, " + LlmComplete("b") + " AS dx FROM docs");
    auto se = SemExtractsIn(*plan);
    ASSERT_EQ(se.size(), 1u);
    EXPECT_TRUE(SemFiltersIn(*plan).empty());  // extract, not filter
}

// Fallback: llm_complete nested inside another expression (root is CONCAT, not the
// call) is not a top-level projected call -> left to the scalar.
TEST(OptimizerRewrite, Extract_NestedIsNoOp) {
    auto plan = Plan("SELECT a, CONCAT(" + LlmComplete("b") + ", 'x') AS dx FROM docs");
    EXPECT_TRUE(SemExtractsIn(*plan).empty());
}

// Fallback: more than one top-level llm_complete in the projection -> out of scope.
TEST(OptimizerRewrite, Extract_TwoCallsNoOp) {
    auto plan = Plan("SELECT " + LlmComplete("a", "p1") + " AS d1, " + LlmComplete("b", "p2") + " AS d2 FROM docs");
    EXPECT_TRUE(SemExtractsIn(*plan).empty());
}

// Fallback: llm_complete outside a projection (here in WHERE) -> no extract.
TEST(OptimizerRewrite, Extract_OutsideProjectionNoOp) {
    auto plan = Plan("SELECT a FROM docs WHERE CAST(" + LlmComplete("b") + " AS VARCHAR) = 'yes'");
    EXPECT_TRUE(SemExtractsIn(*plan).empty());
}

// Baseline (flag off): even the handled projection is left untouched.
TEST(OptimizerRewrite, Extract_FlagOffNoOp) {
    auto plan = Plan("SELECT a, " + LlmComplete("b") + " AS dx FROM docs", /*flag_off=*/true);
    EXPECT_TRUE(SemExtractsIn(*plan).empty());
}

}  // namespace
}  // namespace flock

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    duckdb::DuckDB db(nullptr);  // in-memory
    flock::g_db = &db;
    flock::Config::GetConnection(&*db.instance);

    duckdb::Connection setup(db);
    setup.Query("CREATE SECRET (TYPE OPENAI, API_KEY 'test-key');");
    flock::Model::SetMockProvider(std::make_shared<flock::MockProvider>(flock::ModelDetails{}));
    setup.Query("CREATE TABLE docs(a VARCHAR, b VARCHAR, x INT, y INT);");
    // A few rows so statistics don't prune a relational predicate (e.g. x>5) to an
    // EMPTY_RESULT and collapse the plan we want to inspect.
    setup.Query("INSERT INTO docs VALUES ('a1','b1',1,2), ('a2','b2',7,3), ('a3','b3',9,9), ('a4','b4',4,8);");

    int rc = RUN_ALL_TESTS();

    flock::Model::ResetMockProvider();
    flock::g_db = nullptr;
    return rc;
}
