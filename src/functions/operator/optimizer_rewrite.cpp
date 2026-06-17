#include "flock/functions/operator/optimizer_rewrite.hpp"

#include "duckdb/common/types/value.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "flock/functions/operator/semantic_filter.hpp"
#include "flock/runtime/semantic_settings.h"

#include <utility>

namespace flock {

namespace {

using duckdb::Expression;
using duckdb::ExpressionClass;
using duckdb::ExpressionType;
using duckdb::idx_t;
using duckdb::LogicalOperator;
using duckdb::LogicalOperatorType;
using duckdb::LogicalTypeId;

// A conjunct is an llm_call iff, after peeling a top-level CAST(... AS BOOLEAN)
// (llm_filter returns VARCHAR) and an optional top-level NOT, its root is a
// BoundFunctionExpression named "llm_filter". This is the non-mutating decision;
// `invert` is set true iff a NOT was peeled. A conjunct whose root is anything
// else (an OR/comparison/CONCAT that merely *contains* an llm_filter deeper
// inside) is NOT a semantic call: it stays a residual, and the nested llm_filter
// runs as a scalar function inside the residual executor.
bool ConjunctIsLLMCall(const Expression& expr, bool& invert) {
    const Expression* cur = &expr;
    invert = false;
    for (;;) {
        if (cur->expression_class == ExpressionClass::BOUND_OPERATOR &&
            cur->GetExpressionType() == ExpressionType::OPERATOR_NOT) {
            const auto& op = cur->Cast<duckdb::BoundOperatorExpression>();
            if (op.children.size() == 1) {
                invert = !invert;
                cur = op.children[0].get();
                continue;
            }
        }
        if (cur->expression_class == ExpressionClass::BOUND_CAST &&
            cur->return_type.id() == LogicalTypeId::BOOLEAN) {
            cur = cur->Cast<duckdb::BoundCastExpression>().child.get();
            continue;
        }
        break;
    }
    return cur->expression_class == ExpressionClass::BOUND_FUNCTION &&
           cur->Cast<duckdb::BoundFunctionExpression>().function.name == "llm_filter";
}

// Physically peel to the bare llm_filter expression. Mirrors ConjunctIsLLMCall's
// peel conditions exactly, so it stops at the same node (the function call).
duckdb::unique_ptr<Expression> UnwrapLLMCall(duckdb::unique_ptr<Expression> expr) {
    for (;;) {
        if (expr->expression_class == ExpressionClass::BOUND_OPERATOR &&
            expr->GetExpressionType() == ExpressionType::OPERATOR_NOT) {
            auto& op = expr->Cast<duckdb::BoundOperatorExpression>();
            if (op.children.size() == 1) {
                expr = std::move(op.children[0]);
                continue;
            }
        }
        if (expr->expression_class == ExpressionClass::BOUND_CAST &&
            expr->return_type.id() == LogicalTypeId::BOOLEAN) {
            expr = std::move(expr->Cast<duckdb::BoundCastExpression>().child);
            continue;
        }
        break;
    }
    return expr;
}

// True if `filter` reads from an aggregate (its input is an aggregate, possibly
// behind a chain of projections DuckDB inserts to compute/prune the HAVING
// expressions). Distinguishes a HAVING clause (and any WHERE over aggregated
// output) from a row-level filter (whose descent lands on a scan/join).
bool FilterFeedsFromAggregate(const duckdb::LogicalFilter& filter) {
    if (filter.children.empty()) {
        return false;
    }
    const LogicalOperator* cur = filter.children[0].get();
    while (cur->type == LogicalOperatorType::LOGICAL_PROJECTION && !cur->children.empty()) {
        cur = cur->children[0].get();
    }
    return cur->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY;
}

// If a LogicalFilter (held in `node`) has any top-level llm_filter conjunct,
// replace it with a stack of LogicalSemFilters. No-op otherwise.
void TryRewriteFilter(duckdb::unique_ptr<LogicalOperator>& node) {
    auto& filter = node->Cast<duckdb::LogicalFilter>();

    // Scope contract: a HAVING clause (a filter over aggregated output) is out of
    // scope for v1, so leave its llm_filter as a scalar call.
    if (FilterFeedsFromAggregate(filter)) {
        return;
    }

    // Decide first, without mutating: is there at least one top-level llm_filter?
    bool has_llm_call = false;
    for (auto& conjunct : filter.expressions) {
        bool invert = false;
        if (ConjunctIsLLMCall(*conjunct, invert)) {
            has_llm_call = true;
            break;
        }
    }
    if (!has_llm_call) {
        return;
    }

    // Partition the conjuncts: each top-level llm_filter becomes its own node's
    // call; everything else is a residual. (A second top-level llm_filter never
    // lands in residuals - - -> it gets its own stacked node below.)
    duckdb::vector<duckdb::unique_ptr<Expression>> llm_calls;
    duckdb::vector<bool> inverts;
    duckdb::vector<duckdb::unique_ptr<Expression>> residuals;
    for (auto& conjunct : filter.expressions) {
        bool invert = false;
        if (ConjunctIsLLMCall(*conjunct, invert)) {
            llm_calls.push_back(UnwrapLLMCall(std::move(conjunct)));
            inverts.push_back(invert);
        } else {
            residuals.push_back(std::move(conjunct));
        }
    }

    auto original_child = std::move(filter.children[0]);
    const duckdb::vector<idx_t> projection_map = filter.projection_map;
    const idx_t call_count = llm_calls.size();

    // Bottom node: original child + ALL residuals + the first llm_call. Only the
    // TOP node carries the filter's projection_map (its output must equal the
    // filter's); lower nodes use an empty map so every column passes through and
    // upper nodes still see what they need. Common case (one call): a single node
    // that gets both the projection_map and the residuals.
    duckdb::unique_ptr<LogicalOperator> current = duckdb::make_uniq<LogicalSemFilter>(
            std::move(original_child), std::move(llm_calls[0]), std::move(residuals),
            call_count == 1 ? projection_map : duckdb::vector<idx_t>{}, inverts[0]);

    for (idx_t i = 1; i < call_count; ++i) {
        const bool is_top = (i + 1 == call_count);
        duckdb::vector<duckdb::unique_ptr<Expression>> no_residuals;
        current = duckdb::make_uniq<LogicalSemFilter>(std::move(current), std::move(llm_calls[i]),
                                                      std::move(no_residuals),
                                                      is_top ? projection_map : duckdb::vector<idx_t>{},
                                                      inverts[i]);
    }

    node = std::move(current);
}

// POST-ORDER WALK over unique_ptr<LogicalOperator>& slots so any node (incl. the
// root) can be replaced in place; children are rewritten before their parent.
void RewriteTree(duckdb::unique_ptr<LogicalOperator>& node) {
    for (auto& child : node->children) {
        RewriteTree(child);
    }
    if (node->type == LogicalOperatorType::LOGICAL_FILTER) {
        TryRewriteFilter(node);
    }
}

}  // namespace

void SemanticOptimizeFunction(duckdb::OptimizerExtensionInput& input,
                              duckdb::unique_ptr<LogicalOperator>& plan) {
    duckdb::Value enabled;
    if (input.context.TryGetCurrentSetting(semantic_option::kRewriteEnabled, enabled) &&
        !enabled.GetValue<bool>()) {
        return;  // A/B baseline: flag off -> leave every llm_filter as a scalar call.
    }
    RewriteTree(plan);
}

}  // namespace flock