#include "flock/functions/operator/optimizer_rewrite.hpp"

#include "duckdb/common/types/value.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_any_join.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_cross_product.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_limit.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "flock/functions/operator/semantic_extract.hpp"
#include "flock/functions/operator/semantic_filter.hpp"
#include "flock/runtime/semantic_settings.h"

#include <limits>
#include <utility>
#include <unordered_set>

namespace flock {

namespace {

using duckdb::Expression;
using duckdb::ExpressionClass;
using duckdb::ExpressionType;
using duckdb::idx_t;
using duckdb::LogicalOperator;
using duckdb::LogicalOperatorType;
using duckdb::LogicalTypeId;

// DuckDB 2.0: a cast is a BoundFunctionExpression of type OPERATOR_CAST (the old
// BOUND_CAST class is gone), so test for casts BEFORE treating a function as a call.
bool IsBoolCast(const Expression& expr) {
    return duckdb::BoundCastExpression::IsCast(expr) && expr.GetReturnType().id() == LogicalTypeId::BOOLEAN;
}

// True iff `expr` is a (non-cast) bound scalar call to `name`.
bool IsCallTo(const Expression& expr, const char* name) {
    return expr.GetExpressionClass() == ExpressionClass::BOUND_FUNCTION && !duckdb::BoundCastExpression::IsCast(expr) &&
           expr.Cast<duckdb::BoundFunctionExpression>().Function().GetName().GetIdentifierName() == name;
}

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
        if (cur->GetExpressionClass() == ExpressionClass::BOUND_OPERATOR &&
            cur->GetExpressionType() == ExpressionType::OPERATOR_NOT) {
            const auto& op = cur->Cast<duckdb::BoundOperatorExpression>();
            if (op.GetChildren().size() == 1) {
                invert = !invert;
                cur = op.GetChildren()[0].get();
                continue;
            }
        }
        if (IsBoolCast(*cur)) {
            cur = &duckdb::BoundCastExpression::Child(cur->Cast<duckdb::BoundFunctionExpression>());
            continue;
        }
        break;
    }
    return IsCallTo(*cur, "llm_filter");
}

// Physically peel to the bare llm_filter expression. Mirrors ConjunctIsLLMCall's
// peel conditions exactly, so it stops at the same node (the function call).
duckdb::unique_ptr<Expression> UnwrapLLMCall(duckdb::unique_ptr<Expression> expr) {
    for (;;) {
        if (expr->GetExpressionClass() == ExpressionClass::BOUND_OPERATOR &&
            expr->GetExpressionType() == ExpressionType::OPERATOR_NOT) {
            auto& op = expr->Cast<duckdb::BoundOperatorExpression>();
            if (op.GetChildren().size() == 1) {
                expr = std::move(op.GetChildrenMutable()[0]);
                continue;
            }
        }
        if (IsBoolCast(*expr)) {
            auto child = std::move(duckdb::BoundCastExpression::ChildMutable(expr->Cast<duckdb::BoundFunctionExpression>()));
            expr = std::move(child);
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
    const duckdb::vector<duckdb::ProjectionIndex> projection_map = filter.projection_map;
    const idx_t call_count = llm_calls.size();

    // Bottom node: original child + ALL residuals + the first llm_call. Only the
    // TOP node carries the filter's projection_map (its output must equal the
    // filter's); lower nodes use an empty map so every column passes through and
    // upper nodes still see what they need. Common case (one call): a single node
    // that gets both the projection_map and the residuals.
    duckdb::unique_ptr<LogicalOperator> current = duckdb::make_uniq<LogicalSemFilter>(
            std::move(original_child), std::move(llm_calls[0]), std::move(residuals),
            call_count == 1 ? projection_map : duckdb::vector<duckdb::ProjectionIndex>{}, inverts[0]);

    for (idx_t i = 1; i < call_count; ++i) {
        const bool is_top = (i + 1 == call_count);
        duckdb::vector<duckdb::unique_ptr<Expression>> no_residuals;
        current = duckdb::make_uniq<LogicalSemFilter>(std::move(current), std::move(llm_calls[i]),
                                                      std::move(no_residuals),
                                                      is_top ? projection_map : duckdb::vector<duckdb::ProjectionIndex>{},
                                                      inverts[i]);
    }

    node = std::move(current);
}

// DuckDB 2.0 folds a predicate that reads both join sides into an INNER
// LogicalComparisonJoin as an "arbitrary" (non-comparison) JoinCondition, where
// 1.5 left it in a FILTER above the join. A top-level llm_filter there would run as
// a scalar inside the hash join, so pull it out: keep the key conditions (and any
// other arbitrary condition) in the join, and stack LogicalSemFilters above it,
// exactly the 1.5 plan shape. The join's projection maps may drop columns that
// only the predicate reads, so widen them and let the top SemFilter restore the
// join's original output.
void TryRewriteJoinConditions(duckdb::unique_ptr<LogicalOperator>& node) {
    auto& join = node->Cast<duckdb::LogicalComparisonJoin>();
    if (join.join_type != duckdb::JoinType::INNER || join.children.size() != 2) {
        return;
    }
    bool has_llm_call = false;
    for (auto& cond : join.conditions) {
        bool invert = false;
        if (!cond.IsComparison() && ConjunctIsLLMCall(cond.GetJoinExpression(), invert)) {
            has_llm_call = true;
            break;
        }
    }
    if (!has_llm_call) {
        return;
    }

    const auto original_output = join.GetColumnBindings();

    duckdb::vector<duckdb::unique_ptr<Expression>> llm_calls;
    duckdb::vector<bool> inverts;
    duckdb::vector<duckdb::JoinCondition> kept;
    for (auto& cond : join.conditions) {
        bool invert = false;
        if (!cond.IsComparison() && ConjunctIsLLMCall(cond.GetJoinExpression(), invert)) {
            llm_calls.push_back(UnwrapLLMCall(std::move(cond.JoinExpressionReference())));
            inverts.push_back(invert);
        } else {
            kept.push_back(std::move(cond));
        }
    }

    // Columns the llm calls read.
    duckdb::vector<duckdb::ColumnBinding> needed;
    for (auto& call : llm_calls) {
        duckdb::ExpressionIterator::VisitExpression<duckdb::BoundColumnRefExpression>(
            *call, [&](const duckdb::BoundColumnRefExpression& ref) { needed.push_back(ref.Binding()); });
    }

    duckdb::unique_ptr<LogicalOperator> below;
    if (kept.empty()) {
        // Only semantic predicates: a cross product outputs every column.
        below = duckdb::make_uniq<duckdb::LogicalCrossProduct>(std::move(join.children[0]),
                                                                std::move(join.children[1]));
    } else {
        join.conditions = std::move(kept);
        // Keep every needed column in the join output.
        auto widen = [&](duckdb::LogicalOperator& child, duckdb::vector<duckdb::ProjectionIndex>& map) {
            if (map.empty()) {
                return;  // empty map = all child columns pass through
            }
            const auto child_bindings = child.GetColumnBindings();
            for (const auto& b : needed) {
                for (idx_t i = 0; i < child_bindings.size(); ++i) {
                    if (child_bindings[i] == b) {
                        bool present = false;
                        for (const auto& m : map) {
                            present |= (m.GetIndex() == i);
                        }
                        if (!present) {
                            map.push_back(duckdb::ProjectionIndex(i));
                        }
                    }
                }
            }
        };
        widen(*join.children[0], join.left_projection_map);
        widen(*join.children[1], join.right_projection_map);
        // The join's output changed: refresh its cached types to match.
        join.ResolveOperatorTypes();
        below = std::move(node);
    }

    // Top node restores the original output if the columns below differ.
    duckdb::vector<duckdb::ProjectionIndex> projection_map;
    const auto below_output = below->GetColumnBindings();
    if (below_output != original_output) {
        for (const auto& b : original_output) {
            for (idx_t i = 0; i < below_output.size(); ++i) {
                if (below_output[i] == b) {
                    projection_map.push_back(duckdb::ProjectionIndex(i));
                    break;
                }
            }
        }
    }

    const idx_t call_count = llm_calls.size();
    duckdb::unique_ptr<LogicalOperator> current = std::move(below);
    for (idx_t i = 0; i < call_count; ++i) {
        const bool is_top = (i + 1 == call_count);
        duckdb::vector<duckdb::unique_ptr<Expression>> no_residuals;
        current = duckdb::make_uniq<LogicalSemFilter>(std::move(current), std::move(llm_calls[i]),
                                                      std::move(no_residuals),
                                                      is_top ? projection_map : duckdb::vector<duckdb::ProjectionIndex>{},
                                                      inverts[i]);
    }
    node = std::move(current);
}

// Skein may absorb CROSS JOIN ... WHERE llm_filter(left, right) into an
// INNER LogicalAnyJoin. Only the bare semantic predicate (or its implicit
// BOOLEAN cast) is supported here; extra predicates, NOT and OR are outside
// this rule (other existing filter rewrites may still apply).
void TryRewriteUnkeyedJoin(duckdb::unique_ptr<LogicalOperator>& node) {
    auto& join = node->Cast<duckdb::LogicalAnyJoin>();
    if (join.join_type != duckdb::JoinType::INNER || join.children.size() != 2 ||
        !join.condition || !join.left_projection_map.empty() || !join.right_projection_map.empty()) {
        return;
    }
    const Expression* call = join.condition.get();
    while (IsBoolCast(*call)) {
        call = &duckdb::BoundCastExpression::Child(call->Cast<duckdb::BoundFunctionExpression>());
    }
    if (!IsCallTo(*call, "llm_filter")) {
        return;
    }

    // Require genuine two-sided, local references. In particular, do not turn
    // correlated or one-sided predicates into Cartesian semantic joins.
    std::unordered_set<duckdb::TableIndex> left_tables, right_tables;
    for (const auto& binding : join.children[0]->GetColumnBindings()) {
        left_tables.insert(binding.table_index);
    }
    for (const auto& binding : join.children[1]->GetColumnBindings()) {
        right_tables.insert(binding.table_index);
    }
    bool reads_left = false, reads_right = false, unsupported = false;
    duckdb::ExpressionIterator::VisitExpression<duckdb::BoundColumnRefExpression>(
        *call, [&](const duckdb::BoundColumnRefExpression& ref) {
            const bool on_left = left_tables.count(ref.Binding().table_index) != 0;
            const bool on_right = right_tables.count(ref.Binding().table_index) != 0;
            if (ref.Depth() != 0 || on_left == on_right) {
                unsupported = true;
            }
            reads_left |= on_left;
            reads_right |= on_right;
        });
    if (unsupported || !reads_left || !reads_right) {
        return;
    }

    auto condition = UnwrapLLMCall(std::move(join.condition));
    auto product = duckdb::make_uniq<duckdb::LogicalCrossProduct>(
        std::move(join.children[0]), std::move(join.children[1]));
    duckdb::vector<duckdb::unique_ptr<Expression>> no_residuals;
    node = duckdb::make_uniq<LogicalSemFilter>(std::move(product), std::move(condition),
                                               std::move(no_residuals), duckdb::vector<duckdb::ProjectionIndex>{}, false);
}

// True iff the expression's ROOT is a bound llm_complete(...) call. No CAST/wrapper
// peeling: a call nested inside another expression (CONCAT(llm_complete(...), 'x'),
// CAST(llm_complete(...) AS ...)) has a non-llm_complete root, so it is NOT a
// top-level call and its projection is left to the scalar (the documented fallback).
bool ExprIsLlmComplete(const Expression& expr) {
    return IsCallTo(expr, "llm_complete");
}

// If a LogicalProjection (held in `node`) has EXACTLY ONE top-level llm_complete
// projected expression, replace it with a LogicalSemExtract that subsumes the whole
// projection. No-op otherwise: zero (none, or an llm_complete nested inside another
// expression) or more than one (out of scope for v1) leaves the scalar call in place.
void TryRewriteProjection(duckdb::unique_ptr<LogicalOperator>& node) {
    auto& projection = node->Cast<duckdb::LogicalProjection>();
    if (projection.children.empty()) {
        return;
    }
    idx_t llm_call_index = 0;
    int llm_call_count = 0;
    for (idx_t i = 0; i < projection.expressions.size(); ++i) {
        if (ExprIsLlmComplete(*projection.expressions[i])) {
            llm_call_index = i;
            ++llm_call_count;
        }
    }
    if (llm_call_count != 1) {
        return;
    }
    node = duckdb::make_uniq<LogicalSemExtract>(std::move(projection.children[0]), projection.table_index,
                                                std::move(projection.expressions), llm_call_index);
}

// POST-ORDER WALK over unique_ptr<LogicalOperator>& slots so any node (incl. the
// root) can be replaced in place; children are rewritten before their parent.
void RewriteTree(duckdb::unique_ptr<LogicalOperator>& node) {
    for (auto& child : node->children) {
        RewriteTree(child);
    }
    if (node->type == LogicalOperatorType::LOGICAL_FILTER) {
        TryRewriteFilter(node);
    } else if (node->type == LogicalOperatorType::LOGICAL_ANY_JOIN) {
        TryRewriteUnkeyedJoin(node);
    } else if (node->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
        TryRewriteJoinConditions(node);
    } else if (node->type == LogicalOperatorType::LOGICAL_PROJECTION) {
        TryRewriteProjection(node);
    }
}

// LIMIT early stop: a constant `LIMIT n [OFFSET m]` whose child is a semantic
// filter, possibly through projections and empty filters only (which keep the
// row count), lets that filter stop after n + m output rows. Anything else in
// between (a non-empty filter, ORDER BY, aggregate, DISTINCT, join, ...) changes which or how many
// rows reach the LIMIT, so no hint. ORDER BY ... LIMIT plans as TOP_N, never
// matched here. Only the top node of a stacked filter gets the hint.
void AttachLimitHints(LogicalOperator& node) {
    if (node.type == LogicalOperatorType::LOGICAL_LIMIT && !node.children.empty()) {
        const auto& limit = node.Cast<duckdb::LogicalLimit>();
        const bool const_limit = limit.limit_val.Type() == duckdb::LimitNodeType::CONSTANT_VALUE;
        const bool const_offset = limit.offset_val.Type() == duckdb::LimitNodeType::UNSET ||
                                  limit.offset_val.Type() == duckdb::LimitNodeType::CONSTANT_VALUE;
        if (const_limit && const_offset) {
            const idx_t n = limit.limit_val.GetConstantValue();
            const idx_t m = limit.offset_val.Type() == duckdb::LimitNodeType::CONSTANT_VALUE
                                    ? limit.offset_val.GetConstantValue()
                                    : 0;
            const idx_t hint = (n > (std::numeric_limits<idx_t>::max)() - m) ? 0 : n + m;
            LogicalOperator* cur = node.children[0].get();
            // An empty FILTER (all conjuncts moved into the semantic filter by RewriteTree)
            // passes every row, so it keeps the row count like a projection.
            while (cur->children.size() == 1 &&
                   (cur->type == LogicalOperatorType::LOGICAL_PROJECTION ||
                    (cur->type == LogicalOperatorType::LOGICAL_FILTER && cur->expressions.empty()))) {
                cur = cur->children[0].get();
            }
            if (auto* sem = dynamic_cast<LogicalSemFilter*>(cur)) {
                if (hint > 0 && (sem->limit_hint == 0 || hint < sem->limit_hint)) {
                    sem->limit_hint = hint;
                }
            }
        }
    }
    for (auto& child : node.children) {
        AttachLimitHints(*child);
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
    duckdb::Value early_stop;
    if (!input.context.TryGetCurrentSetting(semantic_option::kLimitEarlyStop, early_stop) ||
        early_stop.GetValue<bool>()) {
        AttachLimitHints(*plan);
    }
}

}  // namespace flock
