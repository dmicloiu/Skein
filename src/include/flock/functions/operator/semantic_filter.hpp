#pragma once

#include "duckdb/planner/column_binding.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "flock/functions/operator/semantic_operator_base.hpp"
#include "flock/functions/operator/semantic_operator_common.hpp"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace flock {

// Logical node that replaces a LogicalFilter whose predicate is an llm_filter(...)
// call (optionally negated, optionally ANDed with non-LLM conjuncts). It mirrors
// LogicalFilter's binding/typing exactly so the parent sees an identical schema;
// the optimizer rewrite that creates it is a separate task that calls this ctor.
class LogicalSemFilter : public duckdb::LogicalExtensionOperator {
public:
    // child       : the operator feeding rows (becomes children[0], unchanged)
    // llm_call    : the UNWRAPPED llm_filter BoundFunctionExpression (cast/NOT peeled)
    // residuals   : the other WHERE conjuncts, ANDed with the LLM verdict
    // projection_map : copied verbatim from the replaced LogicalFilter
    // invert      : true for WHERE NOT llm_filter(...)
    LogicalSemFilter(duckdb::unique_ptr<duckdb::LogicalOperator> child,
                     duckdb::unique_ptr<duckdb::Expression> llm_call,
                     duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> residuals,
                     duckdb::vector<duckdb::idx_t> projection_map, bool invert);

    // Index of the llm_call inside `expressions` (residuals follow it).
    duckdb::idx_t llm_call_index = 0;
    duckdb::vector<duckdb::idx_t> projection_map;
    bool invert = false;

    duckdb::vector<duckdb::ColumnBinding> GetColumnBindings() override;
    bool HasProjectionMap() const override {
        return !projection_map.empty();
    }

    duckdb::PhysicalOperator& CreatePlan(duckdb::ClientContext& context,
                                         duckdb::PhysicalPlanGenerator& planner) override;

    // GetExtensionName intentionally not overridden: the inherited base throws, so
    // serialization is (correctly) unsupported while normal execution is unaffected.

protected:
    void ResolveTypes() override;
};

// Concrete async semantic filter over the dispatch engine. Renders flock-parity
// batch prompts, parses the per-row boolean verdict, applies invert + the output
// projection, and pre-filters residual conjuncts before the LLM round-trip.
class PhysicalSemFilter : public SemanticOperatorBase {
public:
    PhysicalSemFilter(duckdb::PhysicalPlan& physical_plan, duckdb::vector<duckdb::LogicalType> types,
                      duckdb::idx_t estimated_cardinality, std::shared_ptr<ILLMClient> client,
                      std::shared_ptr<EndpointRouter> router, SemanticParams cfg,
                      duckdb::unique_ptr<duckdb::Expression> residual, bool invert,
                      duckdb::vector<duckdb::idx_t> output_projection,
                      std::vector<SemContextColumn> context_columns, std::string prompt_template,
                      std::string tuple_format, std::string served_model, std::string sticky_key);

    std::string RenderPrompt(const std::vector<RowData>& batch) const override;
    void ParseAndEmit(const nlohmann::json& element, const RowData& row, duckdb::DataChunk& out) const override;

    // Guided schema.
    nlohmann::json BuildResponseFormat(size_t batch_rows) const override;

    // Build the per-thread residual executor (the engine's pre-filter slot).
    duckdb::unique_ptr<duckdb::LocalSinkState> GetLocalSinkState(duckdb::ExecutionContext& context) const override;
    // Stamp the served model + sticky key onto the engine sink state.
    duckdb::unique_ptr<duckdb::GlobalSinkState> GetGlobalSinkState(duckdb::ClientContext& context) const override;

    // Interpret one response element as a boolean verdict. Accepts a JSON boolean
    // (json_schema path) or a free-text string / number (free_form fallback).
    static bool ParseVerdict(const nlohmann::json& element);

    // Accessors (resolved config; used by tests / EXPLAIN).
    const std::string& PromptTemplate() const { return prompt_template_; }
    const std::string& TupleFormat() const { return tuple_format_; }
    const std::string& ServedModel() const { return served_model_; }
    const std::string& StickyKey() const { return sticky_key_; }
    bool Invert() const { return invert_; }
    bool HasResidual() const { return residual_ != nullptr; }
    const std::vector<SemContextColumn>& ContextColumns() const { return context_columns_; }
    const duckdb::vector<duckdb::idx_t>& OutputProjection() const { return output_projection_; }
    const SemanticParams& Config() const { return cfg; }

private:
    duckdb::unique_ptr<duckdb::Expression> residual_;  // null when no residual conjuncts
    bool invert_;
    duckdb::vector<duckdb::idx_t> output_projection_;  // empty -> emit all input columns
    std::vector<SemContextColumn> context_columns_;
    std::string prompt_template_;
    std::string tuple_format_;
    std::string served_model_;
    std::string sticky_key_;
};

}  // namespace flock
