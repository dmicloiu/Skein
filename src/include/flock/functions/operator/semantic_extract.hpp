#pragma once

#include "duckdb/planner/column_binding.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "flock/functions/operator/semantic_operator_base.hpp"
#include "flock/functions/operator/semantic_operator_common.hpp"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace flock {

// Logical node that replaces a LogicalProjection whose select list holds exactly
// one top-level llm_complete(...) call. It mirrors LogicalProjection's
// binding/typing exactly (a new table_index with one binding per projected
// column) so the parent sees an identical schema; the optimizer rewrite that
// creates it is a separate task that calls this ctor.
class LogicalSemExtract : public duckdb::LogicalExtensionOperator {
public:
    // child          : the operator feeding rows (becomes children[0], unchanged)
    // table_index    : copied verbatim from the replaced LogicalProjection
    // select_list    : the full projection expression list (llm_complete at
    //                  llm_call_index; the others are plain refs / simple exprs)
    // llm_call_index : index of the llm_complete call within select_list
    LogicalSemExtract(duckdb::unique_ptr<duckdb::LogicalOperator> child, duckdb::idx_t table_index,
                      duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> select_list,
                      duckdb::idx_t llm_call_index);

    duckdb::idx_t table_index;
    duckdb::idx_t llm_call_index = 0;

    duckdb::vector<duckdb::ColumnBinding> GetColumnBindings() override;
    duckdb::vector<duckdb::idx_t> GetTableIndex() const override;

    duckdb::PhysicalOperator& CreatePlan(duckdb::ClientContext& context,
                                         duckdb::PhysicalPlanGenerator& planner) override;

    // GetExtensionName intentionally not overridden: the inherited base throws, so
    // serialization is (correctly) unsupported while normal execution is unaffected.

protected:
    void ResolveTypes() override;
};

// Concrete async semantic extract over the dispatch engine. Unlike the filter it
// SUBSUMES the projection: every input row is emitted with the llm_complete result
// slotted at its projected position, alongside the other (non-LLM) projected
// columns, in projection column order.
//
// The non-LLM projected columns AND the llm_complete context data are materialized
// at Sink into the captured RowData (see Sink) as [siblings...][context data...],
// so ParseAndEmit just slots the completion string and RenderPrompt reads the
// context slice. The engine's dispatch is unchanged.
class PhysicalSemExtract : public SemanticOperatorBase {
public:
    PhysicalSemExtract(duckdb::PhysicalPlan& physical_plan, duckdb::vector<duckdb::LogicalType> types,
                       duckdb::idx_t estimated_cardinality, std::shared_ptr<ILLMClient> client,
                       std::shared_ptr<EndpointRouter> router, SemanticParams cfg,
                       duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> capture_exprs,
                       duckdb::vector<duckdb::LogicalType> capture_types, duckdb::idx_t llm_call_index,
                       std::vector<SemContextColumn> context_columns, std::string prompt_template,
                       std::string tuple_format, std::string served_model, std::string sticky_key);

    std::string RenderPrompt(const std::vector<RowData>& batch) const override;
    void ParseAndEmit(const nlohmann::json& element, const RowData& row, duckdb::DataChunk& out) const override;
    nlohmann::json BuildResponseFormat(size_t batch_rows) const override;

    // Pre-project the child chunk (siblings ++ context data) into the per-thread
    // capture chunk, then hand THAT to the engine's SinkChunk. Overrides the base
    // adapter (which forwards the raw chunk); the engine itself is untouched.
    duckdb::SinkResultType Sink(duckdb::ExecutionContext& context, duckdb::DataChunk& chunk,
                                duckdb::OperatorSinkInput& input) const override;
    // Build the per-thread capture executor + capture chunk.
    duckdb::unique_ptr<duckdb::LocalSinkState> GetLocalSinkState(duckdb::ExecutionContext& context) const override;
    // Stamp the served model + sticky key onto the engine sink state.
    duckdb::unique_ptr<duckdb::GlobalSinkState> GetGlobalSinkState(duckdb::ClientContext& context) const override;

    // Interpret one response element as the completion string. Mirrors the scalar
    // llm_complete: a JSON string is used verbatim; anything else is dumped.
    static std::string ParseCompletion(const nlohmann::json& element);

    // Accessors (resolved config; used by tests / EXPLAIN).
    const std::string& PromptTemplate() const { return prompt_template_; }
    const std::string& TupleFormat() const { return tuple_format_; }
    const std::string& ServedModel() const { return served_model_; }
    const std::string& StickyKey() const { return sticky_key_; }
    duckdb::idx_t LlmCallIndex() const { return llm_call_index_; }
    const std::vector<SemContextColumn>& ContextColumns() const { return context_columns_; }
    const SemanticParams& Config() const { return cfg; }

private:
    // Output column c (c != llm_call_index_) maps to this slot of the captured
    // non-LLM prefix (siblings are captured in output order, skipping the LLM slot).
    duckdb::idx_t CaptureIndex(duckdb::idx_t output_col) const {
        return output_col < llm_call_index_ ? output_col : output_col - 1;
    }

    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> capture_exprs_;  // [siblings...][context refs]
    duckdb::vector<duckdb::LogicalType> capture_types_;
    duckdb::idx_t llm_call_index_;
    std::vector<SemContextColumn> context_columns_;  // data_index points into the capture row
    std::string prompt_template_;
    std::string tuple_format_;
    std::string served_model_;
    std::string sticky_key_;
};

}  // namespace flock