#include "flock/functions/operator/semantic_extract.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "flock/functions/llm_function_bind_data.hpp"
#include "flock/model_manager/model.hpp"
#include "flock/prompt_manager/prompt_manager.hpp"
#include "flock/runtime/extension_state.h"
#include "flock/runtime/semantic_settings.h"

#include <string>
#include <utility>

namespace flock {

using duckdb::ColumnBinding;
using duckdb::DataChunk;
using duckdb::Expression;
using duckdb::idx_t;
using duckdb::LogicalType;
using duckdb::SinkResultType;

// =============================================================================
// LogicalSemExtract -- mirrors LogicalProjection's bindings / typing.
// =============================================================================

LogicalSemExtract::LogicalSemExtract(duckdb::unique_ptr<duckdb::LogicalOperator> child, idx_t table_index_p,
                                     duckdb::vector<duckdb::unique_ptr<Expression>> select_list,
                                     idx_t llm_call_index_p)
    : duckdb::LogicalExtensionOperator(std::move(select_list)), table_index(table_index_p),
      llm_call_index(llm_call_index_p) {
    children.push_back(std::move(child));
}

duckdb::vector<ColumnBinding> LogicalSemExtract::GetColumnBindings() {
    return GenerateColumnBindings(table_index, expressions.size());
}

duckdb::vector<idx_t> LogicalSemExtract::GetTableIndex() const {
    return duckdb::vector<idx_t>{table_index};
}

void LogicalSemExtract::ResolveTypes() {
    for (auto& expr : expressions) {
        types.push_back(expr->return_type);
    }
}

// =============================================================================
// Per-thread sink state: the base engine's local state plus a capture executor
// and its output chunk (the materialized [siblings...][context data...] row).
// =============================================================================

namespace {

class SemExtractLocalSinkState : public SemLocalSinkState {
public:
    duckdb::unique_ptr<duckdb::ExpressionExecutor> capture_executor;
    DataChunk capture_chunk;
};

}  // namespace

// =============================================================================
// CreatePlan
// =============================================================================

duckdb::PhysicalOperator& LogicalSemExtract::CreatePlan(duckdb::ClientContext& context,
                                                        duckdb::PhysicalPlanGenerator& planner) {
    // Reuse the scalar's resolved model + prompt template; do not re-resolve.
    auto& llm_call = expressions[llm_call_index]->Cast<duckdb::BoundFunctionExpression>();
    auto& bind_data = llm_call.bind_info->Cast<LlmFunctionBindData>();
    const std::string prompt_template = bind_data.prompt;

    Model model(bind_data.model_json);
    const auto model_details = model.GetModelDetails();
    const std::string tuple_format = model_details.tuple_format;
    const std::string served_model = model_details.model;

    SemanticParams params = ResolveSemanticParams(context, model_details.model_name);
    auto& extension_state = ExtensionState::Get(context);

    auto context_columns = ExtractContextColumns(context, llm_call, "sem_extract");

    // Sticky key: the per-query-stable template head. Rendering with no tuples
    // leaves the {{TUPLES}} block untouched, yielding the same head for every batch.
    auto rendered_head = std::get<0>(PromptManager::Render(prompt_template, nlohmann::json::array(),
                                                           ScalarFunctionType::COMPLETE, tuple_format));

    // Capture executor expressions: the non-LLM projected exprs in output order,
    // then one ref per context column. Evaluated over the child chunk at Sink, they
    // yield [sibling values...][context data...], which becomes the captured RowData.
    duckdb::vector<duckdb::unique_ptr<Expression>> capture_exprs;
    for (idx_t i = 0; i < expressions.size(); ++i) {
        if (i == llm_call_index) {
            continue;
        }
        capture_exprs.push_back(std::move(expressions[i]));
    }
    const idx_t sibling_count = capture_exprs.size();
    for (idx_t j = 0; j < context_columns.size(); ++j) {
        auto& cc = context_columns[j];
        // Reference the source child column (data_index is still the child index
        // here); ExtractContextColumns validated it is a direct column reference.
        capture_exprs.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(
                cc.data_type, static_cast<duckdb::storage_t>(cc.data_index)));
        // Repoint the context column at its slot in the captured row for RenderPrompt.
        cc.data_index = sibling_count + j;
    }

    duckdb::vector<LogicalType> capture_types;
    capture_types.reserve(capture_exprs.size());
    for (auto& expr : capture_exprs) {
        capture_types.push_back(expr->return_type);
    }

    auto& child_plan = planner.CreatePlan(*children[0]);
    auto& op = planner.Make<PhysicalSemExtract>(types, estimated_cardinality, extension_state.client,
                                                extension_state.router, std::move(params),
                                                std::move(capture_exprs), std::move(capture_types),
                                                llm_call_index, std::move(context_columns), prompt_template,
                                                tuple_format, served_model, std::move(rendered_head))
                       .Cast<PhysicalSemExtract>();
    op.children.push_back(child_plan);
    return op;
}

// =============================================================================
// PhysicalSemExtract
// =============================================================================

PhysicalSemExtract::PhysicalSemExtract(duckdb::PhysicalPlan& physical_plan, duckdb::vector<LogicalType> types,
                                       idx_t estimated_cardinality, std::shared_ptr<ILLMClient> client,
                                       std::shared_ptr<EndpointRouter> router, SemanticParams cfg,
                                       duckdb::vector<duckdb::unique_ptr<Expression>> capture_exprs,
                                       duckdb::vector<LogicalType> capture_types, idx_t llm_call_index,
                                       std::vector<SemContextColumn> context_columns, std::string prompt_template,
                                       std::string tuple_format, std::string served_model, std::string sticky_key)
    : SemanticOperatorBase(physical_plan, duckdb::PhysicalOperatorType::EXTENSION, std::move(types),
                           estimated_cardinality, std::move(client), std::move(router), std::move(cfg)),
      capture_exprs_(std::move(capture_exprs)), capture_types_(std::move(capture_types)),
      llm_call_index_(llm_call_index), context_columns_(std::move(context_columns)),
      prompt_template_(std::move(prompt_template)), tuple_format_(std::move(tuple_format)),
      served_model_(std::move(served_model)), sticky_key_(std::move(sticky_key)) {
}

std::string PhysicalSemExtract::RenderPrompt(const std::vector<RowData>& batch) const {
    // Byte-parity with the scalar llm_complete: the SAME context_columns JSON
    // (metadata verbatim, data = each row's captured value) through the SAME shared
    // builder with the COMPLETE template.
    auto columns = BuildContextColumnsJson(context_columns_, batch);
    if (SemPromptSlim()) {
        // Lean text-only head + the SAME tuples, mirroring the filter's slim
        // path (extract wording: task / one string per row).
        return RenderSlimSemanticPrompt(SlimKind::kExtract, prompt_template_, columns, tuple_format_,
                                        SemVariantsFromEnv(), /*state_output_shape=*/true);
    }
    return std::get<0>(
            PromptManager::Render(prompt_template_, columns, ScalarFunctionType::COMPLETE, tuple_format_));
}

std::string PhysicalSemExtract::ParseCompletion(const nlohmann::json& element) {
    if (element.is_string()) {
        return element.get<std::string>();
    }
    // Non-string (object/number/null): dump, matching the scalar's fallback. A
    // padded null (degraded response) becomes the literal "null", same as scalar.
    return element.dump();
}

void PhysicalSemExtract::ParseAndEmit(const nlohmann::json& element, const RowData& row, DataChunk& out) const {
    // Emit EVERY row (a projection, not a filter): the completion string at the
    // llm_complete position, the captured sibling values elsewhere, in column order.
    const idx_t idx = out.size();
    const idx_t cols = GetTypes().size();
    for (idx_t c = 0; c < cols; ++c) {
        if (c == llm_call_index_) {
            out.SetValue(c, idx, duckdb::Value(ParseCompletion(element)));
        } else {
            out.SetValue(c, idx, row.values[CaptureIndex(c)]);
        }
    }
    out.SetCardinality(idx + 1);
}

nlohmann::json PhysicalSemExtract::BuildResponseFormat(size_t batch_rows) const {
    // One string completion per row (mirrors the scalar's OutputType::STRING schema),
    // but bound each string to its share of the output-token budget (~4 chars/token).
    // The bound scales with the configured budget and R, so it never
    // clips a completion the budget could actually fit.
    const uint64_t per_row_tokens = batch_rows > 0 ? cfg.max_output_tokens / batch_rows : cfg.max_output_tokens;
    const int max_chars = static_cast<int>((per_row_tokens > 0 ? per_row_tokens : 1) * 4);
    nlohmann::json element = {{"type", "string"}, {"maxLength", max_chars}};
    return ItemsResponseFormat("extract_results", std::move(element), batch_rows);
}

SinkResultType PhysicalSemExtract::Sink(duckdb::ExecutionContext& /*context*/, DataChunk& chunk,
                                        duckdb::OperatorSinkInput& input) const {
    auto& local = input.local_state.Cast<SemExtractLocalSinkState>();
    auto& gss = input.global_state.Cast<SemGlobalSinkState>();
    // Fresh chunk (next_row_idx == 0): materialize [siblings...][context data...].
    // A BLOCKED resume (next_row_idx != 0) reuses the cached capture chunk, mirroring
    // the filter's residual-selection cache, so the projection runs once per chunk.
    if (local.next_row_idx == 0) {
        local.capture_chunk.Reset();
        local.capture_executor->Execute(chunk, local.capture_chunk);
    }
    return gss.SinkChunk(local, local.capture_chunk, input.interrupt_state);
}

duckdb::unique_ptr<duckdb::LocalSinkState> PhysicalSemExtract::GetLocalSinkState(
        duckdb::ExecutionContext& context) const {
    auto state = duckdb::make_uniq<SemExtractLocalSinkState>();
    state->capture_executor = duckdb::make_uniq<duckdb::ExpressionExecutor>(context.client, capture_exprs_);
    state->capture_chunk.Initialize(duckdb::Allocator::Get(context.client), capture_types_);
    return std::move(state);
}

duckdb::unique_ptr<duckdb::GlobalSinkState> PhysicalSemExtract::GetGlobalSinkState(
        duckdb::ClientContext& /*context*/) const {
    auto state = CreateGlobalSinkState();  // binds render_prompt + response_schema to this operator
    state->served_model = served_model_;
    state->sticky_key = sticky_key_;
    return std::move(state);
}

}  // namespace flock
