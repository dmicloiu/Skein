#include "flock/functions/operator/semantic_filter.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "flock/functions/llm_function_bind_data.hpp"
#include "flock/model_manager/model.hpp"
#include "flock/prompt_manager/prompt_manager.hpp"
#include "flock/runtime/extension_state.h"
#include "flock/runtime/semantic_settings.h"

#include <cctype>
#include <string>
#include <utility>

namespace flock {

using duckdb::ColumnBinding;
using duckdb::DataChunk;
using duckdb::Expression;
using duckdb::idx_t;
using duckdb::LogicalType;

// =============================================================================
// LogicalSemFilter
// =============================================================================

LogicalSemFilter::LogicalSemFilter(duckdb::unique_ptr<duckdb::LogicalOperator> child,
                                   duckdb::unique_ptr<Expression> llm_call,
                                   duckdb::vector<duckdb::unique_ptr<Expression>> residuals,
                                   duckdb::vector<idx_t> projection_map_p, bool invert_p)
    : duckdb::LogicalExtensionOperator(), projection_map(std::move(projection_map_p)), invert(invert_p) {
    children.push_back(std::move(child));
    llm_call_index = expressions.size();  // 0; residuals follow
    expressions.push_back(std::move(llm_call));
    for (auto& residual : residuals) {
        expressions.push_back(std::move(residual));
    }
}

duckdb::vector<ColumnBinding> LogicalSemFilter::GetColumnBindings() {
    return MapBindings(children[0]->GetColumnBindings(), projection_map);
}

void LogicalSemFilter::ResolveTypes() {
    types = MapTypes(children[0]->types, projection_map);
}

// =============================================================================
// CreatePlan helpers
// =============================================================================

namespace {

// AND the residual conjuncts into a single expression for the pre-filter executor.
duckdb::unique_ptr<Expression> CombineConjunction(duckdb::vector<duckdb::unique_ptr<Expression>> exprs) {
    if (exprs.empty()) {
        return nullptr;
    }
    if (exprs.size() == 1) {
        return std::move(exprs[0]);
    }
    auto conjunction = duckdb::make_uniq<duckdb::BoundConjunctionExpression>(duckdb::ExpressionType::CONJUNCTION_AND);
    for (auto& expr : exprs) {
        conjunction->children.push_back(std::move(expr));
    }
    return std::move(conjunction);
}

}  // namespace

duckdb::PhysicalOperator& LogicalSemFilter::CreatePlan(duckdb::ClientContext& context,
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

    auto context_columns = ExtractContextColumns(context, llm_call, "sem_filter");

    // Sticky key: the per-query-stable template head. Rendering with no tuples
    // leaves the {{TUPLES}} block untouched, yielding the same head for every batch.
    auto rendered_head = std::get<0>(PromptManager::Render(prompt_template, nlohmann::json::array(),
                                                           ScalarFunctionType::FILTER, tuple_format));

    // Residual conjuncts: everything in `expressions` except the llm_call.
    duckdb::vector<duckdb::unique_ptr<Expression>> residual_exprs;
    for (idx_t i = 0; i < expressions.size(); ++i) {
        if (i == llm_call_index) {
            continue;
        }
        residual_exprs.push_back(std::move(expressions[i]));
    }
    auto residual = CombineConjunction(std::move(residual_exprs));

    auto& child_plan = planner.CreatePlan(*children[0]);
    auto& op = planner.Make<PhysicalSemFilter>(types, estimated_cardinality, extension_state.client,
                                               extension_state.router, std::move(params), std::move(residual),
                                               invert, projection_map, std::move(context_columns),
                                               prompt_template, tuple_format, served_model,
                                               std::move(rendered_head))
                       .Cast<PhysicalSemFilter>();
    op.children.push_back(child_plan);
    return op;
}

// =============================================================================
// PhysicalSemFilter
// =============================================================================

PhysicalSemFilter::PhysicalSemFilter(duckdb::PhysicalPlan& physical_plan, duckdb::vector<LogicalType> types,
                                     idx_t estimated_cardinality, std::shared_ptr<ILLMClient> client,
                                     std::shared_ptr<EndpointRouter> router, SemanticParams cfg,
                                     duckdb::unique_ptr<Expression> residual, bool invert,
                                     duckdb::vector<idx_t> output_projection,
                                     std::vector<SemContextColumn> context_columns, std::string prompt_template,
                                     std::string tuple_format, std::string served_model, std::string sticky_key)
    : SemanticOperatorBase(physical_plan, duckdb::PhysicalOperatorType::EXTENSION, std::move(types),
                           estimated_cardinality, std::move(client), std::move(router), std::move(cfg)),
      residual_(std::move(residual)), invert_(invert), output_projection_(std::move(output_projection)),
      context_columns_(std::move(context_columns)), prompt_template_(std::move(prompt_template)),
      tuple_format_(std::move(tuple_format)), served_model_(std::move(served_model)),
      sticky_key_(std::move(sticky_key)) {
}

std::string PhysicalSemFilter::RenderPrompt(const std::vector<RowData>& batch) const {
    // Rebuild the scalar's context_columns JSON for this batch (metadata verbatim,
    // data = each row's value stringified), then render through the shared builder
    // so the bytes match llm_filter at the same batch_size.
    auto columns = BuildContextColumnsJson(context_columns_, batch);
    const SemSchema mode = SemSchemaModeFromEnv();
    if (mode != SemSchema::kBool) {
        // Rec-1 ablation: prepend an explicit batch-local row id per row so the
        // model can anchor each verdict; the guided schema requires it to echo id.
        nlohmann::json id_col;
        id_col["name"] = "row_id";
        auto ids = nlohmann::json::array();
        for (size_t k = 0; k < batch.size(); ++k) {
            ids.push_back(k + 1);
        }
        id_col["data"] = std::move(ids);
        columns.insert(columns.begin(), std::move(id_col));
    }
    std::string prompt;
    if (SemPromptSlim()) {
        // Lean, text-only head + the SAME tuples (byte-parity on the rows),
        // dropping the META_PROMPT image/audio boilerplate.
        prompt = "For each row in the table below, decide whether it satisfies the criterion, "
                 "judging every row independently on its own merits.\n"
                 "Criterion: " + prompt_template_ + "\n\n"
                 + PromptManager::ConstructInputTuples(columns, tuple_format_);
        if (mode == SemSchema::kBool) {
            // full mode carries RESPONSE_FORMAT::FILTER; slim must state the shape.
            prompt += "\n\nReturn a JSON object {\"items\": [...]} with one boolean per row, in row order.";
        }
    } else {
        prompt = std::get<0>(
                PromptManager::Render(prompt_template_, columns, ScalarFunctionType::FILTER, tuple_format_));
    }
    // The bool template says "return true/false"; override it for the object modes
    // (the guided schema enforces shape; this makes the model use id/reason).
    if (mode == SemSchema::kId) {
        prompt += "\n\n## Output (structured)\n"
                  "For EACH row return one object with that row's `row_id` and a boolean `verdict` "
                  "(true iff the review satisfies the user prompt). One object per row, in row order, keyed "
                  "to its row_id. Judge every row independently on its own merits.";
    } else if (mode == SemSchema::kIdReason) {
        prompt += "\n\n## Output (structured, reason first)\n"
                  "For EACH row return one object with that row's `row_id`, then a brief `reason` (<="
                  + std::to_string(SemReasonWords()) +
                  " words) for the judgement, then a boolean `verdict` (true iff the review satisfies "
                  "the user prompt). Write the reason BEFORE the verdict. One object per row, in row order, "
                  "keyed to its row_id. Judge every row independently on its own merits.";
    }
    return prompt;
}

bool PhysicalSemFilter::ParseVerdict(const nlohmann::json& element) {
    if (element.is_boolean()) {
        return element.get<bool>();
    }
    if (element.is_number()) {
        return element.get<double>() != 0.0;
    }
    if (element.is_string()) {
        // free_form fallback: scan free text for a true/false signal.
        std::string lower = element.get<std::string>();
        for (auto& c : lower) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        if (lower.find("true") != std::string::npos || lower.find("yes") != std::string::npos) {
            return true;
        }
        if (lower.find("false") != std::string::npos || lower.find("no") != std::string::npos) {
            return false;
        }
        return lower.find('1') != std::string::npos;
    }
    if (element.is_null()) {
        return true;  // parity with scalar llm_filter: a null/missing verdict -> pass (keep)
    }
    return false;  // object / array -> fail-safe: do not emit
}

nlohmann::json PhysicalSemFilter::BuildResponseFormat(size_t batch_rows) const {
    // FILTER output element. bool: a boolean. The ablation modes make it a
    // per-row object so the model anchors to a row id (kId) and reasons before the
    // verdict (kIdReason); ParseItems reduces either back to a positional verdict
    // array. Property order id,reason,verdict is preserved so the rationale is
    // emitted before the verdict.
    const SemSchema mode = SemSchemaModeFromEnv();
    nlohmann::json element;
    if (mode == SemSchema::kBool) {
        element = {{"type", "boolean"}};
    } else if (mode == SemSchema::kId) {
        element = {{"type", "object"},
                   {"properties", {{"id", {{"type", "integer"}}}, {"verdict", {{"type", "boolean"}}}}},
                   {"required", nlohmann::json::array({"id", "verdict"})},
                   {"additionalProperties", false}};
    } else {  // kIdReason: id, then reason (<= SemReasonWords() words), then verdict
        // ~7 chars/word (incl. space) bounds the string to the word budget so guided
        // decoding forces brevity for the cheap-reason variants.
        const int reason_max_chars = 7 * SemReasonWords();
        element = {{"type", "object"},
                   {"properties", {{"id", {{"type", "integer"}}},
                                   {"reason", {{"type", "string"}, {"maxLength", reason_max_chars}}},
                                   {"verdict", {{"type", "boolean"}}}}},
                   {"required", nlohmann::json::array({"id", "reason", "verdict"})},
                   {"additionalProperties", false}};
    }
    return ItemsResponseFormat("filter_results", std::move(element), batch_rows);
}

void PhysicalSemFilter::ParseAndEmit(const nlohmann::json& element, const RowData& row, DataChunk& out) const {
    // Emit iff verdict XOR invert.
    if (ParseVerdict(element) == invert_) {
        return;
    }
    const idx_t idx = out.size();
    if (output_projection_.empty()) {
        for (idx_t c = 0; c < row.values.size(); ++c) {
            out.SetValue(c, idx, row.values[c]);
        }
    } else {
        for (idx_t c = 0; c < output_projection_.size(); ++c) {
            out.SetValue(c, idx, row.values[output_projection_[c]]);
        }
    }
    out.SetCardinality(idx + 1);
}

duckdb::unique_ptr<duckdb::LocalSinkState> PhysicalSemFilter::GetLocalSinkState(
        duckdb::ExecutionContext& context) const {
    auto state = duckdb::make_uniq<SemLocalSinkState>();
    if (residual_) {
        state->residual_executor = duckdb::make_uniq<duckdb::ExpressionExecutor>(context.client, *residual_);
        state->residual_sel.Initialize(STANDARD_VECTOR_SIZE);
    }
    return std::move(state);
}

duckdb::unique_ptr<duckdb::GlobalSinkState> PhysicalSemFilter::GetGlobalSinkState(
        duckdb::ClientContext& /*context*/) const {
    auto state = CreateGlobalSinkState();  // binds render_prompt to this operator
    state->served_model = served_model_;
    state->sticky_key = sticky_key_;
    return std::move(state);
}

}  // namespace flock
