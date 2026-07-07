#include "flock/functions/operator/semantic_filter.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
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
using duckdb::ExpressionClass;
using duckdb::idx_t;
using duckdb::LogicalType;
using duckdb::StructType;

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

// Find the child expression of a bound struct literal by its field name.
Expression* FindStructField(const duckdb::BoundFunctionExpression& struct_expr, const LogicalType& struct_type,
                            const std::string& field_name) {
    const idx_t fields = StructType::GetChildCount(struct_type);
    for (idx_t i = 0; i < fields && i < struct_expr.children.size(); ++i) {
        if (StructType::GetChildName(struct_type, i) == field_name) {
            return struct_expr.children[i].get();
        }
    }
    return nullptr;
}

// Extract the prompt-context columns from the bound prompt struct (llm_call's 2nd
// argument). v1 supports tabular columns whose `data` is a direct column reference;
// anything else (computed data, image/audio) is rejected loudly.
std::vector<SemContextColumn> ExtractContextColumns(duckdb::ClientContext& context,
                                                    const duckdb::BoundFunctionExpression& llm_call) {
    if (llm_call.children.size() < 2) {
        throw duckdb::InvalidInputException("sem_filter: llm_filter expects (model, prompt) arguments");
    }
    auto& prompt_expr = *llm_call.children[1];
    if (prompt_expr.expression_class != ExpressionClass::BOUND_FUNCTION) {
        throw duckdb::NotImplementedException("sem_filter: prompt argument must be a struct literal");
    }
    auto& prompt_struct = prompt_expr.Cast<duckdb::BoundFunctionExpression>();
    auto* cc_expr = FindStructField(prompt_struct, prompt_expr.return_type, "context_columns");
    if (!cc_expr) {
        throw duckdb::NotImplementedException("sem_filter: llm_filter requires context_columns");
    }
    if (cc_expr->expression_class != ExpressionClass::BOUND_FUNCTION) {
        throw duckdb::NotImplementedException("sem_filter: context_columns must be a list literal");
    }
    auto& list_expr = cc_expr->Cast<duckdb::BoundFunctionExpression>();

    std::vector<SemContextColumn> columns;
    for (auto& element : list_expr.children) {
        if (element->expression_class != ExpressionClass::BOUND_FUNCTION) {
            throw duckdb::NotImplementedException("sem_filter: each context column must be a struct literal");
        }
        auto& col_struct = element->Cast<duckdb::BoundFunctionExpression>();
        const auto& col_type = element->return_type;

        SemContextColumn cc;
        bool has_data = false;
        const idx_t fields = StructType::GetChildCount(col_type);
        for (idx_t j = 0; j < fields && j < col_struct.children.size(); ++j) {
            const auto field = StructType::GetChildName(col_type, j);
            auto& child = *col_struct.children[j];
            if (field == "data") {
                if (child.expression_class != ExpressionClass::BOUND_REF) {
                    throw duckdb::NotImplementedException(
                            "sem_filter: context column 'data' must be a direct column reference");
                }
                cc.data_index = child.Cast<duckdb::BoundReferenceExpression>().index;
                has_data = true;
            } else {
                // Static metadata (name/type/detail): fold the constant, drop NULLs.
                if (!child.IsFoldable()) {
                    throw duckdb::NotImplementedException("sem_filter: context column metadata must be constant");
                }
                auto value = duckdb::ExpressionExecutor::EvaluateScalar(context, child);
                if (value.IsNull()) {
                    continue;
                }
                auto str = value.ToString();
                if (str == "NULL") {
                    continue;
                }
                if (field == "type" && (str == "image" || str == "audio")) {
                    throw duckdb::NotImplementedException(
                            "sem_filter: image/audio context columns are not supported");
                }
                cc.metadata[field] = str;
            }
        }
        if (!has_data) {
            throw duckdb::InvalidInputException("sem_filter: context column missing 'data'");
        }
        columns.push_back(std::move(cc));
    }
    if (columns.empty()) {
        throw duckdb::NotImplementedException("sem_filter: llm_filter requires at least one context column");
    }
    return columns;
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

    auto context_columns = ExtractContextColumns(context, llm_call);

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
    auto columns = nlohmann::json::array();
    for (const auto& cc : context_columns_) {
        nlohmann::json column = cc.metadata;
        auto data = nlohmann::json::array();
        for (const auto& row : batch) {
            data.push_back(row.values[cc.data_index].ToString());
        }
        column["data"] = std::move(data);
        columns.push_back(std::move(column));
    }
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
    std::string prompt = std::get<0>(
            PromptManager::Render(prompt_template_, columns, ScalarFunctionType::FILTER, tuple_format_));
    // The bool template says "return true/false"; override it for the object modes
    // (the guided schema enforces shape; this makes the model use id/reason).
    if (mode == SemSchema::kId) {
        prompt += "\n\n## Output (structured)\n"
                  "For EACH row return one object with that row's `row_id` and a boolean `verdict` "
                  "(true iff the review satisfies the user prompt). One object per row, in row order, keyed "
                  "to its row_id. Judge every row independently on its own merits.";
    } else if (mode == SemSchema::kIdReason) {
        prompt += "\n\n## Output (structured, reason first)\n"
                  "For EACH row return one object with that row's `row_id`, then a brief `reason` "
                  "(<=12 words) for the judgement, then a boolean `verdict` (true iff the review satisfies "
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
