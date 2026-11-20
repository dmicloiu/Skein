#include "flock/functions/operator/semantic_filter.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "flock/functions/llm_function_bind_data.hpp"
#include "flock/functions/operator/embedding_proxy.hpp"
#include "flock/model_manager/model.hpp"
#include "flock/prompt_manager/prompt_manager.hpp"
#include "flock/runtime/extension_state.h"
#include "flock/runtime/semantic_settings.h"

#include <cctype>
#include <cstdio>
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

// Build the embedding proxy stage for this filter, or null (with a note on
// stderr) when the filter does not fit it: the proxy can only reject, so a
// negated filter (where a rejected row would be EMITTED) runs exactly, and the
// ranking needs exactly one image column plus optional text columns.
std::shared_ptr<ProxyStage> MakeEmbeddingProxyStage(const SemanticParams& params,
                                                    const std::vector<SemContextColumn>& columns,
                                                    const std::string& prompt_template, bool invert) {
    const auto skip = [](const std::string& why) -> std::shared_ptr<ProxyStage> {
        std::fprintf(stderr, "[flock proxy] disabled for this filter: %s\n", why.c_str());
        return nullptr;
    };
    if (params.filter_proxy_endpoint.empty()) {
        throw duckdb::InvalidInputException("semantic_filter_proxy_kind='%s' requires semantic_filter_proxy_endpoint",
                                            params.filter_proxy_kind);
    }
    if (invert) {
        return skip("NOT llm_filter (a pruned row would be emitted)");
    }
    EmbeddingProxyConfig cfg;
    size_t images = 0;
    for (const auto& column : columns) {
        const std::string type = column.metadata.value("type", std::string("tabular"));
        if (type.compare(0, 5, "image") == 0) {
            ++images;
            cfg.image_index = column.data_index;
            cfg.image_column = column.metadata;
        } else if (type == "audio") {
            return skip("audio context column");
        } else {
            cfg.text_indices.push_back(column.data_index);
        }
    }
    if (images != 1) {
        return skip("needs exactly one image context column, found " + std::to_string(images));
    }
    cfg.kind = params.filter_proxy_kind;
    cfg.endpoint = params.filter_proxy_endpoint;
    cfg.model = params.filter_proxy_model;
    cfg.text_template = params.filter_proxy_template;
    cfg.fallback_text = prompt_template;
    cfg.min_rows = params.filter_proxy_min_rows;
    cfg.max_in_flight = params.in_flight_cap;
    cfg.topk.k_min = params.filter_proxy_k_min;
    cfg.topk.k_frac = params.filter_proxy_k_frac;
    cfg.topk.gap = params.filter_proxy_gap;
    cfg.topk.audit_near = params.filter_proxy_audit_near;
    cfg.topk.audit_far = params.filter_proxy_audit_far;
    cfg.topk.seed = params.filter_proxy_seed;
    cfg.lotus.recall_target = params.filter_proxy_lotus_recall;
    cfg.lotus.precision_target = params.filter_proxy_lotus_precision;
    cfg.lotus.failure_probability = params.filter_proxy_lotus_failure_prob;
    cfg.lotus.sampling_percentage = params.filter_proxy_lotus_sample_frac;
    cfg.lotus.max_sample_range = params.filter_proxy_lotus_sample_range;
    cfg.lotus.seed = params.filter_proxy_seed;
    return std::make_shared<EmbeddingProxyStage>(std::move(cfg));
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
    std::string proxy_served_model;
    const bool embedding_proxy = !params.filter_proxy_model.empty() && params.filter_proxy_kind != "llm";
    if (!params.filter_proxy_model.empty() && !embedding_proxy) {
        if (params.response_format != "json_schema") {
            throw duckdb::InvalidInputException("semantic filter proxy requires semantic_response_format='json_schema'");
        }
        if (params.filter_proxy_endpoint.empty()) {
            throw duckdb::InvalidInputException("semantic_filter_proxy_model requires semantic_filter_proxy_endpoint");
        }
        Model proxy(nlohmann::json{{"model_name", params.filter_proxy_model}});
        proxy_served_model = proxy.GetModelDetails().model;
        if (proxy_served_model == served_model) {
            throw duckdb::InvalidInputException("semantic_filter_proxy_model must differ from the main model");
        }
    }
    auto& extension_state = ExtensionState::Get(context);

    auto context_columns = ExtractContextColumns(context, llm_call, "sem_filter");

    std::shared_ptr<ProxyStage> proxy_stage;
    if (embedding_proxy) {
        proxy_stage = MakeEmbeddingProxyStage(params, context_columns, prompt_template, invert);
    }

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
                                               std::move(rendered_head), std::move(proxy_served_model),
                                               std::move(proxy_stage))
                       .Cast<PhysicalSemFilter>();
    op.SetLimitHint(limit_hint);
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
                                     std::string tuple_format, std::string served_model, std::string sticky_key,
                                     std::string proxy_served_model, std::shared_ptr<ProxyStage> proxy_stage)
    : SemanticOperatorBase(physical_plan, duckdb::PhysicalOperatorType::EXTENSION, std::move(types),
                           estimated_cardinality, std::move(client), std::move(router), std::move(cfg)),
      residual_(std::move(residual)), invert_(invert), output_projection_(std::move(output_projection)),
      context_columns_(std::move(context_columns)), prompt_template_(std::move(prompt_template)),
      tuple_format_(std::move(tuple_format)), served_model_(std::move(served_model)),
      sticky_key_(std::move(sticky_key)), proxy_served_model_(std::move(proxy_served_model)),
      proxy_stage_(std::move(proxy_stage)) {
}

std::string PhysicalSemFilter::RenderPrompt(const std::vector<RowData>& batch) const {
    // Rebuild the scalar's context_columns JSON for this batch (metadata verbatim,
    // data = each row's value stringified), then render through the shared builder
    // so the bytes match llm_filter at the same batch_size.
    auto columns = BuildContextColumnsJson(context_columns_, batch);
    if (SemPromptSlim() && !HasMediaColumns(columns)) {
        // Lean, text-only head + the SAME tuples, dropping the META_PROMPT
        // image/audio boilerplate. Batch-adaptive: a multi-row batch renders the
        // count/index contract + chunked row-major form; a single-row batch keeps the plain slim form.
        return RenderSlimSemanticPrompt(SlimKind::kFilter, prompt_template_, columns, tuple_format_);
    }
    return std::get<0>(
            PromptManager::Render(prompt_template_, columns, ScalarFunctionType::FILTER, tuple_format_));
}

std::string PhysicalSemFilter::RenderProxyPrompt(const std::vector<RowData>& batch) const {
    const auto columns = BuildContextColumnsJson(context_columns_, batch);
    return "For each row below, estimate the probability from 0 to 1 that the criterion is true. "
           "Judge each row independently. Return a JSON object with an items array containing exactly " +
           std::to_string(batch.size()) + " numeric scores in row order. A score of 0 means certainly false "
           "and 1 means certainly true.\nCriterion: " + prompt_template_ + "\n\n" +
           PromptManager::ConstructInputTuples(columns, tuple_format_);
}

nlohmann::json PhysicalSemFilter::RenderMedia(const std::vector<RowData>& batch) const {
    return BuildImageColumnsJson(BuildContextColumnsJson(context_columns_, batch));
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
    // FILTER output element: one boolean per row.
    return ItemsResponseFormat("filter_results", {{"type", "boolean"}}, batch_rows);
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
    state->proxy_served_model = proxy_served_model_;
    state->proxy_stage = proxy_stage_;
    state->verdict_of = [](const nlohmann::json& element) { return ParseVerdict(element); };
    if (limit_hint_ > 0 && !proxy_stage_ && proxy_served_model_.empty()) {
        state->limit_rows = limit_hint_;
        // An element yields an output row iff verdict XOR invert (as ParseAndEmit).
        state->limit_emits = [this](const nlohmann::json& element) { return ParseVerdict(element) != invert_; };
    }
    if (!proxy_served_model_.empty()) {
        state->render_proxy_prompt = [this](const std::vector<RowData>& batch) {
            return RenderProxyPrompt(batch);
        };
    }
    return std::move(state);
}

}  // namespace flock
