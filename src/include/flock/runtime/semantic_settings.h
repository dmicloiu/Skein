#pragma once

#include "duckdb/main/client_context.hpp"
#include "duckdb/main/database.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace flock {

// Single source of truth for the default values of every semantic_* knob.
//
// Shared between the SET-option registration (RegisterSemanticSettings) and the
// ExtensionState constructor, so the EndpointRouter built at database-open time
// matches exactly what `SET` reports before the user overrides anything. Change
// a default here and both the router and the option pick it up.
struct SemanticDefaults {
    static constexpr bool kRewriteEnabled = true;
    static constexpr const char* kEndpoints = "http://localhost:8000/v1";
    static constexpr const char* kRouting = "round_robin";
    static constexpr int64_t kInFlightCap = 128;
    // Rows packed into one multi-row prompt.
    static constexpr int64_t kBatchSize = 32;
    static constexpr int64_t kCoalesceMaxAgeMs = 500;
    static constexpr const char* kResponseFormat = "json_schema";
    static constexpr int64_t kMaxOutputTokens = 16;
    // Empty disables the experimental two-model filter cascade.
    static constexpr const char* kFilterProxyModel = "";
    static constexpr const char* kFilterProxyEndpoint = "";
    // Safe calibration mode: a score in [0,1] always goes to the main model.
    static constexpr double kFilterProxyNegativeThreshold = -1.0;
    static constexpr double kFilterProxyPositiveThreshold = 2.0;
    // Proxy kind: "llm" (scored by a small chat model, thresholds above),
    // "embedding" (CLIP-style embeddings, reject-only per-group top-k) or
    // "lotus" (embeddings + LOTUS-style learned cascade with auto-accept).
    static constexpr const char* kFilterProxyKind = "llm";
    static constexpr const char* kFilterProxyTemplate = "";
    static constexpr int64_t kFilterProxyKMin = 5;
    static constexpr double kFilterProxyKFrac = 0.0;
    static constexpr int64_t kFilterProxyGap = 2;
    static constexpr double kFilterProxyAuditNear = 0.25;
    static constexpr double kFilterProxyAuditFar = 0.01;
    static constexpr int64_t kFilterProxyMinRows = 1000;
    static constexpr int64_t kFilterProxySeed = 42;
    // LOTUS CascadeArgs as used by the sembench LOTUS runner.
    static constexpr double kFilterProxyLotusRecall = 0.8;
    static constexpr double kFilterProxyLotusPrecision = 0.8;
    static constexpr double kFilterProxyLotusFailureProb = 0.2;
    static constexpr double kFilterProxyLotusSampleFrac = 0.1;
    static constexpr int64_t kFilterProxyLotusSampleRange = 200;

    // Guardrail thresholds: when the resolved in_flight_cap AND batch_size both
    // exceed these, ResolveSemanticParams emits a warning (it never errors).
    static constexpr int64_t kGuardrailInFlightCap = 128;
    static constexpr int64_t kGuardrailBatchSize = 64;
};

// Canonical option names. Used by RegisterSemanticSettings and by
// ResolveSemanticParams (which reads the SET values back via
// ClientContext::TryGetCurrentSetting), so the two never drift.
namespace semantic_option {
inline constexpr const char* kRewriteEnabled = "semantic_rewrite_enabled";
inline constexpr const char* kEndpoints = "semantic_endpoints";
inline constexpr const char* kRouting = "semantic_routing";
inline constexpr const char* kInFlightCap = "semantic_in_flight_cap";
inline constexpr const char* kBatchSize = "semantic_batch_size";
inline constexpr const char* kCoalesceMaxAgeMs = "semantic_coalesce_max_age_ms";
inline constexpr const char* kResponseFormat = "semantic_response_format";
inline constexpr const char* kMaxOutputTokens = "semantic_max_output_tokens";
inline constexpr const char* kFilterProxyModel = "semantic_filter_proxy_model";
inline constexpr const char* kFilterProxyEndpoint = "semantic_filter_proxy_endpoint";
inline constexpr const char* kFilterProxyNegativeThreshold = "semantic_filter_proxy_negative_threshold";
inline constexpr const char* kFilterProxyPositiveThreshold = "semantic_filter_proxy_positive_threshold";
inline constexpr const char* kFilterProxyKind = "semantic_filter_proxy_kind";
inline constexpr const char* kFilterProxyTemplate = "semantic_filter_proxy_template";
inline constexpr const char* kFilterProxyKMin = "semantic_filter_proxy_k_min";
inline constexpr const char* kFilterProxyKFrac = "semantic_filter_proxy_k_frac";
inline constexpr const char* kFilterProxyGap = "semantic_filter_proxy_gap";
inline constexpr const char* kFilterProxyAuditNear = "semantic_filter_proxy_audit_near";
inline constexpr const char* kFilterProxyAuditFar = "semantic_filter_proxy_audit_far";
inline constexpr const char* kFilterProxyMinRows = "semantic_filter_proxy_min_rows";
inline constexpr const char* kFilterProxySeed = "semantic_filter_proxy_seed";
inline constexpr const char* kFilterProxyLotusRecall = "semantic_filter_proxy_lotus_recall";
inline constexpr const char* kFilterProxyLotusPrecision = "semantic_filter_proxy_lotus_precision";
inline constexpr const char* kFilterProxyLotusFailureProb = "semantic_filter_proxy_lotus_failure_prob";
inline constexpr const char* kFilterProxyLotusSampleFrac = "semantic_filter_proxy_lotus_sample_frac";
inline constexpr const char* kFilterProxyLotusSampleRange = "semantic_filter_proxy_lotus_sample_range";
}  // namespace semantic_option

// Semantic settings resolved for one operator invocation. The dispatch knobs
// can be overridden by model_args; proxy settings come from the session only.
struct SemanticParams {
    uint64_t in_flight_cap;
    uint64_t batch_size;  // rows per multi-row prompt
    uint64_t coalesce_max_age_ms;
    uint64_t max_output_tokens;
    std::string response_format;  // "json_schema" | "free_form"
    std::string filter_proxy_model;  // empty disables proxy scoring
    std::string filter_proxy_endpoint;  // separate proxy vLLM /chat/completions URL
    double filter_proxy_negative_threshold = SemanticDefaults::kFilterProxyNegativeThreshold;
    double filter_proxy_positive_threshold = SemanticDefaults::kFilterProxyPositiveThreshold;
    std::string filter_proxy_kind = SemanticDefaults::kFilterProxyKind;
    std::string filter_proxy_template = SemanticDefaults::kFilterProxyTemplate;
    uint64_t filter_proxy_k_min = SemanticDefaults::kFilterProxyKMin;
    double filter_proxy_k_frac = SemanticDefaults::kFilterProxyKFrac;
    uint64_t filter_proxy_gap = SemanticDefaults::kFilterProxyGap;
    double filter_proxy_audit_near = SemanticDefaults::kFilterProxyAuditNear;
    double filter_proxy_audit_far = SemanticDefaults::kFilterProxyAuditFar;
    uint64_t filter_proxy_min_rows = SemanticDefaults::kFilterProxyMinRows;
    uint64_t filter_proxy_seed = SemanticDefaults::kFilterProxySeed;
    double filter_proxy_lotus_recall = SemanticDefaults::kFilterProxyLotusRecall;
    double filter_proxy_lotus_precision = SemanticDefaults::kFilterProxyLotusPrecision;
    double filter_proxy_lotus_failure_prob = SemanticDefaults::kFilterProxyLotusFailureProb;
    double filter_proxy_lotus_sample_frac = SemanticDefaults::kFilterProxyLotusSampleFrac;
    uint64_t filter_proxy_lotus_sample_range = SemanticDefaults::kFilterProxyLotusSampleRange;
};

// Split SemanticDefaults::kEndpoints into the router's initial endpoint list.
// Same splitting/trimming rules as the semantic_endpoints SET callback, so the
// default router and an explicit `SET semantic_endpoints=<default>` agree.
std::vector<std::string> DefaultEndpointList();

// Register semantic_* options as native DuckDB extension options.
// Call once from LoadInternal, after flock::Config::Configure(loader).
void RegisterSemanticSettings(duckdb::DBConfig& config);

// Resolve the model-scoped knobs for a single call.
// Precedence per field: model_args value -> current SET value -> SemanticDefaults.
// A missing model, or a model that omits a field, falls through to the SET value
// (which is itself the session SET or, if never set, the default).
SemanticParams ResolveSemanticParams(duckdb::ClientContext& context, const std::string& model_name);

}  // namespace flock
