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
    static constexpr int64_t kCoalesceSize = 32;
    static constexpr int64_t kCoalesceMaxAgeMs = 500;
    static constexpr const char* kResponseFormat = "json_schema";
    static constexpr int64_t kMaxOutputTokens = 16;

    // Guardrail thresholds: when the resulting in_flight_cap AND coalesce_size
    // both exceed these, the SET callbacks emit a warning (they never error).
    static constexpr int64_t kGuardrailInFlightCap = 128;
    static constexpr int64_t kGuardrailCoalesceSize = 64;
};

// Canonical option names. Used by RegisterSemanticSettings and by
// ResolveSemanticParams (which reads the SET values back via
// ClientContext::TryGetCurrentSetting), so the two never drift.
namespace semantic_option {
inline constexpr const char* kRewriteEnabled = "semantic_rewrite_enabled";
inline constexpr const char* kEndpoints = "semantic_endpoints";
inline constexpr const char* kRouting = "semantic_routing";
inline constexpr const char* kInFlightCap = "semantic_in_flight_cap";
inline constexpr const char* kCoalesceSize = "semantic_coalesce_size";
inline constexpr const char* kCoalesceMaxAgeMs = "semantic_coalesce_max_age_ms";
inline constexpr const char* kResponseFormat = "semantic_response_format";
inline constexpr const char* kMaxOutputTokens = "semantic_max_output_tokens";
}  // namespace semantic_option

// The model-scoped semantic knobs, resolved for one operator invocation.
// rewrite_enabled / endpoints / routing are intentionally absent: they are not
// model-scoped (read directly from the SET surface / live router where needed).
struct SemanticParams {
    uint64_t in_flight_cap;
    uint64_t coalesce_size;
    uint64_t coalesce_max_age_ms;
    uint64_t max_output_tokens;
    std::string response_format;  // "json_schema" | "free_form"
};

// Split SemanticDefaults::kEndpoints into the router's initial endpoint list.
// Same splitting/trimming rules as the semantic_endpoints SET callback, so the
// default router and an explicit `SET semantic_endpoints=<default>` agree.
std::vector<std::string> DefaultEndpointList();

// Register all eight semantic_* options as native DuckDB extension options.
// Call once from LoadInternal, after flock::Config::Configure(loader).
void RegisterSemanticSettings(duckdb::DBConfig& config);

// Resolve the model-scoped knobs for a single call.
// Precedence per field: model_args value -> current SET value -> SemanticDefaults.
// A missing model, or a model that omits a field, falls through to the SET value
// (which is itself the session SET or, if never set, the default).
SemanticParams ResolveSemanticParams(duckdb::ClientContext& context, const std::string& model_name);

}  // namespace flock