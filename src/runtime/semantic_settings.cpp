#include "flock/runtime/semantic_settings.h"

#include "duckdb/common/enums/set_scope.hpp"
#include "duckdb/common/printer.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/vector_size.hpp"
#include "duckdb/main/config.hpp"
#include "flock/core/common.hpp"
#include "flock/core/config.hpp"
#include "flock/runtime/extension_state.h"

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace flock {

namespace {

using duckdb::ClientContext;
using duckdb::DBConfig;
using duckdb::LogicalType;
using duckdb::SetScope;
using duckdb::Value;

// Split on ',', trim surrounding whitespace from each piece, reject empties.
// Throws (so SET fails) if the input yields no endpoints or any blank one.
std::vector<std::string> SplitEndpoints(const std::string& raw) {
    std::vector<std::string> out;
    size_t start = 0;
    while (true) {
        const size_t comma = raw.find(',', start);
        const std::string piece = raw.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        const size_t b = piece.find_first_not_of(" \t\r\n");
        const size_t e = piece.find_last_not_of(" \t\r\n");
        if (b == std::string::npos) {
            throw duckdb::InvalidInputException("semantic_endpoints contains an empty endpoint");
        }
        out.push_back(piece.substr(b, e - b + 1));
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    if (out.empty()) {
        throw duckdb::InvalidInputException("semantic_endpoints must contain at least one endpoint");
    }
    return out;
}

// Shared guardrail: warn (never error) when both knobs are pushed high enough
// to risk overwhelming the endpoint. Evaluated once per operator invocation in
// ResolveSemanticParams, on the resolved (model_args -> SET -> default) values.
void MaybeWarnGuardrail(int64_t in_flight_cap, int64_t batch_size) {
    if (in_flight_cap > SemanticDefaults::kGuardrailInFlightCap &&
        batch_size > SemanticDefaults::kGuardrailBatchSize) {
        duckdb::Printer::Print(
                duckdb::OutputStream::STREAM_STDERR,
                duckdb_fmt::format("Warning: semantic_in_flight_cap={} with semantic_batch_size={} may overwhelm the "
                                   "endpoint (cap>{}, batch>{}).",
                                   in_flight_cap, batch_size, SemanticDefaults::kGuardrailInFlightCap,
                                   SemanticDefaults::kGuardrailBatchSize));
    }
}

int64_t ReadIntSetting(ClientContext& context, const char* name, int64_t fallback) {
    Value v;
    if (context.TryGetCurrentSetting(name, v)) {
        return v.GetValue<int64_t>();
    }
    return fallback;
}

// --- set_option callbacks (plain function pointers -- no captures) ----------

void OnSetEndpoints(ClientContext& context, SetScope, Value& parameter) {
    ExtensionState::Get(context).router->SetEndpoints(SplitEndpoints(parameter.ToString()));
}

void OnSetRouting(ClientContext& context, SetScope, Value& parameter) {
    // ParseStrategy throws on an unknown strategy -> the SET fails.
    const auto strategy = EndpointRouter::ParseStrategy(parameter.ToString());
    ExtensionState::Get(context).router->SetStrategy(strategy);
}

// in_flight_cap and batch_size carry NO set_option callback: the high-N/high-R
// guardrail is evaluated once per invocation in ResolveSemanticParams (on the
// resolved values), not at SET time.

void OnSetResponseFormat(ClientContext&, SetScope, Value& parameter) {
    const std::string format = parameter.ToString();
    if (format != "json_schema" && format != "free_form") {
        throw duckdb::InvalidInputException(
                "semantic_response_format must be 'json_schema' or 'free_form', got '%s'", format);
    }
}

// Look up a model's persisted model_args JSON.
// Returns an empty object if the model does not exist;
// a missing model is not an error here; resolution falls through to SET/default.
nlohmann::json LookupModelArgs(const std::string& model_name) {
    try {
        auto con = Config::GetConnection();
        Config::StorageAttachmentGuard guard(con, true);
        auto result = con.Query(duckdb_fmt::format(
                " SELECT model_args"
                "   FROM flock_storage.flock_config.FLOCKMTL_MODEL_USER_DEFINED_INTERNAL_TABLE"
                "  WHERE model_name = '{}'"
                " UNION ALL "
                " SELECT model_args"
                "   FROM flock_config.FLOCKMTL_MODEL_USER_DEFINED_INTERNAL_TABLE"
                "  WHERE model_name = '{}'"
                " UNION ALL "
                " SELECT model_args"
                "   FROM flock_storage.flock_config.FLOCKMTL_MODEL_DEFAULT_INTERNAL_TABLE"
                "  WHERE model_name = '{}';",
                model_name, model_name, model_name));
        if (!result || result->HasError() || result->RowCount() == 0) {
            return nlohmann::json::object();
        }
        return nlohmann::json::parse(result->GetValue(0, 0).ToString());
    } catch (...) {
        // Tables not configured, parse failure, etc. -> behave as "no overrides".
        return nlohmann::json::object();
    }
}

}  // namespace

std::vector<std::string> DefaultEndpointList() { return SplitEndpoints(SemanticDefaults::kEndpoints); }

void RegisterSemanticSettings(DBConfig& config) {
    config.AddExtensionOption(semantic_option::kRewriteEnabled,
                              "Enable the semantic-operator query rewrite for this session.", LogicalType::BOOLEAN,
                              Value::BOOLEAN(SemanticDefaults::kRewriteEnabled), nullptr, SetScope::SESSION);

    // GLOBAL: mutates the single shared EndpointRouter, so SESSION scope would
    // misrepresent the blast radius.
    config.AddExtensionOption(semantic_option::kEndpoints,
                              "Comma-separated vLLM endpoint URLs (stored verbatim).", LogicalType::VARCHAR,
                              Value(SemanticDefaults::kEndpoints), OnSetEndpoints, SetScope::GLOBAL);

    config.AddExtensionOption(semantic_option::kRouting,
                              "Endpoint routing strategy: single|round_robin|sticky_by_prefix|least_loaded.",
                              LogicalType::VARCHAR, Value(SemanticDefaults::kRouting), OnSetRouting, SetScope::GLOBAL);

    config.AddExtensionOption(semantic_option::kInFlightCap,
                              "Maximum concurrent in-flight requests per semantic operator.", LogicalType::BIGINT,
                              Value::BIGINT(SemanticDefaults::kInFlightCap), nullptr, SetScope::SESSION);

    config.AddExtensionOption(semantic_option::kBatchSize,
                              "Rows packed into one multi-row LLM request (batch size); model_args overrides this.",
                              LogicalType::BIGINT, Value::BIGINT(SemanticDefaults::kBatchSize), nullptr,
                              SetScope::SESSION);

    config.AddExtensionOption(semantic_option::kCoalesceMaxAgeMs,
                              "Max time (ms) a partial coalesce batch waits before being flushed.", LogicalType::BIGINT,
                              Value::BIGINT(SemanticDefaults::kCoalesceMaxAgeMs), nullptr, SetScope::SESSION);

    config.AddExtensionOption(semantic_option::kResponseFormat,
                              "LLM response format: json_schema|free_form.", LogicalType::VARCHAR,
                              Value(SemanticDefaults::kResponseFormat), OnSetResponseFormat, SetScope::SESSION);

    config.AddExtensionOption(semantic_option::kMaxOutputTokens, "Max output tokens requested per LLM call.",
                              LogicalType::BIGINT, Value::BIGINT(SemanticDefaults::kMaxOutputTokens), nullptr,
                              SetScope::SESSION);
}

SemanticParams ResolveSemanticParams(ClientContext& context, const std::string& model_name) {
    // Base each field on the current SET value (which is the session SET, or the
    // registered default if the user never set it).
    SemanticParams params;
    params.in_flight_cap =
            static_cast<uint64_t>(ReadIntSetting(context, semantic_option::kInFlightCap, SemanticDefaults::kInFlightCap));
    params.batch_size =
            static_cast<uint64_t>(ReadIntSetting(context, semantic_option::kBatchSize, SemanticDefaults::kBatchSize));
    params.coalesce_max_age_ms = static_cast<uint64_t>(
            ReadIntSetting(context, semantic_option::kCoalesceMaxAgeMs, SemanticDefaults::kCoalesceMaxAgeMs));
    params.max_output_tokens = static_cast<uint64_t>(
            ReadIntSetting(context, semantic_option::kMaxOutputTokens, SemanticDefaults::kMaxOutputTokens));
    {
        Value v;
        params.response_format = context.TryGetCurrentSetting(semantic_option::kResponseFormat, v)
                                         ? v.ToString()
                                         : std::string(SemanticDefaults::kResponseFormat);
    }

    // Overlay any model_args overrides (highest precedence).
    const nlohmann::json args = LookupModelArgs(model_name);
    if (args.contains("in_flight_cap")) {
        params.in_flight_cap = args["in_flight_cap"].get<uint64_t>();
    }
    if (args.contains("batch_size")) {
        params.batch_size = args["batch_size"].get<uint64_t>();
    }
    if (args.contains("coalesce_max_age_ms")) {
        params.coalesce_max_age_ms = args["coalesce_max_age_ms"].get<uint64_t>();
    }
    if (args.contains("max_output_tokens")) {
        params.max_output_tokens = args["max_output_tokens"].get<uint64_t>();
    }
    if (args.contains("response_format")) {
        params.response_format = args["response_format"].get<std::string>();
    }

    // Authoritative range validation: this is the single chokepoint both the SET
    // surface and model_args flow through. Fail loud rather than clamp -> an
    // out-of-range value is a misconfiguration worth surfacing.
    if (params.batch_size < 1 || params.batch_size > static_cast<uint64_t>(STANDARD_VECTOR_SIZE)) {
        throw duckdb::InvalidInputException("semantic batch_size must be in [1, " +
                                            std::to_string(STANDARD_VECTOR_SIZE) + "], got " +
                                            std::to_string(params.batch_size));
    }
    if (params.response_format == "free_form" && params.batch_size != 1) {
        // Without guided decoding there is no per-row envelope to split a
        // multi-row completion by.
        throw duckdb::InvalidInputException(
                "semantic_response_format='free_form' requires semantic_batch_size=1, got " +
                std::to_string(params.batch_size));
    }
    if (params.in_flight_cap < 1) {
        throw duckdb::InvalidInputException("semantic in_flight_cap must be >= 1, got " +
                                            std::to_string(params.in_flight_cap));
    }

    // Relocated guardrail: warn once here on the fully-resolved values.
    MaybeWarnGuardrail(static_cast<int64_t>(params.in_flight_cap), static_cast<int64_t>(params.batch_size));
    return params;
}

}  // namespace flock