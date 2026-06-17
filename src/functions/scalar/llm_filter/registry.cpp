#include "flock/registry/registry.hpp"
#include "flock/functions/scalar/llm_filter.hpp"

namespace flock {

void ScalarRegistry::RegisterLlmFilter(duckdb::ExtensionLoader& loader) {
    // VOLATILE (an LLM verdict is non-deterministic) so filter pushdown does not fold
    // a single-column WHERE llm_filter(col) into the scan, letting the semantic-operator
    // rewrite engage. Mirrors the fusion_* registrations (positional stability arg);
    // varargs stays INVALID to keep the fixed 2-arg arity. Bind/Execute are unchanged.
    loader.RegisterFunction(duckdb::ScalarFunction("llm_filter",
                                                   {duckdb::LogicalType::ANY, duckdb::LogicalType::ANY},
                                                   duckdb::LogicalType::VARCHAR, LlmFilter::Execute,
                                                   LlmFilter::Bind, nullptr, nullptr, nullptr,
                                                   duckdb::LogicalType(duckdb::LogicalTypeId::INVALID),
                                                   duckdb::FunctionStability::VOLATILE));
}

}// namespace flock
