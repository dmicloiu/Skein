#include "flock/registry/registry.hpp"
#include "flock/functions/scalar/llm_complete.hpp"

namespace flock {

void ScalarRegistry::RegisterLlmComplete(duckdb::ExtensionLoader& loader) {
    // VOLATILE (an LLM completion is non-deterministic) so a projected llm_complete
    // is not const-folded / CSE'd away, letting the semantic-operator rewrite engage
    // on the surviving LogicalProjection.
    loader.RegisterFunction(duckdb::ScalarFunction("llm_complete",
                                                   {duckdb::LogicalType::ANY, duckdb::LogicalType::ANY},
                                                   duckdb::LogicalType::JSON(), LlmComplete::Execute,
                                                   LlmComplete::Bind, nullptr, nullptr, nullptr,
                                                   duckdb::LogicalType(duckdb::LogicalTypeId::INVALID),
                                                   duckdb::FunctionStability::VOLATILE));
}

}// namespace flock
