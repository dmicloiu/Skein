#include "flock/runtime/extension_state.h"

#include "flock/runtime/semantic_settings.h"

#include <memory>

namespace flock {

ExtensionState::ExtensionState()
    : client(std::make_shared<AsyncLLMClient>()),
      router(std::make_shared<EndpointRouter>(DefaultEndpointList(),
                                              EndpointRouter::ParseStrategy(SemanticDefaults::kRouting))) {}

ExtensionState& ExtensionState::Get(duckdb::DatabaseInstance& db) {
    // GetOrCreate locks internally and, because the entry is non-evictable,
    // returns the same instance for the lifetime of the database.
    auto entry = db.GetObjectCache().GetOrCreate<ExtensionState>(ObjectType());
    return *entry;
}

ExtensionState& ExtensionState::Get(duckdb::ClientContext& context) {
    return Get(duckdb::DatabaseInstance::GetDatabase(context));
}

std::string ExtensionState::ObjectType() { return "flock_extension_state"; }

std::string ExtensionState::GetObjectType() { return ObjectType(); }

duckdb::optional_idx ExtensionState::GetEstimatedCacheMemory() const {
    // Invalid index -> non-evictable; the entry is pinned for the DB lifetime.
    return duckdb::optional_idx();
}

}  // namespace flock