#pragma once

#include "duckdb/storage/object_cache.hpp"  // ObjectCache, ObjectCacheEntry
#include "flock/runtime/async_llm_client.h"
#include "flock/runtime/endpoint_router.h"

#include <memory>
#include <string>

namespace flock {

// Database-lifetime place for the shared async runtime objects.
//
// Lives in DatabaseInstance::GetObjectCache() [one instance per
// DatabaseInstance], NOT ClientContext::registered_state, which is
// per-connection and would give each connection its own libcurl IO thread and
// router. The single AsyncLLMClient (one IO thread) and the single
// EndpointRouter must be shared across all connections to the database.
//
// The entry is non-evictable (GetEstimatedCacheMemory returns an invalid index)
// so it persists for the database's lifetime.
class ExtensionState : public duckdb::ObjectCacheEntry {
public:
    // Constructs the client and router with defaults from SemanticDefaults.
    ExtensionState();

    std::shared_ptr<AsyncLLMClient> client;
    std::shared_ptr<EndpointRouter> router;

    // Get-or-create the singleton for the database behind this context/instance.
    // ObjectCache::GetOrCreate is internally locked, so creation is race-free.
    static ExtensionState& Get(duckdb::ClientContext& context);
    static ExtensionState& Get(duckdb::DatabaseInstance& db);

    static std::string ObjectType();
    std::string GetObjectType() override;

    // Non-evictable: invalid index keeps this entry pinned for the DB lifetime.
    duckdb::optional_idx GetEstimatedCacheMemory() const override;
};

}  // namespace flock