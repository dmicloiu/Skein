#pragma once

#include "flock/functions/operator/proxy_policy.hpp"
#include "flock/runtime/llm_client.h"
#include "flock/runtime/pending_request.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace flock {

// Result of planning the embedding proxy over an operator's buffered input.
struct ProxyPlan {
    std::unique_ptr<ProxyPolicy> policy;  // null -> run the operator exactly (every row to the main model)
    std::vector<size_t> row_unit;         // row position (input order) -> unit index
    size_t num_units = 0;
    std::string info;                     // one-line note for the run log (also the fallback reason)
};

// Buffered first stage of a semantic filter (see proxy_policy.hpp). Plan runs
// ONCE, in Finalize, over all input rows; it may block on HTTP calls through
// `client` (tagged with `generation`, so query teardown cancels them).
class ProxyStage {
public:
    virtual ~ProxyStage() = default;
    virtual ProxyPlan Plan(const std::vector<PendingRequest>& rows, ILLMClient& client, uint64_t generation) = 0;
};

}  // namespace flock
