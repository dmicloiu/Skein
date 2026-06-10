#pragma once

#include "duckdb/common/types/value.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace flock {

// A full-row COPY of one input row.
//
// INVARIANT: this is a deep copy of the row's values (never a reference or
// view into a DataChunk). The producing chunk is recycled the instant Sink
// returns NEED_MORE_INPUT, yet a row must survive across an async HTTP
// round-trip before it is emitted. duckdb::Value owns its payload (including the
// string heap), so a vector<Value> is a safe, self-contained snapshot.
struct RowData {
    std::vector<duckdb::Value> values;
};

// One row queued for dispatch, held in submission order.
//
// INVARIANT: row_id is globally unique per query (a monotonic atomic in the
// sink state), NOT chunk-local. That is what keeps the end-to-end alignment
//     prompt[i] <-> row_id[i] <-> row[i] <-> choices[i]
// intact while batches are coalesced across many threads and chunks.
struct PendingRequest {
    uint64_t row_id = 0;
    std::string prompt;
    RowData row;
};

}  // namespace flock