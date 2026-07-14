#pragma once

#include "duckdb/common/types.hpp"
#include "flock/runtime/pending_request.h"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace duckdb {
class ClientContext;
class BoundFunctionExpression;
}  // namespace duckdb

namespace flock {

// Shared building blocks for the Tier-3 semantic operators (sem_filter,
// sem_extract, ...): the prompt-context-column representation, its extraction
// from a bound llm_* call, and the batch column-JSON builder. Outside the 
// engine for separation of concerns.

// One prompt-context column: where its per-row value lives in the captured row,
// the underlying DuckDB type, plus the static metadata (name/type/detail)
// carried verbatim into the rendered batch.
struct SemContextColumn {
    duckdb::idx_t data_index = 0;   // index into RowData.values
    duckdb::LogicalType data_type;  // the source column's type (for re-projection)
    nlohmann::json metadata;        // {name?, type?, detail?} -- never holds "data"
};

// Extract the prompt-context columns from a bound llm_* call's prompt struct (its
// 2nd argument). v1 supports tabular columns whose `data` is a direct column
// reference; anything else (computed data, image/audio) is rejected loudly.
// `fn_label` prefixes error messages (e.g. "sem_filter" / "sem_extract").
// `data_index` is set to the source BoundReferenceExpression index; callers that
// re-project the row may overwrite it with the captured position.
std::vector<SemContextColumn> ExtractContextColumns(duckdb::ClientContext& context,
                                                    const duckdb::BoundFunctionExpression& llm_call,
                                                    const char* fn_label);

// Rebuild the scalar's context_columns JSON for one batch: metadata verbatim,
// data = each row's value at cc.data_index stringified. Byte-parity with the
// scalar path, so a prompt rendered from this matches llm_filter / llm_complete
// at the same batch size.
nlohmann::json BuildContextColumnsJson(const std::vector<SemContextColumn>& context_columns,
                                       const std::vector<RowData>& batch);

}  // namespace flock