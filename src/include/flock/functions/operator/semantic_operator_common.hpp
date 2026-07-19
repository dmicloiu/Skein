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

// Slim-prompt variant flags (accuracy-tweak ablation; see
// analysis/prompt_engineering.md Q3). Effective only with FLOCK_SEM_PROMPT=slim.
struct SemVariants {
    bool rowmajor = false;   // P1: row-major tuples, one {"id": k, ...} object per line
    bool sandwich = false;   // P2: restate the criterion/task after the table
    bool symmetric = false;  // P3: symmetric true/false framing (filter only)
    bool count = false;      // P4: explicit "N rows, i-th answer" count line
    bool example = false;    // P5: fixed 2-row worked example before the table
    bool chunk = false;      // P6: "### Rows a-b" headers every 8 rows (implies rowmajor)
    bool Any() const { return rowmajor || sandwich || symmetric || count || example || chunk; }
};

// Parse a comma-separated variant list ("rowmajor,sandwich"). Unknown tokens
// throw (a typo silently no-oping would corrupt a sweep). chunk sets rowmajor.
SemVariants ParseSemVariants(const std::string& csv);

// Cached FLOCK_SEM_VARIANTS (read once per process, like the other env gates).
const SemVariants& SemVariantsFromEnv();

// Which semantic operator a slim prompt is rendered for; selects the wording
// (criterion/boolean vs task/string).
enum class SlimKind { kFilter, kExtract };

// Render the lean text-only prompt: head + [example] + tuples + [reminder] +
// output-shape tail. With all variants off and kind=kFilter this is
// byte-identical to the committed slim prompt (baseline continuity).
// state_output_shape: emit the trailing "one boolean/string per row" line --
// the filter's id/id_reason schema modes append their own output block instead.
std::string RenderSlimSemanticPrompt(SlimKind kind, const std::string& user_prompt,
                                     const nlohmann::json& columns, const std::string& tuple_format,
                                     const SemVariants& variants, bool state_output_shape);

}  // namespace flock