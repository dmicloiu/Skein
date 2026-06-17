#pragma once

#include "duckdb/optimizer/optimizer_extension.hpp"

namespace flock {

// OptimizerExtension callback (a plain function pointer). Runs
// AFTER DuckDB's built-in optimizers and rewrites WHERE-filter `llm_filter`
// predicates into stacked LogicalSemFilter nodes. Registered once from
// LoadInternal via OptimizerExtension::Register. Gated on the
// semantic_rewrite_enabled SESSION setting (false -> exact no-op, the A/B
// baseline). Purely structural: no model/provider resolution happens here
// (CreatePlan resolves config when the physical plan is built).
void SemanticOptimizeFunction(duckdb::OptimizerExtensionInput& input,
                              duckdb::unique_ptr<duckdb::LogicalOperator>& plan);

}  // namespace flock