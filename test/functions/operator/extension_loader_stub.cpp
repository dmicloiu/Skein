// Link stub for the threading test.
//
// The test links duckdb_static only for its core types (DataChunk / Value /
// InterruptState / Allocator) and never opens a DuckDB. But duckdb_static's
// DuckDB constructor references ExtensionHelper::LoadAllExtensions, which is
// normally provided by duckdb_generated_extension_loader. This
// definition resolves the symbol so the link succeeds; it is never called.
#include "duckdb/main/extension_helper.hpp"

namespace duckdb {

void ExtensionHelper::LoadAllExtensions(DuckDB &) {
}

}  // namespace duckdb