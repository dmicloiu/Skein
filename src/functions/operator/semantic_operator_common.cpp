#include "flock/functions/operator/semantic_operator_common.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "flock/prompt_manager/prompt_manager.hpp"

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <string>

namespace flock {

using duckdb::Expression;
using duckdb::ExpressionClass;
using duckdb::idx_t;
using duckdb::LogicalType;
using duckdb::StructType;

namespace {

// Find the child expression of a bound struct literal by its field name.
Expression* FindStructField(const duckdb::BoundFunctionExpression& struct_expr, const LogicalType& struct_type,
                            const std::string& field_name) {
    const idx_t fields = StructType::GetChildCount(struct_type);
    for (idx_t i = 0; i < fields && i < struct_expr.children.size(); ++i) {
        if (StructType::GetChildName(struct_type, i) == field_name) {
            return struct_expr.children[i].get();
        }
    }
    return nullptr;
}

}  // namespace

std::vector<SemContextColumn> ExtractContextColumns(duckdb::ClientContext& context,
                                                    const duckdb::BoundFunctionExpression& llm_call,
                                                    const char* fn_label) {
    const std::string label(fn_label);
    if (llm_call.children.size() < 2) {
        throw duckdb::InvalidInputException(label + ": llm call expects (model, prompt) arguments");
    }
    auto& prompt_expr = *llm_call.children[1];
    if (prompt_expr.expression_class != ExpressionClass::BOUND_FUNCTION) {
        throw duckdb::NotImplementedException(label + ": prompt argument must be a struct literal");
    }
    auto& prompt_struct = prompt_expr.Cast<duckdb::BoundFunctionExpression>();
    auto* cc_expr = FindStructField(prompt_struct, prompt_expr.return_type, "context_columns");
    if (!cc_expr) {
        throw duckdb::NotImplementedException(label + ": llm call requires context_columns");
    }
    if (cc_expr->expression_class != ExpressionClass::BOUND_FUNCTION) {
        throw duckdb::NotImplementedException(label + ": context_columns must be a list literal");
    }
    auto& list_expr = cc_expr->Cast<duckdb::BoundFunctionExpression>();

    std::vector<SemContextColumn> columns;
    for (auto& element : list_expr.children) {
        if (element->expression_class != ExpressionClass::BOUND_FUNCTION) {
            throw duckdb::NotImplementedException(label + ": each context column must be a struct literal");
        }
        auto& col_struct = element->Cast<duckdb::BoundFunctionExpression>();
        const auto& col_type = element->return_type;

        SemContextColumn cc;
        bool has_data = false;
        const idx_t fields = StructType::GetChildCount(col_type);
        for (idx_t j = 0; j < fields && j < col_struct.children.size(); ++j) {
            const auto field = StructType::GetChildName(col_type, j);
            auto& child = *col_struct.children[j];
            if (field == "data") {
                if (child.expression_class != ExpressionClass::BOUND_REF) {
                    throw duckdb::NotImplementedException(
                            label + ": context column 'data' must be a direct column reference");
                }
                auto& ref = child.Cast<duckdb::BoundReferenceExpression>();
                cc.data_index = ref.index;
                cc.data_type = ref.return_type;
                has_data = true;
            } else {
                // Static metadata (name/type/detail): fold the constant, drop NULLs.
                if (!child.IsFoldable()) {
                    throw duckdb::NotImplementedException(label + ": context column metadata must be constant");
                }
                auto value = duckdb::ExpressionExecutor::EvaluateScalar(context, child);
                if (value.IsNull()) {
                    continue;
                }
                auto str = value.ToString();
                if (str == "NULL") {
                    continue;
                }
                if (field == "type" && (str == "image" || str == "audio")) {
                    throw duckdb::NotImplementedException(label + ": image/audio context columns are not supported");
                }
                cc.metadata[field] = str;
            }
        }
        if (!has_data) {
            throw duckdb::InvalidInputException(label + ": context column missing 'data'");
        }
        columns.push_back(std::move(cc));
    }
    if (columns.empty()) {
        throw duckdb::NotImplementedException(label + ": llm call requires at least one context column");
    }
    return columns;
}

nlohmann::json BuildContextColumnsJson(const std::vector<SemContextColumn>& context_columns,
                                       const std::vector<RowData>& batch) {
    auto columns = nlohmann::json::array();
    for (const auto& cc : context_columns) {
        nlohmann::json column = cc.metadata;
        auto data = nlohmann::json::array();
        for (const auto& row : batch) {
            data.push_back(row.values[cc.data_index].ToString());
        }
        column["data"] = std::move(data);
        columns.push_back(std::move(column));
    }
    return columns;
}

SemVariants ParseSemVariants(const std::string& csv) {
    SemVariants v;
    std::stringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            continue;
        }
        if (token == "rowmajor") {
            v.rowmajor = true;
        } else if (token == "sandwich") {
            v.sandwich = true;
        } else if (token == "symmetric") {
            v.symmetric = true;
        } else if (token == "count") {
            v.count = true;
        } else if (token == "example") {
            v.example = true;
        } else if (token == "chunk") {
            v.chunk = true;
            v.rowmajor = true;  // chunk headers only make sense over row-major rows
        } else {
            throw std::runtime_error("FLOCK_SEM_VARIANTS: unknown variant '" + token +
                                     "' (expected rowmajor|sandwich|symmetric|count|example|chunk)");
        }
    }
    return v;
}

const SemVariants& SemVariantsFromEnv() {
    static const SemVariants v = [] {
        const char* p = std::getenv("FLOCK_SEM_VARIANTS");
        return ParseSemVariants(p ? p : "");
    }();
    return v;
}

namespace {

constexpr int kChunkRows = 8;  // P6: rows per "### Rows a-b" group

// Column display name, matching ConstructInputTuplesJSON's fallback numbering.
std::string ColumnName(const nlohmann::json& column, unsigned& fallback_idx) {
    if (column.contains("name") && column["name"].is_string()) {
        return column["name"].get<std::string>();
    }
    return "COLUMN " + std::to_string(fallback_idx++);
}

// P1/P6 tuple block: one compact {"id": k, "<col>": "<val>", ...} object per
// line (id first; built as a string because nlohmann::json sorts keys). With
// chunk, a "### Rows a-b" header every kChunkRows lines re-anchors attention.
std::string RowMajorTuples(const nlohmann::json& columns, const SemVariants& v) {
    const int n = columns.empty() ? 0 : static_cast<int>(columns[0]["data"].size());
    std::string out;
    for (int i = 0; i < n; ++i) {
        if (v.chunk && i % kChunkRows == 0) {
            out += "### Rows " + std::to_string(i + 1) + "-" +
                   std::to_string(std::min(n, i + kChunkRows)) + "\n";
        }
        out += "{\"id\": " + std::to_string(i + 1);
        unsigned fallback_idx = 1;
        for (const auto& column : columns) {
            const auto& item = column["data"][i];
            const std::string value = item.is_string() ? item.get<std::string>() : item.dump();
            out += ", " + nlohmann::json(ColumnName(column, fallback_idx)).dump() + ": " +
                   nlohmann::json(value).dump();
        }
        out += "}\n";
    }
    return out;
}

// Tuple body in the model-configured format, WITHOUT the count line (the
// variant path owns that line). Mirrors ConstructInputTuples' dispatch.
std::string TupleBody(const nlohmann::json& columns, const std::string& tuple_format) {
    auto body = PromptManager::ConstructInputTuplesHeader(columns, tuple_format);
    switch (stringToTupleFormat(tuple_format)) {
        case TupleFormat::XML:
            return body + PromptManager::ConstructInputTuplesXML(columns);
        case TupleFormat::Markdown:
            return body + PromptManager::ConstructInputTuplesMarkdown(columns);
        case TupleFormat::JSON:
            return body + PromptManager::ConstructInputTuplesJSON(columns);
        default:
            throw std::runtime_error("Invalid tuple format provided `" + tuple_format + "`");
    }
}

}  // namespace

std::string RenderSlimSemanticPrompt(SlimKind kind, const std::string& user_prompt,
                                     const nlohmann::json& columns, const std::string& tuple_format,
                                     const SemVariants& v, bool state_output_shape) {
    const bool filter = kind == SlimKind::kFilter;
    const char* unit = filter ? "boolean" : "string";
    const int n = columns.empty() ? 0 : static_cast<int>(columns[0]["data"].size());

    // Head. The variant-free filter head is byte-identical to the committed slim
    // prompt; results stay comparable across sweeps.
    std::string prompt;
    if (filter) {
        if (v.symmetric) {
            prompt = "For each row in the table below, output true if it satisfies the criterion and "
                     "false if it does not. Judge every row independently; do not let earlier rows "
                     "influence later ones.\n"
                     "Criterion: " + user_prompt + "\n\n";
        } else {
            prompt = "For each row in the table below, decide whether it satisfies the criterion, "
                     "judging every row independently on its own merits.\n"
                     "Criterion: " + user_prompt + "\n\n";
        }
    } else {
        prompt = "For each row in the table below, produce the output the task requests, "
                 "treating every row independently on its own merits.\n"
                 "Task: " + user_prompt + "\n\n";
    }

    // P5: fixed worked example (deliberately generic: an unrelated criterion/task,
    // so it teaches the row->item alignment convention, not the benchmark).
    if (v.example) {
        if (filter) {
            prompt += "Example with a different criterion (\"the text mentions an animal\") and 2 rows:\n"
                      "{\"id\": 1, \"text\": \"The cat sat on the mat.\"}\n"
                      "{\"id\": 2, \"text\": \"Traffic was heavy today.\"}\n"
                      "Correct output: {\"items\": [true, false]}\n\n";
        } else {
            prompt += "Example with a different task (\"name the color mentioned\") and 2 rows:\n"
                      "{\"id\": 1, \"text\": \"The sky turned deep red at dusk.\"}\n"
                      "{\"id\": 2, \"text\": \"She wore a blue coat.\"}\n"
                      "Correct output: {\"items\": [\"red\", \"blue\"]}\n\n";
        }
    }

    // Tuples. Variant-free: the legacy block (count line + header + body),
    // byte-identical to ConstructInputTuples.
    if (v.rowmajor || v.count) {
        if (v.count) {
            prompt += "The table has " + std::to_string(n) + " rows. Return exactly " +
                      std::to_string(n) + " " + unit + "s; the i-th " + unit +
                      " answers row i.\n\n";
        } else {
            prompt += PromptManager::ConstructNumTuples(n);
        }
        prompt += v.rowmajor ? RowMajorTuples(columns, v) : TupleBody(columns, tuple_format);
    } else {
        prompt += PromptManager::ConstructInputTuples(columns, tuple_format);
    }

    // P2: restate the criterion/task right after the rows it applies to.
    if (v.sandwich) {
        prompt += filter ? "\nReminder of the criterion: " + user_prompt + "\n"
                         : "\nReminder of the task: " + user_prompt + "\n";
    }

    if (state_output_shape) {
        prompt += std::string("\n\nReturn a JSON object {\"items\": [...]} with one ") + unit +
                  " per row, in row order.";
    }
    return prompt;
}

}  // namespace flock
