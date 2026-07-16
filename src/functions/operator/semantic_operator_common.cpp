#include "flock/functions/operator/semantic_operator_common.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

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

}  // namespace flock
