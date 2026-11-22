#define DUCKDB_EXTENSION_MAIN

#include "skein_extension.hpp"

#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/statement/extension_statement.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "flock/core/common.hpp"
#include "flock/core/config.hpp"
#include "flock/custom_parser/query_parser.hpp"
#include "flock/functions/operator/optimizer_rewrite.hpp"
#include "flock/runtime/semantic_settings.h"

#include <flock/model_manager/model.hpp>

namespace duckdb {

static void LoadInternal(ExtensionLoader& loader) {
    flock::Config::Configure(loader);

    // Register parser and binder hooks using extension registration APIs.
    auto& config = DBConfig::GetConfig(loader.GetDatabaseInstance());
    flock::RegisterSemanticSettings(config);

    // Plan rewrite (llm_filter WHERE predicates -> LogicalSemFilter). Runs after
    // DuckDB's optimizers; gated on the semantic_rewrite_enabled setting.
    OptimizerExtension semantic_rewrite;
    semantic_rewrite.optimize_function = flock::SemanticOptimizeFunction;
    OptimizerExtension::Register(config, semantic_rewrite);

    DuckParserExtension duck_parser;
    ParserExtension::Register(config, duck_parser);
    OperatorExtension::Register(config, make_shared_ptr<DuckOperatorExtension>());
}

ParserExtensionParseResult duck_parse(ParserExtensionInfo*, const vector<SimpleToken>& tokens) {
    if (tokens.empty()) {
        return {};
    }
    // GET is not a SQL keyword, so match on text rather than token type
    auto first = StringUtil::Upper(tokens[0].text);
    if (first != "CREATE" && first != "DELETE" && first != "UPDATE" && first != "GET") {
        return {};
    }

    std::string query;
    idx_t end = 0;
    for (; end < tokens.size(); end++) {
        auto type = tokens[end].type;
        if (type == TokenType::TERMINATOR || type == TokenType::END_OF_INPUT ||
            type == TokenType::END_OF_INPUT_AUTOCOMPLETE) {
            break;
        }
        if (type == TokenType::COMMENT) {
            continue;
        }
        if (!query.empty()) {
            query += " ";
        }
        query += tokens[end].text;
    }

    flock::QueryParser query_parser;

    // Translate and print SQL queries for each input query
    std::string sql_query = query_parser.ParseQuery(query);

    // Parse and return the statement using DuckDB's parser
    auto parser = Parser::GetBuiltinParser();
    parser.ParseQuery(sql_query);
    auto statements = std::move(parser.statements);

    ParserExtensionParseResult result(
            make_uniq_base<ParserExtensionParseData, DuckParseData>(std::move(statements[0])));
    // Also claim the terminator, as a top-level statement would.
    result.consumed_tokens = NumericCast<int64_t>(end < tokens.size() ? end + 1 : end);
    return result;
}

ParserExtensionPlanResult duck_plan(ParserExtensionInfo*, ClientContext& context,
                                    unique_ptr<ParserExtensionParseData> parse_data) {
    if (auto state = context.registered_state->Get<DuckState>("duck")) {
        context.registered_state->Remove("duck");
    }
    context.registered_state->GetOrCreate<DuckState>("duck", std::move(parse_data));
    throw BinderException("Use duck_bind instead");
}

BoundStatement duck_bind(ClientContext& context, Binder& binder, OperatorExtensionInfo* info, SQLStatement& statement) {
    switch (statement.type) {
        case StatementType::EXTENSION_STATEMENT: {
            auto& extension_statement = dynamic_cast<ExtensionStatement&>(statement);
            if (extension_statement.extension.parse_function == duck_parse) {
                if (const auto duck_state = context.registered_state->Get<DuckState>("duck")) {
                    const auto duck_binder = Binder::CreateBinder(context, &binder);
                    const auto duck_parse_data = dynamic_cast<DuckParseData*>(duck_state->parse_data.get());
                    auto bound_statement = duck_binder->Bind(*(duck_parse_data->statement));
                    BoundStatement result;
                    return bound_statement;
                }
                throw BinderException("Registered state not found");
            }
        }
        default:
            // No-op empty
            return {};
    }
}

void SkeinExtension::Load(ExtensionLoader& loader) { LoadInternal(loader); }

std::string SkeinExtension::Name() { return "skein"; }
std::string SkeinExtension::Version() const { return "v0.3.0"; }

}// namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(skein, loader) {
    duckdb::LoadInternal(loader);
}
}
