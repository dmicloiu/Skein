// sem_extract operator throughput driver: run ONE (threads, rewrite) config
// against a live vLLM fleet and write a --result-out JSON, so the off-cluster
// aggregation reuses the sem_filter driver's path/flag conventions (the morsel
// A/B summarizer, analysis/summarize_morsel_ab.py, reads the same schema).
//
// Sibling of sem_filter_vllm_integration.cpp: same embedded DuckDB + full SQL
// chain, same single A/B knob (semantic_rewrite_enabled), but the workload is a
// PROJECTED llm_complete (extraction/classification) instead of a WHERE filter:
//   * --rewrite on  -> PhysicalSemExtract (the operator) subsumes the projection
//                      and posts batched chat requests via the EndpointRouter.
//   * --rewrite off -> the scalar llm_complete posts /chat/completions via the
//                      openai provider (base_url derived from --endpoints).
// Same query, same vLLM, same batch_size; only the rewrite flag differs.
//
// The timed op materialises the projection (CREATE TABLE ... AS SELECT ...,
// llm_complete(...)) so BOTH arms fully execute every row (a projection emits
// every row -- no filtering). Only that CREATE-TABLE-AS is timed; table/secret/
// model setup + burn-in are excluded.
//
//   flock_sem_extract_vllm_integration
//       --endpoints http://127.0.0.1:8000/v1/chat/completions
//       --model Qwen/Qwen2.5-7B-Instruct
//       --data Reviews.csv --text-col reviewText
//       --prompt 'Classify the sentiment of this movie review as exactly POSITIVE or NEGATIVE.'
//       --rows 262144 --rewrite on --threads 128
//       --inflight 128 --rows-per-request 32 --timeout-ms 120000
//       --row-group-size 2048 --attach-db /tmp/sem_extract.db
//       --result-out result.json

#include "duckdb.hpp"
#include "flock/core/config.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

namespace {

struct Args {
    std::string endpoints_csv;
    std::string model = "Qwen/Qwen2.5-7B-Instruct";
    std::string data;
    std::string text_col = "reviewText";
    std::string prompt = "Classify the sentiment of this movie review as exactly POSITIVE or NEGATIVE.";
    int rows = 2000;
    std::string rewrite = "on";  // on|off
    int threads = 8;
    int inflight = 128;
    int rows_per_request = 32;  // == batch_size (operator) / model batch_size (scalar)
    int timeout_ms = 120000;
    std::string result_out;
    // Morsel control (identical semantics to the filter driver): 0 = in-memory
    // table => one row group => one morsel => single-threaded scan. >0 attaches an
    // on-disk DB with this ROW_GROUP_SIZE (multiple of 2048) so the scan sees
    // rows/row_group_size morsels and the scalar parallelises to min(threads,
    // morsels); the operator stays cap-bound. attach_db backs that DB (fresh).
    int row_group_size = 0;
    std::string attach_db;
    // Per-row output-token budget multiplier: max_output_tokens = mult * R. 16 is
    // ample for a short classification label array (a couple tokens/row).
    int max_out_mult = 16;
    // Tuple encoding of the {{TUPLES}} block. The model default is XML; json is the
    // canonical encoding for the evaluation (and what the sembench cross-system
    // harness uses), so default to it and keep the knob for an XML comparison.
    std::string tuple_format = "json";
    // Skip the untimed burn-in query (never on a timed run: burn-in warms the
    // xgrammar kernel and is what makes the throughput number clean).
    bool skip_burn_in = false;
};

void Usage(const char* prog) {
    std::fprintf(stderr,
                 "Usage: %s --endpoints URL1,URL2,... --model NAME --data PATH\n"
                 "          [--text-col NAME] [--prompt TEXT] [--rows N]\n"
                 "          --rewrite on|off [--threads T] [--inflight N]\n"
                 "          [--rows-per-request N] [--timeout-ms N] [--result-out PATH]\n"
                 "          [--row-group-size N (mult. of 2048; >0 => rows/N morsels)]\n"
                 "          [--attach-db PATH (on-disk DB backing the morsel layout)]\n"
                 "          [--skip-burn-in (untimed only)]\n"
                 "          [--max-out-mult N (max_output_tokens = N*R; default 16)]\n"
                 "          [--tuple-format json|XML|Markdown (default json)]\n",
                 prog);
}

bool ParseArgs(int argc, char** argv, Args* a) {
    for (int i = 1; i < argc; ++i) {
        const char* k = argv[i];
        auto need = [&](const char* name) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (!std::strcmp(k, "--endpoints")) a->endpoints_csv = need("--endpoints");
        else if (!std::strcmp(k, "--model")) a->model = need("--model");
        else if (!std::strcmp(k, "--data")) a->data = need("--data");
        else if (!std::strcmp(k, "--text-col")) a->text_col = need("--text-col");
        else if (!std::strcmp(k, "--prompt")) a->prompt = need("--prompt");
        else if (!std::strcmp(k, "--rows")) a->rows = std::atoi(need("--rows"));
        else if (!std::strcmp(k, "--rewrite")) a->rewrite = need("--rewrite");
        else if (!std::strcmp(k, "--threads")) a->threads = std::atoi(need("--threads"));
        else if (!std::strcmp(k, "--inflight")) a->inflight = std::atoi(need("--inflight"));
        else if (!std::strcmp(k, "--rows-per-request")) a->rows_per_request = std::atoi(need("--rows-per-request"));
        else if (!std::strcmp(k, "--timeout-ms")) a->timeout_ms = std::atoi(need("--timeout-ms"));
        else if (!std::strcmp(k, "--result-out")) a->result_out = need("--result-out");
        else if (!std::strcmp(k, "--row-group-size")) a->row_group_size = std::atoi(need("--row-group-size"));
        else if (!std::strcmp(k, "--attach-db")) a->attach_db = need("--attach-db");
        else if (!std::strcmp(k, "--skip-burn-in")) a->skip_burn_in = true;
        else if (!std::strcmp(k, "--max-out-mult")) a->max_out_mult = std::atoi(need("--max-out-mult"));
        else if (!std::strcmp(k, "--tuple-format")) a->tuple_format = need("--tuple-format");
        else if (!std::strcmp(k, "-h") || !std::strcmp(k, "--help")) {
            Usage(argv[0]);
            return false;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", k);
            Usage(argv[0]);
            return false;
        }
    }
    if (a->endpoints_csv.empty()) {
        std::fprintf(stderr, "--endpoints is required\n");
        return false;
    }
    if (a->data.empty()) {
        std::fprintf(stderr, "--data is required\n");
        return false;
    }
    if (a->rewrite != "on" && a->rewrite != "off") {
        std::fprintf(stderr, "--rewrite must be 'on' or 'off'\n");
        return false;
    }
    if (a->row_group_size != 0) {
        if (a->row_group_size % 2048 != 0) {
            std::fprintf(stderr, "--row-group-size must be a multiple of 2048 (the vector size)\n");
            return false;
        }
        if (a->attach_db.empty()) {
            std::fprintf(stderr, "--row-group-size requires --attach-db PATH (the on-disk DB to back it)\n");
            return false;
        }
    }
    return true;
}

// Double single quotes so the value is safe inside a single-quoted SQL literal.
std::string SqlEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (const char c : s) {
        if (c == '\'') out += "''";
        else out += c;
    }
    return out;
}

// The scalar path's openai base_url. The operator posts to <endpoint> verbatim
// (e.g. .../v1/chat/completions); the openai provider instead appends
// /chat/completions to its base_url, so strip that suffix from the first endpoint
// to recover the shared base (e.g. .../v1).
std::string DeriveBaseUrl(const std::string& endpoints_csv) {
    std::string first = endpoints_csv.substr(0, endpoints_csv.find(','));
    for (const std::string& suffix : {std::string("/chat/completions"), std::string("/completions")}) {
        if (first.size() >= suffix.size() &&
            first.compare(first.size() - suffix.size(), suffix.size(), suffix) == 0) {
            first.erase(first.size() - suffix.size());
            break;
        }
    }
    return first;
}

bool Run(duckdb::Connection& con, const std::string& sql) {
    auto res = con.Query(sql);
    if (res->HasError()) {
        std::fprintf(stderr, "SQL failed: %s\n  -> %s\n", sql.c_str(), res->GetError().c_str());
        return false;
    }
    return true;
}

void WriteResultJson(const Args& a, long long emitted, long long rows_loaded, double elapsed_s) {
    if (a.result_out.empty()) return;
    std::ofstream out(a.result_out);
    if (!out) {
        std::fprintf(stderr, "cannot write result file: %s\n", a.result_out.c_str());
        return;
    }
    const double rows_per_s = elapsed_s > 0.0 ? static_cast<double>(rows_loaded) / elapsed_s : 0.0;
    // Schema is a superset of the sem_filter driver's so summarize_morsel_ab.py
    // reads it unchanged. "passes" = rows emitted; a projection emits every row,
    // so passes == rows (pass_pct == 100 confirms no rows were dropped).
    out << "{\n";
    out << "  \"rewrite\": \"" << a.rewrite << "\",\n";
    out << "  \"threads\": " << a.threads << ",\n";
    out << "  \"rows\": " << rows_loaded << ",\n";
    out << "  \"passes\": " << emitted << ",\n";
    out << "  \"elapsed_s\": " << elapsed_s << ",\n";
    out << "  \"rows_per_s\": " << rows_per_s << ",\n";
    out << "  \"row_group_size\": " << a.row_group_size << ",\n";
    out << "  \"morsels\": " << (a.row_group_size > 0 ? rows_loaded / a.row_group_size : 1) << ",\n";
    out << "  \"inflight\": " << a.inflight << ",\n";
    out << "  \"batch\": " << a.rows_per_request << ",\n";
    out << "  \"model\": \"" << a.model << "\",\n";
    out << "  \"tuple_format\": \"" << a.tuple_format << "\",\n";
    out << "  \"endpoints\": \"" << a.endpoints_csv << "\"\n";
    out << "}\n";
}

// The projected llm_complete extraction over table `t` aliased `r`, materialised
// into `dest` so every row executes. With rewrite on the projection is subsumed
// by PhysicalSemExtract; with rewrite off it runs the scalar llm_complete.
std::string ExtractSelect(const Args& a, const std::string& from_table) {
    return "SELECT r.reviewId, llm_complete("
           "{'model_name': '" + SqlEscape(a.model) + "'}, "
           "{'prompt': '" + SqlEscape(a.prompt) + "', "
           "'context_columns': [{'data': r." + a.text_col + ", 'name': 'review'}]}) AS sentiment "
           "FROM " + from_table + " AS r";
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!ParseArgs(argc, argv, &args)) return 2;

    const bool rewrite_on = (args.rewrite == "on");

    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    flock::Config::GetConnection(&*db.instance);

    // --- session + runtime config (excluded from timing) --------------------
    if (!Run(con, "SET threads=" + std::to_string(args.threads) + ";")) return 1;
    if (!Run(con, "SET semantic_endpoints='" + SqlEscape(args.endpoints_csv) + "';")) return 1;
    if (!Run(con, "SET semantic_in_flight_cap=" + std::to_string(args.inflight) + ";")) return 1;
    if (!Run(con, "SET semantic_batch_size=" + std::to_string(args.rows_per_request) + ";")) return 1;
    if (!Run(con, std::string("SET semantic_rewrite_enabled=") + (rewrite_on ? "true" : "false") + ";")) return 1;

    if (!Run(con, "CREATE SECRET (TYPE OPENAI, API_KEY 'sk-noauth', BASE_URL '" +
                          SqlEscape(DeriveBaseUrl(args.endpoints_csv)) + "');"))
        return 1;
    // Pin the served model id + batch_size so both arms batch identically, and
    // bound the completion length on both surfaces (operator: max_output_tokens;
    // scalar: model_parameters.max_tokens) to max_out_mult*R. See the filter
    // driver for why both are needed.
    const long long max_out_tokens = static_cast<long long>(args.max_out_mult) * args.rows_per_request;
    const std::string tok = std::to_string(max_out_tokens);
    // tuple_format pinned explicitly (the model default is XML) so this driver and
    // the sembench cross-system harness render the same tuple block.
    if (!Run(con, "CREATE MODEL ('" + SqlEscape(args.model) + "', '" + SqlEscape(args.model) +
                          "', 'openai', {\"batch_size\": " + std::to_string(args.rows_per_request) +
                          ", \"max_output_tokens\": " + tok +
                          ", \"tuple_format\": \"" + SqlEscape(args.tuple_format) +
                          "\", \"model_parameters\": {\"max_tokens\": " + tok +
                          ", \"temperature\": 0}});"))
        return 1;

    // Default: in-memory table => one row group => one morsel. With
    // --row-group-size, materialise into an attached on-disk DB whose row groups
    // are that size, so the scan sees rows/row_group_size morsels (node-local /tmp
    // backing -- see the slurm script).
    std::string table = "reviews";
    if (args.row_group_size > 0) {
        std::remove(args.attach_db.c_str());
        std::remove((args.attach_db + ".wal").c_str());
        if (!Run(con, "ATTACH '" + SqlEscape(args.attach_db) + "' AS m (ROW_GROUP_SIZE " +
                              std::to_string(args.row_group_size) + ");"))
            return 1;
        table = "m.reviews";
    }

    if (!Run(con, "CREATE TABLE " + table + " AS SELECT reviewId, " + args.text_col +
                          " FROM read_csv_auto('" + SqlEscape(args.data) + "') LIMIT " +
                          std::to_string(args.rows) + ";"))
        return 1;

    long long rows_loaded = 0;
    {
        auto res = con.Query("SELECT count(*) FROM " + table + ";");
        if (res->HasError()) {
            std::fprintf(stderr, "row count failed: %s\n", res->GetError().c_str());
            return 1;
        }
        rows_loaded = std::stoll(res->GetValue(0, 0).ToString());
    }

    std::fprintf(stderr,
                 "rewrite=%s threads=%d rows=%lld inflight=%d batch=%d tuple_format=%s model=%s "
                 "endpoints=%s\n",
                 args.rewrite.c_str(), args.threads, rows_loaded, args.inflight,
                 args.rows_per_request, args.tuple_format.c_str(), args.model.c_str(),
                 args.endpoints_csv.c_str());

    // --- burn-in (UNTIMED): warm vLLM compute on a fresh endpoint before timing.
    // One full batch of SYNTHETIC rows through the SAME llm_complete path compiles
    // the xgrammar guided-decoding kernel and the batch-sized CUDA graph. Synthetic
    // content keeps the measured rows' per-row prefix-cache entries cold; only the
    // shared instruction/template prefix warms, identically for both arms.
    if (!args.skip_burn_in) {
        if (!Run(con, "CREATE TABLE warmup AS SELECT i AS reviewId, 'warmup review ' || i::VARCHAR AS " +
                              args.text_col + " FROM range(" + std::to_string(args.rows_per_request) + ") t(i);"))
            return 1;
        auto warm = con.Query("CREATE TABLE warmup_out AS " + ExtractSelect(args, "warmup") + ";");
        if (warm->HasError()) {
            std::fprintf(stderr, "burn-in query failed: %s\n", warm->GetError().c_str());
            return 1;
        }
        if (!Run(con, "DROP TABLE warmup_out;")) return 1;
        if (!Run(con, "DROP TABLE warmup;")) return 1;
        std::fprintf(stderr, "burn-in done (%d synthetic rows, untimed)\n", args.rows_per_request);
    }

    // --- timed region: materialise the projected extraction ----------------
    // CREATE TABLE ... AS SELECT forces every row through llm_complete (a
    // projection emits all rows). The destination lives in the default in-memory
    // catalog for both morsel and non-morsel modes (only the SOURCE scan layout
    // varies), so materialisation cost is identical across arms.
    const std::string timed = "CREATE TABLE extract_out AS " + ExtractSelect(args, table) + ";";
    const auto t0 = std::chrono::steady_clock::now();
    if (!Run(con, timed)) return 1;
    const auto t1 = std::chrono::steady_clock::now();

    long long emitted = 0;
    {
        auto res = con.Query("SELECT count(*) FROM extract_out;");
        if (res->HasError()) {
            std::fprintf(stderr, "emitted count failed: %s\n", res->GetError().c_str());
            return 1;
        }
        emitted = std::stoll(res->GetValue(0, 0).ToString());
    }

    const double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    const double rows_per_s = elapsed_s > 0.0 ? static_cast<double>(rows_loaded) / elapsed_s : 0.0;

    std::fprintf(stderr,
                 "DONE rewrite=%s threads=%d rows=%lld emitted=%lld elapsed_s=%.3f rows_per_s=%.2f\n",
                 args.rewrite.c_str(), args.threads, rows_loaded, emitted, elapsed_s, rows_per_s);

    WriteResultJson(args, emitted, rows_loaded, elapsed_s);
    return 0;
}
