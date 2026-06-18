// sem_filter operator throughput driver: run ONE (threads, rewrite) config
// against a live vLLM fleet and write a --result-out JSON, so the off-cluster
// aggregation reuses the endpoint-router driver's path/flag conventions.
//
// Unlike flock_endpoint_router_vllm_integration (which drives AsyncLLMClient
// directly, no DuckDB), this binary embeds a real DuckDB with flock loaded and
// exercises the full SQL chain. The single A/B knob is semantic_rewrite_enabled:
//   * --rewrite on  -> PhysicalSemFilter (the operator) posts batched
//                      /v1/completions through the shared EndpointRouter.
//   * --rewrite off -> the scalar llm_filter posts /chat/completions via the
//                      openai provider (base_url derived from --endpoints).
// Same query, same vLLM, same batch_size; only the rewrite flag differs.
//
// Only the SELECT count(*) is timed; table/secret/model setup is excluded.
//
//   flock_sem_filter_vllm_integration
//       --endpoints http://127.0.0.1:8000/v1/completions
//       --model Qwen/Qwen2.5-7B-Instruct
//       --data reviews.csv --text-col reviewText
//       --prompt 'The following movie review is clearly positive.'
//       --rows 2000 --rewrite on --threads 8
//       --inflight 128 --rows-per-request 32 --timeout-ms 120000
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
    std::string prompt = "The following movie review is clearly positive.";
    int rows = 2000;
    std::string rewrite = "on";  // on|off
    int threads = 8;
    int inflight = 128;
    int rows_per_request = 32;  // == batch_size (operator) / model batch_size (scalar)
    int timeout_ms = 120000;
    std::string result_out;
};

void Usage(const char* prog) {
    std::fprintf(stderr,
                 "Usage: %s --endpoints URL1,URL2,... --model NAME --data PATH\n"
                 "          [--text-col NAME] [--prompt TEXT] [--rows N]\n"
                 "          --rewrite on|off [--threads T] [--inflight N]\n"
                 "          [--rows-per-request N] [--timeout-ms N] [--result-out PATH]\n",
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
// (e.g. .../v1/completions); the openai provider instead appends
// /chat/completions to its base_url, so strip a trailing /completions from the
// first endpoint to recover the shared base (e.g. .../v1).
std::string DeriveBaseUrl(const std::string& endpoints_csv) {
    std::string first = endpoints_csv.substr(0, endpoints_csv.find(','));
    // Strip the request path so the scalar's openai handler can re-append
    // "chat/completions". Handle the chat URL (operator now posts there) and the
    // legacy completions URL; check the longer suffix first.
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

void WriteResultJson(const Args& a, long long passes, long long rows_loaded, double elapsed_s) {
    if (a.result_out.empty()) return;
    std::ofstream out(a.result_out);
    if (!out) {
        std::fprintf(stderr, "cannot write result file: %s\n", a.result_out.c_str());
        return;
    }
    const double rows_per_s = elapsed_s > 0.0 ? static_cast<double>(rows_loaded) / elapsed_s : 0.0;
    out << "{\n";
    out << "  \"rewrite\": \"" << a.rewrite << "\",\n";
    out << "  \"threads\": " << a.threads << ",\n";
    out << "  \"rows\": " << rows_loaded << ",\n";
    out << "  \"passes\": " << passes << ",\n";
    out << "  \"elapsed_s\": " << elapsed_s << ",\n";
    out << "  \"rows_per_s\": " << rows_per_s << ",\n";
    out << "  \"inflight\": " << a.inflight << ",\n";
    out << "  \"batch\": " << a.rows_per_request << ",\n";
    out << "  \"model\": \"" << a.model << "\",\n";
    out << "  \"endpoints\": \"" << a.endpoints_csv << "\"\n";
    out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!ParseArgs(argc, argv, &args)) return 2;

    const bool rewrite_on = (args.rewrite == "on");

    // flock auto-loads when the (statically linked) extension's DB is constructed;
    // Config::GetConnection binds the global connection used for model resolution.
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    flock::Config::GetConnection(&*db.instance);

    // --- session + runtime config (excluded from timing) --------------------
    if (!Run(con, "SET threads=" + std::to_string(args.threads) + ";")) return 1;
    if (!Run(con, "SET semantic_endpoints='" + SqlEscape(args.endpoints_csv) + "';")) return 1;
    if (!Run(con, "SET semantic_in_flight_cap=" + std::to_string(args.inflight) + ";")) return 1;
    if (!Run(con, "SET semantic_batch_size=" + std::to_string(args.rows_per_request) + ";")) return 1;
    if (!Run(con, std::string("SET semantic_rewrite_enabled=") + (rewrite_on ? "true" : "false") + ";")) return 1;

    // Scalar path target: the default openai secret's base_url. The operator path
    // ignores this and uses semantic_endpoints. NOTE: the operator's AsyncLLMClient
    // uses its built-in 60s per-request timeout (no SQL knob); --timeout-ms is
    // recorded for provenance and bounds a batch comfortably at these batch sizes.
    if (!Run(con, "CREATE SECRET (TYPE OPENAI, API_KEY 'sk-noauth', BASE_URL '" +
                          SqlEscape(DeriveBaseUrl(args.endpoints_csv)) + "');"))
        return 1;
    // Pin the served model id (== --model, what vLLM serves) and the batch_size so
    // BOTH paths batch identically (model_args batch_size also overrides the
    // operator's resolved batch_size, keeping the A/B fair).
    //
    // Bound the completion length for BOTH arms to the same value, scaled to the
    // batch. The two arms read DIFFERENT surfaces (see below), so we set both:
    //   - operator (rewrite=on): SemanticParams.max_output_tokens <- model_args
    //     "max_output_tokens"; its default is only 16 (semantic_settings.h),
    //     which truncates a 32-row verdict array -> "completion text is not valid
    //     JSON".
    //   - scalar (rewrite=off): the OpenAI provider only emits "max_tokens" if it
    //     is present in model_parameters (openai.cpp AddCompletionRequest); with
    //     none set the request is unbounded and the model over-generates until the
    //     context limit -> truncated JSON. So we nest "max_tokens" there.
    // 16 tokens/row is ~8x headroom over the ~2 tokens a guided boolean needs, so
    // the cap never clips a valid array while still bounding a runaway generation.
    const long long max_out_tokens = 16LL * args.rows_per_request;
    const std::string tok = std::to_string(max_out_tokens);
    if (!Run(con, "CREATE MODEL ('" + SqlEscape(args.model) + "', '" + SqlEscape(args.model) +
                          "', 'openai', {\"batch_size\": " + std::to_string(args.rows_per_request) +
                          ", \"max_output_tokens\": " + tok +
                          ", \"model_parameters\": {\"max_tokens\": " + tok +
                          ", \"temperature\": 0}});"))
        return 1;

    if (!Run(con, "CREATE TABLE reviews AS SELECT reviewId, " + args.text_col +
                          " FROM read_csv_auto('" + SqlEscape(args.data) + "') LIMIT " +
                          std::to_string(args.rows) + ";"))
        return 1;

    long long rows_loaded = 0;
    {
        auto res = con.Query("SELECT count(*) FROM reviews;");
        if (res->HasError()) {
            std::fprintf(stderr, "row count failed: %s\n", res->GetError().c_str());
            return 1;
        }
        rows_loaded = std::stoll(res->GetValue(0, 0).ToString());
    }

    const std::string timed =
            "SELECT count(*) AS passes FROM reviews AS r "
            "WHERE llm_filter({'model_name': '" + SqlEscape(args.model) + "'}, "
            "{'prompt': '" + SqlEscape(args.prompt) + "', "
            "'context_columns': [{'data': r." + args.text_col + ", 'name': 'review'}]});";

    std::fprintf(stderr,
                 "rewrite=%s threads=%d rows=%lld inflight=%d batch=%d model=%s endpoints=%s\n",
                 args.rewrite.c_str(), args.threads, rows_loaded, args.inflight,
                 args.rows_per_request, args.model.c_str(), args.endpoints_csv.c_str());

    // --- burn-in (UNTIMED): warm vLLM compute on a fresh endpoint before timing.
    // One full batch of SYNTHETIC rows through the SAME llm_filter path compiles
    // the xgrammar guided-decoding kernel (the "JIT during inference" spike) and
    // exercises the batch-sized CUDA graph. Synthetic content != the measured
    // reviews, so the measured rows' per-row prefix-cache entries stay cold (the
    // cold-fleet intent); only the shared instruction/template prefix warms, and
    // it warms identically for both arms -> the A/B stays fair. Result discarded.
    {
        if (!Run(con, "CREATE TABLE warmup AS SELECT 'warmup review ' || i::VARCHAR AS " + args.text_col +
                              " FROM range(" + std::to_string(args.rows_per_request) + ") t(i);"))
            return 1;
        auto warm = con.Query(
                "SELECT count(*) FROM warmup AS r "
                "WHERE llm_filter({'model_name': '" + SqlEscape(args.model) + "'}, "
                "{'prompt': '" + SqlEscape(args.prompt) + "', "
                "'context_columns': [{'data': r." + args.text_col + ", 'name': 'review'}]});");
        if (warm->HasError()) {
            std::fprintf(stderr, "burn-in query failed: %s\n", warm->GetError().c_str());
            return 1;
        }
        if (!Run(con, "DROP TABLE warmup;")) return 1;
        std::fprintf(stderr, "burn-in done (%d synthetic rows, untimed)\n", args.rows_per_request);
    }

    // --- timed region: the filter query only -------------------------------
    const auto t0 = std::chrono::steady_clock::now();
    auto res = con.Query(timed);
    const auto t1 = std::chrono::steady_clock::now();
    if (res->HasError()) {
        std::fprintf(stderr, "filter query failed: %s\n", res->GetError().c_str());
        return 1;
    }
    const double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    const long long passes = std::stoll(res->GetValue(0, 0).ToString());
    const double rows_per_s = elapsed_s > 0.0 ? static_cast<double>(rows_loaded) / elapsed_s : 0.0;

    std::fprintf(stderr,
                 "DONE rewrite=%s threads=%d rows=%lld passes=%lld elapsed_s=%.3f rows_per_s=%.2f\n",
                 args.rewrite.c_str(), args.threads, rows_loaded, passes, elapsed_s, rows_per_s);

    WriteResultJson(args, passes, rows_loaded, elapsed_s);
    return 0;
}
