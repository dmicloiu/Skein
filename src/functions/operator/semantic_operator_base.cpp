#include "flock/functions/operator/semantic_operator_base.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace flock {

using duckdb::DataChunk;
using duckdb::idx_t;
using duckdb::InterruptState;
using duckdb::SinkFinalizeType;
using duckdb::SinkResultType;
using duckdb::SourceResultType;

// Benchmark-only knob: force generation to run to max_tokens (vLLM
// ignore_eos), so output length is a controlled variable for workload-shape
// experiments. Never set in normal operation.
static bool SemIgnoreEos() {
    static const bool on = [] {
        const char* p = std::getenv("FLOCK_SEM_IGNORE_EOS");
        return p && std::string(p) == "1";
    }();
    return on;
}

// Check if prompt slim-ing is on.
bool SemPromptSlim() {
    static const bool slim = [] {
        const char* p = std::getenv("FLOCK_SEM_PROMPT");
        return p && std::string(p) == "slim";
    }();
    return slim;
}

namespace {

std::string ImageMimeType(const nlohmann::json& column, const std::string& path) {
    const auto type = column.value("type", std::string("image"));
    if (type.compare(0, 6, "image/") == 0) {
        return type;
    }
    const auto dot = path.find_last_of('.');
    std::string extension = dot == std::string::npos ? "" : path.substr(dot);
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (extension == ".jpg" || extension == ".jpeg") return "image/jpeg";
    if (extension == ".webp") return "image/webp";
    if (extension == ".gif") return "image/gif";
    return "image/png";
}

std::string EncodeImageFile(const std::string& path) {
    std::FILE* file = std::fopen(path.c_str(), "rb");
    if (!file) throw std::runtime_error("flock semantic operator: cannot open image: " + path);
    std::string encoded;
    static const char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    unsigned char bytes[3];
    size_t count;
    while ((count = std::fread(bytes, 1, 3, file)) != 0) {
        const unsigned int triple = (static_cast<unsigned int>(bytes[0]) << 16) |
                                    (count > 1 ? static_cast<unsigned int>(bytes[1]) << 8 : 0) |
                                    (count > 2 ? bytes[2] : 0);
        encoded.push_back(alphabet[(triple >> 18) & 63]);
        encoded.push_back(alphabet[(triple >> 12) & 63]);
        encoded.push_back(count > 1 ? alphabet[(triple >> 6) & 63] : '=');
        encoded.push_back(count > 2 ? alphabet[triple & 63] : '=');
    }
    const bool read_error = std::ferror(file) != 0;
    std::fclose(file);
    if (read_error || encoded.empty()) {
        throw std::runtime_error("flock semantic operator: cannot read image: " + path);
    }
    return encoded;
}

nlohmann::json BuildChatContent(const std::string& prompt, const nlohmann::json& image_columns) {
    auto content = nlohmann::json::array({{{"type", "text"}, {"text", prompt}}});
    for (const auto& column : image_columns) {
        const std::string name = column.value("name", std::string("image"));
        const std::string detail = column.value("detail", std::string("low"));
        const auto& data = column.at("data");
        for (size_t row = 0; row < data.size(); ++row) {
            if (data[row].is_null()) continue;
            const std::string path = data[row].get<std::string>();
            if (path.empty()) continue;
            const std::string url = ImageDataUrl(column, path);
            content.push_back({{"type", "text"},
                               {"text", "Attachment " + name + ", row " + std::to_string(row + 1) + ":"}});
            content.push_back({{"type", "image_url"}, {"image_url", {{"url", url}, {"detail", detail}}}});
        }
    }
    return content;
}

// Parse one /v1/chat/completions response into the batch's per-row response array.
//
// The batch was sent as ONE multi-row chat prompt, so the response carries ONE
// completion whose message.content is a JSON object holding one element per row
// under the "items" key. This is flock's shared batch envelope (BatchAndComplete
// returns the same {"items":[...]} for FILTER, COMPLETE, ...; see
// scalar/llm_filter/implementation.cpp's CollectCompletions()[0]["items"]), so
// the operator and the scalar agree. Element TYPE (bool, string, struct, ...) is
// the operator's concern -> this function only realigns the array with its rows.
//
// Degraded-response parity with the scalar's BatchAndComplete: a short items
// array is padded with null and a long one truncated, so the result aligns by
// row (ParseVerdict maps null -> pass) instead of failing the query.
//
// Fail-loud policy: a failed HTTP request or an unparseable body/content/missing
// choices/message/items field still throws.
// Transport + chat-envelope validation shared by both response formats:
// returns choices[0].message.content or throws.
std::string ParseCompletionText(const CompletedBatch& batch) {
    if (!batch.response.ok) {
        throw std::runtime_error("flock semantic operator: LLM request failed (http " +
                                 std::to_string(batch.response.http_status) + "): " + batch.response.error);
    }
    nlohmann::json parsed = nlohmann::json::parse(batch.response.body, /*cb=*/nullptr, /*allow_exceptions=*/false);
    if (parsed.is_discarded()) {
        std::fprintf(stderr, "[flock debug] operator: response body not valid JSON (len=%zu):\n%.2000s\n",
                     batch.response.body.size(), batch.response.body.c_str());
        throw std::runtime_error("flock semantic operator: response body is not valid JSON");
    }
    auto cit = parsed.find("choices");
    if (cit == parsed.end() || !cit->is_array() || cit->empty()) {
        throw std::runtime_error("flock semantic operator: response missing a non-empty 'choices' array");
    }
    // ONE chat completion for the whole batch: choices[0].message.content holds
    // the {"items":[...]} envelope (chat/completions, for parity with llm_filter).
    const auto& choice = (*cit)[0];
    auto mit = choice.find("message");
    if (mit == choice.end() || !mit->is_object()) {
        throw std::runtime_error("flock semantic operator: chat choice missing a 'message' object");
    }
    auto ctit = mit->find("content");
    if (ctit == mit->end() || !ctit->is_string()) {
        throw std::runtime_error("flock semantic operator: chat message missing a 'content' string");
    }
    return ctit->get<std::string>();
}

// json_schema contract: the completion text is the {"items":[...]} envelope.
nlohmann::json ParseItems(const CompletedBatch& batch) {
    const std::string completion_text = ParseCompletionText(batch);
    nlohmann::json completion =
            nlohmann::json::parse(completion_text, /*cb=*/nullptr, /*allow_exceptions=*/false);
    if (completion.is_discarded()) {
        std::fprintf(stderr, "[flock debug] operator: message content not valid JSON (len=%zu):\n%.2000s\n",
                     completion_text.size(), completion_text.c_str());
        throw std::runtime_error("flock semantic operator: message content is not valid JSON");
    }
    auto iit = completion.find("items");
    if (iit == completion.end() || !iit->is_array()) {
        throw std::runtime_error("flock semantic operator: completion missing an 'items' array");
    }
    const size_t want = batch.row_ids.size();
    // Pad short with null (-> pass via ParseVerdict), truncate long. Under
    // guided decoding the array is already exactly batch-sized, so this only
    // matters for degraded responses.
    nlohmann::json items = std::move(*iit);
    while (items.size() < want) {
        items.push_back(nullptr);
    }
    if (items.size() > want) {
        items.erase(items.begin() + static_cast<long>(want), items.end());
    }
    return items;
}

// Optional one-shot dump of the first rendered prompt (env FLOCK_PROMPT_DUMP=<path>),
// so a benchmark run can be gated on the prompt it actually sent -- tuple encoding,
// head, variants -- rather than on the flags meant to produce it. Inert unless the
// env var is set; writes one prompt, so it cannot grow with the run.
void DumpPrompt(const std::string& prompt) {
    static std::FILE* out = [] {
        const char* p = std::getenv("FLOCK_PROMPT_DUMP");
        return (p && *p) ? std::fopen(p, "wb") : nullptr;
    }();
    if (!out) {
        return;
    }
    static std::once_flag once;
    std::call_once(once, [&] {
        std::fwrite(prompt.data(), 1, prompt.size(), out);
        std::fflush(out);
    });
}

// Stable diagnostic key for checking that two separate verdict-dump runs
// assigned the same query-local row ID to the same input row. Length framing
// avoids ambiguity when column values contain separators.
uint64_t RowFingerprint(const RowData& row) {
    uint64_t hash = 1469598103934665603ULL;
    const auto add = [&hash](const std::string& value) {
        const uint64_t length = value.size();
        for (unsigned shift = 0; shift < 64; shift += 8) {
            hash ^= static_cast<unsigned char>((length >> shift) & 0xff);
            hash *= 1099511628211ULL;
        }
        for (unsigned char byte : value) {
            hash ^= byte;
            hash *= 1099511628211ULL;
        }
    };
    for (const auto& value : row.values) {
        add(value.type().ToString());
        add(value.IsNull() ? "<NULL>" : value.ToString());
    }
    return hash;
}

// Optional per-row verdict dump for prompt-degradation diagnostics
// (env FLOCK_VERDICT_DUMP=<path>). One JSONL line per row: the query-global
// row_id (== table scan order at threads=1), the row's 0-based position within
// its batch, and the raw model verdict element. Inert unless the env var is set;
// by protocol only ever enabled on an untimed pass, so it cannot perturb a
// measured run. Serialises the raw element generically, so it is not
// filter-specific.
//
// Flushes per batch. It used to rely on stdio buffering with no flush and no
// close, which silently capped every dump at ONE 64 KiB buffer -- the process
// exits without draining the rest, so a 2000-row dump landed at exactly 65536
// bytes (~1900 rows here) with a truncated final line. The loss scales with row
// width, not with a fixed row count, and at high R the surviving rows are an
// interleaved subset rather than a prefix. The flush is free: the dump is
// untimed by protocol.
void DumpVerdicts(const CompletedBatch& batch, const nlohmann::json& items) {
    static std::FILE* out = [] {
        const char* p = std::getenv("FLOCK_VERDICT_DUMP");
        return (p && *p) ? std::fopen(p, "wb") : nullptr;
    }();
    if (!out) {
        return;
    }
    static std::mutex mu;
    std::lock_guard<std::mutex> guard(mu);
    for (size_t i = 0; i < batch.rows.size(); ++i) {
        std::fprintf(out, "{\"id\":%llu,\"pos\":%zu,\"row_hash\":%llu,\"stage\":\"%s\",\"v\":%s}\n",
                     static_cast<unsigned long long>(batch.row_ids[i]), i,
                     static_cast<unsigned long long>(RowFingerprint(batch.rows[i])),
                     batch.proxy ? "proxy" : "main", items[i].dump().c_str());
    }
    std::fflush(out);
}

}  // namespace

std::string ImageDataUrl(const nlohmann::json& column, const std::string& path) {
    if (path.compare(0, 7, "http://") == 0 || path.compare(0, 8, "https://") == 0 ||
        path.compare(0, 5, "data:") == 0) {
        return path;
    }
    return "data:" + ImageMimeType(column, path) + ";base64," + EncodeImageFile(path);
}

SemGlobalSinkState::SemGlobalSinkState(std::shared_ptr<ILLMClient> client,
                                       std::shared_ptr<EndpointRouter> router, SemanticParams cfg)
    : generation(NextGeneration()), client(std::move(client)), router(std::move(router)),
      cfg(std::move(cfg)) {
    if (!this->cfg.filter_proxy_endpoint.empty()) {
        proxy_router = std::make_shared<EndpointRouter>(
                std::vector<std::string>{this->cfg.filter_proxy_endpoint}, EndpointRouter::Strategy::Single);
    }
}

SemGlobalSinkState::~SemGlobalSinkState() {
    // First statement, and WITHOUT the sink lock: the drain blocks until the IO
    // thread's in-flight callback returns, so members are still alive when it
    // finishes + that callback wants the lock, so holding it would deadlock.
    if (client) {
        client->CancelByGeneration(generation.load(std::memory_order_acquire));
    }
}

void SemGlobalSinkState::NoteInFlight(size_t now_in_flight) {
    size_t prev = max_in_flight_seen.load(std::memory_order_relaxed);
    while (now_in_flight > prev &&
           !max_in_flight_seen.compare_exchange_weak(prev, now_in_flight, std::memory_order_relaxed)) {
        // prev reloaded by compare_exchange_weak on failure; retry.
    }
}

std::string SemGlobalSinkState::BuildPayload(const std::string& prompt,
                                              const nlohmann::json& image_columns, size_t batch_rows,
                                              bool proxy) const {
    // ONE multi-row prompt -> ONE /v1/chat/completions request (flock's
    // BatchAndComplete strategy). The single completion returns an "items" array
    // of batch_rows elements, realigned with the rows in ParseItems.
    nlohmann::json body;
    // Chat-completions for parity with the scalar llm_filter path: same single
    // user message with a text content part, so vLLM applies the model's chat
    // template (+ default system prompt) identically for both arms. (Mirrors
    // flock's openai adapter message shape.)
    body["messages"] = nlohmann::json::array(
            {{{"role", "user"}, {"content", BuildChatContent(prompt, image_columns)}}});
    const auto& model = proxy ? proxy_served_model : served_model;
    if (!model.empty()) {
        body["model"] = model;
    }
    // Decimal scores need more output tokens than the boolean main-model path.
    body["max_tokens"] = proxy ? std::max<uint64_t>(cfg.max_output_tokens, batch_rows * 8 + 16)
                                 : cfg.max_output_tokens;
    if (SemIgnoreEos()) {
        body["ignore_eos"] = true;
    }
    // Deterministic, fair A/B: greedy decoding. Explicitly overrides the model's
    // generation_config.json default (temperature=0.7) so the operator arm is
    // reproducible and matches the scalar arm (which pins temperature via
    // model_parameters). temperature=0 -> greedy, so top_p/top_k are no-ops.
    body["temperature"] = 0.0;
    if (cfg.response_format == "json_schema" && (proxy || response_schema)) {
        nlohmann::json rf = proxy
                ? SemanticOperatorBase::ItemsResponseFormat("proxy_scores",
                        {{"type", "number"}, {"minimum", 0.0}, {"maximum", 1.0}}, batch_rows)
                : response_schema(batch_rows);
        if (!rf.is_null()) {
            body["response_format"] = std::move(rf);
        }
    }
    return body.dump();
}

void SemGlobalSinkState::SubmitBatch(std::vector<PendingRequest> batch, bool proxy) {
    if (batch.empty()) {
        return;
    }
    // OUTSIDE the lock: gather the carry-over, render, route, build, submit.
    // Preserve submission order: row_ids[i] <-> rows[i] <-> items[i].
    std::vector<uint64_t> row_ids;
    std::vector<RowData> rows;
    row_ids.reserve(batch.size());
    rows.reserve(batch.size());
    for (auto& p : batch) {
        row_ids.push_back(p.row_id);
        rows.push_back(std::move(p.row));
    }
    // Pack the whole batch into ONE multi-row prompt (render hook is read-only
    // after setup, so it is lock-free here).
    const std::string prompt = proxy ? render_proxy_prompt(rows) : render_prompt(rows);
    const nlohmann::json image_columns = render_media ? render_media(rows) : nlohmann::json::array();
    // Routing key for StickyByPrefix. Prefer the operator-supplied sticky_key (the
    // query-stable template head). Fall back to a bounded leading slice of the
    // prompt, which stays inside flock's shared instruction/template head while
    // excluding the variable per-row tail; keep that slice short so it never
    // reaches the tail and scatters a query's batches across endpoints.
    const std::string key = sticky_key.empty() ? prompt.substr(0, kStickyPrefixBytes) : sticky_key;
    auto& target_router = proxy ? proxy_router : router;
    EndpointRouter::Pick pick = target_router->Choose(key);
    std::string payload = BuildPayload(prompt, image_columns, rows.size(), proxy);
    const uint64_t gen = generation.load(std::memory_order_acquire);
    const size_t endpoint_index = pick.index;
    const std::string request_id = "sem-batch-" + std::to_string(row_ids.front());

    // Move the carry-over into the callback; it fires at most once on the IO
    // thread. The captured `gen` is the generation guard's left-hand side.
    client->Submit(pick.url, payload, gen, request_id,
                   [this, gen, endpoint_index, proxy, ids = std::move(row_ids),
                    rws = std::move(rows)](LLMResponse resp) mutable {
                       OnBatchComplete(gen, endpoint_index, std::move(ids), std::move(rws), std::move(resp), proxy);
                   });
}

std::vector<std::vector<PendingRequest>> SemGlobalSinkState::TakeReadyMainBatches() {
    std::vector<std::vector<PendingRequest>> ready;
    auto guard = Lock();
    while (!pending_main.empty() && in_flight.load(std::memory_order_relaxed) < cfg.in_flight_cap) {
        const size_t take = std::min<size_t>(cfg.batch_size, pending_main.size());
        std::vector<PendingRequest> batch;
        batch.reserve(take);
        for (size_t i = 0; i < take; ++i) {
            batch.push_back(std::move(pending_main.front()));
            pending_main.pop_front();
        }
        NoteInFlight(in_flight.fetch_add(1, std::memory_order_relaxed) + 1);
        ready.push_back(std::move(batch));
    }
    return ready;
}

SinkResultType SemGlobalSinkState::MergeAndCoalesce(SemLocalSinkState& local, idx_t resume_idx,
                                                    InterruptState& interrupt) {
    std::vector<std::vector<PendingRequest>> to_submit;
    bool must_block = false;
    {
        auto guard = Lock();
        // Force-merge the per-thread staging buffer into the global queue.
        for (auto& p : local.pending) {
            pending.push_back(std::move(p));
        }
        local.pending.clear();
        if (proxy_stage) {
            // Embedding proxy: buffer everything; ranking needs the whole input.
            return SinkResultType::NEED_MORE_INPUT;
        }

        // Coalesce as many full batches as the cap allows, in submission order.
        // The read-check-of-in_flight and the increment are BOTH under the lock,
        // so the cap is authoritative - - -> in_flight can never exceed the cap.
        while (pending.size() >= cfg.batch_size) {
            if (in_flight.load(std::memory_order_relaxed) < cfg.in_flight_cap) {
                std::vector<PendingRequest> batch;
                batch.reserve(static_cast<size_t>(cfg.batch_size));
                for (uint64_t k = 0; k < cfg.batch_size; ++k) {
                    batch.push_back(std::move(pending.front()));
                    pending.pop_front();  // O(1) on deque
                }
                const size_t now = in_flight.fetch_add(1, std::memory_order_relaxed) + 1;
                NoteInFlight(now);
                to_submit.push_back(std::move(batch));
            } else {
                // Cap saturated AND a full batch is ready -> block THIS task
                // (per-batch, never per-row). Resume mid-chunk on wakeup.
                local.next_row_idx = resume_idx;
                blocked_sinks.push_back(interrupt);
                blocked_at_least_once.store(true, std::memory_order_relaxed);
                must_block = true;
                break;
            }
        }
    }
    // Submit OUTSIDE the lock.
    for (auto& batch : to_submit) {
        SubmitBatch(std::move(batch), !proxy_served_model.empty());
    }
    return must_block ? SinkResultType::BLOCKED : SinkResultType::NEED_MORE_INPUT;
}

SinkResultType SemGlobalSinkState::SinkChunk(SemLocalSinkState& local, DataChunk& chunk,
                                             InterruptState& interrupt) {
    const idx_t cols = chunk.ColumnCount();
    // Residual pre-filter: drop rows failing the non-LLM conjuncts before the LLM
    // round-trip. Computed ONLY on a fresh chunk (next_row_idx == 0) and cached in
    // `local`; a BLOCKED resume reuses the cached selection. No executor -> identity.
    if (local.next_row_idx == 0) {
        local.survivor_count = local.residual_executor
                                       ? local.residual_executor->SelectExpression(chunk, local.residual_sel)
                                       : chunk.size();
    }
    const idx_t n = local.survivor_count;
    for (idx_t i = local.next_row_idx; i < n; ++i) {
        const idx_t src = local.residual_executor ? local.residual_sel.get_index(i) : i;
        // Full-row COPY: GetValue returns an owning Value, so the snapshot
        // outlives this (soon-recycled) chunk across the async round-trip.
        RowData row;
        row.values.reserve(cols);
        for (idx_t c = 0; c < cols; ++c) {
            row.values.push_back(chunk.GetValue(c, src));
        }
        // Capture the row only; the prompt is rendered per BATCH by the
        // coalescer (SubmitBatch), not per row.
        PendingRequest req;
        req.row_id = next_row_id.fetch_add(1, std::memory_order_relaxed);  // query-global, lock-free
        req.row = std::move(row);
        local.pending.push_back(std::move(req));

        if (local.pending.size() >= LOCAL_MERGE_THRESHOLD) {
            // Resume AFTER this row -> it is already staged locally / merged.
            if (MergeAndCoalesce(local, i + 1, interrupt) == SinkResultType::BLOCKED) {
                return SinkResultType::BLOCKED;
            }
        }
    }
    // Chunk fully consumed: flush the local remainder and coalesce.
    if (MergeAndCoalesce(local, n, interrupt) == SinkResultType::BLOCKED) {
        return SinkResultType::BLOCKED;
    }
    local.next_row_idx = 0;
    return SinkResultType::NEED_MORE_INPUT;
}

void SemGlobalSinkState::MergeLocal(SemLocalSinkState& local) {
    if (local.pending.empty()) {
        return;
    }
    auto guard = Lock();
    for (auto& p : local.pending) {
        pending.push_back(std::move(p));
    }
    local.pending.clear();
}

SinkFinalizeType SemGlobalSinkState::FinalizeFlush(InterruptState& interrupt) {
    std::vector<std::vector<PendingRequest>> to_submit;
    std::vector<InterruptState> to_wake;
    bool must_block = false;
    if (proxy_stage && !proxy_planned) {
        if (!verdict_of) {
            throw std::runtime_error("flock semantic proxy: verdict_of hook is not set");
        }
        std::vector<PendingRequest> rows;
        {
            auto guard = Lock();
            proxy_planned = true;
            rows.reserve(pending.size());
            for (auto& p : pending) rows.push_back(std::move(p));
            pending.clear();
        }
        // Blocking (embedding HTTP calls); Finalize runs on one thread and no row
        // has been submitted yet, so nothing else waits on this state.
        ProxyPlan plan = proxy_stage->Plan(rows, *client, generation.load(std::memory_order_acquire));
        if (!plan.info.empty()) {
            std::fprintf(stderr, "[flock proxy] %s\n", plan.info.c_str());
        }
        if (!plan.policy) {
            // Exact fallback: hand the rows to the ordinary flush below.
            auto guard = Lock();
            for (auto& r : rows) pending.push_back(std::move(r));
        } else {
            if (plan.row_unit.size() != rows.size()) {
                throw std::runtime_error("flock semantic proxy: plan does not cover every row");
            }
            ProxyStep step;
            {
                std::lock_guard<std::mutex> policy_guard(policy_mu);
                policy = std::move(plan.policy);
                step = policy->Start();
            }
            {
                auto guard = Lock();
                unit_rows.assign(plan.num_units, {});
                for (size_t i = 0; i < rows.size(); ++i) {
                    unit_rows.at(plan.row_unit[i]).push_back(i);
                    row_unit_by_id[rows[i].row_id] = plan.row_unit[i];
                }
                proxy_rows = std::move(rows);
                ApplyProxyStepLocked(step);
                input_exhausted = true;
                for (auto& is : blocked_sources) to_wake.push_back(is);
                blocked_sources.clear();
            }
            for (auto& batch : TakeReadyMainBatches()) {
                SubmitBatch(std::move(batch), false);
            }
            for (auto& is : to_wake) {
                is.Callback();
            }
            return SinkFinalizeType::READY;
        }
    }
    {
        auto guard = Lock();
        // Submit ALL remaining rows -- including a trailing partial batch
        // (< batch_size) respecting the cap.
        // [TO DO] coalesce_max_age_ms is honored opportunistically here
        // and in SinkChunk; a dedicated timer thread that flushes aged partial
        // batches during idle / trickle input is deferred.
        while (!pending.empty()) {
            if (in_flight.load(std::memory_order_relaxed) < cfg.in_flight_cap) {
                const size_t take = std::min<size_t>(static_cast<size_t>(cfg.batch_size), pending.size());
                std::vector<PendingRequest> batch;
                batch.reserve(take);
                for (size_t k = 0; k < take; ++k) {
                    batch.push_back(std::move(pending.front()));
                    pending.pop_front();  // O(1) front pop
                }
                const size_t now = in_flight.fetch_add(1, std::memory_order_relaxed) + 1;
                NoteInFlight(now);
                to_submit.push_back(std::move(batch));
            } else {
                // Cap saturated, leftover remains -> block; resume on wakeup.
                blocked_sinks.push_back(interrupt);
                blocked_at_least_once.store(true, std::memory_order_relaxed);
                must_block = true;
                break;
            }
        }
        if (!must_block) {
            // Everything submitted; no more input will arrive. Outstanding
            // batches are already counted in in_flight, so Drain still waits.
            input_exhausted = true;
            for (auto& is : blocked_sources) {
                to_wake.push_back(is);
            }
            blocked_sources.clear();
        }
    }
    for (auto& batch : to_submit) {
        SubmitBatch(std::move(batch), !proxy_served_model.empty());
    }
    for (auto& is : to_wake) {
        is.Callback();
    }
    return must_block ? SinkFinalizeType::BLOCKED : SinkFinalizeType::READY;
}

void SemGlobalSinkState::OnBatchComplete(uint64_t submit_generation, size_t endpoint_index,
                                         std::vector<uint64_t> row_ids, std::vector<RowData> rows,
                                         LLMResponse response, bool proxy) {
    // Defense-in-depth, not the UAF guard (the client's drain is authoritative
    // and per-query generations never collide); a cheap no-op for a cancelled gen.
    // TODO: this skips router->OnComplete below (leaks the endpoint counter)
    // fix if live-query row-level cancellation is added; harmless on teardown.
    if (submit_generation != generation.load(std::memory_order_acquire)) {
        return;
    }
    std::vector<InterruptState> to_wake;
    {
        auto guard = Lock();
        CompletedBatch cb;
        cb.row_ids = std::move(row_ids);
        cb.rows = std::move(rows);
        cb.response = std::move(response);
        cb.endpoint_index = endpoint_index;
        cb.proxy = proxy;
        completed.push(std::move(cb));
        in_flight.fetch_sub(1, std::memory_order_relaxed);
        (proxy ? proxy_router : router)->OnComplete(endpoint_index);
        // Splice out blocked tasks; one completion wakes BOTH a cap-blocked
        // Sink and an empty-blocked Drain, each re-checks and re-blocks if it
        // still cannot make progress.
        for (auto& is : blocked_sinks) {
            to_wake.push_back(is);
        }
        blocked_sinks.clear();
        for (auto& is : blocked_sources) {
            to_wake.push_back(is);
        }
        blocked_sources.clear();
    }
    // Fire callbacks OUTSIDE the lock.
    for (auto& is : to_wake) {
        is.Callback();
    }
}

SourceResultType SemGlobalSinkState::Drain(DataChunk& out, const ParseFn& parse_fn,
                                           InterruptState& interrupt) {
    for (auto& ready : TakeReadyMainBatches()) {
        SubmitBatch(std::move(ready), false);
    }
    // Embedding proxy: rows already decided go out first, one vector at a time.
    if (EmitDecided(out, parse_fn)) {
        return SourceResultType::HAVE_MORE_OUTPUT;
    }
    CompletedBatch batch;
    {
        auto guard = Lock();
        if (completed.empty()) {
            if (!pending_emit.empty()) {
                // Another source thread queued decided rows meanwhile.
                return SourceResultType::HAVE_MORE_OUTPUT;
            }
            // Terminate ONLY once the sink is finalized AND nothing is in flight:
            if (in_flight.load(std::memory_order_relaxed) > 0 || !input_exhausted || !pending_main.empty()) {
                blocked_sources.push_back(interrupt);
                return SourceResultType::BLOCKED;
            }
            if (policy && !proxy_stats_printed.exchange(true)) {
                std::lock_guard<std::mutex> policy_guard(policy_mu);
                std::fprintf(stderr, "[flock proxy] stats %s\n", policy->Stats().c_str());
            }
            return SourceResultType::FINISHED;
        }
        batch = std::move(completed.front());
        completed.pop();
    }
    // Parse + emit OUTSIDE the lock. One batch per call; batch_size is bounded
    // well under STANDARD_VECTOR_SIZE, so a batch always fits in `out`.
    // Fail-loud on any anomaly. The response contract is configuration, not a
    // guess from the payload: free_form has no items envelope -- the whole
    // completion is the single row's element (batch_size pinned to 1 at config
    // resolution; pad defensively regardless).
    nlohmann::json items;
    if (cfg.response_format == "free_form") {
        items = nlohmann::json::array({ParseCompletionText(batch)});
        while (items.size() < batch.row_ids.size()) {
            items.push_back(nullptr);
        }
    } else {
        items = ParseItems(batch);
    }
    DumpVerdicts(batch, items);  // before proxy rows move into the main queue
    if (batch.proxy) {
        std::vector<PendingRequest> uncertain;
        for (size_t i = 0; i < batch.rows.size(); ++i) {
            if (!items[i].is_number()) {
                throw std::runtime_error("flock semantic proxy: score must be a number in [0, 1]");
            }
            const double score = items[i].get<double>();
            if (!std::isfinite(score) || score < 0.0 || score > 1.0) {
                throw std::runtime_error("flock semantic proxy: score must be a number in [0, 1]");
            }
            if (score <= cfg.filter_proxy_negative_threshold) {
                parse_fn(false, batch.rows[i], out);
            } else if (score >= cfg.filter_proxy_positive_threshold) {
                parse_fn(true, batch.rows[i], out);
            } else {
                uncertain.push_back({batch.row_ids[i], std::move(batch.rows[i])});
            }
        }
        if (!uncertain.empty()) {
            auto guard = Lock();
            for (auto& req : uncertain) pending_main.push_back(std::move(req));
        }
        for (auto& ready : TakeReadyMainBatches()) {
            SubmitBatch(std::move(ready), false);
        }
    } else if (proxy_planned && policy) {
        // Embedding proxy: the policy owns every resolution (a unit's verdict is
        // emitted for all of its rows, possibly deferred, e.g. LOTUS sampling).
        std::vector<ProxyStep> steps;
        steps.reserve(batch.rows.size());
        {
            std::lock_guard<std::mutex> policy_guard(policy_mu);
            for (size_t i = 0; i < batch.rows.size(); ++i) {
                const size_t unit = row_unit_by_id.at(batch.row_ids[i]);
                steps.push_back(policy->OnVerdict(unit, verdict_of(items[i])));
            }
        }
        {
            auto guard = Lock();
            for (const auto& step : steps) ApplyProxyStepLocked(step);
        }
        for (auto& ready : TakeReadyMainBatches()) {
            SubmitBatch(std::move(ready), false);
        }
        EmitDecided(out, parse_fn);
    } else {
        for (size_t i = 0; i < batch.rows.size(); ++i) {
            parse_fn(items[i], batch.rows[i], out);
        }
    }
    return SourceResultType::HAVE_MORE_OUTPUT;
}

void SemGlobalSinkState::ApplyProxyStepLocked(const ProxyStep& step) {
    for (size_t unit : step.send) {
        const auto& rep = proxy_rows[unit_rows.at(unit).front()];  // identical prompt for every row of the unit
        PendingRequest req;
        req.row_id = rep.row_id;
        req.row = rep.row;
        pending_main.push_back(std::move(req));
    }
    for (const auto& resolved : step.resolve) {
        for (size_t pos : unit_rows.at(resolved.first)) {
            pending_emit.emplace_back(pos, resolved.second);
        }
    }
}

bool SemGlobalSinkState::EmitDecided(DataChunk& out, const ParseFn& parse_fn) {
    std::vector<std::pair<size_t, bool>> take;
    {
        auto guard = Lock();
        if (pending_emit.empty()) return false;
        // ParseAndEmit appends at most one row per element; leave room so `out`
        // (STANDARD_VECTOR_SIZE) never overflows.
        const size_t room = static_cast<size_t>(out.GetCapacity()) - static_cast<size_t>(out.size());
        const size_t n = (std::min)(room, pending_emit.size());
        take.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            take.push_back(pending_emit.front());
            pending_emit.pop_front();
        }
    }
    for (const auto& item : take) {
        parse_fn(nlohmann::json(item.second), proxy_rows[item.first].row, out);
    }
    return true;
}

// =============================================================================
// SemanticOperatorBase -- thin PhysicalOperator adapters over the engine.
// Each method Casts DuckDB's state to the engine state and forwards; no logic
// lives here. The two pure virtuals (RenderPrompt / ParseAndEmit) are the only
// per-operator behavior, bound into the engine via the render hook + ParseFn.
// =============================================================================

SemanticOperatorBase::SemanticOperatorBase(duckdb::PhysicalPlan& physical_plan,
                                           duckdb::PhysicalOperatorType type,
                                           duckdb::vector<duckdb::LogicalType> types,
                                           duckdb::idx_t estimated_cardinality,
                                           std::shared_ptr<ILLMClient> client,
                                           std::shared_ptr<EndpointRouter> router, SemanticParams cfg)
    : duckdb::PhysicalOperator(physical_plan, type, std::move(types), estimated_cardinality),
      client(std::move(client)), router(std::move(router)), cfg(std::move(cfg)) {
}

nlohmann::json SemanticOperatorBase::BuildResponseFormat(size_t /*batch_rows*/) const {
    // Default: no guided schema. Concrete operators override to supply one.
    return nullptr;
}

nlohmann::json SemanticOperatorBase::ItemsResponseFormat(const std::string& name, nlohmann::json element,
                                                         size_t batch_rows) {
    nlohmann::json items_schema = {{"type", "array"},
                                   {"minItems", batch_rows},
                                   {"maxItems", batch_rows},
                                   {"items", std::move(element)}};
    return {{"type", "json_schema"},
            {"json_schema",
             {{"name", name},
              {"schema",
               {{"type", "object"},
                {"properties", {{"items", std::move(items_schema)}}},
                {"required", nlohmann::json::array({"items"})},
                {"additionalProperties", false}}}}}};
}

duckdb::unique_ptr<SemGlobalSinkState> SemanticOperatorBase::CreateGlobalSinkState() const {
    auto state = duckdb::make_uniq<SemGlobalSinkState>(client, router, cfg);
    // Bind the engine's render + schema hooks to this operator. `this` outlives the
    // sink state (the operator owns sink_state), and both hooks are set once before
    // any Sink call, then only read.
    state->render_prompt = [this](const std::vector<RowData>& batch) {
        auto prompt = RenderPrompt(batch);
        DumpPrompt(prompt);  // no-op unless FLOCK_PROMPT_DUMP is set (diagnostics)
        return prompt;
    };
    state->render_media = [this](const std::vector<RowData>& batch) { return RenderMedia(batch); };
    state->response_schema = [this](size_t batch_rows) { return BuildResponseFormat(batch_rows); };
    return state;
}

nlohmann::json SemanticOperatorBase::RenderMedia(const std::vector<RowData>& /*batch*/) const {
    return nlohmann::json::array();
}

duckdb::unique_ptr<duckdb::GlobalSinkState> SemanticOperatorBase::GetGlobalSinkState(
    duckdb::ClientContext& /*context*/) const {
    return CreateGlobalSinkState();
}

duckdb::unique_ptr<duckdb::LocalSinkState> SemanticOperatorBase::GetLocalSinkState(
    duckdb::ExecutionContext& /*context*/) const {
    return duckdb::make_uniq<SemLocalSinkState>();
}

duckdb::SinkResultType SemanticOperatorBase::Sink(duckdb::ExecutionContext& /*context*/, DataChunk& chunk,
                                                  duckdb::OperatorSinkInput& input) const {
    return input.global_state.Cast<SemGlobalSinkState>().SinkChunk(
        input.local_state.Cast<SemLocalSinkState>(), chunk, input.interrupt_state);
}

duckdb::SinkCombineResultType SemanticOperatorBase::Combine(duckdb::ExecutionContext& /*context*/,
                                                            duckdb::OperatorSinkCombineInput& input) const {
    input.global_state.Cast<SemGlobalSinkState>().MergeLocal(input.local_state.Cast<SemLocalSinkState>());
    return duckdb::SinkCombineResultType::FINISHED;
}

SinkFinalizeType SemanticOperatorBase::Finalize(duckdb::Pipeline& /*pipeline*/, duckdb::Event& /*event*/,
                                                duckdb::ClientContext& /*context*/,
                                                duckdb::OperatorSinkFinalizeInput& input) const {
    return input.global_state.Cast<SemGlobalSinkState>().FinalizeFlush(input.interrupt_state);
}

duckdb::unique_ptr<duckdb::GlobalSourceState> SemanticOperatorBase::GetGlobalSourceState(
    duckdb::ClientContext& /*context*/) const {
    return duckdb::make_uniq<SemGlobalSourceState>(cfg.in_flight_cap);
}

duckdb::unique_ptr<duckdb::LocalSourceState> SemanticOperatorBase::GetLocalSourceState(
    duckdb::ExecutionContext& /*context*/, duckdb::GlobalSourceState& /*gstate*/) const {
    return duckdb::make_uniq<SemLocalSourceState>();
}

SourceResultType SemanticOperatorBase::GetDataInternal(duckdb::ExecutionContext& /*context*/, DataChunk& chunk,
                                                       duckdb::OperatorSourceInput& input) const {
    // The confirmed bridge: the source phase reads the SINK state.
    auto& gss = sink_state->Cast<SemGlobalSinkState>();
    auto parse = [this](const nlohmann::json& element, const RowData& row, DataChunk& out) {
        ParseAndEmit(element, row, out);
    };
    return gss.Drain(chunk, parse, input.interrupt_state);
}

}  // namespace flock
