#include "flock/functions/operator/embedding_proxy.hpp"

#include "flock/functions/operator/semantic_operator_base.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace flock {

namespace {

using Clock = std::chrono::steady_clock;

// Shared with the client callbacks; outlives a timed-out wait.
struct RequestBoard {
    std::mutex mu;
    std::condition_variable cv;
    std::vector<LLMResponse> responses;
    std::vector<bool> done;
    size_t in_flight = 0;
    size_t completed = 0;
};

// Submit `count` requests (payload built on demand) with at most `window` in
// flight; wait for all. Returns false on timeout (responses then incomplete).
bool RunRequests(ILLMClient& client, uint64_t generation, const std::string& endpoint, size_t count,
                 const std::function<std::string(size_t)>& payload_of, size_t window, double timeout_s,
                 std::vector<LLMResponse>& out) {
    auto board = std::make_shared<RequestBoard>();
    board->responses.resize(count);
    board->done.assign(count, false);
    const auto deadline = Clock::now() + std::chrono::milliseconds(static_cast<int64_t>(timeout_s * 1000.0));
    window = std::max<size_t>(1, window);
    for (size_t next = 0; next < count; ++next) {
        {
            std::unique_lock<std::mutex> lock(board->mu);
            if (!board->cv.wait_until(lock, deadline, [&] { return board->in_flight < window; })) {
                return false;
            }
            ++board->in_flight;
        }
        const std::string payload = payload_of(next);
        client.Submit(endpoint, payload, generation, "proxy-emb-" + std::to_string(next),
                      [board, next](LLMResponse response) {
                          {
                              std::lock_guard<std::mutex> lock(board->mu);
                              board->responses[next] = std::move(response);
                              board->done[next] = true;
                              --board->in_flight;
                              ++board->completed;
                          }
                          board->cv.notify_all();
                      });
    }
    std::unique_lock<std::mutex> lock(board->mu);
    if (!board->cv.wait_until(lock, deadline, [&] { return board->completed == count; })) {
        return false;
    }
    out = std::move(board->responses);
    return true;
}

// {"data":[{"index":i,"embedding":[...]}, ...]} -> embeddings by index.
std::vector<std::vector<float>> ParseEmbeddings(const LLMResponse& response, size_t expected) {
    if (!response.ok) {
        throw std::runtime_error("embedding request failed (http " + std::to_string(response.http_status) +
                                 "): " + response.error + " " + response.body.substr(0, 300));
    }
    const auto body = nlohmann::json::parse(response.body);
    const auto& data = body.at("data");
    if (!data.is_array() || data.size() != expected) {
        throw std::runtime_error("embedding response has " + std::to_string(data.size()) + " items, expected " +
                                 std::to_string(expected));
    }
    std::vector<std::vector<float>> out(expected);
    for (const auto& item : data) {
        const size_t index = item.value("index", static_cast<size_t>(0));
        if (index >= expected) throw std::runtime_error("embedding response index out of range");
        out[index] = item.at("embedding").get<std::vector<float>>();
    }
    return out;
}

void Normalize(std::vector<float>& v) {
    double norm = 0.0;
    for (float x : v) norm += static_cast<double>(x) * x;
    norm = std::sqrt(norm);
    if (norm > 0.0) {
        for (float& x : v) x = static_cast<float>(x / norm);
    }
}

double Dot(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) throw std::runtime_error("text and image embeddings differ in dimension");
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += static_cast<double>(a[i]) * b[i];
    return s;
}

std::string ValueText(const duckdb::Value& value) {
    return value.IsNull() ? std::string() : value.ToString();
}

void AppendFramed(std::string& key, const std::string& part) {
    key += std::to_string(part.size());
    key += ':';
    key += part;
}

}  // namespace

std::string EmbeddingProxyStage::RenderText(const std::string& text_template, const std::vector<std::string>& values,
                                            const std::string& fallback) {
    std::string joined;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) joined += ", ";
        joined += values[i];
    }
    if (values.empty()) {
        return text_template.empty() ? fallback : text_template;
    }
    if (text_template.empty()) {
        return joined;
    }
    std::string out;
    size_t start = 0;
    bool replaced = false;
    for (size_t pos; (pos = text_template.find("{}", start)) != std::string::npos; start = pos + 2) {
        out += text_template.substr(start, pos - start);
        out += joined;
        replaced = true;
    }
    out += text_template.substr(start);
    return replaced ? out : out + " " + joined;
}

ProxyPlan EmbeddingProxyStage::Plan(const std::vector<PendingRequest>& rows, ILLMClient& client,
                                    uint64_t generation) {
    ProxyPlan plan;
    const std::string tag = "kind=" + cfg_.kind + " rows=" + std::to_string(rows.size());
    if (rows.size() < cfg_.min_rows) {
        plan.info = tag + " -> exact (fewer than min_rows=" + std::to_string(cfg_.min_rows) + ")";
        return plan;
    }
    const auto t0 = Clock::now();

    // ---- group / unit / image bookkeeping ------------------------------------
    std::unordered_map<std::string, uint32_t> group_of_text;
    std::vector<std::string> group_texts;
    std::unordered_map<std::string, size_t> image_of_path;
    std::vector<std::string> image_paths;
    std::unordered_map<std::string, size_t> unit_of_key;
    std::vector<uint32_t> unit_group;
    std::vector<int64_t> unit_image;  // -1 -> no image (forced)
    plan.row_unit.resize(rows.size());
    std::vector<std::string> texts;
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& values = rows[i].row.values;
        texts.clear();
        std::string key;
        for (auto idx : cfg_.text_indices) {
            texts.push_back(ValueText(values.at(idx)));
            AppendFramed(key, texts.back());
        }
        const std::string path = ValueText(values.at(cfg_.image_index));
        AppendFramed(key, path);

        const std::string text = RenderText(cfg_.text_template, texts, cfg_.fallback_text);
        auto git = group_of_text.find(text);
        if (git == group_of_text.end()) {
            git = group_of_text.emplace(text, static_cast<uint32_t>(group_texts.size())).first;
            group_texts.push_back(text);
        }
        auto uit = unit_of_key.find(key);
        if (uit == unit_of_key.end()) {
            int64_t image = -1;
            if (!path.empty()) {
                auto iit = image_of_path.find(path);
                if (iit == image_of_path.end()) {
                    iit = image_of_path.emplace(path, image_paths.size()).first;
                    image_paths.push_back(path);
                }
                image = static_cast<int64_t>(iit->second);
            }
            uit = unit_of_key.emplace(key, unit_group.size()).first;
            unit_group.push_back(git->second);
            unit_image.push_back(image);
        }
        plan.row_unit[i] = uit->second;
    }
    plan.num_units = unit_group.size();

    // ---- embeddings ------------------------------------------------------------
    std::vector<std::vector<float>> text_emb(group_texts.size());
    std::vector<std::vector<float>> image_emb(image_paths.size());
    try {
        const size_t batch = std::max<uint64_t>(1, cfg_.text_batch);
        const size_t text_requests = (group_texts.size() + batch - 1) / batch;
        std::vector<LLMResponse> responses;
        const auto text_payload = [&](size_t r) {
            nlohmann::json input = nlohmann::json::array();
            for (size_t i = r * batch; i < std::min(group_texts.size(), (r + 1) * batch); ++i) {
                input.push_back(group_texts[i]);
            }
            return nlohmann::json{{"model", cfg_.model}, {"input", input}, {"encoding_format", "float"}}.dump();
        };
        if (!RunRequests(client, generation, cfg_.endpoint, text_requests, text_payload, cfg_.max_in_flight,
                         cfg_.timeout_s, responses)) {
            throw std::runtime_error("text embedding timed out");
        }
        for (size_t r = 0; r < text_requests; ++r) {
            const size_t first = r * batch;
            const size_t n = std::min(group_texts.size(), first + batch) - first;
            auto part = ParseEmbeddings(responses[r], n);
            for (size_t i = 0; i < n; ++i) text_emb[first + i] = std::move(part[i]);
        }
        const auto image_payload = [&](size_t i) {
            const nlohmann::json content = nlohmann::json::array(
                    {{{"type", "image_url"}, {"image_url", {{"url", ImageDataUrl(cfg_.image_column, image_paths[i])}}}}});
            return nlohmann::json{{"model", cfg_.model},
                                  {"encoding_format", "float"},
                                  {"messages", nlohmann::json::array({{{"role", "user"}, {"content", content}}})}}
                    .dump();
        };
        responses.clear();
        if (!RunRequests(client, generation, cfg_.endpoint, image_paths.size(), image_payload, cfg_.max_in_flight,
                         cfg_.timeout_s, responses)) {
            throw std::runtime_error("image embedding timed out");
        }
        for (size_t i = 0; i < image_paths.size(); ++i) {
            image_emb[i] = std::move(ParseEmbeddings(responses[i], 1)[0]);
        }
    } catch (const std::exception& e) {
        plan.policy.reset();
        plan.info = tag + " -> exact (embedding stage failed: " + e.what() + ")";
        return plan;
    }
    for (auto& v : text_emb) Normalize(v);
    for (auto& v : image_emb) Normalize(v);

    std::vector<ProxyUnit> units(plan.num_units);
    for (size_t u = 0; u < units.size(); ++u) {
        units[u].group = unit_group[u];
        if (unit_image[u] < 0) {
            units[u].forced = true;
        } else {
            units[u].score = Dot(text_emb[unit_group[u]], image_emb[static_cast<size_t>(unit_image[u])]);
        }
    }
    const double embed_s = std::chrono::duration<double>(Clock::now() - t0).count();
    std::ostringstream info;
    info << tag << " units=" << units.size() << " groups=" << group_texts.size() << " images=" << image_paths.size()
         << " embed_s=" << embed_s;

    if (cfg_.kind == "lotus") {
        // LOTUS helper-join order: left rows (here: groups by first appearance),
        // each by descending score; ties by input position.
        std::vector<size_t> order;
        order.reserve(rows.size());
        for (size_t i = 0; i < rows.size(); ++i) {
            if (!units[plan.row_unit[i]].forced) order.push_back(i);
        }
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            const auto& ua = units[plan.row_unit[a]];
            const auto& ub = units[plan.row_unit[b]];
            if (ua.group != ub.group) return ua.group < ub.group;
            return ua.score > ub.score;
        });
        std::vector<size_t> pair_units;
        pair_units.reserve(order.size());
        for (size_t i : order) pair_units.push_back(plan.row_unit[i]);
        plan.policy = MakeLotusPolicy(std::move(units), std::move(pair_units), cfg_.lotus);
    } else {
        plan.policy = MakeTopKPolicy(std::move(units), cfg_.topk);
    }
    plan.info = info.str();
    return plan;
}

}  // namespace flock
