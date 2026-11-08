#pragma once

#include "duckdb/common/types.hpp"
#include "flock/functions/operator/proxy_stage.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace flock {

// Configuration of the CLIP-style embedding proxy for one semantic filter.
struct EmbeddingProxyConfig {
    std::string kind;          // "embedding" (reject-only top-k) | "lotus" (LOTUS-style cascade)
    std::string endpoint;      // full URL of an OpenAI-compatible /v1/embeddings endpoint
    std::string model;         // served model id, sent verbatim as body["model"]
    std::string text_template; // "{}" -> joined text values; empty -> raw values
    std::string fallback_text; // text when the filter has no text column (the filter prompt)
    duckdb::idx_t image_index = 0;                // RowData index of the single image column
    nlohmann::json image_column;                  // its metadata (MIME type for data URLs)
    std::vector<duckdb::idx_t> text_indices;      // RowData indices of the text columns
    uint64_t min_rows = 1000;                     // fewer input rows -> exact
    uint64_t max_in_flight = 64;                  // concurrent embedding requests
    uint64_t text_batch = 64;                     // texts per /v1/embeddings request
    double timeout_s = 600.0;                     // whole embedding stage; exceeded -> exact
    TopKOptions topk;
    LotusOptions lotus;
};

class EmbeddingProxyStage : public ProxyStage {
public:
    explicit EmbeddingProxyStage(EmbeddingProxyConfig config) : cfg_(std::move(config)) {}
    ProxyPlan Plan(const std::vector<PendingRequest>& rows, ILLMClient& client, uint64_t generation) override;
    const EmbeddingProxyConfig& Config() const { return cfg_; }

    // Rendered proxy text for a row's text values (exposed for tests).
    static std::string RenderText(const std::string& text_template, const std::vector<std::string>& values,
                                  const std::string& fallback);

private:
    EmbeddingProxyConfig cfg_;
};

}  // namespace flock
