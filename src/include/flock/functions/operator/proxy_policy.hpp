#pragma once

// Pure decision logic for the embedding proxy of the semantic filter (no I/O,
// no DuckDB): which candidate "units" go to the main model, and which are
// decided without it. A unit is one distinct prompt (all context-column values
// equal); the engine maps every input row to its unit and emits all rows of a
// unit with the unit's verdict.
//
//   * TopK  ("embedding" kind): reject-only. Per group (rows sharing the
//     rendered proxy text, e.g. one airline), the top-k candidates by
//     embedding similarity go to the main model; a TRUE at rank r extends the
//     group's window to r + gap; a random audit below the cutoff (denser just
//     below it) catches misses, and an audit TRUE at depth d raises k to d + gap
//     for every group. Unsent units are rejected; only the main model accepts.
//   * Lotus ("lotus" kind, ablation): a port of LOTUS's join cascade
//     (importance sampling + learn_cascade_thresholds). Units scoring at or
//     above the learned positive threshold are ACCEPTED without the main model,
//     at or below the negative threshold rejected, the rest go to the main model.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace flock {

struct ProxyUnit {
    uint32_t group = 0;   // TopK ranking group (rendered proxy text)
    double score = 0.0;   // proxy similarity (cosine)
    bool forced = false;  // no proxy signal (e.g. NULL image) -> always ask the main model
};

struct ProxyStep {
    std::vector<size_t> send;                      // units to ask the main model
    std::vector<std::pair<size_t, bool>> resolve;  // units decided now (verdict for all their rows)
};

class ProxyPolicy {
public:
    virtual ~ProxyPolicy() = default;
    // Called once, before any verdict.
    virtual ProxyStep Start() = 0;
    // Main-model verdict for a unit returned by an earlier `send`. Every sent
    // unit gets exactly one call. The policy returns the unit's resolution
    // (possibly deferred) plus any follow-up sends.
    virtual ProxyStep OnVerdict(size_t unit, bool verdict) = 0;
    // One-line JSON summary for the run log.
    virtual std::string Stats() const = 0;
};

struct TopKOptions {
    uint64_t k_min = 5;
    double k_frac = 0.0;      // window = max(k_min, ceil(k_frac * group size))
    uint64_t gap = 2;
    double audit_near = 0.25; // sampling rate for ranks (k, 2k]
    double audit_far = 0.01;  // sampling rate for ranks > 2k
    uint64_t seed = 42;
};

struct LotusOptions {
    // Defaults = lotus.types.CascadeArgs as used by the sembench LOTUS runner.
    double recall_target = 0.8;
    double precision_target = 0.8;
    double failure_probability = 0.2;
    double sampling_percentage = 0.1;
    uint64_t max_sample_range = 200;  // cascade_IS_max_sample_range
    double is_weight = 0.9;           // cascade_IS_weight
    uint64_t seed = 42;
};

std::unique_ptr<ProxyPolicy> MakeTopKPolicy(std::vector<ProxyUnit> units, const TopKOptions& options);

// `pair_units`: one entry per input pair (row) in LOTUS helper-join order (left
// rows in input order, each left row's candidates by descending score); forced
// units must not appear. Sampling is over pairs, main-model calls over units.
std::unique_ptr<ProxyPolicy> MakeLotusPolicy(std::vector<ProxyUnit> units, std::vector<size_t> pair_units,
                                             const LotusOptions& options);

// Port of lotus.sem_ops.cascade_utils.learn_cascade_thresholds. Returns
// (positive threshold, negative threshold). An empty sample returns (1.0, 0.0),
// LOTUS's fallback when threshold learning fails.
std::pair<double, double> LotusLearnThresholds(const std::vector<double>& scores, const std::vector<bool>& oracle,
                                               const std::vector<double>& corrections, const LotusOptions& options);

}  // namespace flock
