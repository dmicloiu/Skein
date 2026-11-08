#include "flock/functions/operator/proxy_policy.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

namespace flock {

namespace {

// =============================================================================
// TopK (reject-only) policy
// =============================================================================

class TopKPolicy : public ProxyPolicy {
public:
    TopKPolicy(std::vector<ProxyUnit> units, const TopKOptions& options)
        : units_(std::move(units)), opt_(options), rank_(units_.size(), 0), sent_(units_.size(), false),
          audit_(units_.size(), false) {
        if (opt_.k_min < 1) {
            throw std::invalid_argument("proxy k_min must be >= 1");
        }
        uint32_t num_groups = 0;
        for (const auto& u : units_) {
            if (!u.forced) num_groups = std::max<uint32_t>(num_groups, u.group + 1);
        }
        groups_.assign(num_groups, {});
        for (size_t i = 0; i < units_.size(); ++i) {
            if (!units_[i].forced) groups_[units_[i].group].push_back(i);
        }
        for (auto& members : groups_) {
            // Descending score; ties by unit index (deterministic).
            std::stable_sort(members.begin(), members.end(),
                             [this](size_t a, size_t b) { return units_[a].score > units_[b].score; });
            for (size_t r = 0; r < members.size(); ++r) rank_[members[r]] = r + 1;
        }
        window_.assign(groups_.size(), 0);
        k_global_ = opt_.k_min;
    }

    ProxyStep Start() override {
        ProxyStep step;
        for (size_t i = 0; i < units_.size(); ++i) {
            if (units_[i].forced) {
                Send(i, step);
                ++sent_forced_;
            }
        }
        std::mt19937_64 rng(opt_.seed);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        for (size_t g = 0; g < groups_.size(); ++g) {
            const auto& members = groups_[g];
            const auto by_frac = static_cast<uint64_t>(std::ceil(opt_.k_frac * static_cast<double>(members.size())));
            window_[g] = std::max<uint64_t>(opt_.k_min, by_frac);
            for (size_t r = 1; r <= members.size(); ++r) {
                const size_t unit = members[r - 1];
                if (r <= window_[g]) {
                    Send(unit, step);
                    ++sent_topk_;
                    continue;
                }
                // One draw per out-of-window rank keeps the stream deterministic.
                const double draw = uniform(rng);
                const bool near = r <= 2 * window_[g];
                if (draw < (near ? opt_.audit_near : opt_.audit_far)) {
                    audit_[unit] = true;
                    Send(unit, step);
                    ++(near ? sent_audit_near_ : sent_audit_far_);
                }
            }
        }
        return step;
    }

    ProxyStep OnVerdict(size_t unit, bool verdict) override {
        ProxyStep step;
        step.resolve.emplace_back(unit, verdict);
        ++verdicts_;
        if (!verdict) {
            return step;
        }
        ++true_verdicts_;
        if (units_[unit].forced) {
            return step;
        }
        const uint32_t g = units_[unit].group;
        const uint64_t r = rank_[unit];
        Extend(g, r + opt_.gap, step, sent_extend_);
        if (audit_[unit]) {
            ++audit_hits_;
            if (r > k_global_) {
                k_global_ = r + opt_.gap;
                for (size_t other = 0; other < groups_.size(); ++other) {
                    Extend(static_cast<uint32_t>(other), k_global_, step, sent_widen_);
                }
            }
        }
        return step;
    }

    std::string Stats() const override {
        uint64_t max_window = 0;
        for (auto w : window_) max_window = std::max(max_window, w);
        return nlohmann::json{{"mode", "embedding_topk"},
                              {"units", units_.size()},
                              {"groups", groups_.size()},
                              {"k_min", opt_.k_min},
                              {"k_global_final", k_global_},
                              {"max_window_final", max_window},
                              {"sent_total", sent_total_},
                              {"sent_forced", sent_forced_},
                              {"sent_topk", sent_topk_},
                              {"sent_extend", sent_extend_},
                              {"sent_widen", sent_widen_},
                              {"sent_audit_near", sent_audit_near_},
                              {"sent_audit_far", sent_audit_far_},
                              {"audit_hits", audit_hits_},
                              {"verdicts", verdicts_},
                              {"true_verdicts", true_verdicts_}}
                .dump();
    }

private:
    void Send(size_t unit, ProxyStep& step) {
        if (sent_[unit]) return;
        sent_[unit] = true;
        ++sent_total_;
        step.send.push_back(unit);
    }

    void Extend(uint32_t g, uint64_t target, ProxyStep& step, uint64_t& counter) {
        if (target <= window_[g]) return;
        const auto& members = groups_[g];
        const uint64_t limit = std::min<uint64_t>(target, members.size());
        for (uint64_t r = window_[g] + 1; r <= limit; ++r) {
            const size_t unit = members[r - 1];
            if (!sent_[unit]) {
                Send(unit, step);
                ++counter;
            }
        }
        window_[g] = target;
    }

    std::vector<ProxyUnit> units_;
    TopKOptions opt_;
    std::vector<std::vector<size_t>> groups_;
    std::vector<uint64_t> rank_;  // 1-based rank within its group (0 = forced)
    std::vector<bool> sent_;
    std::vector<bool> audit_;
    std::vector<uint64_t> window_;
    uint64_t k_global_ = 0;
    uint64_t sent_total_ = 0, sent_forced_ = 0, sent_topk_ = 0, sent_extend_ = 0, sent_widen_ = 0;
    uint64_t sent_audit_near_ = 0, sent_audit_far_ = 0, audit_hits_ = 0, verdicts_ = 0, true_verdicts_ = 0;
};

// =============================================================================
// LOTUS-style cascade policy (ablation)
// =============================================================================

double Clip01(double x) { return std::min(1.0, std::max(0.0, x)); }

class LotusPolicy : public ProxyPolicy {
public:
    LotusPolicy(std::vector<ProxyUnit> units, std::vector<size_t> pair_units, const LotusOptions& options)
        : units_(std::move(units)), pairs_(std::move(pair_units)), opt_(options), known_(units_.size(), false),
          verdict_(units_.size(), false), sent_(units_.size(), false) {}

    ProxyStep Start() override {
        ProxyStep step;
        for (size_t i = 0; i < units_.size(); ++i) {
            if (units_[i].forced) Send(i, step);
        }
        // importance_sampling(): weights over ALL pairs, draws restricted to the
        // first max_sample_range pairs, with replacement.
        const size_t n = pairs_.size();
        if (n > 0) {
            std::vector<double> w(n);
            double sum = 0.0;
            for (size_t i = 0; i < n; ++i) {
                w[i] = std::sqrt(Clip01(units_[pairs_[i]].score));
                sum += w[i];
            }
            for (size_t i = 0; i < n; ++i) {
                // All-zero scores make LOTUS divide by zero; use the uniform part only.
                const double importance = sum > 0.0 ? w[i] / sum : 0.0;
                w[i] = opt_.is_weight * importance + (1.0 - opt_.is_weight) / static_cast<double>(n);
            }
            const size_t range = std::min<size_t>(opt_.max_sample_range, n);
            const auto sample_size = static_cast<size_t>(opt_.sampling_percentage * static_cast<double>(n));
            std::mt19937_64 rng(opt_.seed);
            std::discrete_distribution<size_t> pick(w.begin(), w.begin() + static_cast<std::ptrdiff_t>(range));
            sample_.reserve(sample_size);
            for (size_t s = 0; s < sample_size; ++s) {
                const size_t idx = pick(rng);
                sample_.push_back(idx);
                correction_.push_back((1.0 / static_cast<double>(n)) / w[idx]);
            }
        }
        for (size_t idx : sample_) {
            const size_t unit = pairs_[idx];
            if (!sent_[unit]) {
                Send(unit, step);
                ++sent_sample_;
                awaiting_.insert(unit);
            }
        }
        if (awaiting_.empty()) {
            Learn(step);
        }
        return step;
    }

    ProxyStep OnVerdict(size_t unit, bool verdict) override {
        ProxyStep step;
        known_[unit] = true;
        verdict_[unit] = verdict;
        if (units_[unit].forced || learned_) {
            step.resolve.emplace_back(unit, verdict);
            return step;
        }
        awaiting_.erase(unit);
        if (awaiting_.empty()) {
            Learn(step);
        }
        return step;
    }

    std::string Stats() const override {
        return nlohmann::json{{"mode", "lotus_cascade"},
                              {"pairs", pairs_.size()},
                              {"units", units_.size()},
                              {"sample_pairs", sample_.size()},
                              {"sent_sample_units", sent_sample_},
                              {"tau_pos", tau_pos_},
                              {"tau_neg", tau_neg_},
                              {"auto_accept_units", auto_accept_},
                              {"auto_reject_units", auto_reject_},
                              {"reused_sample_verdicts", reused_},
                              {"sent_uncertain_units", sent_uncertain_},
                              {"sent_total", sent_total_}}
                .dump();
    }

private:
    void Send(size_t unit, ProxyStep& step) {
        if (sent_[unit]) return;
        sent_[unit] = true;
        ++sent_total_;
        step.send.push_back(unit);
    }

    void Learn(ProxyStep& step) {
        std::vector<double> scores;
        std::vector<bool> oracle;
        scores.reserve(sample_.size());
        oracle.reserve(sample_.size());
        for (size_t idx : sample_) {
            scores.push_back(Clip01(units_[pairs_[idx]].score));
            oracle.push_back(verdict_[pairs_[idx]]);
        }
        std::tie(tau_pos_, tau_neg_) = LotusLearnThresholds(scores, oracle, correction_, opt_);
        learned_ = true;
        for (size_t u = 0; u < units_.size(); ++u) {
            if (units_[u].forced) continue;
            const double s = Clip01(units_[u].score);
            if (s >= tau_pos_) {
                step.resolve.emplace_back(u, true);  // helper positive: accepted without the main model
                ++auto_accept_;
            } else if (s <= tau_neg_) {
                ++auto_reject_;  // helper negative: rejected
            } else if (known_[u]) {
                step.resolve.emplace_back(u, verdict_[u]);  // sample verdict reused
                ++reused_;
            } else if (!sent_[u]) {
                Send(u, step);
                ++sent_uncertain_;
            }
        }
    }

    std::vector<ProxyUnit> units_;
    std::vector<size_t> pairs_;
    LotusOptions opt_;
    std::vector<bool> known_;
    std::vector<bool> verdict_;
    std::vector<bool> sent_;
    std::vector<size_t> sample_;       // pair indices, with replacement
    std::vector<double> correction_;   // per sample draw
    std::unordered_set<size_t> awaiting_;
    bool learned_ = false;
    double tau_pos_ = 1.0, tau_neg_ = 0.0;
    uint64_t sent_total_ = 0, sent_sample_ = 0, sent_uncertain_ = 0;
    uint64_t auto_accept_ = 0, auto_reject_ = 0, reused_ = 0;
};

// --- learn_cascade_thresholds helpers (names follow the Python source) ------

struct ScoredPair {
    double score;
    bool oracle;
    double correction;
};

double Recall(double pos, double neg, const std::vector<ScoredPair>& pairs) {
    double total_correct = 0.0;
    for (const auto& x : pairs) total_correct += (x.oracle ? 1.0 : 0.0) * x.correction;
    if (!(total_correct > 0.0)) return 0.0;
    double accepted = 0.0;
    double oracle = 0.0;
    for (const auto& x : pairs) {
        if (x.score >= pos || x.score <= neg) {
            if (x.score >= pos && x.oracle) accepted += 1.0;
        } else {
            oracle += (x.oracle ? 1.0 : 0.0) * x.correction;
        }
    }
    return (accepted + oracle) / total_correct;
}

double CalculateTauNeg(const std::vector<ScoredPair>& pairs, double tau_pos, double recall_target) {
    bool found = false;
    double best = 0.0;
    for (auto it = pairs.rbegin(); it != pairs.rend(); ++it) {
        if (Recall(tau_pos, it->score, pairs) >= recall_target) {
            best = found ? std::max(best, it->score) : it->score;
            found = true;
        }
    }
    return found ? best : 0.0;
}

void MeanStd(const std::vector<double>& z, double& mean, double& std_dev) {
    mean = 0.0;
    std_dev = 0.0;
    if (z.empty()) return;
    for (double v : z) mean += v;
    mean /= static_cast<double>(z.size());
    double var = 0.0;
    for (double v : z) var += (v - mean) * (v - mean);
    std_dev = std::sqrt(var / static_cast<double>(z.size()));  // numpy population std
}

double Ub(double mean, double std_dev, double s, double delta) {
    return mean + (std_dev / std::sqrt(s)) * std::sqrt(2.0 * std::log(1.0 / delta));
}

double Lb(double mean, double std_dev, double s, double delta) {
    return mean - (std_dev / std::sqrt(s)) * std::sqrt(2.0 * std::log(1.0 / delta));
}

}  // namespace

std::pair<double, double> LotusLearnThresholds(const std::vector<double>& scores, const std::vector<bool>& oracle,
                                               const std::vector<double>& corrections, const LotusOptions& options) {
    if (scores.size() != oracle.size() || scores.size() != corrections.size()) {
        throw std::invalid_argument("LotusLearnThresholds: size mismatch");
    }
    if (scores.empty()) {
        return {1.0, 0.0};  // LOTUS: learning raises (division by zero) -> full join
    }
    std::vector<ScoredPair> pairs;
    pairs.reserve(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) pairs.push_back({scores[i], oracle[i], corrections[i]});
    // sorted(..., key=score, reverse=True) is stable.
    std::stable_sort(pairs.begin(), pairs.end(),
                     [](const ScoredPair& a, const ScoredPair& b) { return a.score > b.score; });
    const double n = static_cast<double>(pairs.size());

    double tau_neg = CalculateTauNeg(pairs, 1.0, options.recall_target);

    std::vector<double> z1, z2;
    for (const auto& x : pairs) {
        (x.score >= tau_neg ? z1 : z2).push_back((x.oracle ? 1.0 : 0.0) * x.correction);
    }
    double m1, s1, m2, s2;
    MeanStd(z1, m1, s1);
    MeanStd(z2, m2, s2);
    const double ub_z1 = Ub(m1, s1, n, options.failure_probability / 2.0);
    const double lb_z2 = Lb(m2, s2, n, options.failure_probability / 2.0);
    double corrected = (ub_z1 + lb_z2 == 0.0) ? 1.0 : ub_z1 / (ub_z1 + lb_z2);
    corrected = std::min(1.0, corrected);
    tau_neg = CalculateTauNeg(pairs, 1.0, corrected);

    double min_candidate = 1.0;
    std::vector<double> z;
    for (const auto& candidate : pairs) {
        z.clear();
        for (const auto& x : pairs) {
            if (x.score >= candidate.score) z.push_back(x.oracle ? 1.0 : 0.0);
        }
        double mean, std_dev;
        MeanStd(z, mean, std_dev);
        const double p_l = Lb(mean, std_dev, static_cast<double>(z.size()), options.failure_probability / n);
        if (p_l > options.precision_target) {
            min_candidate = std::min(min_candidate, candidate.score);
        }
    }
    return {std::max(tau_neg, min_candidate), tau_neg};
}

std::unique_ptr<ProxyPolicy> MakeTopKPolicy(std::vector<ProxyUnit> units, const TopKOptions& options) {
    return std::make_unique<TopKPolicy>(std::move(units), options);
}

std::unique_ptr<ProxyPolicy> MakeLotusPolicy(std::vector<ProxyUnit> units, std::vector<size_t> pair_units,
                                             const LotusOptions& options) {
    return std::make_unique<LotusPolicy>(std::move(units), std::move(pair_units), options);
}

}  // namespace flock
