// Pure tests of the embedding-proxy decision policies (no DuckDB, no I/O):
// reject-only per-group top-k with extension + audit widening, and the
// LOTUS-style cascade (threshold port checked against lotus 1.x Python values).

#include "flock/functions/operator/proxy_policy.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <set>
#include <vector>

namespace flock {
namespace {

// groups x per_group units; unit g*per_group + i has rank i+1 in group g.
std::vector<ProxyUnit> RankedUnits(uint32_t groups, size_t per_group) {
    std::vector<ProxyUnit> units;
    for (uint32_t g = 0; g < groups; ++g) {
        for (size_t i = 0; i < per_group; ++i) {
            units.push_back({g, 1.0 - 0.01 * static_cast<double>(i), false});
        }
    }
    return units;
}

std::set<size_t> AsSet(const std::vector<size_t>& v) { return {v.begin(), v.end()}; }

TopKOptions NoAudit(uint64_t k, uint64_t gap) {
    TopKOptions o;
    o.k_min = k;
    o.gap = gap;
    o.audit_near = 0.0;
    o.audit_far = 0.0;
    return o;
}

TEST(ProxyPolicy, LotusThresholdsMatchLotusPython) {
    // Reference: lotus.sem_ops.cascade_utils.learn_cascade_thresholds on the same
    // inputs (sembench env, 2026-11-05).
    const std::vector<double> s = {0.05, 0.236, 0.422, 0.608, 0.794, 0.98, 0.205, 0.391, 0.577, 0.763,
                                   0.949, 0.174, 0.36, 0.546, 0.732, 0.918, 0.143, 0.329, 0.515, 0.701,
                                   0.887, 0.112, 0.298, 0.484, 0.67, 0.856, 0.081, 0.267, 0.453, 0.639};
    const int o[] = {1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1};
    std::vector<double> c;
    std::vector<bool> oracle;
    for (size_t i = 0; i < s.size(); ++i) {
        c.push_back(0.5 + 0.1 * static_cast<double>(i % 5));
        oracle.push_back(o[i] != 0);
    }
    LotusOptions defaults;
    auto t = LotusLearnThresholds(s, oracle, c, defaults);
    EXPECT_DOUBLE_EQ(t.first, 0.608);
    EXPECT_DOUBLE_EQ(t.second, 0.081);

    LotusOptions strict;
    strict.recall_target = 0.95;
    strict.precision_target = 0.9;
    strict.failure_probability = 0.1;
    t = LotusLearnThresholds(s, oracle, c, strict);
    EXPECT_DOUBLE_EQ(t.first, 0.67);
    EXPECT_DOUBLE_EQ(t.second, 0.67);

    t = LotusLearnThresholds(s, std::vector<bool>(s.size(), false), c, defaults);
    EXPECT_DOUBLE_EQ(t.first, 1.0);
    EXPECT_DOUBLE_EQ(t.second, 0.0);

    t = LotusLearnThresholds({}, {}, {}, defaults);  // LOTUS falls back to the full join
    EXPECT_DOUBLE_EQ(t.first, 1.0);
    EXPECT_DOUBLE_EQ(t.second, 0.0);
}

TEST(ProxyPolicy, TopKStartSendsTopKPerGroup) {
    auto policy = MakeTopKPolicy(RankedUnits(3, 10), NoAudit(2, 2));
    const auto step = policy->Start();
    EXPECT_EQ(AsSet(step.send), (std::set<size_t>{0, 1, 10, 11, 20, 21}));
    EXPECT_TRUE(step.resolve.empty());
}

TEST(ProxyPolicy, TopKGroupSmallerThanK) {
    auto policy = MakeTopKPolicy(RankedUnits(2, 2), NoAudit(5, 2));
    EXPECT_EQ(AsSet(policy->Start().send), (std::set<size_t>{0, 1, 2, 3}));
}

TEST(ProxyPolicy, TopKUsesScoreNotInputOrder) {
    std::vector<ProxyUnit> units = {{0, 0.1, false}, {0, 0.9, false}, {0, 0.5, false}, {0, 0.7, false}};
    auto policy = MakeTopKPolicy(units, NoAudit(2, 0));
    EXPECT_EQ(AsSet(policy->Start().send), (std::set<size_t>{1, 3}));
}

TEST(ProxyPolicy, TopKVerdictResolvesAndTrueExtends) {
    auto policy = MakeTopKPolicy(RankedUnits(2, 10), NoAudit(3, 2));
    policy->Start();
    // FALSE: resolved, no follow-up.
    auto step = policy->OnVerdict(0, false);
    ASSERT_EQ(step.resolve.size(), 1u);
    EXPECT_EQ(step.resolve[0], std::make_pair(size_t{0}, false));
    EXPECT_TRUE(step.send.empty());
    // TRUE at rank 2 -> window r + gap = 4 -> sends rank 4.
    step = policy->OnVerdict(1, true);
    EXPECT_EQ(AsSet(step.send), (std::set<size_t>{3}));
    // TRUE at rank 3 -> window 5 -> sends rank 5 (rank 4 already sent).
    step = policy->OnVerdict(2, true);
    EXPECT_EQ(AsSet(step.send), (std::set<size_t>{4}));
    // The other group is untouched by group-local extension.
    step = policy->OnVerdict(10, true);
    EXPECT_TRUE(step.send.empty());
}

TEST(ProxyPolicy, TopKAuditHitWidensEveryGroup) {
    TopKOptions o = NoAudit(2, 2);
    o.audit_near = 1.0;  // audit all of ranks 3..4
    auto policy = MakeTopKPolicy(RankedUnits(2, 10), o);
    EXPECT_EQ(AsSet(policy->Start().send), (std::set<size_t>{0, 1, 2, 3, 10, 11, 12, 13}));
    // Audit TRUE at rank 4 of group 0 -> k = 6 everywhere.
    const auto step = policy->OnVerdict(3, true);
    EXPECT_EQ(AsSet(step.send), (std::set<size_t>{4, 5, 14, 15}));
    EXPECT_NE(policy->Stats().find("\"audit_hits\":1"), std::string::npos);
    EXPECT_NE(policy->Stats().find("\"k_global_final\":6"), std::string::npos);
}

TEST(ProxyPolicy, TopKForcedUnitsAlwaysAsked) {
    auto units = RankedUnits(1, 6);
    units.push_back({0, 0.0, true});
    auto policy = MakeTopKPolicy(units, NoAudit(1, 0));
    EXPECT_EQ(AsSet(policy->Start().send), (std::set<size_t>{0, 6}));
    const auto step = policy->OnVerdict(6, true);
    EXPECT_TRUE(step.send.empty());
    ASSERT_EQ(step.resolve.size(), 1u);
}

TEST(ProxyPolicy, TopKAuditIsSeededAndRoughlyAtRate) {
    TopKOptions o = NoAudit(5, 2);
    o.audit_near = 0.25;
    o.audit_far = 0.01;
    auto a = MakeTopKPolicy(RankedUnits(100, 200), o);
    auto b = MakeTopKPolicy(RankedUnits(100, 200), o);
    const auto sa = a->Start().send;
    EXPECT_EQ(sa, b->Start().send);
    // 100 groups: 500 top-k, ~125 near (25% of 500), ~190 far (1% of 19,000).
    EXPECT_GT(sa.size(), 815u - 60u);
    EXPECT_LT(sa.size(), 815u + 60u);
    o.seed = 7;
    EXPECT_NE(sa, MakeTopKPolicy(RankedUnits(100, 200), o)->Start().send);
}

TEST(ProxyPolicy, LotusCascadeResolvesEachUnitOnce) {
    // 1 left row x 60 candidates, one pair per unit, descending scores.
    std::vector<ProxyUnit> units;
    std::vector<size_t> pairs;
    for (size_t i = 0; i < 60; ++i) {
        units.push_back({0, 0.95 - 0.015 * static_cast<double>(i), false});
        pairs.push_back(i);
    }
    LotusOptions o;
    o.sampling_percentage = 0.5;  // 30 draws from the first 200 (= all 60) pairs
    auto policy = MakeLotusPolicy(units, pairs, o);
    const auto truth = [&](size_t u) { return units[u].score > 0.6; };

    std::vector<size_t> queue;
    std::map<size_t, bool> resolved;
    std::set<size_t> sent;
    const auto apply = [&](const ProxyStep& step) {
        for (size_t u : step.send) {
            EXPECT_TRUE(sent.insert(u).second) << "unit sent twice: " << u;
            queue.push_back(u);
        }
        for (const auto& r : step.resolve) {
            EXPECT_TRUE(resolved.emplace(r.first, r.second).second) << "unit resolved twice: " << r.first;
        }
    };
    apply(policy->Start());
    EXPECT_FALSE(queue.empty());
    EXPECT_TRUE(resolved.empty());  // sample verdicts are held until thresholds are learned
    while (!queue.empty()) {
        const size_t u = queue.back();
        queue.pop_back();
        apply(policy->OnVerdict(u, truth(u)));
    }
    // Every unit is either resolved or (implicitly) rejected; accepted units are
    // true positives here, since the helper ranks perfectly.
    for (const auto& r : resolved) {
        if (r.second) EXPECT_TRUE(truth(r.first)) << r.first;
    }
    EXPECT_NE(policy->Stats().find("lotus_cascade"), std::string::npos);
}

TEST(ProxyPolicy, LotusAutoAcceptSkipsTheMainModel) {
    // All helper scores high and every sampled verdict TRUE -> tau_pos drops and
    // units above it are accepted WITHOUT being sent (the risky side of LOTUS).
    std::vector<ProxyUnit> units;
    std::vector<size_t> pairs;
    for (size_t i = 0; i < 400; ++i) {
        units.push_back({0, 0.99 - 0.001 * static_cast<double>(i), false});
        pairs.push_back(i);
    }
    auto policy = MakeLotusPolicy(units, pairs, LotusOptions());
    auto step = policy->Start();
    std::vector<size_t> queue = step.send;
    std::set<size_t> sent(queue.begin(), queue.end());
    size_t accepted_unsent = 0;
    while (!queue.empty()) {
        const size_t u = queue.back();
        queue.pop_back();
        step = policy->OnVerdict(u, true);
        for (size_t v : step.send) {
            sent.insert(v);
            queue.push_back(v);
        }
        for (const auto& r : step.resolve) {
            if (r.second && !sent.count(r.first)) ++accepted_unsent;
        }
    }
    EXPECT_GT(accepted_unsent, 0u);
    EXPECT_LT(sent.size(), units.size());
}

}  // namespace
}  // namespace flock
