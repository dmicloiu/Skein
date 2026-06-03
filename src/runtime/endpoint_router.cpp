#include "flock/runtime/endpoint_router.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace flock {
namespace {

// FNV-1a, 64-bit. Fixed offset basis + prime so StickyByPrefix routing is
// reproducible across processes -- important for benchmarking.
uint64_t fnv1a(const std::string& s) {
    uint64_t h = 1469598103934665603ULL;  // offset basis
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ULL;  // prime
    }
    return h;
}

}  // namespace

struct EndpointRouter::Impl {
    mutable std::mutex mu;
    std::vector<std::string> endpoints;
    Strategy strategy;
    uint64_t round_robin_counter = 0;
    std::vector<size_t> loads;  // in-flight count per endpoint, parallel to endpoints

    Impl(std::vector<std::string> eps, Strategy s)
        : endpoints(std::move(eps)), strategy(s), loads(endpoints.size(), 0) {}

    // Computes the target index for the given key. Caller holds mu and has
    // already checked that endpoints is non-empty.
    size_t SelectIndex(const std::string& prefix_key) {
        const size_t n = endpoints.size();
        switch (strategy) {
            case Strategy::Single:
                return 0;
            case Strategy::RoundRobin:
                return round_robin_counter++ % n;
            case Strategy::StickyByPrefix:
                if (prefix_key.empty()) {
                    return round_robin_counter++ % n;  // RR fallback
                }
                return static_cast<size_t>(fnv1a(prefix_key) % n);
            case Strategy::LeastLoaded: {
                size_t best = 0;
                for (size_t i = 1; i < n; ++i) {
                    if (loads[i] < loads[best]) best = i;  // ties -> lowest index
                }
                return best;
            }
        }
        return 0;  // unreachable; silences -Wreturn-type
    }
};

EndpointRouter::Strategy EndpointRouter::ParseStrategy(const std::string& s) {
    if (s == "single") return Strategy::Single;
    if (s == "round_robin") return Strategy::RoundRobin;
    if (s == "sticky_by_prefix") return Strategy::StickyByPrefix;
    if (s == "least_loaded") return Strategy::LeastLoaded;
    throw std::runtime_error("EndpointRouter: unknown routing strategy '" + s + "'");
}

const char* EndpointRouter::StrategyName(Strategy s) {
    switch (s) {
        case Strategy::Single:
            return "single";
        case Strategy::RoundRobin:
            return "round_robin";
        case Strategy::StickyByPrefix:
            return "sticky_by_prefix";
        case Strategy::LeastLoaded:
            return "least_loaded";
    }
    return "single";  // unreachable; silences -Wreturn-type
}

EndpointRouter::EndpointRouter(std::vector<std::string> endpoints, Strategy strategy)
    : impl_(std::make_unique<Impl>(std::move(endpoints), strategy)) {}

EndpointRouter::~EndpointRouter() = default;

EndpointRouter::Pick EndpointRouter::Choose(const std::string& prefix_key) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (impl_->endpoints.empty()) {
        throw std::runtime_error("EndpointRouter::Choose: no endpoints configured");
    }
    const size_t index = impl_->SelectIndex(prefix_key);
    impl_->loads[index]++;
    return Pick{index, impl_->endpoints[index]};  // url copied by value
}

void EndpointRouter::OnComplete(size_t index) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (index < impl_->loads.size() && impl_->loads[index] > 0) {
        impl_->loads[index]--;
    }
}

void EndpointRouter::SetEndpoints(std::vector<std::string> endpoints) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->endpoints = std::move(endpoints);
    impl_->loads.assign(impl_->endpoints.size(), 0);  // reset in-flight counters
}

void EndpointRouter::SetStrategy(Strategy s) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->strategy = s;
}

size_t EndpointRouter::EndpointCount() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->endpoints.size();
}

size_t EndpointRouter::InFlight(size_t index) const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return index < impl_->loads.size() ? impl_->loads[index] : 0;
}

}  // namespace flock
