#include "flock/runtime/extension_state.h"

#include "duckdb.hpp"

#include <gtest/gtest.h>

namespace flock {

// Provided by semantic_test_main.cpp.
duckdb::DuckDB& TestDB();

// Get returns the same instance for the same DatabaseInstance (pointer identity),
// across both the DatabaseInstance and ClientContext overloads.
TEST(ExtensionStateTest, GetIsSingletonPerDatabase) {
    auto& db = TestDB();
    auto& a = ExtensionState::Get(*db.instance);
    auto& b = ExtensionState::Get(*db.instance);
    EXPECT_EQ(&a, &b);

    duckdb::Connection con(db);
    auto& c = ExtensionState::Get(*con.context);
    EXPECT_EQ(&a, &c);

    // A second connection still resolves to the same DB-lifetime instance.
    duckdb::Connection con2(db);
    auto& d = ExtensionState::Get(*con2.context);
    EXPECT_EQ(&a, &d);
}

// client and router are constructed and non-null.
TEST(ExtensionStateTest, ClientAndRouterNonNull) {
    auto& state = ExtensionState::Get(*TestDB().instance);
    EXPECT_NE(state.client, nullptr);
    EXPECT_NE(state.router, nullptr);
}

// The default router has at least the one default endpoint.
TEST(ExtensionStateTest, RouterHasDefaultEndpoint) {
    auto& state = ExtensionState::Get(*TestDB().instance);
    EXPECT_GE(state.router->EndpointCount(), 1u);
}

}  // namespace flock
