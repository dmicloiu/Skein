#include "duckdb.hpp"
#include "flock/core/config.hpp"

#include <gtest/gtest.h>

#include <memory>

namespace flock {

// One DuckDB instance for the whole semantic-config suite. Constructing it
// auto-loads the statically-linked flock extension (load_extensions defaults
// to true), which runs LoadInternal -> RegisterSemanticSettings (so the
// semantic_* options exist) and Config::Configure (so the model tables exist).
// Mirrors test/unit's GlobalTestEnvironment.
class SemanticConfigEnvironment : public ::testing::Environment {
public:
    static duckdb::DuckDB* db;

    void SetUp() override {
        owned_db = std::make_unique<duckdb::DuckDB>("flock_semantic_test.db");
        db = owned_db.get();
        // Point Config's global connection at this instance, like the unit suite.
        Config::GetConnection(&*owned_db->instance);
    }

    void TearDown() override {
        db = nullptr;
        owned_db.reset();
    }

private:
    std::unique_ptr<duckdb::DuckDB> owned_db;
};

duckdb::DuckDB* SemanticConfigEnvironment::db = nullptr;

duckdb::DuckDB& TestDB() { return *SemanticConfigEnvironment::db; }

}  // namespace flock

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new flock::SemanticConfigEnvironment());
    return RUN_ALL_TESTS();
}
