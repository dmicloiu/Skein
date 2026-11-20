#include "flock/core/config.hpp"

#include <vector>
#include "duckdb/common/file_system.hpp"
#include "duckdb/main/extension_helper.hpp"
#include "flock/secret_manager/secret_manager.hpp"
#include <chrono>
#include <thread>
#include <fmt/format.h>

namespace flock {

duckdb::DatabaseInstance* Config::db;

std::string Config::get_schema_name() { return "flock_config"; }

std::filesystem::path Config::get_global_storage_path() {
#ifdef __EMSCRIPTEN__
    return std::filesystem::path("opfs://skein_data/skein.db");
#else
    const auto& home = duckdb::FileSystem::GetHomeDirectory(nullptr);
    if (home.empty()) {
        throw std::runtime_error("Could not find home directory");
    }
    return std::filesystem::path(home) / ".duckdb" / "skein_storage" / "skein.db";
#endif
}

duckdb::Connection Config::GetConnection(duckdb::DatabaseInstance* db) {
    if (db) {
        Config::db = db;
    }
    duckdb::Connection con(*Config::db);
    return con;
}


void Config::SetupGlobalStorageLocation(duckdb::DatabaseInstance* db_instance) {
    if (!db_instance) {
        return;
    }
#ifdef __EMSCRIPTEN__
    // WASM: Client registers OPFS files before loading extension
    return;
#endif
    auto& fs = duckdb::FileSystem::GetFileSystem(*db_instance);
    const auto dir = get_global_storage_path().parent_path();
    // CreateDirectory is not recursive: on a fresh home (no ~/.duckdb yet) create
    // every missing ancestor first, outermost to innermost.
    std::vector<std::filesystem::path> missing;
    for (auto p = dir; !p.empty() && p != p.root_path(); p = p.parent_path()) {
        if (fs.DirectoryExists(p.string())) {
            break;
        }
        missing.push_back(p);
    }
    for (auto it = missing.rbegin(); it != missing.rend(); ++it) {
        try {
            fs.CreateDirectory(it->string());
        } catch (const std::exception& e) {
            std::cerr << "Error creating directory " << it->string() << ": " << e.what() << std::endl;
            return;
        }
    }
}

void Config::ConfigSchema(duckdb::Connection& con, std::string& schema_name) {
    con.Query(duckdb_fmt::format("CREATE SCHEMA IF NOT EXISTS {};", schema_name));
}

void Config::ConfigureGlobal(duckdb::DatabaseInstance* db_instance) {
    if (!db_instance) {
        return;
    }
    // Use the already-attached flock_storage database
    auto con = Config::GetConnection(db_instance);
    // Switch to flock_storage so ConfigureTables creates tables there.
    // We switch back to memory afterward to avoid leaving the connection
    // pointing at flock_storage, which would affect subsequent queries.
    auto use_result = con.Query("USE flock_storage;");
    if (use_result->HasError()) {
        std::cerr << "Failed to USE flock_storage: " << use_result->GetError() << std::endl;
        return;
    }
    ConfigureTables(con, ConfigType::GLOBAL);
    con.Query("USE memory;");
}

void Config::ConfigureLocal(duckdb::DatabaseInstance& db) {
    auto con = Config::GetConnection(&db);
    ConfigureTables(con, ConfigType::LOCAL);

    const std::string global_path = get_global_storage_path().string();
    auto result = con.Query(
            duckdb_fmt::format("ATTACH DATABASE '{}' AS flock_storage;", global_path));
    if (result->HasError()) {
        std::cerr << "Failed to attach flock_storage: " << result->GetError() << std::endl;
    }
}

void Config::ConfigureTables(duckdb::Connection& con, const ConfigType type) {
    con.BeginTransaction();
    std::string schema = Config::get_schema_name();
    ConfigSchema(con, schema);
    ConfigModelTable(con, schema, type);
    ConfigPromptTable(con, schema, type);
    con.Commit();
}

void Config::Configure(duckdb::ExtensionLoader& loader) {
    Registry::Register(loader);
    SecretManager::Register(loader);
    auto& db = loader.GetDatabaseInstance();
    // The config tables use the JSON type. When extensions are linked statically
    // (e.g. DuckDB's unittest runner) json is not necessarily loaded before us,
    // and every CREATE TABLE would fail silently; make sure it is available.
    if (!db.ExtensionIsLoaded("json") && !duckdb::ExtensionHelper::TryAutoLoadExtension(db, "json")) {
#ifndef SKEIN_LOADABLE_BUILD
        // Static builds only (ExtensionHelper::LoadExtension is not exported to loadable
        // extensions): autoloading may be off, but a json linked into this binary can
        // still be loaded directly.
        duckdb::DuckDB handle(db);
        duckdb::ExtensionHelper::LoadExtension(handle, "json");
#endif
    }
    const auto db_path = db.config.options.database_path;
    const std::string global_path = get_global_storage_path().string();

    // If the main database is already at the global storage path, still attach for WASM :memory: case
    if (db_path == global_path) {
        auto con = GetConnection(&db);
        ConfigureTables(con, ConfigType::LOCAL);
        ConfigureTables(con, ConfigType::GLOBAL);
#ifdef __EMSCRIPTEN__
        ConfigureLocal(db);
#endif
        return;
    }

    SetupGlobalStorageLocation(&db);
    ConfigureLocal(db);
    ConfigureGlobal(&db);
}

void Config::AttachToGlobalStorage(duckdb::Connection& con, bool read_only) {
    con.Query(duckdb_fmt::format("ATTACH DATABASE '{}' AS flock_storage {};",
                                 Config::get_global_storage_path().string(), read_only ? "(READ_ONLY)" : ""));
}

void Config::DetachFromGlobalStorage(duckdb::Connection& con) {
    con.Query("DETACH DATABASE flock_storage;");
}

bool Config::StorageAttachmentGuard::TryAttach(bool read_only) {
    try {
        Config::AttachToGlobalStorage(connection, read_only);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool Config::StorageAttachmentGuard::TryDetach() {
    try {
        Config::DetachFromGlobalStorage(connection);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

void Config::StorageAttachmentGuard::Wait(int milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

Config::StorageAttachmentGuard::StorageAttachmentGuard(duckdb::Connection& con, bool read_only)
    : connection(con), attached(false) {
    for (int attempt = 0; attempt < MAX_RETRIES; ++attempt) {
        if (TryAttach(read_only)) {
            attached = true;
            return;
        }
        if (attempt < MAX_RETRIES - 1) {
            Wait(RETRY_DELAY_MS);
        }
    }
    Config::AttachToGlobalStorage(connection, read_only);
    attached = true;
}

Config::StorageAttachmentGuard::~StorageAttachmentGuard() {
    if (attached) {
        try {
            Config::DetachFromGlobalStorage(connection);
        } catch (...) {
        }
    }
}

}// namespace flock
