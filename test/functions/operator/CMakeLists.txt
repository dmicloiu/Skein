find_package(GTest CONFIG REQUIRED)
find_package(Threads REQUIRED)

# Threading test for the async semantic-operator dispatch engine.
#
# It compiles the engine + EndpointRouter directly and drives SemGlobalSinkState
# with a FakeLLMClient, so it needs NO curl (no AsyncLLMClient), but it DOES
# need duckdb's DataChunk / Value / InterruptState, so it links duckdb_static
# (same group recipe as test/runtime's semantic_config_tests, minus
# flock_extension, since the engine sources are compiled in directly here).
add_executable(flock_semantic_operator_tests
    ${PROJECT_SOURCE_DIR}/src/functions/operator/semantic_operator_base.cpp
    ${PROJECT_SOURCE_DIR}/src/runtime/endpoint_router.cpp
    ${PROJECT_SOURCE_DIR}/src/runtime/llm_client.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/semantic_operator_base_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/extension_loader_stub.cpp)

target_include_directories(flock_semantic_operator_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/src/include
    ${CMAKE_CURRENT_SOURCE_DIR})

# The test never opens a database or loads an extension. So we link
# duckdb_static directly, WITHOUT duckdb_generated_extension_loader or the
# bundled extensions. jemalloc is duckdb's allocator on Linux; keep it in the
# group so duckdb_static's allocation symbols resolve.
# The residual test opens an in-memory DuckDB for an ExpressionExecutor context,
# which pulls duckdb_static's strong LogicalType::X constants; -fno-gnu-unique
# makes our objects emit weak linkonce_odr for those, avoiding a multiple-def
# clash (STB_GNU_UNIQUE vs the archive's strong def).
target_compile_options(flock_semantic_operator_tests PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-fno-gnu-unique>)

target_link_libraries(flock_semantic_operator_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
    nlohmann_json::nlohmann_json
    $<$<CXX_COMPILER_ID:GNU>:-Wl,--start-group>
    $<$<PLATFORM_ID:Linux>:jemalloc_extension>
    duckdb_static
    $<$<CXX_COMPILER_ID:GNU>:-Wl,--end-group>
    Threads::Threads)

# Opt-in ThreadSanitizer for the THREADING-CRITICAL component: -DFLOCK_OPERATOR_TSAN=ON.
#
# Unlike the standalone TSan targets in test/runtime, this one links the giant
# duckdb_static archive, which on ARM64 hits the R_AARCH64_CALL26 overflow when
# the whole thing is instrumented (see the FLOCK ARM64 build note). We dodge that
# by instrumenting ONLY this target's own objects (engine + router + test) and
# linking the PREBUILT, UNINSTRUMENTED duckdb_static: TSan intercepts pthread /
# atomic ops at the library boundary regardless, so races in the engine's shared
# state are still detected, while duckdb's
# .text size (and thus the relocation range) is unchanged.
#
# Disable inherited UBSan: TSan + UBSan together trip spurious reports inside
# UBSan's own interceptors at std::thread teardown.
#
# Running the TSan binary on this GH200 kernel needs ASLR lowered, e.g.
#     setarch $(uname -m) -R ./flock_semantic_operator_tests
# (high-entropy ASLR places mappings outside TSan's expected shadow layout).
option(FLOCK_OPERATOR_TSAN "Build flock_semantic_operator_tests with ThreadSanitizer" OFF)
if(FLOCK_OPERATOR_TSAN)
    target_compile_options(flock_semantic_operator_tests PRIVATE
        -fno-sanitize=undefined -fsanitize=thread -fno-omit-frame-pointer -g)
    # At LINK we must keep the UBSan RUNTIME (do NOT pass -fno-sanitize=undefined
    # here): the prebuilt duckdb_static is a UBSan-instrumented debug build and
    # references __ubsan_handle_* symbols. So link BOTH runtimes.
    target_link_options(flock_semantic_operator_tests PRIVATE
        -fsanitize=thread)
endif()

if(FLOCK_OPERATOR_TSAN)
    # Run under setarch -R (lower ASLR entropy so TSan's shadow mapping fits this
    # GH200 kernel) and point TSan at the suppressions file (one documented
    # TSan x UBSan libubsan-internal false positive; see tsan_suppressions.txt).
    add_test(NAME flock_semantic_operator_tests
             COMMAND setarch ${CMAKE_HOST_SYSTEM_PROCESSOR} -R $<TARGET_FILE:flock_semantic_operator_tests>)
    set_tests_properties(flock_semantic_operator_tests PROPERTIES ENVIRONMENT
        "TSAN_OPTIONS=halt_on_error=1:suppressions=${CMAKE_CURRENT_SOURCE_DIR}/tsan_suppressions.txt")
else()
    add_test(NAME flock_semantic_operator_tests COMMAND flock_semantic_operator_tests)
endif()
# -- Semantic filter operator tests (LogicalSemFilter / CreatePlan / PhysicalSemFilter)
#
# Unlike flock_semantic_operator_tests above, the filter operator pulls in the
# prompt builder, model manager and ExtensionState/ResolveSemanticParams, all of
# which need a loaded DuckDB. So this target links the full flock extension +
# duckdb_static, mirroring test/runtime's flock_semantic_config_tests. It is NOT
# TSan-eligible (linking duckdb_static under a sanitizer hits R_AARCH64_CALL26 on
# ARM64); the engine concurrency it relies on, including the residual pre-filter,
# is TSan-covered by flock_semantic_operator_tests.
find_package(CURL REQUIRED)
add_executable(flock_sem_filter_tests
    ${CMAKE_CURRENT_SOURCE_DIR}/semantic_filter_test.cpp)

target_include_directories(flock_sem_filter_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/src/include
    ${CMAKE_CURRENT_SOURCE_DIR})

# duckdb_static (built C++11-style) defines LogicalType::VARCHAR etc. as plain
# strong globals; a C++17 TU that ODR-uses those constants emits them as
# STB_GNU_UNIQUE, which GNU ld rejects as a multiple definition against the
# archive's strong def. -fno-gnu-unique makes our objects emit weak linkonce_odr
# instead, which resolves cleanly against duckdb's definition.
target_compile_options(flock_sem_filter_tests PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-fno-gnu-unique>)

target_link_libraries(flock_sem_filter_tests PRIVATE
    GTest::gtest
    GTest::gmock
    $<$<CXX_COMPILER_ID:GNU>:-Wl,--start-group>
    flock_extension
    duckdb_generated_extension_loader
    core_functions_extension
    json_extension
    parquet_extension
    $<$<PLATFORM_ID:Linux>:jemalloc_extension>
    duckdb_static
    $<$<CXX_COMPILER_ID:GNU>:-Wl,--end-group>
    CURL::libcurl
    Threads::Threads)

add_test(NAME flock_sem_filter_tests COMMAND flock_sem_filter_tests)
