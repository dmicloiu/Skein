find_package(GTest CONFIG REQUIRED)
find_package(CURL REQUIRED)
find_package(Threads REQUIRED)

# Build the runtime tests as a fully standalone executable.
add_executable(flock_runtime_tests
    ${PROJECT_SOURCE_DIR}/src/runtime/async_llm_client.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mock_vllm_server.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/async_llm_client_test.cpp)

target_include_directories(flock_runtime_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/src/include
    ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(flock_runtime_tests PRIVATE
    GTest::gtest
    CURL::libcurl
    Threads::Threads)

# Opt-in ThreadSanitizer for this target only: -DFLOCK_RUNTIME_TSAN=ON.
# Standalone (no DuckDB link), so it does NOT hit the R_AARCH64_CALL26 issue
# that breaks sanitizers when linking the giant duckdb_static archive on ARM64.
option(FLOCK_RUNTIME_TSAN "Build flock_runtime_tests with ThreadSanitizer" OFF)
if(FLOCK_RUNTIME_TSAN)
    # Disable inherited UBSan: TSan + UBSan together produce spurious races
    # inside UBSan's own IsAccessibleMemoryRange / pipe(2) interceptors at
    # std::thread teardown. UBSan coverage stays on for the rest of the
    # project; this target focuses purely on race detection.
    target_compile_options(flock_runtime_tests PRIVATE
        -fno-sanitize=undefined -fsanitize=thread -fno-omit-frame-pointer -g)
    target_link_options(flock_runtime_tests PRIVATE
        -fno-sanitize=undefined -fsanitize=thread)
endif()

add_test(NAME flock_runtime_tests COMMAND flock_runtime_tests)

# -- EndpointRouter tests ----------------------------------------------------
#
# Pure policy object: no HTTP, no curl, no DuckDB. Standalone executable so the
# TSan run below does NOT hit the R_AARCH64_CALL26 issue that breaks sanitizers
# when linking the giant duckdb_static archive on ARM64.
add_executable(flock_endpoint_router_tests
    ${PROJECT_SOURCE_DIR}/src/runtime/endpoint_router.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/endpoint_router_test.cpp)

target_include_directories(flock_endpoint_router_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/src/include)

target_link_libraries(flock_endpoint_router_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
    Threads::Threads)

# Opt-in ThreadSanitizer for this target only: -DFLOCK_RUNTIME_TSAN=ON.
# Shares the flag with flock_runtime_tests; see the rationale above for why
# inherited UBSan is disabled here.
#
# NOTE (running the TSan binary on this GH200 / recent kernel): TSan aborts at
# startup with "FATAL: ThreadSanitizer: unexpected memory mapping" because the
# kernel's high-entropy ASLR places mappings outside TSan's expected shadow
# layout. Run the resulting binary with ASLR disabled, e.g.
#     setarch $(uname -m) -R ./flock_endpoint_router_tests
# (or lower it globally: sysctl vm.mmap_rnd_bits=28). This is environmental,
# not a test failure -- CI must apply the wrapper or every TSan run dies before
# the first test executes.
if(FLOCK_RUNTIME_TSAN)
    target_compile_options(flock_endpoint_router_tests PRIVATE
        -fno-sanitize=undefined -fsanitize=thread -fno-omit-frame-pointer -g)
    target_link_options(flock_endpoint_router_tests PRIVATE
        -fno-sanitize=undefined -fsanitize=thread)
endif()

add_test(NAME flock_endpoint_router_tests COMMAND flock_endpoint_router_tests)

# -- vLLM integration test binary --------------------------------------------
#
# Drives AsyncLLMClient against a real vLLM endpoint (no DuckDB, no mock).
# Validates the network layer's behaviour on the same workload that sembench's
# vLLM performance analysis uses, so we can compare throughput head-to-head
# before the operator stack is built on top.
add_executable(flock_async_vllm_integration
    ${PROJECT_SOURCE_DIR}/src/runtime/async_llm_client.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/async_llm_client_vllm_integration.cpp)

target_include_directories(flock_async_vllm_integration PRIVATE
    ${PROJECT_SOURCE_DIR}/src/include)

target_link_libraries(flock_async_vllm_integration PRIVATE
    CURL::libcurl
    Threads::Threads)

# -- EndpointRouter vLLM integration binary ----------------------------------
add_executable(flock_endpoint_router_vllm_integration
    ${PROJECT_SOURCE_DIR}/src/runtime/async_llm_client.cpp
    ${PROJECT_SOURCE_DIR}/src/runtime/endpoint_router.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/endpoint_router_vllm_integration.cpp)

target_include_directories(flock_endpoint_router_vllm_integration PRIVATE
    ${PROJECT_SOURCE_DIR}/src/include)

target_link_libraries(flock_endpoint_router_vllm_integration PRIVATE
    CURL::libcurl
    Threads::Threads)

# -- sem_filter operator throughput driver (DuckDB-embedding A/B) -------------
#
# Unlike the standalone router driver above, this one OPENS a real DuckDB with
# flock loaded and runs the full SQL chain, so it links the full extension +
# duckdb_static (same recipe + -fno-gnu-unique as test/functions/operator's
# flock_sem_filter_e2e_tests). It is a plain main() driver (no gtest, not
# registered with ctest; built and run manually on the cluster).
add_executable(flock_sem_filter_vllm_integration
    ${CMAKE_CURRENT_SOURCE_DIR}/sem_filter_vllm_integration.cpp)

target_include_directories(flock_sem_filter_vllm_integration PRIVATE
    ${PROJECT_SOURCE_DIR}/src/include)

target_compile_options(flock_sem_filter_vllm_integration PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-fno-gnu-unique>)

target_link_libraries(flock_sem_filter_vllm_integration PRIVATE
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

# -- sem_extract operator throughput driver (llm_complete projection A/B) ------
#
# Sibling of flock_sem_filter_vllm_integration above (same recipe + -fno-gnu-unique
# + full extension link). Plain main() driver, not a ctest; built + run manually on
# the cluster. Times a materialised projected llm_complete (PhysicalSemExtract vs
# the scalar) across the morsel sweep.
add_executable(flock_sem_extract_vllm_integration
    ${CMAKE_CURRENT_SOURCE_DIR}/sem_extract_vllm_integration.cpp)

target_include_directories(flock_sem_extract_vllm_integration PRIVATE
    ${PROJECT_SOURCE_DIR}/src/include)

target_compile_options(flock_sem_extract_vllm_integration PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-fno-gnu-unique>)

target_link_libraries(flock_sem_extract_vllm_integration PRIVATE
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

# -- Semantic config tests (ExtensionState, SET surface, CREATE MODEL) --------
#
# Unlike the standalone targets above, ExtensionState
# (DatabaseInstance::GetObjectCache), the semantic_* SET surface and
# ResolveSemanticParams all REQUIRE a loaded DuckDB. So this target links the
# full flock extension + duckdb_static, mirroring test/unit's link block, and is
# NOT eligible for the FLOCK_RUNTIME_TSAN target above: linking the
# duckdb_static archive under a sanitizer hits the R_AARCH64_CALL26 overflow on
# ARM64. The concurrency primitives these tests lean on (ObjectCache's internal
# lock, EndpointRouter's internal lock) are already TSan-covered by
# flock_endpoint_router_tests and DuckDB's own suite.
add_executable(flock_semantic_config_tests
    ${CMAKE_CURRENT_SOURCE_DIR}/semantic_test_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/extension_state_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/semantic_config_test.cpp)

target_include_directories(flock_semantic_config_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/src/include)

# GNU ld needs --start-group/--end-group for the circular deps; jemalloc is
# only built on Linux. Same recipe as test/unit/CMakeLists.txt.
target_link_libraries(flock_semantic_config_tests PRIVATE
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

add_test(NAME flock_semantic_config_tests COMMAND flock_semantic_config_tests)