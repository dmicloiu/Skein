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

# -- Standalone benchmark binary ---------------------------------------------
#
# Drives AsyncLLMClient against a real vLLM endpoint (no DuckDB, no mock). Used
# to validate that the network layer can reach the throughput ceiling observed
# in the E1/E5 study (~800 rows/s on GH200, ~100 rows/s on RTX 3090) before the
# operator stack is built on top.
add_executable(flock_async_bench
    ${PROJECT_SOURCE_DIR}/src/runtime/async_llm_client.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/async_llm_client_bench.cpp)

target_include_directories(flock_async_bench PRIVATE
    ${PROJECT_SOURCE_DIR}/src/include)

target_link_libraries(flock_async_bench PRIVATE
    CURL::libcurl
    Threads::Threads)