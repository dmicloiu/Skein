#!/bin/bash
#SBATCH --account=infra02
#SBATCH --job-name=flock-build-setup
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --output=logs/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# One-shot DuckDB+flock build setup for the Clariden analysis jobs: configures
# and builds a single flock integration driver inside the NGC container and
# drops it at build/<target>. Override the target with the TARGET env var, e.g.
#       TARGET=flock_endpoint_router_vllm_integration sbatch <this script>
# Default builds the sem_filter driver that analysis/slurm/sem_filter_ab_clariden.sh
# (and the other *_clariden.sh experiments) depend on.
#
# Must run INSIDE the NGC container (the EDF) because the toolchain (ninja, the
# arm64 compilers) lives only there -- the login node has no ninja. CPU-only
# build, no GPU requested (eases scheduling).
#
# OFFLINE / NETWORK SPLIT: Clariden compute nodes have no internet, so vcpkg
# cannot fetch+build flock'\''s manifest deps here. They are installed ONCE on the
# login node (which does have internet) into vcpkg/installed/arm64-linux:
#       cd $HOME && \
#       $HOME/projects/flock/vcpkg/vcpkg install nlohmann-json curl gtest \
#           --triplet arm64-linux
# (classic mode -- run from outside the manifest dir so vcpkg.json is ignored.)
# This script then does a CLASSIC-mode configure (toolchain file + triplet, NO
# manifest flags): the vcpkg toolchain adds vcpkg/installed/arm64-linux to
# CMAKE_PREFIX_PATH, so find_package() resolves the prebuilt deps with no network.
#
# WHY THE PRIOR RUN FAILED: the configure ran classic mode against an EMPTY
# vcpkg/installed, so find_package(nlohmann_json) failed at CMakeLists.txt:22.
# The fix is populating vcpkg/installed (above), not changing the configure. We
# still wipe build/release because the old cache has nlohmann_json_DIR cached as
# NOTFOUND and would skip the re-search.
#
# We build ONLY the one target (not a full `make release` of all of duckdb) to
# stay within walltime; the test/runtime target is defined unconditionally in
# non-WASM builds (CMakeLists.txt:60-66), so no extra LOAD_TESTS flag is needed.
# -----------------------------------------------------------------------------
set -euo pipefail
cd "$HOME/projects/flock"
mkdir -p logs
EDF="$HOME/projects/sembench/ngc-pytorch-vllm.toml"

srun -ul --environment="$EDF" bash -c '
    set -euo pipefail
    FLOCK="$HOME/projects/flock"; cd "$FLOCK"
    # Which integration driver to build; override via the TARGET env var.
    TARGET="${TARGET:-flock_sem_filter_vllm_integration}"
    echo "building target: $TARGET"

    echo "==================== TOOLCHAIN ===================="
    cmake --version | head -1
    command -v ninja && ninja --version
    nproc

    echo "==================== PREBUILT VCPKG DEPS (offline) ===================="
    # Populated on the login node (see header). No network here, so fail loudly
    # if the install step was skipped.
    INST="$FLOCK/vcpkg/installed/arm64-linux"
    [ -f "$INST/share/nlohmann_json/nlohmann_jsonConfig.cmake" ] || {
        echo "ERROR: vcpkg/installed/arm64-linux missing nlohmann_json -- run the"
        echo "       login-node install in this scripts header first."; exit 1; }
    ls "$INST/share" | grep -iE "nlohmann|curl|gtest" || true

    echo "==================== WIPE STALE CACHE ===================="
    # Old cache has nlohmann_json_DIR cached as NOTFOUND; wipe to force a fresh
    # find_package against the now-populated vcpkg/installed.
    rm -rf build/release

    echo "==================== CONFIGURE (classic mode -> uses vcpkg/installed, offline) ===================="
    # Toolchain file + triplet in CLASSIC mode (no manifest flags) puts
    # vcpkg/installed/arm64-linux on CMAKE_PREFIX_PATH; deps resolve with no network.
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_TOOLCHAIN_FILE="$FLOCK/vcpkg/scripts/buildsystems/vcpkg.cmake" \
        -DVCPKG_TARGET_TRIPLET=arm64-linux \
        -DEXTENSION_STATIC_BUILD=1 \
        -DDUCKDB_EXTENSION_CONFIGS="$FLOCK/extension_config.cmake" \
        -S duckdb -B build/release

    echo "==================== BUILD ===================="
    cmake --build build/release --target "$TARGET" -j"$(nproc)"

    echo "==================== PLACE BINARY ===================="
    if [ "$TARGET" = "flock_loadable_extension" ]; then
        # The loadable target'\''s OUTPUT file is flock.duckdb_extension (name != the
        # target name), and cross_system_analysis_clariden.sh reads it IN PLACE via
        # FLOCK_EXTENSION_PATH -- so there is nothing to find-by-name or copy; just
        # verify the artifact exists where the experiment expects it.
        OUT_EXT="build/release/extension/flock/flock.duckdb_extension"
        [ -f "$OUT_EXT" ] || { echo "ERROR: loadable not produced at $OUT_EXT"; exit 1; }
        ls -la "$OUT_EXT"
        file "$OUT_EXT" || true
        echo "DONE: loadable ready at $FLOCK/$OUT_EXT"
    else
        # Test drivers: output filename == target name. The experiment scripts expect
        # the driver at build/<target> (same place the other integration binaries
        # live). Find what ninja produced and copy it.
        SRC="$(find build/release -name "$TARGET" -type f -perm -u+x 2>/dev/null | head -1)"
        if [ -z "$SRC" ]; then
            echo "ERROR: built target not found under build/release"; exit 1
        fi
        echo "built: $SRC"
        cp -f "$SRC" "build/$TARGET"
        chmod +x "build/$TARGET"
        ls -la "build/$TARGET"
        file "build/$TARGET" || true
        echo "DONE: driver ready at build/$TARGET"
    fi
'
