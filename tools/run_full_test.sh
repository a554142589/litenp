#!/usr/bin/env bash
# Full test runner for litenp.
#
# Runs every layer of the test plan:
#   L1  build + C++ unit tests (Release, Debug, ASan/UBSan)
#   L3  NumPy behavioral oracle
#   L3  libtorch behavioral oracle (if PyTorch is installed)
#   L4  performance benchmark vs NumPy / Eigen / libtorch + pass/fail report
#
# Usage:
#   ./tools/run_full_test.sh
#
# Env (optional):
#   CXX            C++ compiler (default: g++)
#   BUILD_JOBS     parallel jobs (default: nproc)
#   SKIP_BENCH=1   skip the L4 benchmark stage
#   SKIP_TORCH=1   skip libtorch oracle even if torch is present

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

CXX="${CXX:-g++}"
JOBS="${BUILD_JOBS:-$(nproc 2>/dev/null || echo 2)}"
TMP="${TMPDIR:-/tmp}/litenp_full_test"
mkdir -p "$TMP"

THREAD_ENV="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1"

pass=0
fail=0
stage() {
    echo ""
    echo "============================================================"
    echo "== $1"
    echo "============================================================"
}

run() {
    if "$@"; then
        pass=$((pass+1))
        return 0
    else
        fail=$((fail+1))
        return 1
    fi
}

# ---- L1/L2: C++ unit tests -------------------------------------------------
stage "L1/L2: C++ unit tests (Release)"
run bash -c "cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$CXX && \
             cmake --build build_release -j$JOBS && \
             ctest --test-dir build_release --output-on-failure"

stage "L1/L2: C++ unit tests (Debug)"
run bash -c "cmake -S . -B build_debug -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=$CXX && \
             cmake --build build_debug -j$JOBS && \
             ctest --test-dir build_debug --output-on-failure"

stage "L1/L2: C++ unit tests (ASan + UBSan)"
run bash -c "cmake -S . -B build_asan -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=$CXX \
               -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
               -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined' && \
             cmake --build build_asan -j$JOBS && \
             ASAN_OPTIONS=detect_leaks=1:verify_asan_link_order=0 \
             UBSAN_OPTIONS=print_stacktrace=1 \
             ctest --test-dir build_asan --output-on-failure"

# ---- L3: NumPy oracle ------------------------------------------------------
stage "L3: NumPy behavioral oracle"
if python3 -c "import numpy" 2>/dev/null; then
    run bash -c "CXX=$CXX python3 tests/test_numpy_oracle.py"
else
    echo "SKIP: numpy not installed"
fi

# ---- L3: libtorch oracle ---------------------------------------------------
if [[ "${SKIP_TORCH:-0}" != "1" ]] && python3 -c "import torch" 2>/dev/null; then
    stage "L3: libtorch behavioral oracle"
    run bash -c "CXX=$CXX python3 tests/test_libtorch_oracle.py"
else
    stage "L3: libtorch behavioral oracle"
    echo "SKIP: PyTorch not installed (or SKIP_TORCH=1)"
fi

# ---- L4: benchmark comparison ---------------------------------------------
if [[ "${SKIP_BENCH:-0}" != "1" ]]; then
    stage "L4: benchmark vs NumPy / libtorch"
    TORCH_PREFIX="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)' 2>/dev/null || true)"
    if [[ -n "$TORCH_PREFIX" ]]; then
        BENCH_OPTS="-DCMAKE_PREFIX_PATH=$TORCH_PREFIX"
    else
        BENCH_OPTS=""
    fi
    run bash -c "cmake -S . -B build_perf -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$CXX \
                   -DLITENP_BUILD_BENCHMARKS=ON -DLITENP_NATIVE_ARCH=ON $BENCH_OPTS && \
                 cmake --build build_perf --target litenp_bench -j$JOBS"
    if [[ $? -eq 0 ]]; then
        env $THREAD_ENV python3 benchmarks/bench_numpy.py --out "$TMP/numpy.json"
        env $THREAD_ENV ./build_perf/litenp_bench > "$TMP/cpp.txt"
        python3 tools/compare_benchmarks.py \
            --manifest benchmarks/benchmark_manifest.json \
            --cpp "$TMP/cpp.txt" \
            --numpy "$TMP/numpy.json" \
            --out "$TMP/report.md"
        echo "Benchmark report: $TMP/report.md"
        tail -8 "$TMP/report.md"
    fi
else
    stage "L4: benchmark"
    echo "SKIP: SKIP_BENCH=1"
fi

# ---- summary ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "Full test summary: pass=$pass fail=$fail"
echo "============================================================"
[[ $fail -eq 0 ]]
