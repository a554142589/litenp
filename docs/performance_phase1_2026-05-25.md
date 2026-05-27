# litenp phase 1 performance pass 2026-05-25

## Scope

This pass implements the first low-risk kernel cleanup:

- add a 2D strided-to-contiguous copy helper with a tiled path for transposed
  materialization
- add a contiguous fast path for `astype`
- extend benchmark matrix rows to print best/median timing
- add regression tests for non-square transpose materialization, 2D strided
  contiguous copy, and strided `astype`

The public API and default dependency surface remain unchanged.

## Verification

Commands run:

```bash
cmake --build build_torch -j
ctest --test-dir build_torch --output-on-failure
cmake --build build_lite -j
ctest --test-dir build_lite --output-on-failure
cmake --build build_warn -j
ctest --test-dir build_warn --output-on-failure
cmake --build build_asan -j
LD_PRELOAD="$(gcc -print-file-name=libasan.so)" ctest --test-dir build_asan --output-on-failure
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ./build_torch/litenp_bench > /tmp/litenp_bench_phase1.txt
```

All listed test runs passed.

## Benchmark Evidence

Baseline source: `/tmp/litenp_bench_goal_current.txt`.

Phase 1 source: `/tmp/litenp_bench_phase1.txt`.

Single-thread key rows:

| case | before best | after best/median | note |
| --- | ---: | ---: | --- |
| `transpose materialize 2048^2` | `28.7680 ms` | `9.5964 / 12.7329 ms` | tiled 2D strided copy |
| `as_contiguous transpose 4M` | `26.8758 ms` | `9.1247 ms` | same fast path in all-feature section |
| `astype f32->f64 4M` | `13.1417 ms` | `1.9791 ms` | contiguous cast fast path |
| `where_into 16M` | `7.1091 ms` | `7.0143 / 7.0264 ms` | unchanged code, still healthy |
| `matmul 384` | `1.1490 ms` | `0.7784 / 0.8065 ms` | no GEMM code change; likely run variance |
| `matmul 512` | `2.9955 ms` | `2.2116 / 2.2463 ms` | no GEMM code change; still behind Eigen |
| `matmul 768` | `8.9268 ms` | `9.0418 / 9.0831 ms` | still behind Eigen |

## Remaining Work

- GEMM still needs a deliberate second-phase design for medium/large square
  matrices.
- Allocating `where 16M` remains much slower than Eigen because allocation and
  first-touch dominate; `where_into` is the preferred hot path.
- The benchmark now reports best/median for matrix comparison rows, but the
  non-matrix feature sections still print best timing only.
