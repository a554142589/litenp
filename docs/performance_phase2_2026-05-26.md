# litenp phase 2 GEMM pass 2026-05-26

## Scope

This pass implements the approved GEMM cleanup:

- add an internal `GemmBackend` planner for the fast contiguous row-major
  `matmul_into` path
- route medium `float32` GEMM through the `4x16` AVX2/FMA kernel up to
  `768^3` work
- keep the `8x8` kernel for larger divisible `float32` GEMM beyond that range
- keep optional CBLAS for large matrices only; it remains disabled by default
- add benchmark diagnostic rows comparing public `litenp`, direct `4x16`,
  direct `8x8`, and CBLAS paths

The public API and default dependency surface remain unchanged.

## Verification

Commands run:

```bash
cmake --build build_torch -j
ctest --test-dir build_torch --output-on-failure
cmake --build build_lite -j && ctest --test-dir build_lite --output-on-failure
cmake --build build_warn -j && ctest --test-dir build_warn --output-on-failure
cmake --build build_asan -j && LD_PRELOAD="$(gcc -print-file-name=libasan.so)" ctest --test-dir build_asan --output-on-failure
./build_lite/litenp_basic
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ./build_torch/litenp_bench > /tmp/litenp_bench_phase2.txt
```

All listed test and example runs passed.

## Benchmark Evidence

Phase 1 source: `/tmp/litenp_bench_phase1.txt`.

Phase 2 source: `/tmp/litenp_bench_phase2b.txt`.

Single-thread matrix rows:

| case | phase 1 litenp best/median | phase 2 litenp best/median | phase 2 Eigen best/median | phase 2 libtorch best/median | note |
| --- | ---: | ---: | ---: | ---: | --- |
| `matmul 384` | `0.7784 / 0.8065 ms` | `0.7690 / 0.7720 ms` | `0.7761 / 0.7771 ms` | `2.0958 / 2.3001 ms` | tied with Eigen, faster than libtorch |
| `matmul 512` | `2.2116 / 2.2463 ms` | `2.0648 / 2.0895 ms` | `1.8691 / 1.8951 ms` | `5.2381 / 5.3097 ms` | planner selects `4x16`, closer to Eigen |
| `matmul 768` | `9.0418 / 9.0831 ms` | `8.4570 / 8.5176 ms` | `6.2111 / 6.2683 ms` | `17.9405 / 17.9932 ms` | improved but still behind Eigen, faster than libtorch |

Diagnostic rows:

| case | public | direct `4x16` | direct `8x8` | CBLAS | note |
| --- | ---: | ---: | ---: | ---: | --- |
| `gemm 384` | `0.7712 / 0.7989 ms` | `0.7640 / 0.7671 ms` | `0.8980 / 0.8996 ms` | `2.0495 / 2.2222 ms` | CBLAS is not a good medium-size choice here |
| `gemm 512` | `2.0472 / 2.1263 ms` | `2.0671 / 2.0761 ms` | `2.2929 / 2.3490 ms` | `5.0675 / 5.2449 ms` | public path matches `4x16` |
| `gemm 768` | `8.6124 / 8.7309 ms` | `9.1714 / 9.2367 ms` | `9.1877 / 9.1919 ms` | `17.5475 / 17.6840 ms` | public path is faster than the direct diagnostic calls in this run |

## Remaining Work

- `matmul 768` still trails Eigen. Closing that gap likely requires a packed
  macro-kernel rather than threshold tuning.
- Allocating `where 16M` remains allocation-bound; `where_into` is still the
  preferred hot path.
- This pass improves GEMM structure and medium-size performance, but the broad
  objective should stay open until the remaining Eigen gap is either closed or
  explicitly accepted as out of scope.
