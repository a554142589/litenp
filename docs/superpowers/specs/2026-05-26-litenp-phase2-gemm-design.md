# litenp phase 2 GEMM design

## Goal

Improve the medium-size `float32` GEMM path without changing the public API or
making CBLAS/OpenBLAS mandatory. The default build remains header-first and
lightweight; optional CBLAS is still only used when explicitly enabled.

## Evidence

After phase 1, transpose and `astype` hot paths improved, but GEMM still trails
Eigen at common square sizes. A temporary path probe showed that simply lowering
the CBLAS threshold is not a reliable fix on this machine: single-thread
OpenBLAS was slower than the built-in kernels for `512`, `768`, and `1024`.
Repeated benchmark diagnostics in the full build showed the built-in `4x16`
kernel is the best available lightweight path through `768`.

## Design

Add a small internal GEMM planner that centralizes backend selection for the
fast contiguous row-major case:

- use optional CBLAS only for large matrices where its call overhead and backend
  behavior are more likely to pay off;
- use the built-in `4x16` kernel for small and medium `float32` matrices where
  wider `n` reuse is beneficial;
- use the built-in `8x8` kernel for larger divisible `float32` matrices beyond
  the measured `4x16` range;
- keep the existing scalar/AXPY fallback for unsupported types, shapes, and
  non-AVX builds.

The planner keeps `matmul_into` easier to read by replacing scattered threshold
checks with one named decision. It also makes benchmark evidence explicit by
adding optional diagnostic rows that compare the public `litenp` path against
the built-in kernels and CBLAS.

## Scope

- Public API unchanged.
- No new required dependencies.
- No header split.
- No attempt to replace Eigen/OpenBLAS with a full packed BLIS-style GEMM in
  this phase.

## Verification

Run the existing test matrix:

```bash
cmake --build build_torch -j
ctest --test-dir build_torch --output-on-failure
cmake --build build_lite -j && ctest --test-dir build_lite --output-on-failure
cmake --build build_warn -j && ctest --test-dir build_warn --output-on-failure
cmake --build build_asan -j && LD_PRELOAD="$(gcc -print-file-name=libasan.so)" ctest --test-dir build_asan --output-on-failure
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ./build_torch/litenp_bench
```

Key rows:

- `matmul 384`, `matmul 512`, `matmul 768`
- GEMM diagnostic rows for built-in vs CBLAS path choice
