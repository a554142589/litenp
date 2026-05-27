# litenp phase 3 packed GEMM design

## Goal

Close more of the remaining Eigen gap for medium and large contiguous
`float32` GEMM while keeping `litenp` header-first, API-compatible, and light.
The default build must not gain any required dependency.

## Evidence

Phase 2 centralized GEMM selection and improved the built-in `4x16` path. It
made `384` competitive with Eigen and kept `512` much closer, but `768` still
lags Eigen while remaining much faster than local libtorch. Diagnostics also
showed that lowering the CBLAS threshold is not a good fix on this machine:
CBLAS is slower than the built-in path for the tested medium sizes.

## Design

Add one internal packed-B backend for the fast contiguous row-major
`float32` case:

- Pack each `k x 16` B column panel into a small contiguous scratch panel.
- Reuse that packed panel across all 4-row A blocks.
- Keep the existing `4x16`, `8x8`, `4x8`, CBLAS, and AXPY paths as fallbacks.
- Enable the packed path only for AVX2/FMA builds and matrix sizes large enough
  to amortize packing cost.
- Keep scratch storage local to the kernel call, with one panel buffer per
  executing thread when OpenMP is active.

This keeps the design modest: it improves B locality without introducing a full
BLIS-style packing hierarchy or a new dependency.

## Alternatives Considered

- Lower CBLAS threshold: rejected by diagnostics because local OpenBLAS is
  slower for the target sizes.
- Larger hand-written register kernels: higher risk and more code, with less
  clear reuse than B panel packing.
- Full packed A/B macro-kernel: likely faster, but too large for this phase and
  less aligned with the lightweight goal.

## Verification

Run:

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
- GEMM diagnostics including public, direct `4x16`, direct `8x8`, packed-B,
  and CBLAS paths
