# litenp performance matrix design

## Goal

Optimize the current `lite_np` CPU implementation and expand benchmarks into a
larger scale matrix that compares `litenp` with Eigen and libtorch. The target is
to match or beat Eigen/libtorch on the main ndarray hot paths where a lightweight
header-only implementation can reasonably compete.

## Scope

This pass stays on CPU Release builds and preserves the public API. It focuses on
`float`, `double`, and `int32_t` array workloads that are already implemented:
elementwise arithmetic, broadcasting, mixed dtype arithmetic, comparisons,
`where`, reductions, combine operations, and 2D matmul.

GPU, autograd, expression-template rewrites, full NumPy compatibility, FFT, IO,
and random distributions remain out of scope.

## Optimization Plan

- Add internal fast paths that avoid per-element `linear_to_index` work for
  common 1D and 2D contiguous/broadcasted layouts.
- Speed up contiguous mixed dtype binary arithmetic by using direct pointer loops
  with optional OpenMP.
- Add contiguous comparison and `where` loops, with SIMD for `float`/`double`
  comparisons where useful and scalar fallback for all dtypes.
- Add contiguous total `max` and 2D row/column reduction fast paths.
- Use block copies for contiguous `concatenate` and `stack` cases.
- For non-contiguous matmul, materialize contiguous operands and reuse the
  existing optimized matmul backend when that is cheaper than the generic
  strided triple loop.

## Benchmark Matrix

The benchmark should keep the existing comprehensive coverage and add matrix
families:

- 1D elementwise/compare/where sizes: `1K`, `64K`, `1M`, `4M`, `16M`.
- 2D broadcast/reduction sizes: `256x256`, `512x512`, `1024x1024`,
  `2048x2048`.
- Matmul square sizes: `64`, `128`, `256`, `384`, `512`, `768`.

Rows should include `litenp`, Eigen, and libtorch when available, plus a speed
ratio when a baseline is present.

## Verification

- `cmake --build build_torch -j`
- `ctest --test-dir build_torch --output-on-failure`
- `./build_torch/litenp_bench`

The completion audit must compare the final artifacts against this design and
record any case where Eigen/libtorch still wins.
