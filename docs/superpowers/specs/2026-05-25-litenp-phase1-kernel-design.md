# litenp phase 1 kernel cleanup design

## Goal

Improve the current `litenp` implementation while preserving the public API and
the default lightweight build: C++17, header-first, and no required OpenMP,
CBLAS, Eigen, libtorch, or SIMD abstraction dependency.

This phase targets obvious high-cost paths that are currently implemented with
generic index expansion:

- materializing a 2D transpose through `as_contiguous`
- `astype` on contiguous input
- benchmark timing quality for future optimization decisions

## Scope

### Tiled 2D Transpose Copy

`as_contiguous(transpose(view))` currently calls `linear_to_index` and
`offset_for_index` for every element. Add a private helper that recognizes
2D views whose strides match a simple transpose of a contiguous row-major
matrix, then copy with a small tile loop. The helper must preserve the same
output layout and should fall back to the existing generic loop for all other
views.

### Contiguous Astype

`astype` currently uses the generic strided loop even when the source is
contiguous. Add a linear contiguous fast path that casts source element `i`
directly into output element `i`.

### Benchmark Timing

The benchmark currently reports only best-of timing. Keep best timing for
continuity, but also collect median timing. Matrix comparison rows should print
both best and median for `litenp`, Eigen, and libtorch where available.

## Out Of Scope

- Changing public API semantics
- Making any third-party dependency mandatory
- Splitting the header into multiple installed headers
- Replacing the existing GEMM kernel selection
- Adding a new SIMD abstraction library

## Verification

Run:

```bash
cmake --build build_torch -j
ctest --test-dir build_torch --output-on-failure
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ./build_torch/litenp_bench
```

Key benchmark rows:

- `transpose materialize 2048^2`
- `as_contiguous transpose 4M`
- `astype f32->f64 4M`
- `where 16M` / `where_into 16M`
- `matmul 384`, `matmul 512`, `matmul 768`
