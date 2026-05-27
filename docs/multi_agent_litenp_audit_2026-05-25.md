# litenp multi-agent audit 2026-05-25

## Scope

Four independent audit passes reviewed the current `lite_np` implementation:

- performance and benchmark alignment
- lightweight build and dependency surface
- correctness tests and bug prevention
- API-level correctness risks

## Fixed Immediately

- Release tests now keep `assert` active by undefining `NDEBUG` inside
  `tests/test_litenp.cpp`.
- Default-constructed `Array` and `ArrayView` now represent an empty 1D view
  with shape `{0}`, instead of an accidental scalar-like shape.
- Broadcasting now handles zero-size dimensions such as `{0, 3}` with `{3}` or
  `{1, 3}` without producing a non-empty output.
- `_into` APIs now guard against partial input/output aliasing by computing into
  a temporary array and copying back when the memory ranges may overlap.
- `matmul_into` now handles aliased output safely.
- `mean(a, axis)` now rejects empty reduction axes instead of dividing by zero.
- `detail::is_contiguous` no longer allocates a temporary stride vector.
- CMake defaults are now lightweight: benchmarks, OpenMP, CBLAS, and
  `-march=native` are opt-in.

## Added Tests

- Default empty arrays and reductions on empty arrays.
- Zero-size broadcast for arithmetic and comparison.
- Sum over an empty axis and mean rejection for empty axes.
- Partial aliasing for binary, unary, and clip `_into` paths.
- `matmul_into` with output aliasing an input matrix.

## Remaining Performance Work

- Add a tiled materialized transpose path for `as_contiguous(transpose(...))`.
- Add a contiguous fast path for `astype`.
- Add allocation-free `where_into` for hot loops.
- Improve `stack` for last-axis interleave patterns.
- Revisit GEMM thresholds and benchmark both built-in and CBLAS paths with
  pinned thread counts.

## Remaining Correctness Work

- Decide and document lifetime rules for views made from temporaries.
- Decide signed integer overflow semantics for `negative`, `abs`, and arithmetic.
- Pin AVX/scalar semantics for NaN and signed-zero behavior in min/max/clip/relu.
- Add property-style cross-checks against scalar reference implementations.

## Benchmark Methodology Follow-ups

- Report single-thread and matched-thread suites separately.
- Print OpenMP, BLAS, and torch thread counts in benchmark output.
- Add paired allocation and preallocated rows for every library.
- Use warmup plus median and tail latency, not only best-of timing.
- Include checksums for Eigen and libtorch outputs wherever possible.
