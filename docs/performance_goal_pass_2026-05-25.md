# litenp Eigen/libtorch goal pass 2026-05-25

> Archived earlier pass: this document records the 2026-05-25 optimization
> work and benchmark evidence. It is superseded for the active Eigen/libtorch
> goal by `docs/completion_audit.md`, `docs/status.md`, and
> `docs/phase4_approval_gate_2026-05-26.md`, which show the current broad
> objective is still open.

## Scope

This pass continues the Eigen/libtorch-oriented optimization work with three
goals:

- speed up hot CPU ndarray paths without adding required runtime dependencies;
- keep the header-first default lightweight;
- expand tests around the new fast paths and aliasing cases.

## Changes

- Added `detail::use_openmp_for` to avoid OpenMP work when only one thread is
  available and to allow higher per-kernel thresholds.
- Added AVX2 contiguous `sum` kernels for `float` and `double`.
- Added AVX2 contiguous compare kernels for `float` and `double`, using
  `movemask` plus byte-mask lookup to reduce mask packing overhead.
- Reused contiguous sum kernels in total reductions and 2D `sum(axis=1)`.
- Added allocation-free `where_into`, including overlap protection when the
  output aliases the `uint8_t` mask.
- Added an internal uninitialized `Array` allocation path for kernels that fully
  overwrite their outputs, while keeping public `Array(shape)` zero-initialized.
- Added `Array::to_vector()` as a standard-allocator copy-out path, keeping
  `from_vector`/`to_vector` symmetric even though internal storage now uses a
  no-init allocator.
- Enabled the existing `8x8` float GEMM micro-kernel, skipped unnecessary output
  zero-fill for GEMM paths that fully overwrite the output, and raised the GEMM
  micro-kernel OpenMP threshold to avoid small-matrix thread overhead.
- Added a `4x16` float GEMM micro-kernel for small/mid-size matrices where it
  improves over the `8x8` kernel, while keeping large matrices on `8x8`.
- Raised 2D broadcast/reduction OpenMP thresholds so medium-sized row kernels do
  not pay thread startup cost.
- Unified the remaining OpenMP `if` clauses behind `detail::use_openmp_for` so
  fallback loops also avoid parallel-region overhead when only one thread is
  active.
- Added benchmark rows for `where_into`.

## Tests

Commands run:

```bash
cmake --build build_torch -j
ctest --test-dir build_torch --output-on-failure
LD_PRELOAD="$(gcc -print-file-name=libasan.so)" ctest --test-dir build_asan --output-on-failure
cmake --build build_lite -j
ctest --test-dir build_lite --output-on-failure
cmake --build build_warn -j
ctest --test-dir build_warn --output-on-failure
./build_lite/litenp_basic
cmake --install build_lite --prefix /tmp/litenp-install-goal
cmake --find-package -DNAME=litenp -DCOMPILER_ID=GNU -DLANGUAGE=CXX \
  -DMODE=EXIST -DCMAKE_PREFIX_PATH=/tmp/litenp-install-goal
cat <<'CPP' | g++ -std=c++17 -I/tmp/litenp-install-goal/include \
  -x c++ -o /tmp/litenp_external_smoke -
#include <cassert>
#include <cstdint>
#include "litenp/litenp.hpp"
int main() {
    auto a = litenp::Array<float>::from_vector({2, 2}, {1, 2, 3, 4});
    a.values()[0] = 5.0f;
    auto copied = a.to_vector();
    assert(copied[0] == 5.0f);
    auto mask = litenp::greater<float>(
        a.view(), litenp::Array<float>::from_vector({1}, {2.0f}).view());
    litenp::Array<float> out({2, 2});
    litenp::where_into<float>(
        mask.view(), a.view(), litenp::Array<float>({1}, -1.0f).view(), out.view());
    assert(out({0, 0}) == 5.0f);
    assert(out({0, 1}) == -1.0f);
    assert(litenp::sum(out, 1)({1}) == 7.0f);
}
CPP
/tmp/litenp_external_smoke
```

Results:

- `build_torch`: `100% tests passed, 0 tests failed out of 1`.
- `build_asan`: passed with explicit `libasan` preload.
- `build_lite`: `100% tests passed, 0 tests failed out of 1`.
- `build_warn`: `100% tests passed, 0 tests failed out of 1`.
- `litenp_basic`: printed `row_sum: 25, 31` and `projected(1,1): 11`.
- install/find-package smoke: `litenp found.`
- installed-header external compile/run smoke: passed for `Array`, `values()`,
  `to_vector()`, `greater`, `where_into`, and `sum(axis)`.

New test coverage:

- `where_into` contiguous, row-broadcast, and `uint8_t` mask/output overlap.
- `where_into` partial overlap with `x` and `y` source views.
- Public `Array(shape)` still zero-initializes despite the internal
  uninitialized allocation path; `values()` and `to_vector()` remain covered by
  tests.
- NaN comparison behavior for vectorized `equal`, `not_equal`, and ordered
  comparisons, including vector-width chunks and scalar tails for `float` and
  `double`.
- vector-length total and axis reductions.
- `8x8` GEMM micro-kernel through an identity-matmul regression case.
- `4x16` GEMM micro-kernel through a `4x16 * 16x16 identity` regression case.

## Benchmark Evidence

Single-thread command:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ./build_torch/litenp_bench
```

Before this pass (`/tmp/litenp_bench_goal_20260525_1t.txt`) vs after
(`/tmp/litenp_bench_after_compare_threshold_1t.txt`):

| case | before | after | note |
| --- | ---: | ---: | --- |
| `sum all 2048x2048` | `1.5950 ms` | `0.3204 ms` | AVX2 contiguous sum |
| `sum axis1 2048x2048` | `1.5311 ms` | `0.2972 ms` | row reduction fast path |
| `sum axis1 2048^2` scale row | `3.0150 ms` | `0.3280 ms` | faster than latest Eigen/libtorch row |
| `greater 4M` scale row | `3.6443 ms` | `1.0050 ms` | AVX2 compare and lookup mask packing |
| `greater 16M` scale row | `15.7112 ms` | `5.6137 ms` | now roughly matches Eigen and beats libtorch |
| `matmul_into 384^3 f32` | `2.6227 ms` | `1.2573 ms` | `8x8` kernel and no pre-clear |
| `where_into 16M` | n/a | `7.1148 ms` | allocation-free path; Eigen `6.6082 ms`, libtorch `20.9667 ms` |

Additional GEMM micro-kernel check (`/tmp/litenp_bench_after_gemm4x16_1t.txt`):

| case | before | after | note |
| --- | ---: | ---: | --- |
| `matmul 256` | `0.2441 ms` | `0.2295 ms` | `4x16` kernel slightly beats Eigen in that run |
| `matmul 384` | `1.2110 ms` | `1.1686 ms` | closer to Eigen while still faster than libtorch |

Additional allocation-path check after the uninitialized internal allocation
pass (`/tmp/litenp_bench_after_uninit_fill_1t.txt`):

| case | before | after | note |
| --- | ---: | ---: | --- |
| `Array(shape) 4M f32` | `0.7709 ms` | `0.8287 ms` | public zero-fill preserved, slight allocator cost |
| `add alloc 4M f32` | `2.2760 ms` | `1.9785 ms` | full-overwrite output avoids pre-clear |
| `clip alloc 4M f32` | `2.2556 ms` | `1.0594 ms` | full-overwrite output avoids pre-clear |
| `where 16M` | `22.5578 ms` | `18.6617 ms` | allocating `where` improved but still alloc-bound |

Final single-thread sanity after the API/test/threshold cleanup
(`/tmp/litenp_bench_final_1t.txt`):

| case | final | Eigen | libtorch | note |
| --- | ---: | ---: | ---: | --- |
| `add 4M` | `1.5384 ms` | `1.5032 ms` | `1.5461 ms` | in the Eigen/libtorch band |
| `greater 4M` | `1.0032 ms` | `1.0176 ms` | `2.0177 ms` | slightly faster than Eigen in this run |
| `where_into 16M` | `7.2528 ms` | `6.5732 ms` | `20.8947 ms` | close to Eigen, much faster than libtorch |
| `sum axis0 2048^2` | `0.6186 ms` | `1.3367 ms` | `0.5115 ms` | faster than Eigen, near libtorch |
| `sum axis1 2048^2` | `0.4182 ms` | `0.3654 ms` | `0.3631 ms` | slightly behind both |
| `matmul 384` | `1.1742 ms` | `0.8374 ms` | `2.2959 ms` | still behind Eigen, faster than libtorch |
| `matmul_into 384^3 f32` | `1.1401 ms` | `0.7949 ms` | `2.4031 ms` | dedicated row stays faster than libtorch |

Default threaded benchmark remains noisy because Eigen, libtorch, OpenMP, and
OpenBLAS thread pools interact in the same process. Targeted threshold fixes
reduced avoidable thread overhead:

- `matmul 64` improved from `4.5205 ms` in
  `/tmp/litenp_bench_after_where_mt.txt` to `0.0109 ms` in
  `/tmp/litenp_bench_after_matmul_threshold_mt.txt`.
- `broadcast 1024^2` improved from `5.5560 ms` in
  `/tmp/litenp_bench_after_matmul_threshold_mt.txt` to `0.3640 ms` in
  `/tmp/litenp_bench_after_reduce_threshold_mt.txt`.
- `sum axis1 2048^2` improved from `5.9978 ms` in
  `/tmp/litenp_bench_after_matmul_threshold_mt.txt` to `0.4136 ms` in
  `/tmp/litenp_bench_after_reduce_threshold_mt.txt`.

## Remaining Gaps

- Allocating `where` remains much slower than `where_into` at 16M because it
  includes allocation and first-touch cost, though it improved after internal
  uninitialized output allocation.
- Large broadcast allocation still shows noisy Eigen/libtorch comparisons in
  the combined threaded benchmark.
- GEMM now beats local libtorch CPU rows in the tested single-thread matrix, but
  Eigen remains faster for several single-thread square GEMM sizes.
- Compare rows are much closer to Eigen after lookup mask packing, but timing is
  still noisy in the combined benchmark process.
- `max` reductions still use scalar float loops to preserve conservative
  `std::max` behavior, including NaN-sensitive behavior.
- The internal uninitialized allocation optimization changes the concrete
  `values()` return type to `Array<T>::storage_type`; code that explicitly
  binds it as `std::vector<T>&` will need to use `auto&`, the alias, or
  `to_vector()` when a standard-allocator copy is needed.
