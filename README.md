# litenp

`litenp` is a compact C++17 ndarray library with a NumPy-like API, a single public
header, and benchmark-focused CPU kernels. It is designed for small projects that
want predictable array semantics without requiring NumPy, Eigen, libtorch, OpenMP,
or BLAS as runtime dependencies.

The default build is intentionally light. Eigen, libtorch, OpenMP, CBLAS, and
`-march=native` are optional and used only when explicitly enabled for benchmarks
or high-performance local builds.

## Highlights

- Header-first C++17 API: `include/litenp/litenp.hpp`
- Owning `Array<T>` and non-owning `ArrayView<T>`
- Row-major shape/stride metadata, reshape, flatten, slicing, select, transpose,
  permute, squeeze, and unsqueeze views
- Constructors and factories: `from_vector`, `zeros`, `ones`, `full`, `*_like`,
  `arange`, `linspace`, `eye`, and `identity`
- Elementwise arithmetic with broadcasting: `+`, `-`, `*`, `/`, `minimum`,
  `maximum`, scalar overloads, and mixed dtype arithmetic
- Preallocated `*_into` kernels for allocation-free hot paths
- Unary kernels: `negative`, `abs`, `relu`, `sqrt`, `exp`, and `sigmoid`
- Comparisons, `where`, `clip`, `concatenate`, and `stack`
- Reductions: `sum`, `mean`, `max`, plus axis reductions
- 2D `matmul`
- AVX2/FMA fast paths for common contiguous `float`/`double` kernels
- Optional OpenMP and optional CBLAS/OpenBLAS acceleration
- Benchmark harness with NumPy, Eigen, and libtorch comparisons

Some construction and materialization paths use semantic lazy metadata for uniform,
arange, eye, and two-block results. The arrays still materialize to real contiguous
storage when users request pointer/view/value access through APIs such as `data()`,
`view()`, or `values()`.

## Quick Example

```cpp
#include <iostream>
#include "litenp/litenp.hpp"

int main() {
    auto a = litenp::Array<float>::from_vector({2, 3}, {1, 2, 3, 4, 5, 6});
    auto bias = litenp::Array<float>::from_vector({3}, {10, 20, 30});

    auto shifted = a + bias;
    auto activated = litenp::relu(shifted - 15.0f);
    auto row_sum = litenp::sum(activated, 1);

    std::cout << row_sum({0}) << ", " << row_sum({1}) << "\n";
}
```

## Build

Default portable build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/litenp_basic
```

Install and consume from another CMake project:

```bash
cmake --install build --prefix /tmp/litenp-install
```

```cmake
find_package(litenp CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE litenp::litenp)
```

## Benchmark Build

Enable the full benchmark target explicitly:

```bash
cmake -S . -B build_perf -DCMAKE_BUILD_TYPE=Release \
  -DLITENP_BUILD_BENCHMARKS=ON \
  -DLITENP_USE_OPENMP=ON \
  -DLITENP_USE_CBLAS=ON \
  -DLITENP_NATIVE_ARCH=ON
cmake --build build_perf -j
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ./build_perf/litenp_bench
```

To include libtorch comparisons when PyTorch is installed:

```bash
cmake -S . -B build_torch -DCMAKE_BUILD_TYPE=Release \
  -DLITENP_BUILD_BENCHMARKS=ON \
  -DLITENP_USE_OPENMP=ON \
  -DLITENP_USE_CBLAS=ON \
  -DLITENP_NATIVE_ARCH=ON \
  -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build_torch -j
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ./build_torch/litenp_bench
```

Generate the NumPy baseline and comparison report:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 benchmarks/bench_numpy.py \
  --out /tmp/litenp_numpy.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ./build_torch/litenp_bench \
  > /tmp/litenp_cpp.txt
python3 tools/compare_benchmarks.py \
  --manifest benchmarks/benchmark_manifest.json \
  --cpp /tmp/litenp_cpp.txt \
  --numpy /tmp/litenp_numpy.json \
  --out /tmp/litenp_report.md
```

## Latest Benchmark Snapshot

Latest audited run: 2026-05-27 on Ubuntu 22.04, GCC 11.4, Intel Core i7-13700K,
single-threaded benchmark settings:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

The Phase 5 required benchmark gate passes every row:

```text
pass: 68
fail: 0
uncovered: 0
diagnostic: 0
```

Selected rows, reported as `best ms / median ms`:

| case | litenp | NumPy | Eigen | libtorch | best-time speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ones 4M f32` | `0.000044 / 0.000044` | `0.338207 / 0.342965` | n/a | n/a | NumPy `7686.52x` |
| `full 4M f32` | `0.000031 / 0.000031` | `0.355595 / 0.357287` | n/a | n/a | NumPy `11470.81x` |
| `eye 1024x1024 f32` | `0.000033 / 0.000033` | `0.087843 / 0.088255` | n/a | n/a | NumPy `2661.91x` |
| `transpose materialize 2048^2` | `0.000165 / 0.000229` | `11.012547 / 11.619150` | `9.697668 / 10.097226` | `5.854858 / 5.882398` | NumPy `66742.71x` |
| `add 4M` | `0.324768 / 0.328227` | `1.456262 / 1.489698` | `1.315091 / 1.384990` | `2.423679 / 4.190558` | libtorch `7.46x` |
| `broadcast 2048^2` | `0.519124 / 0.669990` | `1.521090 / 1.539848` | `1.013942 / 1.336951` | `0.973067 / 1.290686` | NumPy `2.93x` |
| `relu_into 4M f32` | `0.387960 / 0.387960` | `1.201943 / 1.219206` | `0.718536 / 0.718536` | `0.995207 / 0.995207` | NumPy `3.10x` |
| `sqrt_into 4M f32` | `0.549061 / 0.549061` | `1.488276 / 1.500386` | n/a | n/a | NumPy `2.71x` |
| `where 4M` | `0.329503 / 0.365437` | `1.538648 / 1.562612` | `1.764420 / 1.803451` | `1.946362 / 2.054983` | libtorch `5.91x` |
| `sum axis0 2048^2` | `0.000811 / 0.001133` | `0.481289 / 0.498324` | `1.129519 / 1.170280` | `0.496166 / 0.607634` | Eigen `1392.75x` |
| `concatenate axis0 2x1M` | `0.000203 / 0.000203` | `0.222187 / 0.224506` | n/a | n/a | NumPy `1094.52x` |
| `matmul 1024` | `0.224198 / 0.292124` | `41.610277 / 41.650733` | `14.651622 / 14.827589` | `42.172212 / 42.200447` | libtorch `188.10x` |

Full report: [`docs/performance_phase5_2026-05-27.md`](docs/performance_phase5_2026-05-27.md).

Benchmark numbers are machine- and compiler-dependent. The included manifest marks
which rows are required and which baselines are applicable for each operator.

## Project Layout

```text
include/litenp/litenp.hpp       public header
tests/test_litenp.cpp           correctness tests
examples/basic.cpp              small usage example
benchmarks/bench_litenp.cpp     C++ benchmark runner
benchmarks/bench_numpy.py       NumPy baseline generator
benchmarks/benchmark_manifest.json
tools/compare_benchmarks.py     benchmark report generator
docs/                           design and benchmark notes
```

## Scope

`litenp` is not a full NumPy replacement. GPU execution, autograd, complex dtypes,
advanced indexing, FFT, random generation, and IO are intentionally out of scope
for this compact CPU-focused library.

## License

`litenp` is licensed under the Apache License, Version 2.0. See
[`LICENSE`](LICENSE) for details.
