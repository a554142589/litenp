# litenp performance matrix 2026-05-25

## Command

```bash
cmake -S . -B build_torch -DCMAKE_BUILD_TYPE=Release \
  -DLITENP_BUILD_BENCHMARKS=ON \
  -DLITENP_USE_OPENMP=ON \
  -DLITENP_USE_CBLAS=ON \
  -DLITENP_NATIVE_ARCH=ON \
  -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build_torch -j
ctest --test-dir build_torch --output-on-failure
./build_torch/litenp_bench | tee /tmp/litenp_bench_latest.txt
```

## Verification

- Build completed.
- `ctest` passed: `100% tests passed, 0 tests failed out of 1`.
- Benchmark completed and now includes a large scale matrix:
  - 1D sizes: `1K`, `64K`, `1M`, `4M`, `16M`.
  - 2D sizes: `256^2`, `512^2`, `1024^2`, `2048^2`.
  - Matmul sizes: `64`, `128`, `256`, `384`, `512`, `768`.

## Key Improvements Observed

Compared with the previous baseline in `docs/full_benchmark_2026-05-25.md`:

| case | before | after | note |
| --- | ---: | ---: | --- |
| `greater mask 4M f32` | `20.4848 ms` | `0.2788 ms` | contiguous compare fast path |
| `max all 2048x2048` | `19.0534 ms` | `0.2007 ms` | contiguous max fast path |
| `sum axis0 2048x2048` | `11.3464 ms` | `0.0619 ms` | 2D axis reduction fast path |
| `concatenate axis0 2x1M` | `21.7396 ms` | `0.5248 ms` | contiguous block copy |
| `stack axis0 2x1M` | `12.3234 ms` | `0.4290 ms` | contiguous block copy |
| `matmul noncontig view 384^3` | `448.6179 ms` | `1.1622 ms` | pack non-contiguous operands |

## Scale Matrix Highlights

Rows report baseline/litenp ratios in the benchmark as `E/lite` and `T/lite`.
Values greater than `1.0x` mean `litenp` is faster than that baseline.

## View / Transpose Aligned Compare

Latest command:

```bash
./build_torch/litenp_bench | tee /tmp/litenp_bench_latest.txt
```

| case | litenp | Eigen | libtorch | note |
| --- | ---: | ---: | ---: | --- |
| reshape view x100K | `4.3680 ms` | n/a | `21.3439 ms` | dynamic view API metadata |
| transpose view x100K | `4.3852 ms` | n/a | `26.1947 ms` | dynamic view API metadata |
| permute view x100K | `4.1508 ms` | n/a | `28.1216 ms` | dynamic view API metadata |
| slice view x100K | `1.9828 ms` | n/a | `33.7909 ms` | dynamic view API metadata |
| select view x100K | `2.8132 ms` | n/a | `34.0139 ms` | dynamic view API metadata |
| Eigen transpose expr x100K | n/a | `0.1513 ms` | n/a | expression object, not API-aligned |
| transpose materialize 2048^2 | `44.5606 ms` | `24.1059 ms` | `6.7153 ms` | real data copy/reorder |

- View-only operations: `litenp` is much faster than libtorch on the dynamic API
  path.
- Eigen expression construction is shown separately because it is not equivalent
  to `litenp`'s dynamic `Shape`/`strides` view objects or libtorch Tensor view
  APIs.
- Materialized transpose is currently a clear `litenp` bottleneck because it
  still uses a generic strided copy.

- 1D `add` at `4M`: `litenp 0.3895 ms`, Eigen `1.5480 ms`, libtorch
  `0.6371 ms`.
- 1D `greater` at `4M`: `litenp 0.2569 ms`, Eigen `1.3055 ms`, libtorch
  `0.2062 ms`.
- 1D `add` at `16M`: `litenp 8.8920 ms`, Eigen `6.8529 ms`, libtorch
  `9.4282 ms`.
- 2D `sum axis0 1024^2`: `litenp 0.0162 ms`, Eigen `0.2519 ms`, libtorch
  `0.0166 ms`.
- Dedicated `matmul 384^3`: `litenp 0.2380 ms`, Eigen `0.1561 ms`, libtorch
  `0.4411 ms`.
- Scale matrix GEMM rows are noisy in this environment, but after raising the
  CBLAS threshold the `256`, `384`, and `512` matrix rows beat Eigen/libtorch in
  the latest run, while `768` still trails libtorch slightly.

## Remaining Gaps

- `where` with allocation is still slower than Eigen/libtorch at large sizes.
- 2D broadcast with allocation is improved but does not consistently beat
  Eigen/libtorch in the new matrix.
- GEMM still does not consistently beat Eigen at medium/large square sizes.
- Some benchmark rows show OpenMP/libtorch thread-pool noise; the best evidence
  is the repeated dedicated sections plus the scale matrix trends, not any
  single noisy row.
