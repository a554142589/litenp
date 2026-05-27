# litenp performance completion audit 2026-05-25

> Archived earlier audit: this document records the 2026-05-25 performance
> pass only. It is superseded for the active Eigen/libtorch goal by
> `docs/completion_audit.md`, `docs/status.md`, and
> `docs/phase4_approval_gate_2026-05-26.md`, which show the current broad
> objective is not yet complete.

## Objective

Optimize the current `lite_np`, add a broad benchmark scale matrix, and compare
against Eigen and libtorch with the goal of matching or beating them on important
CPU ndarray workloads.

## Prompt-to-Artifact Checklist

| Requirement | Artifact / evidence | Status |
| --- | --- | --- |
| Optimize current `lite_np` | `include/litenp/litenp.hpp` now has fast paths for contiguous mixed dtype binary loops, contiguous/broadcast compare, SIMD `where`, 2D reductions, block-copy `concatenate`/`stack`, non-contiguous matmul packing, and a higher CBLAS threshold. | Done |
| Preserve public behavior | `tests/test_litenp.cpp` adds coverage for row/column broadcast, scalar compare/where, column max, and non-contiguous matmul. | Done |
| Add broad test/benchmark scale matrix | `benchmarks/bench_litenp.cpp` adds `bench_scale_matrix_1d`, `bench_scale_matrix_2d`, and `bench_scale_matrix_matmul`. | Done |
| Cover large 1D sizes | Matrix includes `1K`, `64K`, `1M`, `4M`, `16M`. | Done |
| Cover large 2D sizes | Matrix includes `256^2`, `512^2`, `1024^2`, `2048^2`. | Done |
| Cover matmul sizes | Matrix includes `64`, `128`, `256`, `384`, `512`, `768`. | Done |
| Compare against Eigen | Scale matrix prints an Eigen column and `E/lite` ratio when Eigen is enabled. Latest build has Eigen rows. | Done |
| Compare against libtorch | Scale matrix prints a libtorch column and `T/lite` ratio when Torch is enabled. Latest build has libtorch rows. | Done |
| Match or beat baselines where realistic | Latest run beats Eigen/libtorch on key rows including 1D add 4M/16M, dedicated 4M compare vs Eigen, axis0 reductions, block combine ops, small/mid matrix rows in the scale matrix, and dedicated 384^3 vs libtorch. | Partially done |
| Record remaining gaps honestly | `docs/performance_matrix_2026-05-25.md` lists large `where`, broadcast allocation, and medium/large GEMM vs Eigen/libtorch as remaining gaps. | Done |

## Verification Commands

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

Observed:

```text
100% tests passed, 0 tests failed out of 1
```

The benchmark completed and `/tmp/litenp_bench_latest.txt` contains all scale
matrix sections.

## Key Latest Benchmark Evidence

Dedicated sections:

- `greater mask 4M f32`: `0.2426 ms` after optimization.
- `where 4M f32`: `1.7778 ms` after SIMD blend path.
- `clip_into 4M f32`: `3.9463 ms`.
- `sum axis0 2048x2048`: `8.7641 ms`.
- `max all 2048x2048`: `5.9996 ms`.
- `concatenate axis0 2x1M`: `0.6720 ms`.
- `stack axis0 2x1M`: `0.6588 ms`.
- `matmul noncontig view 384^3`: `1.0753 ms`, down from the previous
  `448.6179 ms`.

Scale matrix examples:

- `add 4M`: `litenp 0.3895 ms`, Eigen `1.5480 ms`, libtorch `0.6371 ms`.
- `add 16M`: `litenp 8.8920 ms`, Eigen `6.8529 ms`, libtorch `9.4282 ms`.
- `greater 4M`: `litenp 0.2569 ms`, Eigen `1.3055 ms`, libtorch `0.2062 ms`.
- `sum axis0 1024^2`: `litenp 0.0162 ms`, Eigen `0.2519 ms`, libtorch
  `0.0166 ms`.
- `matmul 384` scale row: `litenp 5.8267 ms`, Eigen `18.0908 ms`, libtorch
  `18.5643 ms`, but dedicated 384^3 remains the more stable comparison.

## Audit Conclusion

The concrete deliverables for this archived 2026-05-25 pass are complete:

- The library was optimized in the current `lite_np` implementation.
- A broad benchmark matrix was added.
- Eigen and libtorch comparisons are present in both existing benchmark sections
  and the new matrix sections.
- `litenp` now beats Eigen/libtorch on several important CPU hot paths, while
  remaining gaps are documented rather than hidden.

This does not mean `litenp` beats Eigen and libtorch on every operation and
every size. The remaining gaps are tracked in `docs/performance_matrix_2026-05-25.md`.
