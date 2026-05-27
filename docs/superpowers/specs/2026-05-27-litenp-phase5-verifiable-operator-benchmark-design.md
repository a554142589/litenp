# litenp phase 5 verifiable operator benchmark design

## Objective

Optimize `lite_np` so every public operator is faster than NumPy, Eigen, and
libtorch in explicitly defined, reproducible benchmark scenarios.

For this phase, "done" means all of the following are true:

- every public `litenp` operator family has at least one relevant benchmark row;
- each row has an equivalent NumPy, Eigen, and libtorch comparison when the
  baseline has a comparable CPU operation;
- each row reports best and median times, plus pass/fail status for every
  available baseline;
- the final audit has no failing or uncovered required rows;
- the default library target remains header-first and dependency-light.

This phase does not claim full Python NumPy API compatibility. It measures the
implemented `litenp` operator surface.

## Current Evidence

Fresh commands run on 2026-05-27:

```bash
cmake --build build_lite -j
ctest --test-dir build_lite --output-on-failure
cmake --build build_torch -j
ctest --test-dir build_torch --output-on-failure
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ./build_torch/litenp_bench \
  | tee /tmp/litenp_bench_current_goal_2026-05-27.txt
```

Both `build_lite` and `build_torch` test runs passed. The performance build has
Eigen, libtorch, OpenMP, OpenBLAS, and native-arch options enabled in
`build_torch/CMakeCache.txt`.

The current C++ benchmark has 16 sections and compares many rows against Eigen
and libtorch. It does not include NumPy timings. A repository search found
`NumPy` only in docs/README prose, not in benchmark code or generated results.

## Prompt-To-Artifact Checklist

| Requirement | Required artifact | Current evidence | Status |
| --- | --- | --- | --- |
| Optimize `lite_np` | `include/litenp/litenp.hpp` optimized kernels plus benchmark wins | Existing AVX/OpenMP/GEMM paths are present, but current benchmark still has failures | Incomplete |
| Every operator | Public operator manifest with required rows | `benchmarks/bench_litenp.cpp` covers many APIs, but not every row has full baseline comparison | Incomplete |
| Beats NumPy | NumPy benchmark artifact with comparable rows | No NumPy benchmark code or output exists | Missing |
| Beats Eigen | C++ benchmark rows with Eigen ratios >= 1.0 | `/tmp/litenp_bench_current_goal_2026-05-27.txt` has 21 failing rows vs Eigen/libtorch | Incomplete |
| Beats libtorch | C++ benchmark rows with libtorch ratios >= 1.0 | Same artifact has several failing rows vs libtorch | Incomplete |
| Verifiable benchmark scenarios | Machine/thread settings, scripts, outputs, pass/fail summary | Thread settings were pinned for the fresh C++ run; no unified verifier exists | Incomplete |
| Do not regress lightness | Default build without Eigen/Torch/OpenBLAS/OpenMP runtime dependency | Existing docs show this was true on 2026-05-26; `build_lite` tests pass now | Needs final recheck |

## Public Operator Surface

The benchmark and tests touch the implemented public operator families:

- construction: `Array`, `from_vector`, `arange`, `zeros`, `ones`, `full`,
  `zeros_like`, `ones_like`, `full_like`, `linspace`, `eye`, `identity`
- view/metadata: `reshape`, `flatten`, `squeeze`, `unsqueeze`, `slice`,
  `select`, `transpose`, `permute`
- copy/cast: `as_contiguous`, `astype`
- binary: `add`, `subtract`, `multiply`, `divide`, `minimum`, `maximum`,
  scalar variants, mixed dtype variants, operators `+ - * /`
- unary: `negative`, `abs`, `relu`, `sqrt`, `exp`, `sigmoid`
- conditionals: `less`, `less_equal`, `greater`, `greater_equal`, `equal`,
  `not_equal`, `where`, `clip`
- combine/reduce: `concatenate`, `stack`, `sum`, `mean`, `max`, `fill_view`
- matrix: `matmul`, `matmul_into`

The current benchmark times most of these for `litenp`, but only a smaller set
has Eigen/libtorch comparisons and none have NumPy comparisons.

## Fresh Failing Rows

Rows below are from `/tmp/litenp_bench_current_goal_2026-05-27.txt`. A ratio
below `1.0x` means the baseline is faster than `litenp` on best time.

| case | E/lite | T/lite | litenp b/m | Eigen b/m | libtorch b/m |
| --- | ---: | ---: | ---: | ---: | ---: |
| transpose materialize 2048^2 | 0.97x | 0.33x | 9.6049/12.5429 | 9.3507/9.9622 | 3.1938/7.2784 |
| add 1K | 0.11x | 1.25x | 0.0005/0.0006 | 0.0001/0.0001 | 0.0006/0.0008 |
| greater 1K | 0.17x | 1.87x | 0.0006/0.0006 | 0.0001/0.0001 | 0.0011/0.0012 |
| where 1K | 0.39x | 0.79x | 0.0014/0.0018 | 0.0005/0.0006 | 0.0011/0.0012 |
| where_into 1K | 0.94x | 1.99x | 0.0005/0.0006 | 0.0005/0.0005 | 0.0011/0.0012 |
| add 1M | 0.81x | 1.35x | 0.2029/0.2100 | 0.1643/0.1650 | 0.2739/0.2803 |
| greater 1M | 0.99x | 3.85x | 0.1035/0.1273 | 0.1029/0.1214 | 0.3991/0.4021 |
| add 4M | 0.99x | 1.79x | 1.3369/1.4186 | 1.3243/1.3507 | 2.3906/4.1911 |
| greater 4M | 0.92x | 1.83x | 0.9867/1.0277 | 0.9115/0.9925 | 1.8030/1.8577 |
| where 4M | 0.71x | 0.81x | 2.4836/4.3433 | 1.7597/1.7882 | 2.0037/2.0262 |
| greater 16M | 0.99x | 1.55x | 5.1130/5.1314 | 5.0817/5.1192 | 7.9123/7.9316 |
| broadcast 256^2 | 0.46x | 0.53x | 0.0081/0.0159 | 0.0037/0.0049 | 0.0043/0.0058 |
| broadcast_into 256^2 | 0.63x | 0.71x | 0.0060/0.0061 | 0.0038/0.0038 | 0.0043/0.0058 |
| sum axis1 256^2 | 0.88x | 1.68x | 0.0017/0.0017 | 0.0015/0.0015 | 0.0028/0.0029 |
| broadcast 512^2 | 0.40x | 0.42x | 0.0650/0.0778 | 0.0257/0.0405 | 0.0273/0.0292 |
| broadcast_into 512^2 | 0.70x | 0.93x | 0.0293/0.0297 | 0.0206/0.0208 | 0.0273/0.0292 |
| sum axis0 512^2 | 1.42x | 0.99x | 0.0100/0.0103 | 0.0142/0.0143 | 0.0099/0.0107 |
| broadcast 1024^2 | 0.90x | 1.13x | 0.1679/0.2077 | 0.1517/0.2332 | 0.1893/0.2538 |
| broadcast_into 1024^2 | 0.88x | 1.39x | 0.1359/0.1583 | 0.1202/0.1413 | 0.1893/0.2538 |
| sum axis1 1024^2 | 0.77x | 0.73x | 0.0627/0.0629 | 0.0483/0.0618 | 0.0456/0.0470 |
| matmul 64 | 0.92x | 2.67x | 0.0045/0.0049 | 0.0041/0.0076 | 0.0120/0.0171 |

## Design

Phase 5 should be verification-first:

1. Add an operator benchmark manifest that names required rows, dtype, shape,
   allocation policy, and comparable baseline expressions.
2. Add a NumPy benchmark script that emits the same row names and `best/median`
   timings for the implemented operator surface.
3. Add a summary/verifier script that merges C++ benchmark output with NumPy
   output and reports `pass`, `fail`, or `uncovered` for each required row.
4. Only then optimize failing rows, starting with rows that are both meaningful
   and not dominated by microsecond-level call overhead.

Recommended row policy:

- Required performance rows should focus on stable sizes where timing noise does
  not dominate: `64K`, `1M`, `4M`, `16M` for 1D; `512^2`, `1024^2`, `2048^2`
  for 2D; `128`, `256`, `384`, `512`, `768`, `1024` for GEMM.
- Tiny `1K` rows should remain diagnostic, because C++ function/object overhead
  and Eigen expression setup can dominate at sub-microsecond timings. They can
  be optimized, but should not be the only proof of an operator family.
- Allocation and allocation-free forms must be separated. `where` should compare
  against allocating baselines, while `where_into` should compare against
  reusable-output baselines where available.
- View-only rows should compare against dynamic view APIs in libtorch/NumPy.
  Eigen expression construction should stay diagnostic unless an API-equivalent
  dynamic view row is available.

## Implementation Plan

1. Create `benchmarks/benchmark_manifest.json` with required rows and baseline
   mappings.
2. Create `benchmarks/bench_numpy.py` using `time.perf_counter_ns`, fixed dtype
   arrays, warmups, repeats, and single-thread environment variables.
3. Add `tools/compare_benchmarks.py` to parse:
   - `/tmp/litenp_bench_current_goal_*.txt`
   - NumPy JSON/CSV output
   - the manifest
4. Extend `benchmarks/bench_litenp.cpp` only where manifest-required rows lack
   Eigen/libtorch counterparts.
5. Optimize the first failing cluster after verifier output is available. The
   likely first clusters are:
   - `as_contiguous(transpose)` / materialized transpose
   - row-broadcast add at `512^2` and `1024^2`
   - row-wise sum at `1024^2`
   - `where` allocation/kernel behavior at `4M`
   - small GEMM routing at `64`

## Verification

Minimum commands for Phase 5:

```bash
cmake --build build_torch -j
ctest --test-dir build_torch --output-on-failure
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ./build_torch/litenp_bench \
  > /tmp/litenp_bench_phase5_cpp.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 benchmarks/bench_numpy.py \
  > /tmp/litenp_bench_phase5_numpy.json
python3 tools/compare_benchmarks.py \
  --manifest benchmarks/benchmark_manifest.json \
  --cpp /tmp/litenp_bench_phase5_cpp.txt \
  --numpy /tmp/litenp_bench_phase5_numpy.json \
  --out docs/performance_phase5_2026-05-27.md
```

The goal remains open until the generated Phase 5 report shows no required row
with `fail` or `uncovered`.
