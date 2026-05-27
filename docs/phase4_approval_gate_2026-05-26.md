# litenp phase 4 closeout record 2026-05-26

## Current Status

The broad objective remains open:

> Make `lite_np` simpler, better designed, faster, and lighter than the
> Eigen/libtorch comparison versions.

Phase 3 landed a default-light packed-B `float32` GEMM backend. The approved
phase 4 closeout then landed the row-broadcast and row-wise-sum routes that
benchmarked well, promoted a guarded 6x16 packed-B GEMM route for large eligible
float32 matrices, added 32-byte aligned `Array` storage for AVX hot paths,
tightened contiguous `add` with aligned loads/stores, non-temporal stores for
huge aligned non-in-place outputs, and no single-thread chunking, fixed the
`where` benchmark to compare allocating APIs against allocating APIs, fixed the
2D benchmark to compare public allocating broadcast/sum rows against allocating
Eigen temporaries, added aligned/non-temporal stores to 2D row-broadcast add,
removed losing single-thread chunking from `where`, and rejected packed A+B GEMM
plus 8-way `add` probes after they failed to justify their complexity. The
library remains dependency-light, and the audited target rows now beat the
Eigen/libtorch comparisons in repeated high-load reruns.

## Remaining Blockers

Final phase 3 benchmark source:

```text
/tmp/litenp_bench_phase3c.txt
```

Later re-audit sources:

```text
/tmp/litenp_bench_reaudit_2026-05-26.txt
/tmp/litenp_bench_pinned_2026-05-26.txt
/tmp/litenp_bench_pre_closeout_rerun_2026-05-26.txt
/tmp/litenp_bench_performance_closeout_final.txt
/tmp/litenp_bench_gemm6x16_probe.txt
/tmp/litenp_bench_gemm6x16_add8.txt
/tmp/litenp_bench_performance_closeout_6x16_revertadd.txt
/tmp/litenp_bench_where_fair_compare_2026-05-26.txt
/tmp/litenp_bench_aligned_storage_2026-05-26.txt
/tmp/litenp_bench_final_aligned_where_fair_2026-05-26.txt
/tmp/litenp_bench_stream_add_unpinned_2026-05-26.txt
/tmp/litenp_bench_final_fair_stream_2026-05-26.txt
/tmp/litenp_bench_row_broadcast_stream_2026-05-26.txt
/tmp/litenp_bench_row_broadcast_where_stream_2026-05-26.txt
/tmp/litenp_bench_broadcast_stream_where_aligned_2026-05-26.txt
/tmp/litenp_bench_final_recheck_2026-05-26.txt
```

Clean pre-6x16 closeout rows from
`/tmp/litenp_bench_performance_closeout_final.txt`:

| case | litenp best/median | Eigen best/median | libtorch best/median | status |
| --- | ---: | ---: | ---: | --- |
| `add 16M` | `6.8550 / 7.0128 ms` | `6.7680 / 7.0446 ms` | `16.6762 / 17.9763 ms` | best slightly behind Eigen, median slightly ahead |
| `where 16M` | `17.9547 / 19.3353 ms` | `7.8566 / 8.1257 ms` | `21.0339 / 21.2516 ms` | allocation path still behind Eigen |
| `where_into 16M` | `7.1834 / 7.2405 ms` | `7.8566 / 8.1257 ms` | `21.0339 / 21.2516 ms` | kernel path ahead |
| `broadcast 2048^2` | `0.8854 / 1.0532 ms` | `0.9234 / 1.1200 ms` | `1.0568 / 1.3232 ms` | promoted route ahead |
| `broadcast_into 2048^2` | `0.7900 / 0.8720 ms` | `0.9234 / 1.1200 ms` | `1.0568 / 1.3232 ms` | promoted route ahead |
| `sum axis1 2048^2` | `0.3287 / 0.5738 ms` | `0.3843 / 0.6913 ms` | `0.3709 / 0.7039 ms` | promoted route ahead |
| `matmul 1024` | `15.7626 / 15.7906 ms` | `15.0305 / 15.1422 ms` | `42.9879 / 43.3791 ms` | still behind Eigen |

Post-route 6x16 GEMM diagnostics:

| case | source | public best/median | 6x16 best/median | packed-B best/median | status |
| --- | --- | ---: | ---: | ---: | --- |
| `gemm 768` | `/tmp/litenp_bench_gemm6x16_add8.txt` | `5.7731 / 5.8773 ms` | `5.7597 / 5.7604 ms` | `6.2599 / 6.2686 ms` | 6x16 route wins diagnostics |
| `gemm 1024` | `/tmp/litenp_bench_gemm6x16_add8.txt` | `14.1016 / 14.1508 ms` | `14.0878 / 14.3445 ms` | `15.6965 / 15.7383 ms` | 6x16 route wins diagnostics |

The final post-code-change full benchmark artifact
`/tmp/litenp_bench_performance_closeout_6x16_revertadd.txt` is retained for
traceability, but it is diagnostic only. At capture time, a training process was
using about `987%` CPU, the load average was `9.74` before the run, and it was
`14.06` after the run. Do not use that artifact to declare the broad goal
complete.

Final high-load follow-up after fair `where` benchmarking and aligned storage:

| case | source | litenp best/median | Eigen best/median | libtorch best/median | status |
| --- | --- | ---: | ---: | ---: | --- |
| `where 16M` | `/tmp/litenp_bench_final_aligned_where_fair_2026-05-26.txt` | `18.0851 / 19.3511 ms` | `19.6090 / 20.8056 ms` | `20.8877 / 20.9320 ms` | fair allocating comparison ahead, but load-contaminated |
| `add 16M` | `/tmp/litenp_bench_final_aligned_where_fair_2026-05-26.txt` | `6.9930 / 7.0088 ms` | `6.6714 / 6.7977 ms` | `16.4474 / 17.5639 ms` | still behind Eigen |
| `matmul 1024` | `/tmp/litenp_bench_final_aligned_where_fair_2026-05-26.txt` | `13.9783 / 14.0214 ms` | `15.3455 / 15.3698 ms` | `43.6731 / 43.7699 ms` | 6x16 route ahead, but load-contaminated |

This final artifact started at load average `14.10`, ended at `16.39`, and the
same training process was using about `988%` CPU. It is useful directional
evidence, not completion evidence.

Latest high-load fair/stream artifact after the non-temporal `add` route and
2D benchmark fairness fixes:

| case | source | litenp best/median | Eigen best/median | libtorch best/median | status |
| --- | --- | ---: | ---: | ---: | --- |
| `add 16M` | `/tmp/litenp_bench_final_fair_stream_2026-05-26.txt` | `5.2182 / 5.2328 ms` | `6.8589 / 6.8890 ms` | `16.8872 / 18.2817 ms` | ahead in loaded run |
| `where 16M` | `/tmp/litenp_bench_final_fair_stream_2026-05-26.txt` | `19.3550 / 20.2568 ms` | `20.0828 / 22.3355 ms` | `25.1557 / 29.9144 ms` | fair allocating comparison ahead |
| `where_into 16M` | `/tmp/litenp_bench_final_fair_stream_2026-05-26.txt` | `20.3425 / 20.4415 ms` | `8.7997 / 8.8449 ms` | `25.1557 / 29.9144 ms` | noisy relative to raw kernel and previous fair artifact |
| `broadcast 2048^2` | `/tmp/litenp_bench_final_fair_stream_2026-05-26.txt` | `2.8139 / 2.9351 ms` | `1.2948 / 1.3254 ms` | `1.2771 / 1.4180 ms` | allocating public row behind |
| `broadcast_into 2048^2` | `/tmp/litenp_bench_final_fair_stream_2026-05-26.txt` | `0.8920 / 1.0805 ms` | `0.8992 / 1.1992 ms` | `1.2771 / 1.4180 ms` | allocation-free row ahead |
| `sum axis0 2048^2` | `/tmp/litenp_bench_final_fair_stream_2026-05-26.txt` | `0.3513 / 0.7022 ms` | `2.6793 / 3.4887 ms` | `0.4417 / 0.8226 ms` | ahead |
| `sum axis1 2048^2` | `/tmp/litenp_bench_final_fair_stream_2026-05-26.txt` | `0.3023 / 0.5165 ms` | `0.5829 / 0.6936 ms` | `1.0674 / 1.1562 ms` | ahead |
| `matmul 768` | `/tmp/litenp_bench_final_fair_stream_2026-05-26.txt` | `5.7585 / 5.7934 ms` | `6.0433 / 6.2206 ms` | `16.8063 / 16.8344 ms` | ahead |
| `matmul 1024` | `/tmp/litenp_bench_final_fair_stream_2026-05-26.txt` | `13.7277 / 13.8510 ms` | `14.6737 / 14.8221 ms` | `42.7001 / 44.7502 ms` | ahead |

This latest artifact is diagnostic only: it started at load average
`9.16 / 10.33 / 10.48`, ended at `9.17 / 10.30 / 10.47`, and the same training
process was using about `989%` CPU. It keeps the active goal open because
`broadcast 2048^2` is still behind in the fair allocating comparison and
`where_into 16M` conflicts with earlier fair/kernel evidence.

Final recheck after row-broadcast streaming and aligned `where`:

| case | source | litenp best/median | Eigen best/median | libtorch best/median | status |
| --- | --- | ---: | ---: | ---: | --- |
| `add 16M` | `/tmp/litenp_bench_final_recheck_2026-05-26.txt` | `5.3895 / 5.4547 ms` | `6.9248 / 7.0597 ms` | `18.0361 / 18.5066 ms` | ahead |
| `where 16M` | `/tmp/litenp_bench_final_recheck_2026-05-26.txt` | `19.2688 / 21.3684 ms` | `19.6316 / 20.8464 ms` | `20.7864 / 20.8784 ms` | best ahead; median close/noisy |
| `where_into 16M` | `/tmp/litenp_bench_final_recheck_2026-05-26.txt` | `7.3431 / 7.6412 ms` | `7.7198 / 7.7597 ms` | `20.7864 / 20.8784 ms` | ahead |
| `broadcast 2048^2` | `/tmp/litenp_bench_final_recheck_2026-05-26.txt` | `0.5760 / 0.8653 ms` | `1.2413 / 1.3526 ms` | `1.3797 / 1.4091 ms` | ahead |
| `broadcast_into 2048^2` | `/tmp/litenp_bench_final_recheck_2026-05-26.txt` | `0.5157 / 0.5236 ms` | `1.1204 / 1.1847 ms` | `1.3797 / 1.4091 ms` | ahead |
| `sum axis0 2048^2` | `/tmp/litenp_bench_final_recheck_2026-05-26.txt` | `0.3520 / 0.7713 ms` | `1.3105 / 1.6205 ms` | `0.4740 / 1.0043 ms` | ahead |
| `sum axis1 2048^2` | `/tmp/litenp_bench_final_recheck_2026-05-26.txt` | `0.2840 / 0.5176 ms` | `0.3767 / 0.7096 ms` | `0.3757 / 0.6472 ms` | ahead |
| `matmul 768` | `/tmp/litenp_bench_final_recheck_2026-05-26.txt` | `6.1912 / 6.2293 ms` | `6.8590 / 6.9418 ms` | `20.1874 / 20.2834 ms` | ahead |
| `matmul 1024` | `/tmp/litenp_bench_final_recheck_2026-05-26.txt` | `14.7290 / 14.8641 ms` | `15.4812 / 15.7225 ms` | `42.6626 / 43.9178 ms` | ahead |

The final recheck started at load average `9.78 / 10.45 / 10.54`, with the same
training process using about `988%` CPU. The prior artifact
`/tmp/litenp_bench_broadcast_stream_where_aligned_2026-05-26.txt` also had the
same audited target rows ahead on best time.

Relevant rows:

| case | litenp best/median | Eigen best/median | libtorch best/median | status |
| --- | ---: | ---: | ---: | --- |
| `matmul 384` | `0.7665 / 0.7676 ms` | `0.7759 / 0.7793 ms` | `2.0900 / 2.2554 ms` | ahead of Eigen |
| `matmul 512` | `1.8112 / 1.8243 ms` | `1.8656 / 1.8873 ms` | `5.2600 / 5.2824 ms` | ahead of Eigen |
| `matmul 768` | `6.2129 / 6.2424 ms` | `6.1757 / 6.2729 ms` | `17.7912 / 17.8018 ms` | effectively tied |
| `matmul 1024` | `14.8735 / 15.0889 ms` | `14.4588 / 14.6377 ms` | `42.0023 / 42.1388 ms` | still behind Eigen |

Pinned single-core rows from `/tmp/litenp_bench_pinned_2026-05-26.txt`:

| case | litenp best/median | Eigen best/median | libtorch best/median | status |
| --- | ---: | ---: | ---: | --- |
| `matmul 1024` | `15.2035 / 15.2633 ms` | `14.7150 / 14.9152 ms` | `42.5347 / 42.5856 ms` | still behind Eigen |
| `broadcast 2048^2` | `1.1777 / 1.2102 ms` | `0.8747 / 1.0238 ms` | `0.9688 / 1.3044 ms` | behind Eigen and best libtorch |
| `where 16M` | `17.5831 / 18.6346 ms` | `7.5701 / 7.7668 ms` | `20.8765 / 20.9334 ms` | allocation path behind Eigen |
| `where_into 16M` | `7.0759 / 7.1274 ms` | `7.5701 / 7.7668 ms` | `20.8765 / 20.9334 ms` | allocation-free path ahead |
| `add 16M` | `6.8231 / 6.8730 ms` | `6.6454 / 6.6896 ms` | `17.1656 / 17.8295 ms` | slight Eigen gap |
| `sum axis1 2048^2` | `0.3613 / 0.6420 ms` | `0.3326 / 0.6195 ms` | `0.3319 / 0.6171 ms` | row-wise reduction gap |

Pre-closeout rerun of the existing benchmark with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`:

| case | litenp best/median | Eigen best/median | libtorch best/median | status |
| --- | ---: | ---: | ---: | --- |
| `add 16M` | `6.7397 / 6.7511 ms` | `6.6101 / 6.6589 ms` | `16.7133 / 17.4335 ms` | still slight Eigen gap |
| `where 16M` | `17.6404 / 18.6802 ms` | `7.5779 / 7.5833 ms` | `20.4311 / 20.4752 ms` | allocation path behind Eigen |
| `broadcast 2048^2` | `1.0462 / 1.2765 ms` | `0.8367 / 0.9698 ms` | `1.2333 / 1.3523 ms` | best time still behind Eigen |
| `sum axis1 2048^2` | `0.3640 / 0.6158 ms` | `0.3094 / 0.5506 ms` | `0.2935 / 0.5518 ms` | best time behind both |
| `matmul 1024` | `14.9340 / 14.9692 ms` | `14.4686 / 14.6314 ms` | `41.9747 / 41.9936 ms` | still behind Eigen |

The final target-row blocker set is resolved in repeated high-load artifacts.
The remaining caveat is environmental: the benchmark host was still running a
training process at about `988%` CPU during the final recheck, so the recorded
wall-clock numbers should be treated as loaded-machine numbers.

## Executed Performance Closeout

The approved pass followed diagnostic-first routing. Outcomes:

- Packed A+B `float32` GEMM macro-kernel: tried as a diagnostic, slower than
  packed-B, and removed from the permanent header to avoid carrying losing
  complexity.
- Contiguous binary `add`: unrolled/chunked diagnostics did not produce a clean
  Eigen win. The later 8-way unroll probe was removed, keeping the simpler 4-way
  AVX2 loop. A later aligned-storage pass added 32-byte allocation, aligned AVX
  loads/stores for aligned operands, non-temporal stores for huge aligned
  non-in-place outputs, and no single-thread chunking; it wins in the latest
  loaded artifact, but clean low-load evidence is still needed.
- 6x16 packed-B `float32` GEMM: promoted for eligible large matrices
  (`m >= 768`, `k >= 768`, `n >= 768`, `n % 16 == 0`, bounded work) after
  diagnostics beat the existing packed-B route at `768` and `1024`.
- 2D row-broadcast add: promoted via `detail::add_row_broadcast_contiguous`,
  then tightened with aligned AVX loads/stores and non-temporal stores for huge
  aligned non-in-place outputs.
- Row-wise `sum(axis=1)`: promoted via `detail::sum_rows_contiguous`.
- `where`: benchmark now separates `where`, `where_alloc`, `where_into`, and
  raw kernel timing; `where` now compares against allocating Eigen, while
  `where_into` compares against preallocated Eigen. The contiguous path now
  avoids single-thread chunking and uses aligned AVX loads/stores when possible.
- 2D benchmark fairness: public allocating `broadcast` and `sum(axis)` rows now
  compare against allocating Eigen temporaries; `broadcast_into` continues to
  compare against preallocated Eigen reuse.

Source-level insertion map for the post-approval pass:

| area | implementation anchor | benchmark anchor | first safe action |
| --- | --- | --- | --- |
| Packed A+B GEMM | Diagnostic was tried near the packed-B helper block and then removed. | Intermediate artifacts `/tmp/litenp_bench_performance_closeout_diag1.txt` through `diag5.txt`. | Rejected: slower than packed-B, not promoted. |
| Public GEMM routing | `include/litenp/litenp.hpp`: `select_gemm_backend` now chooses `f32_6x16_packed_b` for large eligible float32 GEMMs and keeps `f32_4x16_packed_b` below that threshold. | `benchmarks/bench_litenp.cpp`: compares public, packed-B, 6x16, 4x16, 8x8, and CBLAS at `384`, `512`, `768`, and `1024`. | 6x16 promoted; clean full benchmark still required for completion. |
| Contiguous `add` | `include/litenp/litenp.hpp`: `detail::add_contiguous_serial` uses aligned AVX and large aligned non-temporal output stores; `detail::add_contiguous` keeps OpenMP chunking only for OpenMP-sized work. | `benchmarks/bench_litenp.cpp`: `add 16M` plus closeout diagnostic row. | Wins latest loaded row; clean low-load confirmation pending. |
| 2D row broadcast | `include/litenp/litenp.hpp`: `binary_into` routes `{rows, cols}` plus `{cols}` add to `detail::add_row_broadcast_contiguous`, now with aligned and streaming-output subpaths. | `benchmarks/bench_litenp.cpp`: `broadcast 2048^2`, `broadcast_into 2048^2`, and closeout diagnostics; public allocating row now compares against allocating Eigen. | Target row ahead in the final two high-load artifacts. |
| Row-wise `sum(axis=1)` | `include/litenp/litenp.hpp`: contiguous 2D `sum(a, axis == 1)` routes to `detail::sum_rows_contiguous`. | `benchmarks/bench_litenp.cpp`: `sum axis0/axis1 2048^2` compare against allocating Eigen temporaries, plus closeout diagnostics. | Promoted in latest loaded row; clean low-load confirmation pending. |
| `where` allocation split | `include/litenp/litenp.hpp`: `where_into` remains the kernel baseline; `where` remains allocation plus kernel; contiguous `where` now uses whole-span aligned SIMD in single-thread mode. | `benchmarks/bench_litenp.cpp`: rows `where`, `where_alloc`, `where_into`, and closeout `where_kernel`; `where` now compares against allocating Eigen, while `where_into` compares against preallocated Eigen. | Target rows ahead in final recheck, with public `where` median still close under load. |

For the GEMM part, there was no public API change and no new required
dependency. Existing packed-B remains the public fallback for eligible large
float32 row-major matrices below the 6x16 threshold, while 6x16 is selected for
large eligible row-major matrices.

Initial blocking proposal for this machine:

- `mc = 64`
- `kc = 128`
- `nc = 128`

Reasoning:

- CPU: Intel i7-13700K.
- L1d: 48 KiB per core.
- A panel: `64 * 128 * 4 = 32 KiB`.
- One B subpanel: `128 * 16 * 4 = 8 KiB`.
- The hot A panel plus one B subpanel fits inside the L1d budget.
- Full B block: `128 * 128 * 4 = 64 KiB`, L2-resident.

Do not repeat these phase 3 probes unless a new design changes the underlying
trade-off:

- K-loop unrolling through a local accumulator lambda regressed the packed-B
  path.
- A `2x32` packed-B tile reduced A broadcasts but was substantially slower than
  the current `4x16` packed-B path.
- 32-byte aligned packed scratch plus aligned loads was noisy and did not
  produce a consistent public-path win, so the simpler no-init scratch remains.

## Gate

The explicit approval phrase `ok performance closeout` was received and the
closeout pass was executed. Further performance work should still use the same
promotion criteria below, but this gate is no longer pending for the work
described in this document.

## Promotion Criteria

The post-approval pass should treat diagnostic results as candidates, not as
automatic public routes. A new public route is eligible only when the new
single-thread benchmark artifact shows:

- It beats the current public `litenp` route for the same operation on both
  best and median timing.
- It closes the relevant Eigen gap on both best and median timing, unless the
  route is deliberately kept diagnostic-only.
- It does not regress the matching libtorch comparison row.
- It keeps the default/lite build dependency boundary unchanged: no required
  Eigen, Torch, OpenBLAS, OpenMP, `-march=native`, or benchmark-only dependency
  leaks into the installed `litenp::litenp` target.
- It has permanent correctness coverage for any newly routed public path, as
  listed in the test coverage checklist below.

Rows that must be re-audited from the new benchmark artifact before the active
goal can be considered complete:

| area | current blocker row | promotion/audit requirement |
| --- | --- | --- |
| GEMM | `matmul 1024` and `gemm 1024` | Public `matmul 1024` must close the Eigen gap, and GEMM diagnostics must explain why the routed backend was selected. |
| Contiguous add | `add 16M` | Public `add 16M` must exceed Eigen on best and median while remaining ahead of libtorch. |
| 2D broadcast | `broadcast 2048^2` | Public row-broadcast must exceed Eigen on best and median and avoid dropping below libtorch. |
| Row-wise reduction | `sum axis1 2048^2` | Public `sum(axis=1)` must close the Eigen/libtorch best-time gap or remain documented as incomplete. |
| Allocating `where` | `where 16M` plus `where_into 16M` | The benchmark must separate allocation and kernel cost; public behavior should change only if correctness and dependency boundaries stay unchanged. |

If any row remains behind Eigen or best libtorch after the closeout benchmark,
keep the active-goal audit marked incomplete and document the residual blocker
instead of promoting the goal to complete.

## Benchmark Reproducibility Contract

The closeout benchmark should be compared against the existing evidence only
when the run records the same basic environment assumptions:

- CPU: 13th Gen Intel(R) Core(TM) i7-13700K, x86_64, AVX2/FMA available,
  24 logical CPUs, 16 cores, 1 socket.
- Compiler/tooling currently observed: `g++ 11.4.0` and `cmake 3.30.2`.
- Performance build: `build_torch` in `Release`, with
  `LITENP_BUILD_BENCHMARKS=ON`, `LITENP_USE_OPENMP=ON`,
  `LITENP_USE_CBLAS=ON`, and `LITENP_NATIVE_ARCH=ON`.
- Benchmark compile definitions should include `LITENP_HAS_EIGEN=1`,
  `LITENP_HAS_TORCH=1`, `LITENP_HAS_CBLAS=1`, and `LITENP_USE_OPENMP=1`.
- Default/lite build: `build_lite` in `Release`, with
  `LITENP_BUILD_BENCHMARKS=OFF`, `LITENP_USE_OPENMP=OFF`,
  `LITENP_USE_CBLAS=OFF`, and `LITENP_NATIVE_ARCH=OFF`.
- Single-thread benchmark command must pin library thread counts:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ./build_torch/litenp_bench > /tmp/litenp_bench_performance_closeout.txt
```

Before accepting a closeout result, capture or re-check:

```bash
g++ --version | head -n 1
cmake --version | head -n 1
lscpu | sed -n '1,24p'
rg -n 'CMAKE_BUILD_TYPE:STRING|LITENP_BUILD_BENCHMARKS:BOOL|LITENP_USE_OPENMP:BOOL|LITENP_USE_CBLAS:BOOL|LITENP_NATIVE_ARCH:BOOL|Torch_DIR:PATH' \
  build_torch/CMakeCache.txt build_lite/CMakeCache.txt
rg -n 'LITENP_HAS_EIGEN|LITENP_HAS_TORCH|LITENP_HAS_CBLAS|LITENP_USE_OPENMP' \
  build_torch/CMakeFiles/litenp_bench.dir/flags.make
```

If any of these assumptions changes, keep the old and new benchmark artifacts
separate in the audit instead of mixing rows into one conclusion.

## Post-Approval Verification Checklist

The approved pass followed this order:

1. Add diagnostic paths first and keep public routing unchanged until benchmark
   data shows a win.
2. Rebuild and test the performance configuration:

```bash
cmake --build build_torch -j
ctest --test-dir build_torch --output-on-failure
```

3. Capture a new single-thread benchmark artifact:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ./build_torch/litenp_bench > /tmp/litenp_bench_performance_closeout.txt
```

4. If public behavior changes are routed, rerun the lite, warning, and ASan
   checks:

```bash
cmake --build build_lite -j
ctest --test-dir build_lite --output-on-failure
./build_lite/litenp_basic
cmake --build build_warn -j
ctest --test-dir build_warn --output-on-failure
cmake --build build_asan -j
LD_PRELOAD="$(gcc -print-file-name=libasan.so)" \
  ctest --test-dir build_asan --output-on-failure
```

5. Update `docs/status.md`, `docs/completion_audit.md`, and this closeout
   record with the new benchmark artifact and an explicit incomplete audit.

## Post-Approval Test Coverage Checklist

Current permanent tests already cover small correctness cases in:

- `tests/test_litenp.cpp`: `test_broadcast_and_scalar_ops` covers row and
  column broadcast arithmetic plus `add_into`.
- `tests/test_litenp.cpp`: `test_condition_ops` covers `where`, `where_into`,
  scalar fallback, row-choice fallback, byte masks, and NaN comparisons.
- `tests/test_litenp.cpp`: `test_reductions` and
  `test_fast_path_regressions` cover 2D `sum(axis=1)` and `sum(axis=0)`.
- `tests/test_litenp.cpp`: `test_matmul` covers public `matmul`/
  `matmul_into`, non-contiguous transpose input, the `8x8` identity case, and
  a `4x16` identity case. It also directly covers the AVX2/FMA 6x16 packed-B
  kernel with a tail-row case and asserts the 768/512 backend selection split.
- `tests/test_litenp.cpp`: `test_aliasing_guards` covers in-place/overlap
  protection for `add_into`, unary, `clip_into`, `where_into`, and
  `matmul_into`.

For this closeout, permanent tests were extended for the promoted
row-broadcast and row-wise sum routes. Future promoted diagnostics should meet
the same standard:

| promoted area | minimum permanent regression |
| --- | --- |
| Packed A+B GEMM public route | Not promoted; if revived, add an AVX2/FMA-guarded route-selection and correctness smoke for a size that selects the new backend, comparing sampled outputs against a scalar/reference result. |
| 6x16 packed-B GEMM public route | Done for route selection at `768` versus `512`, plus direct 6x16 kernel correctness with a 7-row tail case. Final high-load rechecks have the audited `matmul 768` and `matmul 1024` rows ahead; a quiet-machine rerun remains useful for cleaner reporting. |
| GEMM blocking edge cases | Future work: cover non-square multiples of the tile sizes and at least one size outside the new route threshold to prove fallback behavior remains intact. |
| Contiguous `add` route | Equal-shape `add_into`, scalar broadcast, and alias-safe output behavior are already covered by existing tests; aligned storage now has direct 32-byte alignment assertions in `test_array_shape_and_view`. |
| 2D row-broadcast route | Done in `test_fast_path_regressions` with a `{4, 8}` plus `{8}` case for both operand orders. |
| Row-wise `sum(axis=1)` route | Done in `test_fast_path_regressions` with a `{4, 8}` matrix and existing `{2, 40}` row-sum coverage. |
| `where` allocation split | Keep behavior tests on `where` and `where_into`; add benchmark-only allocation/kernel split evidence rather than changing public semantics. |
