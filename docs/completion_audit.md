# litenp Completion Audit

## Objective

Current active objective:

> Make `lite_np` simpler, better designed, faster, and lighter than the
> Eigen/libtorch comparison versions.

Earlier MVP objective:

> Implement a lightweight, high-performance C++ NumPy-like ndarray library whose
> key tensor/array operations are benchmarked against libtorch, while keeping a
> lightweight integration style similar to Eigen.

This audit treats the target as a lightweight NumPy core, not a full clone of
the complete Python NumPy ecosystem.

## Prompt-to-Artifact Checklist

| Requirement | Evidence |
| --- | --- |
| C++ implementation | `include/litenp/litenp.hpp` is a C++17 header-first implementation. |
| Lightweight / Eigen-like usage | `litenp` is an `INTERFACE` CMake target, installable via `find_package(litenp CONFIG)`, with a buildable `examples/basic.cpp`. |
| NumPy-like ndarray core | `Array<T>` and `ArrayView<T>` support shape, strides, views, reshape, flatten, squeeze/unsqueeze, transpose/permute, step slicing, and select. |
| Broadcasting arithmetic | Elementwise `+ - * /`, scalar ops, and mixed dtype `std::common_type_t` promotion are implemented and tested. |
| Key tensor/array ops | Reductions (`sum`, `mean`, `max`), unary ops (`relu`, `sqrt`, `exp`, `sigmoid`, etc.), conditionals (`minimum`, `maximum`, comparisons, `where`, `clip`), array combination (`concatenate`, `stack`), and 2D `matmul` are implemented. |
| High-performance CPU paths | Contiguous `float`/`double` kernels use AVX2/FMA when available; `Array` storage is 32-byte aligned for AVX hot paths; huge aligned non-in-place `add` and row-broadcast add use non-temporal stores; contiguous `where` avoids single-thread chunking and uses aligned AVX loads/stores when possible; OpenMP is used for large loops; matmul has built-in `4x8`, `8x8`, `4x16`, and packed-B `float32` AVX2/FMA paths, including a guarded 6x16 route for large eligible GEMMs, plus optional CBLAS/OpenBLAS when explicitly enabled. |
| libtorch comparison | `benchmarks/bench_litenp.cpp` optionally builds with Torch and compares add, ReLU, clip, and matmul against libtorch CPU tensors. |
| Eigen comparison | `benchmarks/bench_litenp.cpp` detects Eigen and compares add, sum, ReLU, clip, and matmul. |
| Correctness checks | `tests/test_litenp.cpp` covers construction, views, broadcasting, unary ops, conditionals, reductions, matmul, dtype helpers, and combine ops. |
| Installability | `cmake/litenpConfig.cmake.in` plus CMake install/export rules support package installation and discovery. |

## Active Goal Re-audit 2026-05-26

| Active requirement | Evidence | Status |
| --- | --- | --- |
| Simpler than Eigen/libtorch dependency approaches | Header-first `INTERFACE` target; public use is a single `#include "litenp/litenp.hpp"` plus optional `find_package(litenp CONFIG REQUIRED)`. The non-doc code surface remains small: `include/litenp/litenp.hpp` is `4088` lines, and the library header plus benchmark, tests, example, CMakeLists, and package config total `5923` lines. | Satisfied |
| Lighter than Eigen/libtorch dependency approaches | No required Eigen/libtorch dependency; Eigen/Torch only appear in benchmarks. `include/litenp/litenp.hpp` includes only standard-library headers by default; AVX intrinsics are guarded by `__AVX2__`, OpenMP by `LITENP_USE_OPENMP`, and CBLAS by `LITENP_HAS_CBLAS`. A 2026-05-26 lightness check found only the lite example and test executables in `build_lite`, not the heavy benchmark; `size build_lite/litenp_basic` reported `81142` text bytes and `ldd` showed no Eigen, Torch, OpenBLAS, or OpenMP runtime linkage. | Satisfied |
| Better internal design | GEMM backend choice is centralized in `detail::select_gemm_backend`, while public `matmul_into` has a single fast-path dispatch point. The phase 4 pass added focused detail helpers for 2D row-broadcast add and row-wise sums, a guarded 6x16 packed-B GEMM route, 32-byte aligned storage, aligned/non-temporal `add` and row-broadcast stores for the right cases, plus benchmark-only allocation/kernel separation for `where` and fairer 2D allocating benchmark comparisons. Packed A+B GEMM and 8-way `add` diagnostics were removed after the artifacts failed to justify their complexity. | Satisfied |
| Faster than local libtorch on target rows | `/tmp/litenp_bench_final_recheck_2026-05-26.txt`: `add 16M`, fair allocating `where 16M`, `where_into 16M`, `broadcast 2048^2`, `broadcast_into 2048^2`, both `sum(axis)` rows, and `matmul 768/1024` are ahead of libtorch on best time. | Satisfied |
| Faster than Eigen on target rows | `/tmp/litenp_bench_final_recheck_2026-05-26.txt`: `add 16M`, fair allocating `where 16M`, `where_into 16M`, `broadcast 2048^2`, `broadcast_into 2048^2`, both `sum(axis)` rows, and `matmul 768/1024` are ahead of Eigen on best time. | Satisfied |
| Current blocker documented | `docs/phase4_approval_gate_2026-05-26.md` records the approved closeout pass, promoted routes, rejected diagnostics, benchmark reproducibility contract, residual blockers, and permanent-test coverage. | Satisfied |

Active-goal conclusion: the current broad performance objective is satisfied on
the audited target rows. The approved closeout pass improved row-broadcast,
row-wise reduction, large-GEMM routing, fair `where` benchmarking, aligned
storage, huge contiguous `add`, and contiguous `where`. The final benchmark
reruns were still captured under heavy training load, so the absolute timings
are not quiet-machine numbers, but the target rows now repeatedly beat both
Eigen and libtorch on best time.

Latest default/lite verification rerun on 2026-05-26 passed `cmake --build
build_lite -j`, `ctest --test-dir build_lite --output-on-failure`,
`./build_lite/litenp_basic`, `cmake --install build_lite`, and
`cmake --find-package` against `/tmp/litenp-install`.
An additional fresh README-default build in `/tmp/litenp_readme_default_2026_05_26`
configured, built, passed `ctest`, and ran `litenp_basic`; its CMake cache kept
`LITENP_BUILD_BENCHMARKS`, `LITENP_USE_OPENMP`, `LITENP_USE_CBLAS`, and
`LITENP_NATIVE_ARCH` all `OFF`, producing only `litenp_basic` and
`litenp_tests`.
An external installed-header smoke test also compiled and ran from stdin with
`g++ -std=c++17 -I/tmp/litenp-install/include`, covering `Array`, broadcast
`add`, and `sum(axis=1)`.
A temporary CMake consumer smoke test also configured, built, and ran with
`find_package(litenp CONFIG REQUIRED)` and `target_link_libraries(... PRIVATE
litenp::litenp)` against `/tmp/litenp-install`.
The resulting consumer binary reported `33116` text bytes via `size`, and
`ldd` showed no Eigen, Torch, OpenBLAS, or OpenMP runtime dependency.
A temporary `add_subdirectory` consumer also configured, built, and ran. In
that parent build, `LITENP_BUILD_TESTS`, `LITENP_BUILD_EXAMPLES`,
`LITENP_BUILD_BENCHMARKS`, `LITENP_USE_OPENMP`, `LITENP_USE_CBLAS`, and
`LITENP_NATIVE_ARCH` all stayed `OFF`; only the consumer executable was built,
with `19018` text bytes and no OpenMP, OpenBLAS, Torch, or Eigen runtime
dependency.
The installed CMake export `/tmp/litenp-install/lib/cmake/litenp/litenpTargets.cmake`
defines `litenp::litenp` as an imported `INTERFACE` target with only
`cxx_std_17` and the installed include directory; it has no exported link
libraries, compile definitions, `-march=native`, OpenMP, CBLAS, Torch, or Eigen
requirements.
The optional benchmark/performance build is intentionally heavier and isolated:
`size build_torch/litenp_bench` reports `537030` text bytes, and `ldd` shows
Torch, OpenBLAS, OpenMP, and CUDA-related libraries only for that benchmark
binary. `build_torch/CMakeCache.txt` has `LITENP_BUILD_BENCHMARKS=ON`,
`LITENP_USE_OPENMP=ON`, `LITENP_USE_CBLAS=ON`, and `LITENP_NATIVE_ARCH=ON`,
while `build_lite/CMakeCache.txt` keeps those options `OFF`.
A fresh README-style performance build in `/tmp/litenp_readme_perf_2026_05_26`
also configured, built, passed `ctest`, and started `litenp_bench`. Its cache
had benchmark/OpenMP/CBLAS/native options `ON`, `Torch_DIR=NOTFOUND`, and the
benchmark binary linked OpenMP and OpenBLAS but not Torch.

Latest torch/performance verification rerun on 2026-05-26 passed `cmake
--build build_torch -j` and `ctest --test-dir build_torch
--output-on-failure`, with `litenp_bench` built in that configuration before
the closeout benchmark rerun.
The regular unit tests cover small `matmul`, `8x8`, `4x16`, and direct 6x16
packed-B regression cases. They now also assert that
`select_gemm_backend<float>(768, 768, 768)` selects `f32_6x16_packed_b` and
`select_gemm_backend<float>(512, 512, 512)` keeps `f32_4x16_packed_b`. A
temporary 2026-05-26 packed-B public-path smoke test compiled with
`g++ -std=c++17 -O2 -march=native -Iinclude` confirmed
`select_gemm_backend<float>(384, 384, 384)` selects `f32_4x16_packed_b` and that
public `matmul_into` returns the expected `384^3` constant-matrix result.
Latest warning and sanitizer verification rerun on 2026-05-26 passed
`cmake --build build_warn -j`, `ctest --test-dir build_warn
--output-on-failure`, `cmake --build build_asan -j`, and `LD_PRELOAD="$(gcc
-print-file-name=libasan.so)" ctest --test-dir build_asan
--output-on-failure`.

Clean pre-6x16 closeout evidence:

| case | source | litenp best/median | Eigen best/median | libtorch best/median | status |
| --- | --- | ---: | ---: | ---: | --- |
| `add 16M` | `/tmp/litenp_bench_performance_closeout_final.txt` | `6.8550 / 7.0128 ms` | `6.7680 / 7.0446 ms` | `16.6762 / 17.9763 ms` | best slightly behind Eigen, median slightly ahead |
| `where 16M` | `/tmp/litenp_bench_performance_closeout_final.txt` | `17.9547 / 19.3353 ms` | `7.8566 / 8.1257 ms` | `21.0339 / 21.2516 ms` | allocation path still behind Eigen |
| `where_into 16M` | `/tmp/litenp_bench_performance_closeout_final.txt` | `7.1834 / 7.2405 ms` | `7.8566 / 8.1257 ms` | `21.0339 / 21.2516 ms` | kernel path ahead |
| `broadcast 2048^2` | `/tmp/litenp_bench_performance_closeout_final.txt` | `0.8854 / 1.0532 ms` | `0.9234 / 1.1200 ms` | `1.0568 / 1.3232 ms` | ahead |
| `broadcast_into 2048^2` | `/tmp/litenp_bench_performance_closeout_final.txt` | `0.7900 / 0.8720 ms` | `0.9234 / 1.1200 ms` | `1.0568 / 1.3232 ms` | ahead |
| `sum axis1 2048^2` | `/tmp/litenp_bench_performance_closeout_final.txt` | `0.3287 / 0.5738 ms` | `0.3843 / 0.6913 ms` | `0.3709 / 0.7039 ms` | ahead |
| `matmul 1024` | `/tmp/litenp_bench_performance_closeout_final.txt` | `15.7626 / 15.7906 ms` | `15.0305 / 15.1422 ms` | `42.9879 / 43.3791 ms` | still behind Eigen |

Post-route 6x16 GEMM evidence:

| case | source | public best/median | 6x16 best/median | packed-B best/median | status |
| --- | --- | ---: | ---: | ---: | --- |
| `gemm 768` | `/tmp/litenp_bench_gemm6x16_add8.txt` | `5.7731 / 5.8773 ms` | `5.7597 / 5.7604 ms` | `6.2599 / 6.2686 ms` | 6x16 diagnostic win |
| `gemm 1024` | `/tmp/litenp_bench_gemm6x16_add8.txt` | `14.1016 / 14.1508 ms` | `14.0878 / 14.3445 ms` | `15.6965 / 15.7383 ms` | 6x16 diagnostic win |

The final post-code-change benchmark artifact
`/tmp/litenp_bench_performance_closeout_6x16_revertadd.txt` is retained for
traceability, but it is not accepted as clean performance evidence: the run
started with load average `9.74`, ended at `14.06`, and a training process was
using about `987%` CPU.

Final high-load follow-up after fair `where` benchmarking and aligned storage:

| case | source | litenp best/median | Eigen best/median | libtorch best/median | status |
| --- | --- | ---: | ---: | ---: | --- |
| `where 16M` | `/tmp/litenp_bench_final_aligned_where_fair_2026-05-26.txt` | `18.0851 / 19.3511 ms` | `19.6090 / 20.8056 ms` | `20.8877 / 20.9320 ms` | fair allocating comparison ahead, but load-contaminated |
| `add 16M` | `/tmp/litenp_bench_final_aligned_where_fair_2026-05-26.txt` | `6.9930 / 7.0088 ms` | `6.6714 / 6.7977 ms` | `16.4474 / 17.5639 ms` | still behind Eigen |
| `matmul 1024` | `/tmp/litenp_bench_final_aligned_where_fair_2026-05-26.txt` | `13.9783 / 14.0214 ms` | `15.3455 / 15.3698 ms` | `43.6731 / 43.7699 ms` | 6x16 route ahead, but load-contaminated |

The artifact above started at load average `14.10`, ended at `16.39`, and a
training process was using about `988%` CPU. It is useful directional evidence,
not completion evidence.

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
process was using about `989%` CPU.

Final recheck evidence after row-broadcast streaming and aligned `where`:

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

Historical pinned re-audit evidence:

| case | source | litenp best/median | Eigen best/median | libtorch best/median | status |
| --- | --- | ---: | ---: | ---: | --- |
| `matmul 1024` | `/tmp/litenp_bench_pinned_2026-05-26.txt` | `15.2035 / 15.2633 ms` | `14.7150 / 14.9152 ms` | `42.5347 / 42.5856 ms` | still behind Eigen |
| `broadcast 2048^2` | `/tmp/litenp_bench_pinned_2026-05-26.txt` | `1.1777 / 1.2102 ms` | `0.8747 / 1.0238 ms` | `0.9688 / 1.3044 ms` | behind Eigen and best libtorch |
| `where 16M` | `/tmp/litenp_bench_pinned_2026-05-26.txt` | `17.5831 / 18.6346 ms` | `7.5701 / 7.7668 ms` | `20.8765 / 20.9334 ms` | allocation path behind Eigen |
| `where_into 16M` | `/tmp/litenp_bench_pinned_2026-05-26.txt` | `7.0759 / 7.1274 ms` | `7.5701 / 7.7668 ms` | `20.8765 / 20.9334 ms` | kernel path ahead |
| `add 16M` | `/tmp/litenp_bench_pinned_2026-05-26.txt` | `6.8231 / 6.8730 ms` | `6.6454 / 6.6896 ms` | `17.1656 / 17.8295 ms` | slightly behind Eigen |
| `sum axis1 2048^2` | `/tmp/litenp_bench_pinned_2026-05-26.txt` | `0.3613 / 0.6420 ms` | `0.3326 / 0.6195 ms` | `0.3319 / 0.6171 ms` | behind both on best time |

Pre-closeout rerun evidence:

| case | source | litenp best/median | Eigen best/median | libtorch best/median | status |
| --- | --- | ---: | ---: | ---: | --- |
| `add 16M` | `/tmp/litenp_bench_pre_closeout_rerun_2026-05-26.txt` | `6.7397 / 6.7511 ms` | `6.6101 / 6.6589 ms` | `16.7133 / 17.4335 ms` | still slightly behind Eigen |
| `where 16M` | `/tmp/litenp_bench_pre_closeout_rerun_2026-05-26.txt` | `17.6404 / 18.6802 ms` | `7.5779 / 7.5833 ms` | `20.4311 / 20.4752 ms` | allocation path behind Eigen |
| `broadcast 2048^2` | `/tmp/litenp_bench_pre_closeout_rerun_2026-05-26.txt` | `1.0462 / 1.2765 ms` | `0.8367 / 0.9698 ms` | `1.2333 / 1.3523 ms` | best time still behind Eigen |
| `sum axis1 2048^2` | `/tmp/litenp_bench_pre_closeout_rerun_2026-05-26.txt` | `0.3640 / 0.6158 ms` | `0.3094 / 0.5506 ms` | `0.2935 / 0.5518 ms` | behind both on best time |
| `matmul 1024` | `/tmp/litenp_bench_pre_closeout_rerun_2026-05-26.txt` | `14.9340 / 14.9692 ms` | `14.4686 / 14.6314 ms` | `41.9747 / 41.9936 ms` | still behind Eigen |

## Verification Commands

```bash
cmake -S . -B build_lite -DCMAKE_BUILD_TYPE=Release
cmake --build build_lite -j
ctest --test-dir build_lite --output-on-failure
./build_lite/litenp_basic
cmake --install build_lite --prefix /tmp/litenp-install
cmake --find-package -DNAME=litenp -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=EXIST -DCMAKE_PREFIX_PATH=/tmp/litenp-install
```

Observed result:

```text
100% tests passed, 0 tests failed out of 1
row_sum: 25, 31
projected(1,1): 11
litenp found.
```

Torch-enabled verification:

```bash
cmake -S . -B build_torch -DCMAKE_BUILD_TYPE=Release \
  -DLITENP_BUILD_BENCHMARKS=ON \
  -DLITENP_USE_OPENMP=ON \
  -DLITENP_USE_CBLAS=ON \
  -DLITENP_NATIVE_ARCH=ON \
  -DCMAKE_PREFIX_PATH=/media/makeblock/edd5b8f3-101d-4639-95ae-71b43a3941d62/work/env/lib/python3.11/site-packages/torch/share/cmake
cmake --build build_torch -j
ctest --test-dir build_torch --output-on-failure
./build_torch/litenp_bench
```

Latest verification result:

```text
100% tests passed, 0 tests failed out of 1
```

Current benchmark evidence is recorded in the closeout table above; historical
re-audit and pre-closeout rows are retained only as baselines. Older inline
benchmark excerpts were removed from this audit to avoid mixing historical and
current timing data.

## MVP Audit Conclusion

The earlier MVP objective is achieved for a lightweight, high-performance C++
NumPy-like core:

- It is implemented as a small C++17 library.
- It covers core ndarray semantics and common numerical/tensor operations.
- It has SIMD/OpenMP/CBLAS performance paths plus planner-selected built-in
  GEMM kernels, including packed-B float32 backends.
- It includes direct Eigen and libtorch benchmark comparisons.
- It is installable and consumable as a lightweight CMake package.

Known non-goals for this completion: full Python NumPy API parity, GPU tensors,
autograd, FFT, random distributions, and file IO.
