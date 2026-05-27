# litenp phase 3 packed GEMM pass 2026-05-26

## Scope

This pass implements the approved packed-B GEMM backend:

- add an internal `float32` AVX2/FMA packed-B path for contiguous row-major
  `matmul_into`
- pack one `k x 16` B panel into a small scratch buffer and reuse it across all
  4-row A tiles
- keep scratch local to the call, with one panel buffer per OpenMP worker
- route eligible `384^3` through `1024^3` work to packed-B before optional
  CBLAS, because local CBLAS/OpenBLAS is slower for these rows
- keep the public API and default dependency surface unchanged
- keep GEMM diagnostics for public, packed-B, direct `4x16`, direct `8x8`, and
  CBLAS paths

The final code also factors the packed path into small helpers for packing one
B panel and computing one packed `4x16` tile, avoiding duplicate OpenMP and
single-thread tile bodies. A follow-up micro-pass hoists A row pointers inside
the packed tile, marks private packed-GEMM pointers as non-aliasing for
GNU/Clang builds, and uses no-init scratch storage for the packed B panel.

## Verification

Commands run:

```bash
cmake --build build_torch -j
ctest --test-dir build_torch --output-on-failure
cmake --build build_lite -j
ctest --test-dir build_lite --output-on-failure
cmake --build build_warn -j
ctest --test-dir build_warn --output-on-failure
cmake --build build_asan -j
LD_PRELOAD=/usr/lib/gcc/x86_64-linux-gnu/11/libasan.so ctest --test-dir build_asan --output-on-failure
./build_lite/litenp_basic
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ./build_torch/litenp_bench > /tmp/litenp_bench_phase3c.txt
```

All listed test and example runs passed. Running ASan ctest without
`LD_PRELOAD` fails in this build layout because the ASan runtime is not first
in the loader list; the preloaded run above passed.

## Benchmark Evidence

Phase 2 source: `/tmp/litenp_bench_phase2b.txt`.

Phase 3 source: `/tmp/litenp_bench_phase3c.txt`.

Single-thread matrix rows:

| case | phase 2 litenp best/median | phase 3 litenp best/median | phase 3 Eigen best/median | phase 3 libtorch best/median | note |
| --- | ---: | ---: | ---: | ---: | --- |
| `matmul 384` | `0.7690 / 0.7720 ms` | `0.7665 / 0.7676 ms` | `0.7759 / 0.7793 ms` | `2.0900 / 2.2554 ms` | faster than Eigen and libtorch in this run |
| `matmul 512` | `2.0648 / 2.0895 ms` | `1.8112 / 1.8243 ms` | `1.8656 / 1.8873 ms` | `5.2600 / 5.2824 ms` | faster than Eigen and 2.9x faster than libtorch |
| `matmul 768` | `8.4570 / 8.5176 ms` | `6.2129 / 6.2424 ms` | `6.1757 / 6.2729 ms` | `17.7912 / 17.8018 ms` | essentially tied with Eigen, 2.9x faster than libtorch |
| `matmul 1024` | n/a | `14.8735 / 15.0889 ms` | `14.4588 / 14.6377 ms` | `42.0023 / 42.1388 ms` | close to Eigen, 2.8x faster than libtorch |

Diagnostic rows:

| case | public | packed-B | direct `4x16` | direct `8x8` | CBLAS | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `gemm 384` | `0.7617 / 0.7658 ms` | `0.7594 / 0.7640 ms` | `1.1493 / 1.1504 ms` | `1.2104 / 1.2135 ms` | `2.0443 / 2.2459 ms` | public path selects packed-B |
| `gemm 512` | `1.8242 / 1.8420 ms` | `1.8219 / 1.8261 ms` | `1.8398 / 1.8510 ms` | `2.2102 / 2.2142 ms` | `5.0443 / 5.2346 ms` | packed-B beats direct built-in kernels |
| `gemm 768` | `6.2205 / 6.3116 ms` | `6.1906 / 6.1920 ms` | `8.9710 / 9.2280 ms` | `8.8826 / 9.0710 ms` | `17.5945 / 17.6075 ms` | public packed path remains far ahead of prior kernels and CBLAS |
| `gemm 1024` | `14.8807 / 15.0729 ms` | `14.8620 / 14.9931 ms` | `32.7288 / 34.3277 ms` | `36.7372 / 36.8555 ms` | `41.8666 / 41.9499 ms` | packed-B avoids the old large-size CBLAS fallback |

Rejected probes:

- K-loop unrolling via a local accumulator lambda regressed the packed path.
- A `2x32` packed-B tile reduced A broadcasts but was substantially slower than
  the `4x16` packed path.
- 32-byte aligned packed scratch plus aligned loads was noise-prone and not a
  consistent public-path win, so the simpler no-init scratch remains.

## Remaining Work

- The broad objective should remain open: `matmul 768` is effectively tied and
  `matmul 1024` remains close to, but behind, Eigen.
- Allocating `where 16M` remains allocation-bound; `where_into 16M` is faster
  than Eigen in the final run (`6.9922 / 7.0299 ms` vs `7.6386 / 7.6660 ms`).
- Later re-audits expanded the closeout scope beyond GEMM. Pinned single-core
  evidence in `/tmp/litenp_bench_pinned_2026-05-26.txt` also shows gaps in
  `broadcast 2048^2`, `add 16M`, and `sum axis1 2048^2`.
- A next GEMM phase would likely need packed A+B blocking or a tuned macro-kernel
  to move from "near Eigen" to "consistently faster than Eigen" at 768/1024.
- Broader implementation remains blocked until explicit approval:

```text
ok performance closeout
```
