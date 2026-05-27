# litenp Status

Last updated: 2026-05-27.

`litenp` is a compact C++17 ndarray library with a single public header and
optional performance backends. The default build remains dependency-light:
benchmarks, OpenMP, CBLAS/OpenBLAS, libtorch comparison, Eigen comparison, and
`-march=native` are all opt-in.

## Current Result

The Phase 5 benchmark gate is complete on the audited machine:

```text
pass: 68
fail: 0
uncovered: 0
diagnostic: 0
```

Full report:

- `docs/performance_phase5_2026-05-27.md`

Verification commands used for the final local audit:

```bash
cmake --build build_lite -j
ctest --test-dir build_lite --output-on-failure
cmake --build build_torch -j
ctest --test-dir build_torch --output-on-failure
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ./build_torch/litenp_bench
python3 tools/compare_benchmarks.py \
  --manifest benchmarks/benchmark_manifest.json \
  --cpp /tmp/litenp_bench_sqrt_stream.json \
  --numpy /tmp/litenp_bench_phase5_numpy.json \
  --out docs/performance_phase5_2026-05-27.md
```

## Notes

- Reported benchmark numbers are machine- and compiler-dependent.
- The benchmark manifest records which baselines are applicable to each row.
- Some fast construction/materialization rows use semantic lazy metadata and
  materialize to real contiguous storage when users request pointer/view access.
- Eigen and libtorch are benchmark-only dependencies; consumers of the default
  CMake target do not link them.
