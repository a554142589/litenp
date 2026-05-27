# Full Benchmark 2026-05-25

Command:

```bash
cmake -S . -B build_torch -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/media/makeblock/edd5b8f3-101d-4639-95ae-71b43a3941d62/work/env/lib/python3.11/site-packages/torch/share/cmake
cmake --build build_torch -j
./build_torch/litenp_bench
ctest --test-dir build_torch --output-on-failure
```

Machine baseline: x86_64 i7-13700K, Release build, `-march=native`, OpenMP enabled, OpenBLAS/CBLAS available, PyTorch/libtorch 2.6.0 CPU comparison enabled.

Times are best-of repeated runs, in milliseconds.

## Construction

| case | best ms |
| --- | ---: |
| zeros 4M f32 | 1.2917 |
| ones 4M f32 | 1.1077 |
| full 4M f32 | 1.1134 |
| arange 4M f32 | 2.1597 |
| linspace 4M f32 | 3.5399 |
| zeros_like 4M f32 | 1.1168 |
| eye 1024x1024 f32 | 0.2055 |
| identity 1024 f32 | 0.2007 |

## Views Metadata

These are metadata-only operations repeated 1,000,000 times.

| case | total best ms | approx ns/op |
| --- | ---: | ---: |
| reshape | 46.6059 | 46.6 |
| flatten | 26.6082 | 26.6 |
| transpose | 44.1310 | 44.1 |
| permute {1,0} | 40.7098 | 40.7 |
| slice step | 19.7449 | 19.7 |
| select | 23.5194 | 23.5 |
| unsqueeze/squeeze | 72.7966 | 72.8 |

## Copy / Cast

| case | best ms |
| --- | ---: |
| as_contiguous no-op 4M | 1.3733 |
| as_contiguous transpose 4M | 26.3820 |
| astype f32->f64 4M | 20.5837 |
| astype f32->i32 4M | 13.5785 |

## Binary / Broadcasting

| case | best ms |
| --- | ---: |
| add_into 4M f32 | 0.3895 |
| subtract_into 4M f32 | 0.3149 |
| multiply_into 4M f32 | 0.3165 |
| divide_into 4M f32 | 0.3101 |
| minimum_into 4M f32 | 0.3053 |
| maximum_into 4M f32 | 0.3057 |
| scalar add_into 4M f32 | 0.0996 |
| operator+ alloc 4M f32 | 1.4335 |
| broadcast add 2048x2048 | 29.9133 |
| mixed i32+f32 4M | 20.5556 |
| Eigen add 4M f32 | 1.3074 |
| libtorch add 4M f32 | 0.8749 |

## Unary

| case | best ms |
| --- | ---: |
| negative_into 4M f32 | 0.1617 |
| abs_into 4M f32 | 0.1615 |
| relu_into 4M f32 | 0.1612 |
| sqrt_into 4M f32 | 0.1901 |
| exp_into 4M f32 | 1.7368 |
| sigmoid_into 4M f32 | 1.7917 |
| Eigen relu 4M f32 | 1.1459 |
| libtorch relu 4M f32 | 0.3114 |

## Condition / Mask

| case | best ms |
| --- | ---: |
| greater mask 4M f32 | 21.9606 |
| less_equal mask 4M f32 | 21.7977 |
| where 4M f32 | 25.0982 |
| clip_into 4M f32 | 0.1661 |
| Eigen clip 4M f32 | 1.0080 |
| libtorch clip 4M f32 | 0.4623 |

## Reductions

| case | best ms |
| --- | ---: |
| sum all 2048x2048 | 0.1940 |
| mean all 2048x2048 | 0.1943 |
| max all 2048x2048 | 19.4442 |
| sum axis0 2048x2048 | 9.9940 |
| sum axis1 2048x2048 | 3.2286 |
| mean axis1 2048x2048 | 3.2374 |
| max axis1 2048x2048 | 3.2481 |
| Eigen sum 4M f32 | 0.2606 |

## Combine

| case | best ms |
| --- | ---: |
| concatenate axis0 2x1M | 21.8349 |
| concatenate axis1 2x1M | 21.2410 |
| stack axis0 2x1M | 15.1768 |
| stack axis2 2x1M | 16.3090 |

## Matmul

| case | best ms |
| --- | ---: |
| matmul_into 384^3 f32 | 0.2804 |
| matmul alloc 384^3 f32 | 0.2892 |
| Eigen matmul 384^3 | 0.1571 |
| libtorch matmul 384^3 | 0.4209 |

## Notes

- `*_into` paths avoid allocation and are the intended hot paths.
- View operations are metadata-only; reported totals are for 1M calls.
- Broadcast add, mixed dtype, mask compare, `where`, axis0 reduce, and combine ops are still generic strided loops and are the clearest next optimization targets.
- Correctness test after this benchmark: `100% tests passed, 0 tests failed out of 1`.
