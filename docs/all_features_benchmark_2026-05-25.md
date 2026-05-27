# lite_np all-feature benchmark

Date: 2026-05-25

Command:

```bash
ctest --test-dir build_torch --output-on-failure
./build_torch/litenp_bench
```

Result: tests passed, benchmark completed.

Notes:

- Timings are best-of runs printed by `benchmarks/bench_litenp.cpp`.
- The benchmark covers all current public feature families: construction, metadata/accessors, views including `transpose` and `permute`, copy/cast, unary, binary, broadcasting, mixed dtype, masks, `where`, `clip`, reductions, combine ops, `fill_view`, and matmul.
- Eigen and libtorch rows are included when the build enables those dependencies.
- The non-contiguous matmul row intentionally exercises the generic strided view path.

## Construction

| case | best ms | checksum |
|---|---:|---:|
| default Array x1M | 1.7140 | 1e+06 |
| Array(shape) 4M f32 | 0.7298 | 0 |
| zeros 4M f32 | 0.8903 | 0 |
| ones 4M f32 | 0.9739 | 2 |
| full 4M f32 | 0.9763 | 6 |
| from_vector 1M f32 | 0.1102 | 4 |
| arange 4M f32 | 2.1079 | 4.1943e+06 |
| arange start/step 4M f32 | 2.1264 | 2.09715e+06 |
| linspace 4M f32 | 2.4065 | 1 |
| zeros_like 4M f32 | 1.1199 | 0 |
| ones_like 4M f32 | 1.1592 | 2 |
| full_like 4M f32 | 0.9685 | 8 |
| zeros_like view 4M f32 | 1.1845 | 0 |
| ones_like view 4M f32 | 1.0599 | 2 |
| full_like view 4M f32 | 1.0996 | 10 |
| eye 1024x1024 f32 | 0.1842 | 2 |
| identity 1024 f32 | 0.1844 | 2 |

## Metadata And Accessors

| case | best ms | checksum |
|---|---:|---:|
| numel(shape) x1M | 7.7204 | 5.24288e+12 |
| Array shape/strides x1M | 0.3769 | 5.24288e+12 |
| Array ndim/size x1M | 0.3711 | 5.24288e+12 |
| Array empty/contig x1M | 0.7557 | 5e+06 |
| Array data/values x1M | 0.3769 | 5e+06 |
| Array view x1M | 18.6105 | 5.24288e+12 |
| Array at/operator() x1M | 57.8219 | 1.5e+07 |
| Array operator[] x1M | 0.3781 | 1.5e+07 |
| View shape/stride x1M | 0.3781 | 1.024e+10 |
| View ndim/size/contig x1M | 10.9150 | 5.2429e+12 |
| View offset/at x1M | 16.2152 | 1.026e+10 |
| View operator() x1M | 8.1043 | 1.0255e+10 |

## Views

| case | best ms | checksum |
|---|---:|---:|
| reshape x1M | 53.8641 | 5.12e+09 |
| flatten x1M | 33.5525 | 5.24288e+12 |
| transpose x1M | 44.1445 | 5e+06 |
| free transpose x1M | 63.4233 | 5e+06 |
| Array transpose x1M | 66.4106 | 5e+06 |
| free transpose Array x1M | 67.8514 | 5e+06 |
| permute {1,0} x1M | 42.7084 | 5.12e+09 |
| free permute {1,0} x1M | 63.3813 | 5.12e+09 |
| Array permute {1,0} x1M | 65.7809 | 5.12e+09 |
| free permute Array x1M | 65.3360 | 5.12e+09 |
| slice x1M | 24.1086 | 2.56e+09 |
| slice step x1M | 24.2243 | 2.56e+09 |
| select x1M | 28.6411 | 5.12e+09 |
| unsqueeze/squeeze x1M | 75.3735 | 5.12e+09 |
| squeeze all x1M | 143.2845 | 1.024e+10 |

## Copy And Cast

| case | best ms | checksum |
|---|---:|---:|
| as_contiguous no-op 4M | 1.3552 | 4.1943e+06 |
| as_contiguous transpose 4M | 26.0132 | 4.1943e+06 |
| astype f32->f64 4M | 20.5091 | 4.1943e+06 |
| astype view f32->f64 4M | 20.8564 | 4.1943e+06 |
| astype f32->i32 4M | 13.8161 | 4.1943e+06 |

## Binary And Broadcasting

| case | best ms | checksum |
|---|---:|---:|
| add_into 4M f32 | 0.2858 | 8 |
| binary_into generic add 4M | 0.2738 | 8 |
| subtract_into 4M f32 | 0.2828 | -3 |
| multiply_into 4M f32 | 0.2924 | 6.875 |
| divide_into 4M f32 | 0.2853 | 4.4 |
| minimum_into 4M f32 | 0.2725 | 2.5 |
| maximum_into 4M f32 | 0.2772 | 5.5 |
| scalar add_into 4M f32 | 0.1220 | 6.5 |
| binary_scalar_into add 4M | 0.1196 | 6.5 |
| scalar subtract_into 4M f32 | 0.1198 | -1.5 |
| scalar multiply_into 4M f32 | 0.1199 | 5 |
| scalar divide_into 4M f32 | 0.1576 | 1.25 |
| scalar minimum_into 4M f32 | 0.1218 | 2.5 |
| scalar maximum_into 4M f32 | 0.1197 | 4 |
| operator+ alloc 4M f32 | 1.3715 | 8 |
| operator- alloc 4M f32 | 1.3449 | -3 |
| operator* alloc 4M f32 | 1.3438 | 6.875 |
| operator/ alloc 4M f32 | 1.3304 | 4.4 |
| binary alloc generic add 4M | 1.3281 | 8 |
| add alloc 4M f32 | 1.3613 | 8 |
| subtract alloc 4M f32 | 1.3320 | -3 |
| multiply alloc 4M f32 | 1.3929 | 6.875 |
| divide alloc 4M f32 | 1.3456 | 4.4 |
| minimum alloc 4M f32 | 1.3430 | 2.5 |
| maximum alloc 4M f32 | 1.3367 | 5.5 |
| scalar operator+ alloc 4M | 0.7468 | 6.5 |
| scalar operator- alloc 4M | 0.6828 | -1.5 |
| scalar operator* alloc 4M | 0.6846 | 5 |
| scalar operator/ alloc 4M | 0.6736 | 1.25 |
| binary_scalar alloc add 4M | 0.6678 | 6.5 |
| broadcast add 2048x2048 | 34.8333 | 3 |
| mixed op+ i32+f32 4M | 23.1378 | 6.5 |
| mixed op- i32-f32 4M | 22.8692 | 1.5 |
| mixed op* i32*f32 4M | 23.1571 | 5 |
| mixed op/ i32/f32 4M | 23.0024 | 3.2 |
| mixed add fn i32+f32 4M | 23.0010 | 6.5 |
| mixed subtract fn 4M | 23.0514 | 1.5 |
| mixed multiply fn 4M | 22.6489 | 5 |
| mixed divide fn 4M | 22.9593 | 3.2 |
| Eigen add 4M f32 | 1.3338 | 8 |
| libtorch add 4M f32 | 1.0719 | 8 |

## Unary

| case | best ms | checksum |
|---|---:|---:|
| negative_into 4M f32 | 0.1614 | 0.5 |
| unary_into generic neg 4M | 0.1607 | 0.5 |
| abs_into 4M f32 | 0.1608 | 1.5 |
| relu_into 4M f32 | 0.1610 | 0.5 |
| sqrt_into 4M f32 | 0.1902 | 1.70711 |
| exp_into 4M f32 | 1.9057 | 2.0166 |
| sigmoid_into 4M f32 | 1.8404 | 0.891401 |
| negative alloc 4M f32 | 1.2616 | 0.5 |
| unary alloc generic neg 4M | 1.3141 | 0.5 |
| negative view alloc 4M | 1.3150 | 0.5 |
| abs alloc 4M f32 | 1.2954 | 1.5 |
| relu alloc 4M f32 | 1.2891 | 0.5 |
| sqrt alloc 4M f32 | 1.3317 | 1.70711 |
| exp alloc 4M f32 | 2.7238 | 2.0166 |
| sigmoid alloc 4M f32 | 2.7098 | 0.891401 |
| Eigen relu 4M f32 | 0.9788 | 0.5 |
| libtorch relu 4M f32 | 0.5311 | 0.5 |

## Condition And Mask

| case | best ms | checksum |
|---|---:|---:|
| compare generic greater 4M | 20.5316 | 1 |
| greater mask 4M f32 | 20.4801 | 1 |
| greater Array overload 4M | 20.4652 | 1 |
| greater_equal mask 4M | 21.8683 | 1 |
| less mask 4M f32 | 21.8181 | 1 |
| less_equal mask 4M f32 | 22.1852 | 1 |
| equal mask 4M f32 | 22.4421 | 0 |
| not_equal mask 4M f32 | 21.9531 | 2 |
| where 4M f32 | 23.3976 | -2 |
| clip_into 4M f32 | 0.1644 | 1 |
| clip alloc 4M f32 | 0.8642 | 1 |
| clip view alloc 4M f32 | 1.2372 | 1 |
| Eigen clip 4M f32 | 1.1055 | 1 |
| libtorch clip 4M f32 | 0.4511 | 1 |

## Reductions

| case | best ms | checksum |
|---|---:|---:|
| sum all 2048x2048 | 0.1945 | 4.1943e+06 |
| sum view all 2048x2048 | 0.1940 | 4.1943e+06 |
| mean all 2048x2048 | 0.1940 | 1 |
| mean view all 2048x2048 | 0.1942 | 1 |
| max all 2048x2048 | 18.9404 | 1 |
| max view all 2048x2048 | 18.9037 | 1 |
| sum axis0 2048x2048 | 10.5077 | 4096 |
| sum axis1 2048x2048 | 3.2312 | 4096 |
| mean axis0 2048x2048 | 10.1084 | 2 |
| mean axis1 2048x2048 | 3.2566 | 2 |
| max axis0 2048x2048 | 10.0865 | 2 |
| max axis1 2048x2048 | 3.2029 | 2 |
| sum view axis0 2048x2048 | 10.2029 | 4096 |
| sum view axis1 2048x2048 | 3.2429 | 4096 |
| mean view axis0 2048x2048 | 9.9545 | 2 |
| mean view axis1 2048x2048 | 3.2673 | 2 |
| max view axis0 2048x2048 | 10.3377 | 2 |
| max view axis1 2048x2048 | 3.2041 | 2 |
| Eigen sum 4M f32 | 0.2492 | 4.1943e+06 |

## Combine

| case | best ms | checksum |
|---|---:|---:|
| concatenate axis0 2x1M | 21.7443 | 3 |
| concatenate axis1 2x1M | 21.3356 | 3 |
| stack axis0 2x1M | 12.5613 | 3 |
| stack axis2 2x1M | 12.5222 | 3 |
| fill_view 64x64 | 0.0001 | 10 |

## Matmul

| case | best ms | checksum |
|---|---:|---:|
| matmul_into 384^3 f32 | 0.2825 | 384 |
| matmul alloc 384^3 f32 | 0.2880 | 384 |
| matmul view alloc 384^3 | 0.2867 | 384 |
| matmul noncontig view 384^3 | 449.8490 | 384 |
| Eigen matmul 384^3 | 0.1599 | 384 |
| libtorch matmul 384^3 | 0.4336 | 384 |
