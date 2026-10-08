# litenp MVP Design

## Goal

Build a lightweight C++ ndarray library that covers the core NumPy-style data
model while keeping dependency weight close to Eigen. The first milestone must
be small enough to compile quickly, test locally, and benchmark against Eigen
and libtorch when those libraries are available.

## Scope

The MVP contains:

- `Array<T>` as the owning row-major container.
- `ArrayView<T>` as a non-owning shape/stride view.
- Common construction helpers such as `zeros`, `ones`, `full`, `linspace`,
  and `eye`.
- Shape utilities, contiguous strides, reshape, and slicing.
- **Signed strides** (`Strides = std::vector<std::ptrdiff_t>`) so reverse
  slices and negative-step traversal use well-defined C++ pointer
  arithmetic instead of an unsigned two's-complement hack.
- Step slicing (positive and negative; `litenp::SLICE_OPEN` sentinel for
  omitted `begin`/`end`, mirroring Python slice semantics) and `select`
  views without copies.
- Transpose and permutation views.
- Shape helpers (`flatten`, `squeeze`, `unsqueeze`) and `astype`.
- Elementwise arithmetic with broadcasting via a unified `BroadcastPlan`
  (output shape and per-operand strides precomputed once and reused by
  binary ops, comparisons, and `where`). `broadcast_to` rejects
  lower-rank targets.
- Mixed dtype arithmetic via a NumPy-compatible `promote_type` (kind+bits
  rule; e.g. `int32 + float -> double`, `int64 + uint64 -> double`,
  `int32 + uint32 -> int64`), replacing `std::common_type_t`.
- `true_divide` for integral operands promotes to `double` (NumPy
  semantics); `divide` keeps C++ integer semantics for same-typed integral
  arrays.
- Common unary kernels for inference-style workloads.
- Mask-style conditionals: comparisons, `where`, and `clip`.
- Array combination with `concatenate` and `stack`.
- Allocation-free `*_into` variants for hot paths.
- Reductions over all elements and by axis.
- 2D matrix multiplication.
- SIMD fast paths for contiguous `float` and `double` arithmetic.
- Optional OpenMP parallel loops for large kernels.
- A small benchmark harness.

The MVP explicitly excludes autograd, GPU, expression templates, BLAS bindings,
advanced NumPy indexing, and full dtype coverage. Those should be added only
after the data model and benchmark baseline are stable.

## API Direction

The API favors simple function calls and operators:

```cpp
litenp::Array<float> a({2, 3});
litenp::Array<float> b({3});
auto c = a + b;
auto r = litenp::sum(c, 1);
auto y = litenp::matmul(x, w);
```

This keeps usage compact like Eigen while preserving ndarray semantics such as
broadcasting and views.

## Performance Direction

The first fast path is contiguous CPU memory:

- use row-major contiguous storage by default
- detect contiguous same-shape elementwise operations
- use AVX2/FMA intrinsics when compiling for capable x86 CPUs
- use OpenMP for coarse-grained loops where available
- use CBLAS/OpenBLAS as an optional large-GEMM backend when available, while
  keeping built-in AVX2/FMA fallback kernels for lightweight embedding and
  lower-overhead small/mid-sized matmul
- keep generic strided/broadcast loops correct first, then optimize targeted
  hot paths based on benchmark output
