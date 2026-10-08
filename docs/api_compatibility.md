# API Compatibility Scope

`litenp` intentionally targets a compact subset of NumPy/libtorch CPU tensor
semantics. The goal is not full replacement; the goal is that every advertised
operator behaves like the matching NumPy/libtorch operation inside the supported
subset.

## Supported Semantics

| Area | `litenp` API | NumPy/libtorch equivalent | Notes |
| --- | --- | --- | --- |
| Owning array | `Array<T>` | `numpy.ndarray` / `torch.Tensor` CPU storage | Row-major contiguous owner. |
| View | `ArrayView<T>` | ndarray/tensor view | Non-owning, shape plus strides. |
| Shape metadata | `shape`, `strides`, `ndim`, `size` | `.shape`, `.strides`, `.ndim`, `.size` | Strides are in element counts, not bytes. |
| Reshape/flatten | `reshape`, `flatten` | `reshape`, `ravel`/`flatten` | Requires contiguous input. |
| Slice/select | `slice`, `select` | basic slicing / `select` | `step` may be positive or negative (reverse). Pass `litenp::SLICE_OPEN` (sentinel `PTRDIFF_MIN`) to omit `begin` or `end`, mirroring Python's omitted slice bounds. |
| Transpose/permute | `transpose`, `permute` | `.T`, `transpose`, `permute` | View operation unless materialized. |
| Squeeze/unsqueeze | `squeeze`, `unsqueeze` | `squeeze`, `expand_dims` / `unsqueeze` | Runtime shape checks. |
| Construction | `zeros`, `ones`, `full`, `arange`, `linspace`, `eye`, `identity` | matching constructors | Some constructors may be lazy. |
| Elementwise binary | `+`, `-`, `*`, `/`, `minimum`, `maximum` | ufuncs / tensor ops | NumPy-style trailing-axis broadcasting via unified `BroadcastPlan` (output shape and per-operand strides precomputed once). |
| Scalar binary | scalar overloads | scalar-array ops | Mixed arithmetic promotes with NumPy-compatible `promote_type` (kind+bits rule; e.g. `int32 + float -> double`, `int32 + float32 -> float64`). |
| Unary | `negative`, `abs`, `relu`, `sqrt`, `exp`, `sigmoid` | NumPy ufuncs / torch ops | `relu` and `sigmoid` mirror common tensor behavior. |
| Comparison | `less`, `less_equal`, `greater`, `greater_equal`, `equal`, `not_equal` | comparison ufuncs | Output is `Array<std::uint8_t>` mask, not `bool`. |
| Selection | `where` / `where_into` | `np.where` / `torch.where` | Mask uses nonzero as true. |
| Clip | `clip` / `clip_into` | `np.clip` / `torch.clamp` | Requires `low <= high`. |
| Reductions | `sum`, `mean`, `max` | reductions | Supports all-elements and one-axis reductions. |
| Matrix multiply | `matmul`, `matmul_into` | 2D `matmul` | 2D matrices only. |
| Combine | `concatenate`, `stack` | matching NumPy ops | Explicit axis; runtime shape checks. |
| Cast | `astype` | `astype` / `.to(dtype)` | Numeric casts follow C++ cast semantics. |

## Explicit Differences

- `ArrayView<T>` is non-owning. A view is valid only while its source array and
  storage remain alive.
- `Array<T>::operator[](std::size_t) const` returns by value so virtual arrays
  can serve reads without materializing.
- Comparison masks use `std::uint8_t` values `0` and `1`, not C++ `bool`.
- `zeros<T>` is restricted to arithmetic element types.
- Empty `sum` returns `0`; empty `mean` and `max` throw `std::invalid_argument`.
- Error handling is runtime-based and throws standard exceptions such as
  `std::invalid_argument` and `std::out_of_range`.
- Broadcasting follows NumPy trailing-axis compatibility, but there is no
  symbolic shape type or compile-time shape checking.
- `matmul` supports only rank-2 matrices. Batched matmul is out of scope.
- There is no GPU execution, autograd, random API, complex dtype, FFT, advanced
  indexing, masked array type, IO, or sparse storage.

## Compatibility Verification

The C++ unit tests cover edge cases and aliasing. In addition, CI runs two
complementary NumPy behavioral oracles, each compiling a small C++ program
against the installed header and comparing `litenp` results with NumPy:

- `tests/test_numpy_oracle.py` — representative fixed cases for every operator
  (construction, views, unary, binary, comparison, `where`/`clip`, axis
  reductions, `concatenate`/`stack`, `matmul`, `astype`, broadcasting,
  `cumsum`, NaN propagation).
- `tests/test_differential.py` — property-based random differential tests.
  Each run seeds a Python RNG, generates randomized inputs (varying shape,
  dtype, NaN injection, empty arrays, broadcasting, negative-step slicing),
  and verifies that every supported operator matches NumPy element-for-element
  or matches NumPy's raised exception (e.g. empty `max`/`mean`). Override the
  seed with `SEED=` and case count with `CASES=`; CI runs three fixed seeds.

Both oracles are intentionally behavioral. They do not use benchmark special
cases as proof of functional equivalence.

