#!/usr/bin/env python3
"""Property-based differential tests: litenp vs NumPy.

Generates randomized inputs (varying shape, dtype, NaN injection, empty
arrays, broadcasting, slicing) and checks that litenp produces the same
result as NumPy for every supported operator.

Run:
    python3 tests/test_differential.py
Env:
    CXX      C++ compiler (default: c++)
    SEED     RNG seed (default: 20261008)
    CASES    number of random cases per category (default: 64)
"""

from __future__ import annotations

import os
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

SEED = int(os.environ.get("SEED", "20261008"))
CASES = int(os.environ.get("CASES", "64"))


# ---------------------------------------------------------------------------
# dtype bridge: numpy <-> C++ <-> oracle tag
# ---------------------------------------------------------------------------
# tag: short name used in result identifiers
DTYPE_INFO = {
    "f32": (np.float32, "float", "std::int32_t"),  # (np_dtype, cpp_name, _)
    "f64": (np.float64, "double", "std::int32_t"),
    "i32": (np.int32, "std::int32_t", "std::int32_t"),
}


def np_dtype(tag: str) -> np.dtype:
    return np.dtype(DTYPE_INFO[tag][0])


def cpp_name(tag: str) -> str:
    return DTYPE_INFO[tag][1]


def is_float(tag: str) -> bool:
    return tag in ("f32", "f64")


# ---------------------------------------------------------------------------
# value formatting for C++ literals
# ---------------------------------------------------------------------------

def fmt_value(v, tag: str) -> str:
    """Format a single numeric value as a C++ literal of the right dtype."""
    if tag == "i32":
        return f"{int(v)}"
    # float types: handle NaN/Inf explicitly
    fv = float(v)
    if np.isnan(fv):
        return "NP_NAN"
    if np.isinf(fv):
        return "NP_INF" if fv > 0 else "NP_NEG_INF"
    # use repr to keep precision, suffix float with f
    if tag == "f32":
        return f"{fv!r}f"
    return f"{fv!r}"


def fmt_list(values, tag: str) -> str:
    return ", ".join(fmt_value(v, tag) for v in values)


# ---------------------------------------------------------------------------
# random input generators
# ---------------------------------------------------------------------------

def rand_shape(rng: random.Random, ndim: int | None = None,
               max_dim: int = 5, max_size: int = 24, allow_empty: bool = True):
    """Random shape; sometimes inject size-1 axes for broadcasting, sometimes empty."""
    if ndim is None:
        ndim = rng.randint(1, 3)
    shape = []
    for _ in range(ndim):
        # bias toward small dims; ~15% chance of size 1 (broadcast-friendly)
        r = rng.random()
        if r < 0.15:
            shape.append(1)
        elif allow_empty and r < 0.22:
            shape.append(0)
        else:
            shape.append(rng.randint(1, max_dim))
    total = 1
    for d in shape:
        total *= d
    # bound total size to keep compile/run cheap
    while total > max_size and len(shape) > 1:
        # shrink the largest non-1 axis
        idx = max(range(len(shape)), key=lambda i: shape[i] if shape[i] > 1 else -1)
        if shape[idx] <= 1:
            break
        shape[idx] = max(1, shape[idx] // 2)
        total = 1
        for d in shape:
            total *= d
    return tuple(shape)


def rand_array(rng: random.Random, tag: str, shape: tuple[int, ...],
              nan_prob: float = 0.0, int_range: int = 6):
    """Generate a random NumPy array of given dtype/shape; optionally inject NaN."""
    n = 1
    for d in shape:
        n *= d
    if tag == "i32":
        data = [rng.randint(-int_range, int_range) for _ in range(n)]
        return np.array(data, dtype=np.int32).reshape(shape)
    # float: pick from a small set including 0, negatives, fractions
    pool = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, -3.0]
    data = [rng.choice(pool) for _ in range(n)]
    if nan_prob > 0.0:
        for i in range(n):
            if rng.random() < nan_prob:
                data[i] = float("nan")
    return np.array(data, dtype=np_dtype(tag)).reshape(shape)


# ---------------------------------------------------------------------------
# case categories: each returns (cpp_emit, np_check) for one case
# ---------------------------------------------------------------------------

def gen_unary_cases(rng: random.Random, n: int):
    ops = ["negative", "abs", "relu", "sqrt", "exp", "sigmoid"]
    cases = []
    for i in range(n):
        tag = rng.choice(["f32", "f64"])
        shape = rand_shape(rng)
        arr = rand_array(rng, tag, shape, nan_prob=0.05)
        op = rng.choice(ops)
        cases.append(("unary", tag, op, arr))
    return cases


def gen_binary_cases(rng: random.Random, n: int):
    ops = ["add", "subtract", "multiply", "divide", "minimum", "maximum"]
    cases = []
    for i in range(n):
        tag = rng.choice(["f32", "f64"])
        shape = rand_shape(rng)
        a = rand_array(rng, tag, shape, nan_prob=0.05)
        b = rand_array(rng, tag, shape, nan_prob=0.05)
        # avoid div-by-zero issues in oracle: shift zeros to 1 for divide
        if rng.random() < 0.3:
            b = b.copy()
            b[b == 0] = 1
        op = rng.choice(ops)
        cases.append(("binary", tag, op, a, b))
    return cases


def gen_broadcast_cases(rng: random.Random, n: int):
    ops = ["add", "subtract", "multiply", "divide", "minimum", "maximum"]
    cases = []
    for i in range(n):
        tag = rng.choice(["f32", "f64"])
        shape = rand_shape(rng, ndim=2)
        a = rand_array(rng, tag, shape, nan_prob=0.05)
        # build a broadcastable b: same ndim, some axis = 1
        b_shape = list(shape)
        # set ~half the axes to size 1
        for j in range(len(b_shape)):
            if rng.random() < 0.4 and b_shape[j] != 0:
                b_shape[j] = 1
        b = rand_array(rng, tag, tuple(b_shape), nan_prob=0.05)
        if rng.random() < 0.3:
            b = b.copy()
            b[b == 0] = 1
        op = rng.choice(ops)
        cases.append(("broadcast", tag, op, a, b))
    return cases


def gen_comparison_cases(rng: random.Random, n: int):
    ops = ["greater", "less", "greater_equal", "less_equal", "equal", "not_equal"]
    cases = []
    for i in range(n):
        tag = rng.choice(["f32", "f64"])
        shape = rand_shape(rng)
        a = rand_array(rng, tag, shape, nan_prob=0.1)
        b = rand_array(rng, tag, shape, nan_prob=0.1)
        op = rng.choice(ops)
        cases.append(("compare", tag, op, a, b))
    return cases


def gen_reduction_cases(rng: random.Random, n: int):
    cases = []
    for i in range(n):
        tag = rng.choice(["f32", "f64"])
        shape = rand_shape(rng, allow_empty=True)
        arr = rand_array(rng, tag, shape, nan_prob=0.05)
        op = rng.choice(["sum", "mean", "max", "min"])
        # axis: None (all-reduce) or an axis
        if len(shape) == 0 or rng.random() < 0.3:
            axis = None
        else:
            axis = rng.randint(0, len(shape) - 1)
        cases.append(("reduce", tag, op, arr, axis))
    return cases


def gen_slice_cases(rng: random.Random, n: int):
    cases = []
    for i in range(n):
        tag = rng.choice(["f32", "f64"])
        shape = rand_shape(rng, ndim=2, allow_empty=False)
        arr = rand_array(rng, tag, shape, nan_prob=0.05)
        axis = rng.randint(0, len(shape) - 1)
        dim = shape[axis]
        # pick step in [-3, -1] U [1, 3] (step 0 invalid)
        step = rng.choice([1, 2, 3, -1, -2, -3])
        # pick begin/end compatible with step direction
        if step > 0:
            begin = rng.randint(0, dim)
            end = rng.randint(begin, dim)
            if rng.random() < 0.3:
                begin = "open"
            if rng.random() < 0.3:
                end = "open"
        else:
            begin = rng.randint(-dim, dim - 1)
            end = rng.randint(-dim, dim)
            # ensure end < begin for negative step
            if isinstance(begin, int) and isinstance(end, int) and end >= begin:
                end = begin - 1
            if rng.random() < 0.3:
                begin = "open"
            if rng.random() < 0.3:
                end = "open"
        cases.append(("slice", tag, arr, axis, begin, end, step))
    return cases


def gen_astype_cases(rng: random.Random, n: int):
    cases = []
    for i in range(n):
        tag = rng.choice(["f32", "f64", "i32"])
        shape = rand_shape(rng)
        arr = rand_array(rng, tag, shape, nan_prob=0.05 if is_float(tag) else 0.0)
        target = rng.choice(["f32", "f64", "i32"])
        cases.append(("astype", tag, target, arr))
    return cases


# ---------------------------------------------------------------------------
# C++ harness generation
# ---------------------------------------------------------------------------

CPP_HEADER = r"""
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "litenp/litenp.hpp"

template <typename T>
void emit(const std::string& name, const litenp::Array<T>& a) {
    std::cout << name;
    for (auto v : a.to_vector()) std::cout << ' ' << static_cast<double>(v);
    std::cout << '\n';
    std::cout << name << "_shape";
    for (auto d : a.shape()) std::cout << ' ' << d;
    std::cout << '\n';
}

template <typename T>
void emit_scalar(const std::string& name, T v) {
    std::cout << name << "_scalar " << static_cast<double>(v) << '\n';
}

template <typename T>
void emit_mask(const std::string& name, const litenp::Array<T>& a) {
    std::cout << name;
    for (auto v : a.to_vector()) std::cout << ' ' << static_cast<int>(v);
    std::cout << '\n';
    std::cout << name << "_shape";
    for (auto d : a.shape()) std::cout << ' ' << d;
    std::cout << '\n';
}

int main() {
    std::cout << std::setprecision(14);
    using namespace litenp;
    const float NP_NAN = std::numeric_limits<float>::quiet_NaN();
    const float NP_INF = std::numeric_limits<float>::infinity();
    const float NP_NEG_INF = -std::numeric_limits<float>::infinity();
"""

CPP_FOOTER = r"""
    return 0;
}
"""


def wrap_try(idx: int, body: str) -> str:
    """Wrap a case body in try/catch so empty/invalid reductions report an
    error sentinel instead of aborting the whole harness."""
    return (f"    try {{\n{body}"
            f"    }} catch (const std::exception& e__) {{"
            f' std::cout << "case_{idx}_error " << e__.what() << "\\n"; }}\n')


def shape_cpp(shape: tuple[int, ...]) -> str:
    return "{" + ", ".join(str(d) for d in shape) + "}"


def make_array_decl(name: str, tag: str, arr: np.ndarray) -> str:
    """C++ statement: auto name = Array<T>::from_vector({...}, {...});"""
    ty = cpp_name(tag)
    sh = shape_cpp(tuple(int(d) for d in arr.shape))
    flat = arr.reshape(-1).tolist()
    vals = fmt_list(flat, tag)
    return f"    auto {name} = Array<{ty}>::from_vector({sh}, {{ {vals} }});\n"


def unary_cpp(idx: int, case) -> str:
    _, tag, op, arr = case
    name = f"u{idx}"
    ty = cpp_name(tag)
    decl = make_array_decl(name, tag, arr)
    return wrap_try(idx, decl + (
        f'    emit("case_{idx}", {op}<{ty}>({name}.view()));\n'
    ))


def binary_cpp(idx: int, case) -> str:
    _, tag, op, a, b = case
    na, nb = f"ba{idx}", f"bb{idx}"
    ty = cpp_name(tag)
    da = make_array_decl(na, tag, a)
    db = make_array_decl(nb, tag, b)
    return wrap_try(idx, da + db + (
        f'    emit("case_{idx}", {op}<{ty}>({na}.view(), {nb}.view()));\n'
    ))


def broadcast_cpp(idx: int, case) -> str:
    _, tag, op, a, b = case
    na, nb = f"ra{idx}", f"rb{idx}"
    ty = cpp_name(tag)
    da = make_array_decl(na, tag, a)
    db = make_array_decl(nb, tag, b)
    return wrap_try(idx, da + db + (
        f'    emit("case_{idx}", {op}<{ty}>({na}.view(), {nb}.view()));\n'
    ))


def compare_cpp(idx: int, case) -> str:
    _, tag, op, a, b = case
    na, nb = f"ca{idx}", f"cb{idx}"
    ty = cpp_name(tag)
    da = make_array_decl(na, tag, a)
    db = make_array_decl(nb, tag, b)
    return wrap_try(idx, da + db + (
        f'    emit_mask("case_{idx}", {op}<{ty}>({na}.view(), {nb}.view()));\n'
    ))


def reduce_cpp(idx: int, case) -> str:
    _, tag, op, arr, axis = case
    na = f"rd{idx}"
    ty = cpp_name(tag)
    da = make_array_decl(na, tag, arr)
    if axis is None:
        body = da + f'    emit_scalar("case_{idx}", {op}<{ty}>({na}));\n'
    else:
        body = da + f'    emit("case_{idx}", {op}<{ty}>({na}, {int(axis)}));\n'
    return wrap_try(idx, body)


def slice_cpp(idx: int, case) -> str:
    _, tag, arr, axis, begin, end, step = case
    na = f"sl{idx}"
    ty = cpp_name(tag)
    da = make_array_decl(na, tag, arr)
    def arg(v):
        if v == "open":
            return "litenp::SLICE_OPEN"
        return str(int(v))
    return wrap_try(idx, da + (
        f'    emit("case_{idx}", as_contiguous<{ty}>({na}.view().slice({int(axis)}, {arg(begin)}, {arg(end)}, {int(step)})));\n'
    ))


def astype_cpp(idx: int, case) -> str:
    _, tag, target, arr = case
    na = f"as{idx}"
    da = make_array_decl(na, tag, arr)
    tty = cpp_name(target)
    return wrap_try(idx, da + (
        f'    emit("case_{idx}", astype<{tty}>({na}));\n'
    ))


# ---------------------------------------------------------------------------
# NumPy oracle for each case
# ---------------------------------------------------------------------------

def np_unary(op, a):
    if op == "negative": return -a
    if op == "abs": return np.abs(a)
    if op == "relu": return np.maximum(a, 0)
    if op == "sqrt": return np.sqrt(a)
    if op == "exp": return np.exp(a)
    if op == "sigmoid": return 1.0 / (1.0 + np.exp(-a))
    raise ValueError(op)


def np_binary(op, a, b):
    if op == "add": return a + b
    if op == "subtract": return a - b
    if op == "multiply": return a * b
    if op == "divide": return a / b
    if op == "minimum": return np.minimum(a, b)
    if op == "maximum": return np.maximum(a, b)
    raise ValueError(op)


def np_compare(op, a, b):
    if op == "greater": r = a > b
    elif op == "less": r = a < b
    elif op == "greater_equal": r = a >= b
    elif op == "less_equal": r = a <= b
    elif op == "equal": r = a == b
    elif op == "not_equal": r = a != b
    else: raise ValueError(op)
    return r.astype(np.uint8)


def np_reduce(op, a, axis):
    if op == "sum": return a.sum(axis=axis) if axis is not None else a.sum()
    if op == "mean":
        return a.mean(axis=axis) if axis is not None else a.mean()
    if op == "max":
        return a.max(axis=axis) if axis is not None else a.max()
    if op == "min":
        return a.min(axis=axis) if axis is not None else a.min()
    raise ValueError(op)


def np_slice(arr, axis, begin, end, step):
    sl = [slice(None)] * arr.ndim
    def conv(v):
        if v == "open":
            return None
        return v
    b = conv(begin)
    e = conv(end)
    sl[axis] = slice(b, e, step)
    return np.ascontiguousarray(arr[tuple(sl)])


def np_astype(arr, target):
    return arr.astype(np_dtype(target))


# ---------------------------------------------------------------------------
# comparison helpers
# ---------------------------------------------------------------------------

def parse_output(text: str) -> dict[str, list]:
    rows: dict[str, list] = {}
    for raw in text.splitlines():
        parts = raw.split()
        if not parts:
            continue
        key = parts[0]
        rest = parts[1:]
        if key.endswith("_error"):
            # error sentinel: store the message tokens as-is
            rows[key] = rest
            continue
        try:
            rows[key] = [float(x) for x in rest]
        except ValueError:
            rows[key] = rest
    return rows


def get_array(rows, name):
    vals = np.array(rows[name], dtype=np.float64)
    shape = tuple(int(d) for d in rows[f"{name}_shape"])
    return vals.reshape(shape)


def get_scalar(rows, name):
    return rows[f"{name}_scalar"][0]


def get_mask(rows, name):
    vals = np.array(rows[name], dtype=np.int64)
    shape = tuple(int(d) for d in rows[f"{name}_shape"])
    return vals.reshape(shape).astype(np.uint8)


def compare_array(name, got, exp, nan_safe=False):
    got = np.asarray(got, dtype=np.float64)
    exp = np.asarray(exp, dtype=np.float64)
    if got.shape != exp.shape:
        raise AssertionError(f"{name}: shape {got.shape} vs {exp.shape}")
    if nan_safe:
        ok = np.allclose(got, exp, atol=1e-5, rtol=1e-5, equal_nan=True)
    else:
        ok = np.allclose(got, exp, atol=1e-5, rtol=1e-5)
    if not ok:
        raise AssertionError(f"{name}: mismatch\n  got {got}\n  exp {exp}")


def compare_scalar(name, got, exp, nan_safe=False):
    if nan_safe and (np.isnan(got) or np.isnan(exp)):
        if np.isnan(got) != np.isnan(exp):
            raise AssertionError(f"{name}: got {got}, expected NaN")
        return
    if abs(float(got) - float(exp)) > 1e-5:
        raise AssertionError(f"{name}: got {got}, expected {exp}")


def compare_mask(name, got, exp):
    got = np.asarray(got)
    exp = np.asarray(exp)
    if got.shape != exp.shape or not np.array_equal(got, exp):
        raise AssertionError(f"{name}: got {got}, expected {exp}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = random.Random(SEED)
    np.random.seed(SEED)

    unary = gen_unary_cases(rng, CASES)
    binary = gen_binary_cases(rng, CASES)
    broadcast = gen_broadcast_cases(rng, CASES)
    compare = gen_comparison_cases(rng, CASES)
    reduce = gen_reduction_cases(rng, CASES)
    slc = gen_slice_cases(rng, CASES)
    astype = gen_astype_cases(rng, CASES)

    # build C++ source (global case index across all categories)
    parts = [CPP_HEADER]
    idx = 0
    for c in unary:
        parts.append(unary_cpp(idx, c)); idx += 1
    for c in binary:
        parts.append(binary_cpp(idx, c)); idx += 1
    for c in broadcast:
        parts.append(broadcast_cpp(idx, c)); idx += 1
    for c in compare:
        parts.append(compare_cpp(idx, c)); idx += 1
    for c in reduce:
        parts.append(reduce_cpp(idx, c)); idx += 1
    for c in slc:
        parts.append(slice_cpp(idx, c)); idx += 1
    for c in astype:
        parts.append(astype_cpp(idx, c)); idx += 1
    parts.append(CPP_FOOTER)
    source = "".join(parts)

    compiler = os.environ.get("CXX", "c++")
    with tempfile.TemporaryDirectory(prefix="litenp_diff_") as tmp:
        tmp_path = Path(tmp)
        src = tmp_path / "diff.cpp"
        bin_ = tmp_path / "diff"
        src.write_text(source)
        subprocess.run(
            [compiler, "-std=c++17", "-O2", "-I", str(REPO / "include"),
             str(src), "-o", str(bin_)],
            check=True,
        )
        output = subprocess.check_output([str(bin_)], text=True)

    rows = parse_output(output)
    failures = 0
    total = 0

    def has_error(name):
        return f"{name}_error" in rows

    def check_array_thunk(name, thunk, nan_safe=False):
        nonlocal failures, total
        total += 1
        try:
            exp = thunk()
        except (ValueError, ZeroDivisionError, FloatingPointError):
            # NumPy raised: expect C++ to have thrown too
            if not has_error(name):
                failures += 1
                print(f"FAIL [{name}]: NumPy raised but C++ succeeded")
            return
        exp = np.asarray(exp)
        if exp.ndim == 0:
            check_scalar_val(name, float(exp), nan_safe=nan_safe)
            return
        try:
            got = get_array(rows, name)
            compare_array(name, got, exp, nan_safe=nan_safe)
        except KeyError:
            if not has_error(name):
                failures += 1
                print(f"FAIL [{name}]: missing result, no error sentinel")
        except AssertionError as e:
            failures += 1
            print(f"FAIL [{name}]\n{e}\n")

    def check_scalar_val(name, exp, nan_safe=False):
        nonlocal failures
        # NumPy returns 0-d for axis-None or 1D axis-0 reductions.
        # litenp represents a fully-reduced result as shape {1} (not 0-d),
        # so a NumPy 0-d result is canonicalized to (1,) for comparison.
        if f"{name}_scalar" in rows:
            got = get_scalar(rows, name)
            try:
                compare_scalar(name, got, exp, nan_safe=nan_safe)
            except AssertionError as e:
                failures += 1
                print(f"FAIL [{name}]\n{e}\n")
            return
        if name in rows:
            got = get_array(rows, name)
            exp_arr = np.asarray(exp)
            if exp_arr.ndim == 0:
                exp_arr = exp_arr.reshape(1)
            try:
                compare_array(name, got, exp_arr, nan_safe=nan_safe)
            except AssertionError as e:
                failures += 1
                print(f"FAIL [{name}]\n{e}\n")
            return
        if not has_error(name):
            failures += 1
            print(f"FAIL [{name}]: missing scalar/array, no error sentinel")

    def check_mask_thunk(name, thunk):
        nonlocal failures, total
        total += 1
        try:
            exp = thunk()
        except (ValueError, ZeroDivisionError):
            if not has_error(name):
                failures += 1
                print(f"FAIL [{name}]: NumPy raised but C++ succeeded")
            return
        try:
            got = get_mask(rows, name)
            compare_mask(name, got, exp)
        except KeyError:
            if not has_error(name):
                failures += 1
                print(f"FAIL [{name}]: missing mask, no error sentinel")
        except AssertionError as e:
            failures += 1
            print(f"FAIL [{name}]\n{e}\n")

    for i, c in enumerate(unary):
        _, tag, op, arr = c
        check_array_thunk(f"case_{i}", lambda op=op, arr=arr: np_unary(op, arr), nan_safe=True)

    n0 = len(unary)
    for i, c in enumerate(binary):
        _, tag, op, a, b = c
        check_array_thunk(f"case_{n0 + i}", lambda op=op, a=a, b=b: np_binary(op, a, b), nan_safe=True)

    n0 += len(binary)
    for i, c in enumerate(broadcast):
        _, tag, op, a, b = c
        check_array_thunk(f"case_{n0 + i}", lambda op=op, a=a, b=b: np_binary(op, a, b), nan_safe=True)

    n0 += len(broadcast)
    for i, c in enumerate(compare):
        _, tag, op, a, b = c
        check_mask_thunk(f"case_{n0 + i}", lambda op=op, a=a, b=b: np_compare(op, a, b))

    n0 += len(compare)
    for i, c in enumerate(reduce):
        _, tag, op, arr, axis = c
        nan_safe = is_float(tag)
        check_array_thunk(f"case_{n0 + i}",
                           lambda op=op, arr=arr, axis=axis: np_reduce(op, arr, axis),
                           nan_safe=nan_safe)

    n0 += len(reduce)
    for i, c in enumerate(slc):
        _, tag, arr, axis, begin, end, step = c
        check_array_thunk(f"case_{n0 + i}",
                          lambda arr=arr, axis=axis, begin=begin, end=end, step=step: np_slice(arr, axis, begin, end, step),
                          nan_safe=True)

    n0 += len(slc)
    for i, c in enumerate(astype):
        _, tag, target, arr = c
        check_array_thunk(f"case_{n0 + i}",
                          lambda arr=arr, target=target: np_astype(arr, target),
                          nan_safe=is_float(target))

    if failures:
        print(f"\n{failures}/{total} differential checks FAILED")
        raise SystemExit(1)
    print(f"differential oracle passed ({total} random checks, seed={SEED})")


if __name__ == "__main__":
    main()
