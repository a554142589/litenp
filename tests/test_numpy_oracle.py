#!/usr/bin/env python3
"""Full-coverage behavioral oracle: litenp vs NumPy.

Compiles a C++ harness that exercises every public operator in the supported
subset (construction, views, unary, binary, comparison, where/clip, reductions,
combine, matmul, astype) and compares the materialized results against NumPy.

Run:
    python3 tests/test_numpy_oracle.py
Env:
    CXX      C++ compiler (default: c++)
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# C++ harness: prints one line per result. Format:
#   <name> <elem0> <elem1> ...          (array, flattened row-major)
#   <name>_shape <dim0> <dim1> ...      (shape, optional)
#   <name>_scalar <value>               (scalar result)
# Mask arrays use uint8 {0,1}.
# ---------------------------------------------------------------------------

CPP_SOURCE = r"""
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
void emit(const std::string& name, litenp::ArrayView<const T> view) {
    emit(name, litenp::as_contiguous<T>(view));
}

template <typename T>
void emit(const std::string& name, litenp::ArrayView<T> view) {
    emit(name, litenp::as_contiguous<T>(view));
}

template <typename T>
void emit_scalar(const std::string& name, T v) {
    std::cout << name << "_scalar " << static_cast<double>(v) << '\n';
}

int main() {
    std::cout << std::setprecision(12);
    using namespace litenp;

    // ---- construction ----
    emit("zeros", zeros<float>({2, 3}));
    emit("ones", ones<float>({2, 3}));
    emit("full", full<float>({2, 2}, 3.5f));
    emit("zeros_like", zeros_like<float>(full<float>({2, 3}, 1.0f)));
    emit("ones_like", ones_like<float>(full<float>({2, 3}, 1.0f)));
    emit("full_like", full_like<float>(full<float>({2, 3}, 1.0f), 7.0f));
    emit("arange", Array<float>::arange(6));
    emit("linspace", linspace<double>(0.0, 1.0, 5));
    emit("eye", eye<float>(3, 2));
    emit("identity", identity<float>(3));

    // ---- views ----
    auto base = Array<float>::from_vector({2, 3}, {1, 2, 3, 4, 5, 6});
    emit("reshape", base.reshape({3, 2}));
    emit("flatten", base.flatten());
    emit("transpose", as_contiguous<float>(base.transpose()));
    emit("permute", as_contiguous<float>(permute(base, {1, 0})));
    emit("slice_step", as_contiguous<float>(base.view().slice(1, 0, 3, 2)));
    emit("select", as_contiguous<float>(base.view().select(0, 1)));
    auto sq = Array<float>::from_vector({1, 2, 1, 3}, {1, 2, 3, 4, 5, 6});
    emit("squeeze", sq.squeeze());
    emit("squeeze_axis", sq.squeeze(2));
    emit("unsqueeze", base.unsqueeze(1));

    // ---- unary ----
    auto u = Array<float>::from_vector({4}, {-2.0f, -1.0f, 0.0f, 4.0f});
    emit("negative", negative<float>(u.view()));
    emit("abs", abs<float>(u.view()));
    emit("relu", relu<float>(u.view()));
    emit("sqrt", sqrt<float>(abs<float>(u.view())));
    emit("exp", exp<float>(zeros<float>({3}).view()));
    emit("sigmoid", sigmoid<float>(u.view()));

    // ---- binary (broadcasting + scalar + mixed) ----
    auto a = Array<float>::from_vector({2, 3}, {1, 2, 3, 4, 5, 6});
    auto row = Array<float>::from_vector({3}, {10, 20, 30});
    auto col = Array<float>::from_vector({2, 1}, {100.0f, 200.0f});
    emit("add_bc", add<float>(a.view(), row.view()));
    emit("sub_bc", subtract<float>(a.view(), row.view()));
    emit("mul_bc", multiply<float>(a.view(), row.view()));
    emit("div_bc", divide<float>(a.view(), row.view()));
    emit("add_col", add<float>(a.view(), col.view()));
    emit("min_bc", minimum<float>(a.view(), row.view()));
    emit("max_bc", maximum<float>(a.view(), row.view()));
    emit("scalar_add", a + 2.0f);
    emit("scalar_sub", a - 1.0f);
    emit("scalar_mul", a * 3.0f);
    emit("scalar_div", a / 2.0f);
    emit("scalar_left", 10.0f + a);
    auto ia = Array<std::int32_t>::from_vector({2, 2}, {1, 2, 3, 4});
    auto fb = Array<float>::from_vector({2, 2}, {0.5f, 1.5f, 2.5f, 3.5f});
    emit("mixed_add", ia + fb);

    // ---- comparison ----
    auto cmp = Array<float>::from_vector({2, 3}, {0, 2, 4, 3, 7, 1});
    emit("greater", greater<float>(a.view(), cmp.view()));
    emit("less", less<float>(a.view(), cmp.view()));
    emit("greater_equal", greater_equal<float>(a.view(), cmp.view()));
    emit("less_equal", less_equal<float>(a.view(), cmp.view()));
    emit("equal", equal<float>(a.view(), cmp.view()));
    emit("not_equal", not_equal<float>(a.view(), cmp.view()));
    // NaN semantics
    const float nan = std::numeric_limits<float>::quiet_NaN();
    auto n1 = Array<float>::from_vector({4}, {nan, 1.0f, nan, 2.0f});
    auto n2 = Array<float>::from_vector({4}, {0.0f, 1.0f, nan, nan});
    emit("nan_equal", equal<float>(n1.view(), n2.view()));
    emit("nan_not_equal", not_equal<float>(n1.view(), n2.view()));

    // ---- where / clip ----
    auto mask = greater<float>(a.view(), cmp.view());
    auto fallback = full<float>({2, 3}, -1.0f);
    emit("where", where<float>(mask.view(), a.view(), fallback.view()));
    emit("clip", clip(a, 2.0f, 5.0f));

    // ---- reductions ----
    emit_scalar("sum_all", sum(a));
    emit("sum_axis0", sum(a, 0));
    emit("sum_axis1", sum(a, 1));
    emit_scalar("mean_all", mean(a));
    emit("mean_axis0", mean(a, 0));
    emit_scalar("max_all", max(a));
    emit("max_axis1", max(a, 1));
    // empty sum returns 0
    emit_scalar("sum_empty", sum(Array<float>({0, 3})));

    // ---- combine ----
    auto c1 = Array<float>::from_vector({2, 2}, {1, 2, 3, 4});
    auto c2 = Array<float>::from_vector({2, 2}, {5, 6, 7, 8});
    emit("concat_axis0", concatenate<float>({c1.view(), c2.view()}, 0));
    emit("concat_axis1", concatenate<float>({c1.view(), c2.view()}, 1));
    emit("stack_axis0", stack<float>({c1.view(), c2.view()}, 0));
    emit("stack_axis2", stack<float>({c1.view(), c2.view()}, 2));

    // ---- matmul ----
    auto ml = Array<float>::from_vector({2, 3}, {1, 2, 3, 4, 5, 6});
    auto mr = Array<float>::from_vector({3, 2}, {7, 8, 9, 10, 11, 12});
    emit("matmul", matmul(ml, mr));
    // non-square
    auto mw = Array<float>::from_vector({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
    auto mwr = Array<float>::from_vector({4, 1}, {1, 2, 3, 4});
    emit("matmul_2x4_4x1", matmul(mw, mwr));

    // ---- astype ----
    emit("astype_i32", astype<std::int32_t>(a));
    emit("astype_f64", astype<double>(a));

    // ---- broadcast_to (multi-dimensional) ----
    auto bt2d = Array<float>::from_vector({2, 1}, {1.0f, 4.0f});
    emit("broadcast_to_2x1_2x3", broadcast_to<float>(bt2d.view(), {2, 3}));
    auto bt1d = Array<float>::from_vector({3}, {10.0f, 20.0f, 30.0f});
    emit("broadcast_to_1x3_2x3", broadcast_to<float>(bt1d.view(), {2, 3}));
    auto bt3d = Array<float>::from_vector({1, 2, 1}, {1.0f, 2.0f});
    emit("broadcast_to_1x2x1_3x2x4", broadcast_to<float>(bt3d.view(), {3, 2, 4}));

    // ---- cumsum ----
    auto cs1d = Array<float>::from_vector({5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    emit("cumsum_1d", cumsum<float>(cs1d.view(), 0));
    emit("cumsum_2d_axis0", cumsum<float>(a.view(), 0));
    emit("cumsum_2d_axis1", cumsum<float>(a.view(), 1));

    // ---- NaN propagation: minimum/maximum ----
    auto nm1 = Array<float>::from_vector({4}, {nan, 1.0f, nan, 2.0f});
    auto nm2 = Array<float>::from_vector({4}, {0.0f, 1.0f, nan, nan});
    emit("nan_minimum", minimum<float>(nm1.view(), nm2.view()));
    emit("nan_maximum", maximum<float>(nm1.view(), nm2.view()));

    // ---- NaN propagation: min/max reduction ----
    auto nan_arr = Array<float>::from_vector({4}, {3.0f, nan, 1.0f, 2.0f});
    emit_scalar("nan_min", min<float>(nan_arr.view()));
    emit_scalar("nan_max", max<float>(nan_arr.view()));
    auto nan_2d = Array<float>::from_vector({2, 3}, {1.0f, nan, 3.0f, 4.0f, 5.0f, 6.0f});
    emit("nan_min_axis0", min<float>(nan_2d.view(), 0));
    emit("nan_max_axis1", max<float>(nan_2d.view(), 1));

    return 0;
}
"""


def parse_output(text: str) -> dict[str, list[float]]:
    rows: dict[str, list[float]] = {}
    for raw in text.splitlines():
        parts = raw.split()
        if not parts:
            continue
        rows[parts[0]] = [float(x) for x in parts[1:]]
    return rows


def get_array(rows: dict[str, list[float]], name: str) -> np.ndarray:
    vals = np.array(rows[name], dtype=np.float64)
    shape = tuple(int(d) for d in rows[f"{name}_shape"])
    return vals.reshape(shape)


def check_close(rows: dict[str, list[float]], name: str, expected: np.ndarray,
                atol: float = 1e-5, rtol: float = 1e-5) -> None:
    got = get_array(rows, name)
    exp = np.asarray(expected, dtype=np.float64)
    if got.shape != exp.shape or not np.allclose(got, exp, atol=atol, rtol=rtol):
        raise AssertionError(f"{name}: shape {got.shape} vs {exp.shape}\n  got {got}\n  exp {exp}")


def check_close_nan(rows: dict[str, list[float]], name: str, expected: np.ndarray,
                    atol: float = 1e-5, rtol: float = 1e-5) -> None:
    """Like check_close but treats NaN==NaN as equal (for NaN propagation tests)."""
    got = get_array(rows, name)
    exp = np.asarray(expected, dtype=np.float64)
    if got.shape != exp.shape or not np.allclose(got, exp, atol=atol, rtol=rtol, equal_nan=True):
        raise AssertionError(f"{name}: shape {got.shape} vs {exp.shape}\n  got {got}\n  exp {exp}")


def check_scalar(rows: dict[str, list[float]], name: str, expected: float,
                 atol: float = 1e-5) -> None:
    got = rows[f"{name}_scalar"][0]
    if abs(got - expected) > atol:
        raise AssertionError(f"{name}: got {got}, expected {expected}")


def check_scalar_nan(rows: dict[str, list[float]], name: str) -> None:
    """Check that a scalar result is NaN."""
    got = rows[f"{name}_scalar"][0]
    if not np.isnan(got):
        raise AssertionError(f"{name}: expected NaN, got {got}")


def check_mask(rows: dict[str, list[float]], name: str, expected: np.ndarray) -> None:
    got = get_array(rows, name)
    exp = np.asarray(expected, dtype=np.float64)
    if got.shape != exp.shape or not np.array_equal(got, exp):
        raise AssertionError(f"{name}: got {got}, expected {exp}")


def main() -> None:
    compiler = os.environ.get("CXX", "c++")
    with tempfile.TemporaryDirectory(prefix="litenp_numpy_oracle_") as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "oracle.cpp"
        binary = tmp_path / "oracle"
        source.write_text(CPP_SOURCE)
        subprocess.run(
            [compiler, "-std=c++17", "-O2", "-I", str(REPO / "include"),
             str(source), "-o", str(binary)],
            check=True,
        )
        output = subprocess.check_output([str(binary)], text=True)

    rows = parse_output(output)

    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    row = np.array([10, 20, 30], dtype=np.float32)
    col = np.array([[100.0], [200.0]], dtype=np.float32)
    cmp_rhs = np.array([[0, 2, 4], [3, 7, 1]], dtype=np.float32)

    # construction
    check_close(rows, "zeros", np.zeros((2, 3), dtype=np.float32))
    check_close(rows, "ones", np.ones((2, 3), dtype=np.float32))
    check_close(rows, "full", np.full((2, 2), 3.5, dtype=np.float32))
    check_close(rows, "zeros_like", np.zeros_like(a))
    check_close(rows, "ones_like", np.ones_like(a))
    check_close(rows, "full_like", np.full_like(a, 7.0))
    check_close(rows, "arange", np.arange(6, dtype=np.float32))
    check_close(rows, "linspace", np.linspace(0.0, 1.0, 5, dtype=np.float64))
    check_close(rows, "eye", np.eye(3, 2, dtype=np.float32))
    check_close(rows, "identity", np.identity(3, dtype=np.float32))

    # views
    check_close(rows, "reshape", a.reshape(3, 2))
    check_close(rows, "flatten", a.ravel())
    check_close(rows, "transpose", np.ascontiguousarray(a.T))
    check_close(rows, "permute", np.ascontiguousarray(np.transpose(a, (1, 0))))
    check_close(rows, "slice_step", a[:, 0:3:2])
    check_close(rows, "select", a[1, :])
    sq = np.array([[[[1, 2, 3]], [[4, 5, 6]]]], dtype=np.float32)  # shape (1,2,1,3)
    check_close(rows, "squeeze", sq.squeeze())
    check_close(rows, "squeeze_axis", sq.squeeze(2))
    check_close(rows, "unsqueeze", np.expand_dims(a, 1))

    # unary
    u = np.array([-2.0, -1.0, 0.0, 4.0], dtype=np.float32)
    check_close(rows, "negative", -u)
    check_close(rows, "abs", np.abs(u))
    check_close(rows, "relu", np.maximum(u, 0.0))
    check_close(rows, "sqrt", np.sqrt(np.abs(u)))
    check_close(rows, "exp", np.exp(np.zeros(3, dtype=np.float32)))
    check_close(rows, "sigmoid", 1.0 / (1.0 + np.exp(-u)))

    # binary
    check_close(rows, "add_bc", a + row)
    check_close(rows, "sub_bc", a - row)
    check_close(rows, "mul_bc", a * row)
    check_close(rows, "div_bc", a / row)
    check_close(rows, "add_col", a + col)
    check_close(rows, "min_bc", np.minimum(a, row))
    check_close(rows, "max_bc", np.maximum(a, row))
    check_close(rows, "scalar_add", a + 2.0)
    check_close(rows, "scalar_sub", a - 1.0)
    check_close(rows, "scalar_mul", a * 3.0)
    check_close(rows, "scalar_div", a / 2.0)
    check_close(rows, "scalar_left", 10.0 + a)
    ia = np.array([[1, 2], [3, 4]], dtype=np.int32)
    fb = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    check_close(rows, "mixed_add", ia + fb, atol=1e-5)

    # comparison (uint8 masks)
    check_mask(rows, "greater", (a > cmp_rhs).astype(np.uint8))
    check_mask(rows, "less", (a < cmp_rhs).astype(np.uint8))
    check_mask(rows, "greater_equal", (a >= cmp_rhs).astype(np.uint8))
    check_mask(rows, "less_equal", (a <= cmp_rhs).astype(np.uint8))
    check_mask(rows, "equal", (a == cmp_rhs).astype(np.uint8))
    check_mask(rows, "not_equal", (a != cmp_rhs).astype(np.uint8))
    n1 = np.array([np.nan, 1.0, np.nan, 2.0], dtype=np.float32)
    n2 = np.array([0.0, 1.0, np.nan, np.nan], dtype=np.float32)
    check_mask(rows, "nan_equal", (n1 == n2).astype(np.uint8))
    check_mask(rows, "nan_not_equal", (n1 != n2).astype(np.uint8))

    # where / clip
    mask = a > cmp_rhs
    check_close(rows, "where", np.where(mask, a, -1.0))
    check_close(rows, "clip", np.clip(a, 2.0, 5.0))

    # reductions
    check_scalar(rows, "sum_all", float(a.sum()))
    check_close(rows, "sum_axis0", a.sum(axis=0))
    check_close(rows, "sum_axis1", a.sum(axis=1))
    check_scalar(rows, "mean_all", float(a.mean()), atol=1e-4)
    check_close(rows, "mean_axis0", a.mean(axis=0), atol=1e-4)
    check_scalar(rows, "max_all", float(a.max()))
    check_close(rows, "max_axis1", a.max(axis=1))
    check_scalar(rows, "sum_empty", 0.0)

    # combine
    c1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    c2 = np.array([[5, 6], [7, 8]], dtype=np.float32)
    check_close(rows, "concat_axis0", np.concatenate((c1, c2), axis=0))
    check_close(rows, "concat_axis1", np.concatenate((c1, c2), axis=1))
    check_close(rows, "stack_axis0", np.stack((c1, c2), axis=0))
    check_close(rows, "stack_axis2", np.stack((c1, c2), axis=2))

    # matmul
    ml = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    mr = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32)
    check_close(rows, "matmul", ml @ mr, atol=1e-3)
    mw = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
    mwr = np.array([[1], [2], [3], [4]], dtype=np.float32)
    check_close(rows, "matmul_2x4_4x1", mw @ mwr, atol=1e-3)

    # astype
    check_close(rows, "astype_i32", a.astype(np.int32))
    check_close(rows, "astype_f64", a.astype(np.float64))

    # broadcast_to (multi-dimensional)
    bt2d = np.array([[1.0], [4.0]], dtype=np.float32)
    check_close(rows, "broadcast_to_2x1_2x3", np.broadcast_to(bt2d, (2, 3)))
    bt1d = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    check_close(rows, "broadcast_to_1x3_2x3", np.broadcast_to(bt1d, (2, 3)))
    bt3d = np.array([[[1.0], [2.0]]], dtype=np.float32)
    check_close(rows, "broadcast_to_1x2x1_3x2x4", np.broadcast_to(bt3d, (3, 2, 4)))

    # cumsum
    cs1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    check_close(rows, "cumsum_1d", np.cumsum(cs1d, axis=0))
    check_close(rows, "cumsum_2d_axis0", np.cumsum(a, axis=0))
    check_close(rows, "cumsum_2d_axis1", np.cumsum(a, axis=1))

    # NaN propagation: minimum/maximum
    nm1 = np.array([np.nan, 1.0, np.nan, 2.0], dtype=np.float32)
    nm2 = np.array([0.0, 1.0, np.nan, np.nan], dtype=np.float32)
    check_close_nan(rows, "nan_minimum", np.minimum(nm1, nm2))
    check_close_nan(rows, "nan_maximum", np.maximum(nm1, nm2))

    # NaN propagation: min/max reduction
    nan_arr = np.array([3.0, np.nan, 1.0, 2.0], dtype=np.float32)
    check_scalar_nan(rows, "nan_min")
    check_scalar_nan(rows, "nan_max")
    nan_2d = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    check_close_nan(rows, "nan_min_axis0", np.min(nan_2d, axis=0))
    check_close_nan(rows, "nan_max_axis1", np.max(nan_2d, axis=1))

    print(f"numpy oracle passed ({len([k for k in rows if not k.endswith('_shape') and not k.endswith('_scalar')])} array + scalar checks)")


if __name__ == "__main__":
    main()
