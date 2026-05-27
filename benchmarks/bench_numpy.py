#!/usr/bin/env python3
"""NumPy side of the litenp Phase 5 benchmark matrix."""

from __future__ import annotations

import json
import os
import platform
import statistics
import time
from typing import Callable, Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np


Row = Dict[str, float | str | int]


def time_stats(fn: Callable[[], object], repeats: int = 5, warmups: int = 2) -> tuple[float, float]:
    for _ in range(warmups):
        fn()
    samples: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1_000_000.0)
    samples.sort()
    return samples[0], statistics.median(samples)


def add_row(rows: List[Row], name: str, fn: Callable[[], object], repeats: int = 5) -> None:
    best, median = time_stats(fn, repeats=repeats)
    rows.append({"name": name, "best_ms": best, "median_ms": median, "repeats": repeats})


def run_loop(fn: Callable[[], object], iterations: int) -> float:
    check = 0.0
    for _ in range(iterations):
        view = fn()
        check += float(view.shape[0])
    return check


def size_label(n: int) -> str:
    if n % (1 << 20) == 0:
        return f"{n >> 20}M"
    if n % (1 << 10) == 0:
        return f"{n >> 10}K"
    return str(n)


def main() -> None:
    rows: List[Row] = []
    n4m = 1 << 22

    add_row(rows, "zeros 4M f32", lambda: np.zeros(n4m, dtype=np.float32))
    add_row(rows, "ones 4M f32", lambda: np.ones(n4m, dtype=np.float32))
    add_row(rows, "full 4M f32", lambda: np.full(n4m, 3.0, dtype=np.float32))
    add_row(rows, "arange 4M f32", lambda: np.arange(n4m, dtype=np.float32))
    add_row(rows, "linspace 4M f32", lambda: np.linspace(0.0, 1.0, n4m, dtype=np.float32))
    add_row(rows, "eye 1024x1024 f32", lambda: np.eye(1024, 1024, dtype=np.float32))

    base = np.arange(1024 * 1024, dtype=np.float32)
    matrix = base.reshape(1024, 1024)
    add_row(rows, "reshape view x100K", lambda: run_loop(lambda: base.reshape(1024, 1024), 100000), repeats=3)
    add_row(rows, "transpose view x100K", lambda: run_loop(lambda: matrix.T, 100000), repeats=3)
    add_row(rows, "permute view x100K", lambda: run_loop(lambda: np.transpose(matrix, (1, 0)), 100000), repeats=3)
    add_row(rows, "slice view x100K", lambda: run_loop(lambda: matrix[:, :512], 100000), repeats=3)
    add_row(rows, "select view x100K", lambda: run_loop(lambda: matrix[8, :], 100000), repeats=3)

    material = np.full((2048, 2048), 1.0, dtype=np.float32)
    add_row(rows, "transpose materialize 2048^2", lambda: np.ascontiguousarray(material.T), repeats=3)
    base4m = np.arange(n4m, dtype=np.float32)
    add_row(rows, "as_contiguous no-op 4M", lambda: np.array(base4m, copy=True, order="C"), repeats=5)
    add_row(rows, "astype f32->f64 4M", lambda: base4m.astype(np.float64), repeats=5)

    for n in (1 << 16, 1 << 20, 1 << 22, 1 << 24):
        repeats = 3 if n >= (1 << 24) else 5
        label = size_label(n)
        a = np.full(n, 1.25, dtype=np.float32)
        b = np.full(n, 0.75, dtype=np.float32)
        out = np.empty_like(a)
        mask = np.empty(n, dtype=bool)
        add_row(rows, f"add {label}", lambda a=a, b=b, out=out: np.add(a, b, out=out), repeats=repeats)
        add_row(rows, f"greater {label}", lambda a=a, b=b, out=mask: np.greater(a, b, out=out), repeats=repeats)
        mask = a > b
        add_row(rows, f"where {label}", lambda mask=mask, a=a, b=b: np.where(mask, a, b), repeats=repeats)
        where_out = np.empty_like(a)
        add_row(
            rows,
            f"where_into {label}",
            lambda mask=mask, a=a, b=b, out=where_out: (np.copyto(out, b), np.copyto(out, a, where=mask)),
            repeats=repeats,
        )

    a = np.empty(n4m, dtype=np.float32)
    idx = np.arange(n4m, dtype=np.int32)
    a[:] = ((idx % 17) - 8).astype(np.float32) / 8.0
    positive = np.abs(a)
    out = np.empty_like(a)
    add_row(rows, "subtract_into 4M f32", lambda: np.subtract(a, 0.25, out=out))
    add_row(rows, "multiply_into 4M f32", lambda: np.multiply(a, 0.25, out=out))
    add_row(rows, "divide_into 4M f32", lambda: np.divide(a, 0.25, out=out))
    add_row(rows, "minimum_into 4M f32", lambda: np.minimum(a, 0.25, out=out))
    add_row(rows, "maximum_into 4M f32", lambda: np.maximum(a, 0.25, out=out))
    add_row(rows, "negative_into 4M f32", lambda: np.negative(a, out=out))
    add_row(rows, "abs_into 4M f32", lambda: np.abs(a, out=out))
    add_row(rows, "relu_into 4M f32", lambda: np.maximum(a, 0.0, out=out))
    add_row(rows, "sqrt_into 4M f32", lambda: np.sqrt(positive, out=out))
    add_row(rows, "exp_into 4M f32", lambda: np.exp(a, out=out))
    add_row(rows, "sigmoid_into 4M f32", lambda: np.divide(1.0, np.add(1.0, np.exp(-a)), out=out))

    b = np.full(n4m, 0.25, dtype=np.float32)
    bout = np.empty(n4m, dtype=bool)
    add_row(rows, "less mask 4M f32", lambda: np.less(a, b, out=bout))
    add_row(rows, "equal mask 4M f32", lambda: np.equal(a, b, out=bout))
    add_row(rows, "not_equal mask 4M f32", lambda: np.not_equal(a, b, out=bout))
    add_row(rows, "clip_into 4M f32", lambda: np.clip(a, -1.0, 2.0, out=out))

    for side in (512, 1024, 2048):
        repeats = 3 if side >= 2048 else 5
        matrix = np.full((side, side), 1.0, dtype=np.float32)
        row = np.full((side,), 0.5, dtype=np.float32)
        out2 = np.empty_like(matrix)
        add_row(rows, f"broadcast {side}^2", lambda m=matrix, r=row: m + r, repeats=repeats)
        add_row(rows, f"broadcast_into {side}^2", lambda m=matrix, r=row, out=out2: np.add(m, r, out=out), repeats=repeats)
        add_row(rows, f"sum axis0 {side}^2", lambda m=matrix: m.sum(axis=0), repeats=repeats)
        add_row(rows, f"sum axis1 {side}^2", lambda m=matrix: m.sum(axis=1), repeats=repeats)

    reduce_matrix = np.full((2048, 2048), 1.0, dtype=np.float32)
    add_row(rows, "sum all 2048x2048", lambda: reduce_matrix.sum())
    add_row(rows, "mean all 2048x2048", lambda: reduce_matrix.mean())
    add_row(rows, "max all 2048x2048", lambda: reduce_matrix.max())

    ca = np.full((1024, 1024), 1.0, dtype=np.float32)
    cb = np.full((1024, 1024), 2.0, dtype=np.float32)
    add_row(rows, "concatenate axis0 2x1M", lambda: np.concatenate((ca, cb), axis=0), repeats=5)
    add_row(rows, "stack axis0 2x1M", lambda: np.stack((ca, cb), axis=0), repeats=5)

    for side in (128, 256, 384, 512, 768, 1024):
        repeats = 2 if side >= 512 else 3
        ma = np.full((side, side), 1.0, dtype=np.float32)
        mb = np.full((side, side), 0.5, dtype=np.float32)
        mout = np.empty((side, side), dtype=np.float32)
        add_row(rows, f"matmul {side}", lambda a=ma, b=mb, out=mout: np.matmul(a, b, out=out), repeats=repeats)

    print(json.dumps({
        "metadata": {
            "benchmark": "numpy",
            "numpy_version": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "thread_env": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS")
            }
        },
        "rows": rows
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
