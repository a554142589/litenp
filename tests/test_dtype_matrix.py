#!/usr/bin/env python3
"""Dtype promotion matrix test: litenp vs NumPy.

Builds a Cartesian 10x10 dtype table and verifies that litenp's
`detail::promote_type<A, B>` matches `np.result_type(A, B)` for every pair.
This is the test that earns the "NumPy-compatible dtype promotion" claim.

Run:
    python3 tests/test_dtype_matrix.py
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

# 10 supported dtypes, in NumPy <-> C++ form.
# (numpy_dtype, cpp_type)
DTYPES = [
    (np.uint8,    "std::uint8_t"),
    (np.int8,     "std::int8_t"),
    (np.uint16,   "std::uint16_t"),
    (np.int16,    "std::int16_t"),
    (np.uint32,   "std::uint32_t"),
    (np.int32,    "std::int32_t"),
    (np.uint64,   "std::uint64_t"),
    (np.int64,    "std::int64_t"),
    (np.float32,  "float"),
    (np.float64,  "double"),
]

# Map a C++ type name back to a canonical tag for the static_assert message.
def cpp_tag(cpp: str) -> str:
    return cpp


def np_to_cpp(np_dtype, default_int="std::int64_t") -> str:
    """Map a NumPy result dtype to the matching C++ type litenp would use."""
    dt = np.dtype(np_dtype)
    name = dt.name
    mapping = {
        "uint8":   "std::uint8_t",
        "int8":    "std::int8_t",
        "uint16":  "std::uint16_t",
        "int16":   "std::int16_t",
        "uint32":  "std::uint32_t",
        "int32":   "std::int32_t",
        "uint64":  "std::uint64_t",
        "int64":   "std::int64_t",
        "float32": "float",
        "float64": "double",
    }
    if name in mapping:
        return mapping[name]
    # NumPy may promote e.g. int32+uint32 to int64 — np.result_type already
    # reports the canonical name, so this should not happen for our 10-dtype grid.
    raise ValueError(f"unexpected NumPy result dtype: {name}")


def main() -> None:
    compiler = os.environ.get("CXX", "c++")
    # Build a C++ program with one static_assert per (A, B) pair checking
    # promote_type<A, B> equals the C++ type that np.result_type(A, B) maps to.
    lines = [
        "#include <type_traits>",
        "#include \"litenp/litenp.hpp\"",
        "",
        "static_assert(std::is_same<litenp::detail::promote_type<std::int32_t, float>, double>::value,",
        "              \"sanity: int32+float->double\");",
        "",
    ]
    failures_compile: list[str] = []
    # We emit the asserts unconditionally; if any fails, compilation breaks and
    # we report the first mismatch. To produce a readable per-pair report,
    # instead emit one `struct check_AB { static constexpr bool ok = ...; }`
    # and a runtime print of ok. That way one failing pair doesn't abort the
    # rest.
    checks = []
    for i, (na, ca) in enumerate(DTYPES):
        for j, (nb, cb) in enumerate(DTYPES):
            res_np = np.result_type(na, nb)
            expected_cpp = np_to_cpp(res_np)
            tag = f"{ca.replace('std::','')}_{cb.replace('std::','')}"
            var = f"ok_{i}_{j}"
            lines.append(
                f"  static constexpr bool {var} = "
                f"std::is_same<litenp::detail::promote_type<{ca}, {cb}>, {expected_cpp}>::value;"
            )
            checks.append((i, j, ca, cb, expected_cpp, var, tag))

    # runtime print of all ok flags
    lines.append("")
    lines.append("int main() {")
    for i, j, ca, cb, expected_cpp, var, tag in checks:
        lines.append(f'  std::cout << "{tag} " << ({var} ? "OK" : "FAIL") << "\\n";')
    lines.append("  return 0;")
    lines.append("}")
    lines.append("")
    # need iostream
    lines.insert(0, "#include <iostream>")
    source = "\n".join(lines)

    with tempfile.TemporaryDirectory(prefix="litenp_dtype_") as tmp:
        tmp_path = Path(tmp)
        src = tmp_path / "dtype_matrix.cpp"
        bin_ = tmp_path / "dtype_matrix"
        src.write_text(source)
        compile = subprocess.run(
            [compiler, "-std=c++17", "-O2",
             "-I", str(REPO / "include"),
             str(src), "-o", str(bin_),
             "-Wno-unused-variable"],
            capture_output=True, text=True,
        )
        if compile.returncode != 0:
            # A static_assert failure means a mismatched promotion rule.
            print("COMPILE FAILURE — a dtype promotion rule disagrees with NumPy:")
            print(compile.stderr[-3000:])
            raise SystemExit(1)
        output = subprocess.check_output([str(bin_)], text=True)

    mismatches = 0
    total = 0
    for raw in output.splitlines():
        parts = raw.split()
        if len(parts) != 2:
            continue
        total += 1
        if parts[1] != "OK":
            mismatches += 1
            print(f"MISMATCH: {parts[0]} -> {parts[1]}")

    if mismatches:
        print(f"\n{mismatches}/{total} dtype-pair promotions disagree with NumPy")
        raise SystemExit(1)
    print(f"dtype matrix passed ({total} pairs, all match np.result_type)")


if __name__ == "__main__":
    main()
