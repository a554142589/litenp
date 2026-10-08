#!/usr/bin/env python3
"""Behavioral oracle: litenp vs libtorch (C++ side-by-side comparison).

Compiles a C++ harness that links against both litenp and libtorch, runs every
public operator in the supported subset with identical inputs, and compares the
materialized results element-wise inside the process (avoids float parsing).

Requires a CPU PyTorch install providing the libtorch C++ API.

Run:
    python3 tests/test_libtorch_oracle.py
Env:
    CXX      C++ compiler (default: c++; must support C++20 for libtorch 2.x)
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CPP_SOURCE = r"""
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "litenp/litenp.hpp"
#include <torch/torch.h>

namespace {

int g_failures = 0;

bool close_to(double a, double b, double atol, double rtol) {
    return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}

// Compare a litenp Array (materialized) against a torch tensor.
template <typename T>
void check_close(const std::string& name, const litenp::Array<T>& got,
                 const torch::Tensor& expected, double atol = 1e-5, double rtol = 1e-5) {
    auto g = got.to_vector();
    auto exp = expected.contiguous().flatten();
    if (static_cast<std::size_t>(exp.numel()) != g.size()) {
        std::cout << "FAIL " << name << " size " << g.size()
                  << " vs torch " << exp.numel() << "\n";
        ++g_failures;
        return;
    }
    for (std::size_t i = 0; i < g.size(); ++i) {
        double a = static_cast<double>(g[i]);
        double b = exp[i].item<double>();
        if (!close_to(a, b, atol, rtol)) {
            std::cout << "FAIL " << name << " at " << i
                      << ": litenp " << a << " vs torch " << b << "\n";
            ++g_failures;
            return;
        }
    }
    std::cout << "PASS " << name << "\n";
}

void check_scalar(const std::string& name, double got, double expected,
                  double atol = 1e-5) {
    if (!close_to(got, expected, atol, 1e-5)) {
        std::cout << "FAIL " << name << ": litenp " << got << " vs torch " << expected << "\n";
        ++g_failures;
        return;
    }
    std::cout << "PASS " << name << "\n";
}

// Compare a litenp uint8 mask against a torch bool/byte tensor.
void check_mask(const std::string& name, const litenp::Array<std::uint8_t>& got,
                const torch::Tensor& expected) {
    auto g = got.to_vector();
    auto exp = expected.to(torch::kUInt8).contiguous().flatten();
    if (static_cast<std::size_t>(exp.numel()) != g.size()) {
        std::cout << "FAIL " << name << " size mismatch\n";
        ++g_failures;
        return;
    }
    for (std::size_t i = 0; i < g.size(); ++i) {
        if (g[i] != static_cast<std::uint8_t>(exp[i].item<int>())) {
            std::cout << "FAIL " << name << " at " << i << "\n";
            ++g_failures;
            return;
        }
    }
    std::cout << "PASS " << name << "\n";
}

torch::Tensor from_vector(std::vector<float> v, std::vector<int64_t> shape) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided);
    return torch::from_blob(v.data(), shape, opts).clone().contiguous();
}

}  // namespace

int main() {
    using namespace litenp;
    const float nan = std::numeric_limits<float>::quiet_NaN();

    // ---- construction ----
    check_close("zeros", zeros<float>({2, 3}), torch::zeros({2, 3}));
    check_close("ones", ones<float>({2, 3}), torch::ones({2, 3}));
    check_close("full", full<float>({2, 2}, 3.5f), torch::full({2, 2}, 3.5f));
    {
        auto like_ref = full<float>({2, 3}, 1.0f);
        auto t_ref = torch::full({2, 3}, 1.0f);
        check_close("zeros_like", zeros_like<float>(like_ref), torch::zeros_like(t_ref));
        check_close("ones_like", ones_like<float>(like_ref), torch::ones_like(t_ref));
        check_close("full_like", full_like<float>(like_ref, 7.0f), torch::full_like(t_ref, 7.0f));
    }
    check_close("arange", Array<float>::arange(6), torch::arange(0, 6, torch::kFloat32));
    check_close("linspace", linspace<double>(0.0, 1.0, 5),
                torch::linspace(0.0, 1.0, 5, torch::kFloat64));
    check_close("eye", eye<float>(3, 2), torch::eye(3, 2));
    check_close("identity", identity<float>(3), torch::eye(3));

    // ---- views ----
    auto base = Array<float>::from_vector({2, 3}, {1, 2, 3, 4, 5, 6});
    auto t_base = from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    check_close("reshape", as_contiguous<float>(base.reshape({3, 2})), t_base.reshape({3, 2}));
    check_close("flatten", as_contiguous<float>(base.flatten()), t_base.flatten());
    check_close("transpose", as_contiguous<float>(base.transpose()), t_base.t().contiguous());
    check_close("permute", as_contiguous<float>(permute(base, {1, 0})),
                t_base.permute({1, 0}).contiguous());
    check_close("slice_step", as_contiguous<float>(base.view().slice(1, 0, 3, 2)),
                t_base.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3, 2)}).contiguous());
    check_close("select", as_contiguous<float>(base.view().select(0, 1)),
                t_base[1].contiguous());
    {
        auto sq = Array<float>::from_vector({1, 2, 1, 3}, {1, 2, 3, 4, 5, 6});
        auto t_sq = from_vector({1, 2, 3, 4, 5, 6}, {1, 2, 1, 3});
        check_close("squeeze", as_contiguous<float>(sq.squeeze()), t_sq.squeeze().contiguous());
        check_close("squeeze_axis", as_contiguous<float>(sq.squeeze(2)),
                    t_sq.squeeze(2).contiguous());
    }
    check_close("unsqueeze", as_contiguous<float>(base.unsqueeze(1)),
                t_base.unsqueeze(1).contiguous());

    // ---- unary ----
    auto u = Array<float>::from_vector({4}, {-2.0f, -1.0f, 0.0f, 4.0f});
    auto t_u = from_vector({-2, -1, 0, 4}, {4});
    check_close("negative", negative<float>(u.view()), torch::neg(t_u));
    check_close("abs", abs<float>(u.view()), torch::abs(t_u));
    check_close("relu", relu<float>(u.view()), torch::relu(t_u));
    check_close("sqrt", sqrt<float>(abs<float>(u.view())), torch::sqrt(torch::abs(t_u)));
    check_close("exp", exp<float>(zeros<float>({3}).view()), torch::exp(torch::zeros({3})));
    check_close("sigmoid", sigmoid<float>(u.view()), torch::sigmoid(t_u));

    // ---- binary (broadcasting + scalar + mixed) ----
    auto a = Array<float>::from_vector({2, 3}, {1, 2, 3, 4, 5, 6});
    auto t_a = from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    auto row = Array<float>::from_vector({3}, {10, 20, 30});
    auto t_row = from_vector({10, 20, 30}, {3});
    auto col = Array<float>::from_vector({2, 1}, {100.0f, 200.0f});
    auto t_col = from_vector({100, 200}, {2, 1});
    check_close("add_bc", add<float>(a.view(), row.view()), t_a + t_row);
    check_close("sub_bc", subtract<float>(a.view(), row.view()), t_a - t_row);
    check_close("mul_bc", multiply<float>(a.view(), row.view()), t_a * t_row);
    check_close("div_bc", divide<float>(a.view(), row.view()), t_a / t_row);
    check_close("add_col", add<float>(a.view(), col.view()), t_a + t_col);
    check_close("min_bc", minimum<float>(a.view(), row.view()), torch::minimum(t_a, t_row));
    check_close("max_bc", maximum<float>(a.view(), row.view()), torch::maximum(t_a, t_row));
    check_close("scalar_add", a + 2.0f, t_a + 2.0f);
    check_close("scalar_sub", a - 1.0f, t_a - 1.0f);
    check_close("scalar_mul", a * 3.0f, t_a * 3.0f);
    check_close("scalar_div", a / 2.0f, t_a / 2.0f);
    check_close("scalar_left", 10.0f + a, 10.0f + t_a);

    // ---- comparison ----
    auto cmp = Array<float>::from_vector({2, 3}, {0, 2, 4, 3, 7, 1});
    auto t_cmp = from_vector({0, 2, 4, 3, 7, 1}, {2, 3});
    check_mask("greater", greater<float>(a.view(), cmp.view()), t_a > t_cmp);
    check_mask("less", less<float>(a.view(), cmp.view()), t_a < t_cmp);
    check_mask("greater_equal", greater_equal<float>(a.view(), cmp.view()), t_a >= t_cmp);
    check_mask("less_equal", less_equal<float>(a.view(), cmp.view()), t_a <= t_cmp);
    check_mask("equal", equal<float>(a.view(), cmp.view()), t_a == t_cmp);
    check_mask("not_equal", not_equal<float>(a.view(), cmp.view()), t_a != t_cmp);
    {
        auto n1 = Array<float>::from_vector({4}, {nan, 1.0f, nan, 2.0f});
        auto n2 = Array<float>::from_vector({4}, {0.0f, 1.0f, nan, nan});
        auto t_n1 = from_vector({nan, 1, nan, 2}, {4});
        auto t_n2 = from_vector({0, 1, nan, nan}, {4});
        check_mask("nan_equal", equal<float>(n1.view(), n2.view()), t_n1 == t_n2);
        check_mask("nan_not_equal", not_equal<float>(n1.view(), n2.view()), t_n1 != t_n2);
    }

    // ---- where / clip ----
    auto mask = greater<float>(a.view(), cmp.view());
    auto t_mask = t_a > t_cmp;
    auto fallback = full<float>({2, 3}, -1.0f);
    check_close("where", where<float>(mask.view(), a.view(), fallback.view()),
                torch::where(t_mask, t_a, torch::full_like(t_a, -1.0f)));
    check_close("clip", clip(a, 2.0f, 5.0f), torch::clamp(t_a, 2.0f, 5.0f));

    // ---- reductions ----
    check_scalar("sum_all", static_cast<double>(sum(a)), t_a.sum().item<double>());
    check_close("sum_axis0", sum(a, 0), t_a.sum(0));
    check_close("sum_axis1", sum(a, 1), t_a.sum(1));
    check_scalar("mean_all", mean(a), t_a.mean().item<double>(), 1e-4);
    check_close("mean_axis0", mean(a, 0), t_a.mean(0), 1e-4);
    check_scalar("max_all", static_cast<double>(max(a)), t_a.max().item<double>());
    check_close("max_axis1", max(a, 1), std::get<0>(t_a.max(1)));
    check_scalar("sum_empty", static_cast<double>(sum(Array<float>({0, 3}))),
                 torch::zeros({0, 3}).sum().item<double>());

    // ---- combine ----
    auto c1 = Array<float>::from_vector({2, 2}, {1, 2, 3, 4});
    auto c2 = Array<float>::from_vector({2, 2}, {5, 6, 7, 8});
    auto t_c1 = from_vector({1, 2, 3, 4}, {2, 2});
    auto t_c2 = from_vector({5, 6, 7, 8}, {2, 2});
    check_close("concat_axis0", concatenate<float>({c1.view(), c2.view()}, 0),
                torch::cat({t_c1, t_c2}, 0));
    check_close("concat_axis1", concatenate<float>({c1.view(), c2.view()}, 1),
                torch::cat({t_c1, t_c2}, 1));
    check_close("stack_axis0", stack<float>({c1.view(), c2.view()}, 0),
                torch::stack({t_c1, t_c2}, 0));
    check_close("stack_axis2", stack<float>({c1.view(), c2.view()}, 2),
                torch::stack({t_c1, t_c2}, 2));

    // ---- matmul ----
    auto ml = Array<float>::from_vector({2, 3}, {1, 2, 3, 4, 5, 6});
    auto mr = Array<float>::from_vector({3, 2}, {7, 8, 9, 10, 11, 12});
    auto t_ml = from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    auto t_mr = from_vector({7, 8, 9, 10, 11, 12}, {3, 2});
    check_close("matmul", matmul(ml, mr), torch::matmul(t_ml, t_mr), 1e-3);
    auto mw = Array<float>::from_vector({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
    auto mwr = Array<float>::from_vector({4, 1}, {1, 2, 3, 4});
    auto t_mw = from_vector({1, 2, 3, 4, 5, 6, 7, 8}, {2, 4});
    auto t_mwr = from_vector({1, 2, 3, 4}, {4, 1});
    check_close("matmul_2x4_4x1", matmul(mw, mwr), torch::matmul(t_mw, t_mwr), 1e-3);

    // ---- astype ----
    check_close("astype_i32", astype<std::int32_t>(a), t_a.to(torch::kInt32));
    check_close("astype_f64", astype<double>(a), t_a.to(torch::kFloat64));

    std::cout << (g_failures == 0 ? "libtorch oracle passed\n" : "libtorch oracle FAILED\n");
    return g_failures == 0 ? 0 : 1;
}
"""


def main() -> int:
    try:
        import torch
    except ImportError:
        print("SKIP: PyTorch not installed; install with "
              "`pip install torch --index-url https://download.pytorch.org/whl/cpu`")
        return 0

    from torch.utils import cpp_extension
    compiler = os.environ.get("CXX", "c++")
    include_dirs = cpp_extension.include_paths()
    lib_dirs = cpp_extension.library_paths()

    with tempfile.TemporaryDirectory(prefix="litenp_torch_oracle_") as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "oracle.cpp"
        binary = tmp_path / "oracle"
        source.write_text(CPP_SOURCE)

        cmd = [
            compiler, "-std=c++20", "-O2",
            f"-I{REPO / 'include'}",
            *[f"-I{d}" for d in include_dirs],
            str(source), "-o", str(binary),
            *[f"-L{d}" for d in lib_dirs],
            "-ltorch", "-ltorch_cpu", "-lc10",
            f"-Wl,-rpath,{lib_dirs[0]}" if lib_dirs else "",
        ]
        cmd = [c for c in cmd if c]  # drop empty rpath token
        subprocess.run(cmd, check=True)

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = lib_dirs[0] + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        result = subprocess.run([str(binary)], env=env, text=True, capture_output=True)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        return result.returncode


if __name__ == "__main__":
    sys.exit(main())
