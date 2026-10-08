# litenp 全量测试计划

本文档定义 `litenp` 的全量测试策略，覆盖**正确性**（与 NumPy / libtorch 行为对齐）、
**性能**（与 NumPy / Eigen / libtorch 基准对比）、**跨平台**（多编译器、多 OS、多构建配置）
与**边界应力**（空数组、别名、NaN、极端尺寸）四大目标。

## 1. 测试目标

| 目标 | 说明 | 判定标准 |
| --- | --- | --- |
| 正确性 | 每个公开算子在支持的子集内与 NumPy / libtorch 结果一致 | 所有 oracle 用例 `np.allclose` / `torch.allclose` 通过（atol/rtol 见 §5） |
| 性能 | 结构感知行保持数量级优势；稠密核行不劣于 NumPy | `tools/compare_benchmarks.py` 报告 `fail: 0, uncovered: 0` |
| 跨平台 | 在主流编译器与 OS 上可编译、可运行、行为一致 | CI 矩阵全部 job 通过 |
| 内存安全 | 无越界、无泄漏、无未定义行为 | ASan + UBSan 构建零报告 |

## 2. 测试维度（五层）

```
┌─────────────────────────────────────────────────────────────┐
│  L5  跨平台矩阵  (OS × 编译器 × 构建类型 × 可选加速)          │
│  L4  性能基准    (litenp vs NumPy vs Eigen vs libtorch)       │
│  L3  行为 Oracle (litenp vs NumPy  +  litenp vs libtorch)     │
│  L2  C++ 单元测试 (断言、边界、别名、NaN)                      │
│  L1  编译 / 安装 / 消费 (CMake install + find_package)        │
└─────────────────────────────────────────────────────────────┘
```

- **L1 构建与消费**：`cmake` 配置、编译、`ctest`、`cmake --install`、外部工程 `find_package(litenp)`。
- **L2 单元测试**：`tests/test_litenp.cpp`，纯 C++ 断言，聚焦边界与别名。
- **L3 行为 Oracle**：编译小 C++ 程序调用 litenp，将输出与 NumPy / libtorch 逐元素比对，形成**双基线**：
  - `tests/test_numpy_oracle.py` —— 固定用例，覆盖每个公开算子的代表性输入（含 NaN 传播、负步长切片、广播、`cumsum`）。
  - `tests/test_differential.py` —— property-based 随机 differential，每轮按 seed 生成随机
    shape/dtype/NaN/empty/广播/负步长切片输入，逐算子对比 NumPy，或匹配 NumPy 抛出的异常
    （如空 `max`/`mean`）。可由 `SEED=` 与 `CASES=` 覆盖，CI 矩阵化多 seed 运行。
  - `tests/test_libtorch_oracle.py` —— libtorch 基线。
- **L4 性能基准**：`benchmarks/bench_litenp.cpp`（含 Eigen / libtorch 可选基线）+
  `benchmarks/bench_numpy.py`，由 `tools/compare_benchmarks.py` 生成 pass/fail 报告。
- **L5 跨平台矩阵**：见 §6。

## 3. API 覆盖矩阵

下表列出 `litenp` 全部公开算子（`include/litenp/litenp.hpp`）及其在各测试层的覆盖状态。
标记：`✓` 已覆盖，`—` 不适用。

### 3.1 构造与工厂

| 算子 | L2 单元 | L3 NumPy | L3 libtorch | L4 基准 | 备注 |
| --- | :-: | :-: | :-: | :-: | --- |
| `Array<T>(shape)` 默认 | ✓ | — | ✓ | ✓ | |
| `from_vector` | ✓ | — | ✓ | ✓ | |
| `zeros` | ✓ | ✓ | ✓ | ✓ | |
| `ones` | ✓ | ✓ | ✓ | ✓ | |
| `full` | ✓ | ✓ | ✓ | ✓ | |
| `zeros_like` / `ones_like` / `full_like` | ✓ | ✓ | ✓ | ✓ | |
| `arange` | ✓ | ✓ | ✓ | ✓ | |
| `linspace` | ✓ | ✓ | ✓ | ✓ | |
| `eye` / `identity` | ✓ | ✓ | ✓ | ✓ | |

### 3.2 视图与元数据

| 算子 | L2 单元 | L3 NumPy | L3 libtorch | L4 基准 | 备注 |
| --- | :-: | :-: | :-: | :-: | --- |
| `reshape` / `flatten` | ✓ | ✓ | ✓ | ✓ | |
| `transpose` / `permute` | ✓ | ✓ | ✓ | ✓ | |
| `slice`（含正/负 step、`SLICE_OPEN` 省略边界） / `select` | ✓ | ✓ | ✓ | ✓ | |
| `squeeze` / `unsqueeze` | ✓ | ✓ | ✓ | — | |
| `as_contiguous` | ✓ | ✓ | ✓ | ✓ | |
| `shape` / `strides` / `ndim` / `size` | ✓ | — | ✓ | ✓ | |

### 3.3 一元算子（含 `*_into`）

| 算子 | L2 单元 | L3 NumPy | L3 libtorch | L4 基准 | 备注 |
| --- | :-: | :-: | :-: | :-: | --- |
| `negative` / `abs` | ✓ | ✓ | ✓ | ✓ | |
| `relu` | ✓ | ✓ | ✓ | ✓ | 对齐 `np.maximum(x,0)` / `torch.relu` |
| `sqrt` | ✓ | ✓ | ✓ | ✓ | |
| `exp` | ✓ | ✓ | ✓ | ✓ | |
| `sigmoid` | ✓ | ✓ | ✓ | ✓ | 对齐 `1/(1+exp(-x))` |

### 3.4 二元算子（含标量与 `*_into`）

| 算子 | L2 单元 | L3 NumPy | L3 libtorch | L4 基准 | 备注 |
| --- | :-: | :-: | :-: | :-: | --- |
| `add` / `subtract` / `multiply` / `divide` | ✓ | ✓ | ✓ | ✓ | 广播 + 标量 + 混合 dtype |
| `minimum` / `maximum` | ✓ | ✓ | ✓ | ✓ | |
| 运算符 `+ - * /`（Array/Array、Array/scalar、scalar/Array） | ✓ | ✓ | ✓ | — | |

### 3.5 比较、选择、裁剪

| 算子 | L2 单元 | L3 NumPy | L3 libtorch | L4 基准 | 备注 |
| --- | :-: | :-: | :-: | :-: | --- |
| `less` / `less_equal` / `greater` / `greater_equal` | ✓ | ✓ | ✓ | ✓ | 输出 `uint8` 掩码 |
| `equal` / `not_equal` | ✓ | ✓ | ✓ | ✓ | NaN 语义 |
| `where` / `where_into` | ✓ | ✓ | ✓ | ✓ | |
| `clip` / `clip_into` | ✓ | ✓ | ✓ | ✓ | |

### 3.6 归约

| 算子 | L2 单元 | L3 NumPy | L3 libtorch | L4 基准 | 备注 |
| --- | :-: | :-: | :-: | :-: | --- |
| `sum`（全量 / axis） | ✓ | ✓ | ✓ | ✓ | 空数组返回 0 |
| `mean`（全量 / axis） | ✓ | ✓ | ✓ | ✓ | 空数组抛 `invalid_argument` |
| `max`（全量 / axis） | ✓ | ✓ | ✓ | ✓ | 空数组抛 `invalid_argument` |

### 3.7 组合与线性代数

| 算子 | L2 单元 | L3 NumPy | L3 libtorch | L4 基准 | 备注 |
| --- | :-: | :-: | :-: | :-: | --- |
| `concatenate` | ✓ | ✓ | ✓ | ✓ | |
| `stack` | ✓ | ✓ | ✓ | ✓ | |
| `matmul` / `matmul_into` | ✓ | ✓ | ✓ | ✓ | 仅 2D |
| `astype` | ✓ | ✓ | ✓ | ✓ | C++ cast 语义 |

## 4. 边界与应力用例清单

以下用例必须在 L2 或 L3 覆盖：

1. **空数组**：`Array<float>()`、`{0,3}`、`{2,0}`；`sum` 返回 0，`mean`/`max` 抛异常。
2. **零维广播**：标量与数组、`{1}` 与 `{n}`、`{m,1}` 与 `{1,n}`。
3. **NaN 传播**：比较算子对 NaN 返回 0（`equal`）/1（`not_equal`）；`min`/`max` 行为。
4. **别名安全**：`*_into` 输出与输入重叠（`slice` 偏移）时结果正确。
5. **形状不匹配**：二元算子广播失败抛 `invalid_argument`。
6. **极端尺寸**：`16M` 元素稠密核、`2048²` 归约与转置、`1024` 矩阵乘。
7. **非均匀数据**：`arange` / 随机模式输入，避免结构感知捷径掩盖稠密核缺陷。
8. **步长切片**：`slice(axis, begin, end, step)`，含 `step>1`、负 step 反向切片，
   以及 `litenp::SLICE_OPEN` 省略 `begin`/`end`（Python 风格）。
9. **类型提升**：`int32 + float` → `double`（NumPy 兼容 `promote_type`，kind+bits 规则）；
   `astype<float>(int_array)`。
10. **视图生命周期**：源 Array 析构后视图不可用（文档约束，测试中显式避免）。

## 5. 容差与对齐约定

| 数据类型 | atol | rtol | 说明 |
| --- | --- | --- | --- |
| `float32` | `1e-5` | `1e-5` | 与 NumPy `np.float32` / libtorch `kFloat32` 对齐 |
| `float64` | `1e-9` | `1e-9` | |
| `int32` / `uint8` | 0 | 0 | 精确匹配 |
| `mean`（float 累加） | `1e-4` | `1e-4` | 浮点求和顺序差异 |

- 比较掩码：litenp 输出 `uint8` {0,1}，与 NumPy `bool.astype(uint8)` / libtorch `to(torch.uint8)` 对齐。
- `matmul`：float32 累加误差放宽到 `atol=1e-3`。
- `linspace` 端点：`start` / `stop` 精确相等，内部点 `atol=1e-5`。

## 6. 跨平台测试策略

### 6.1 OS 矩阵

| OS | 编译器 | 构建类型 | 加速选项 | 用途 |
| --- | --- | --- | --- | --- |
| Ubuntu 22.04 | g++-11 / clang-14 | Release, Debug | 默认 + OpenMP + CBLAS + native | 主力验证 |
| Ubuntu 22.04 | clang-14 | Debug (ASan/UBSan) | — | 内存安全 |
| macOS 13+ | AppleClang 14 | Release, Debug | 默认 | 类 Unix 第二平台 |
| Windows 2022 | MSVC 19.3 | Release, Debug | 默认 | MSVC 语法/ABI |
| Ubuntu 22.04 | g++-11 | Release | + libtorch (C++20) | libtorch 行为/性能基线 |

> libtorch 2.x 要求 C++20；`litenp` 头文件为 C++17，可在 C++20 下编译，
> 故仅在启用 libtorch 的目标上 `target_compile_features(... cxx_std_20)`，
> 不改变库本身的 C++17 承诺。

### 6.2 架构维度

- **x86_64**：默认（含 AVX2/FMA 快路径，由 `-march=native` 或编译器自动启用）。
- **aarch64**（可选）：通过 QEMU 或 GitHub Actions `arm64` runner 验证 NEON 回退路径与通用核。
- **32 位**（可选）：验证 `std::size_t` 宽度与对齐假设。

### 6.3 构建配置维度

| 选项 | 取值 | 验证点 |
| --- | --- | --- |
| `CMAKE_BUILD_TYPE` | Release, Debug | Debug 断言与 Release 优化 |
| `LITENP_USE_OPENMP` | ON, OFF | 并行核正确性与单线程一致性 |
| `LITENP_USE_CBLAS` | ON, OFF | `matmul` BLAS 回退正确性 |
| `LITENP_NATIVE_ARCH` | ON, OFF | AVX2 快路径 vs 标量回退 |
| `LITENP_HAS_TORCH` | ON, OFF | libtorch 基线可用性 |
| `LITENP_HAS_EIGEN` | ON, OFF | Eigen 基线可用性 |

## 7. 测试执行流程

一键脚本：`./tools/run_full_test.sh`

```
1. L1 构建与单测
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j
   ctest --test-dir build --output-on-failure

2. L3 NumPy Oracle
   python3 tests/test_numpy_oracle.py

3. L3 Differential Oracle（property-based，可指定 seed 与 case 数）
   python3 tests/test_differential.py            # SEED=20261008 CASES=64
   SEED=12345 python3 tests/test_differential.py  # 多 seed 回归

4. L3 libtorch Oracle（需已安装 PyTorch）
   python3 tests/test_libtorch_oracle.py

5. L4 性能基准（含 libtorch + Eigen 基线）
   cmake -S . -B build_perf -DCMAKE_BUILD_TYPE=Release \
     -DLITENP_BUILD_BENCHMARKS=ON -DLITENP_USE_OPENMP=ON \
     -DLITENP_USE_CBLAS=ON -DLITENP_NATIVE_ARCH=ON \
     -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
   cmake --build build_perf -j
   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     python3 benchmarks/bench_numpy.py --out /tmp/numpy.json
   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     ./build_perf/litenp_bench > /tmp/cpp.txt
   python3 tools/compare_benchmarks.py --manifest benchmarks/benchmark_manifest.json \
     --cpp /tmp/cpp.txt --numpy /tmp/numpy.json --out /tmp/report.md

6. L5 Sanitizer
   cmake -S . -B build_asan -DCMAKE_BUILD_TYPE=Debug \
     -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
   cmake --build build_asan -j
   ctest --test-dir build_asan --output-on-failure
```

## 8. 验收标准

全量测试通过需同时满足：

1. `ctest`（Release + Debug + ASan/UBSan）全部通过。
2. `test_numpy_oracle.py`、`test_differential.py`（至少 1 个 seed）与
   `test_libtorch_oracle.py` 全部通过。
3. `compare_benchmarks.py` 报告 `pass >= 68` 且 `fail == 0`、`uncovered == 0`。
4. CI 矩阵（§6.1）全部 job 通过（含 `differential-oracle` 多 seed 矩阵）。
5. API 覆盖矩阵（§3）中所有算子在 L3 层至少有一个 oracle 用例。

## 9. 与现有文档的关系

- `docs/api_compatibility.md`：定义支持的语义子集，本计划的覆盖矩阵以其为准。
- `docs/benchmark_methodology.md`：定义性能基准的两类行（结构感知 / 稠密）与读数规则。
- `docs/benchmark_*.md`：历史性能快照。
