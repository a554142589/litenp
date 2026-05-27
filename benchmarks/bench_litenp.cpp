#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "litenp/litenp.hpp"

#if defined(LITENP_HAS_EIGEN)
#include <Eigen/Core>
#endif

#if defined(LITENP_HAS_TORCH)
#include <torch/torch.h>
#endif

namespace {

volatile double g_sink = 0.0;

struct TimingStats {
    double best = -1.0;
    double median = -1.0;
};

template <typename Fn>
TimingStats time_stats(Fn&& fn, int repeats = 5) {
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    for (int i = 0; i < repeats; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    const std::size_t mid = samples.size() / 2;
    const double median = samples.size() % 2 == 0 ? (samples[mid - 1] + samples[mid]) * 0.5 : samples[mid];
    return {samples.front(), median};
}

template <typename Fn>
double time_ms(Fn&& fn, int repeats = 5) {
    return time_stats(std::forward<Fn>(fn), repeats).best;
}

template <typename Fn>
TimingStats time_loop_stats(Fn&& fn, std::size_t iterations, int repeats = 5) {
    return time_stats([&] {
        for (std::size_t i = 0; i < iterations; ++i) {
            fn();
        }
    }, repeats);
}

template <typename Fn>
double time_loop_ms(Fn&& fn, std::size_t iterations, int repeats = 5) {
    return time_loop_stats(std::forward<Fn>(fn), iterations, repeats).best;
}

void print_header(const std::string& title) {
    std::cout << "\n[" << title << "]\n";
    std::cout << std::left << std::setw(34) << "case"
              << std::right << std::setw(12) << "best ms"
              << std::setw(16) << "checksum" << "\n";
}

void print_result(const std::string& name, double ms, double checksum) {
    g_sink += checksum;
    std::cout << std::left << std::setw(34) << name
              << std::right << std::setw(12) << std::fixed << std::setprecision(6) << ms
              << std::setw(18) << std::defaultfloat << std::setprecision(6) << checksum << "\n";
}

std::string format_ms(double ms) {
    if (ms < 0.0) {
        return "n/a";
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << ms;
    return oss.str();
}

std::string format_stats(TimingStats stats) {
    if (stats.best < 0.0) {
        return "n/a";
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << stats.best << "/" << stats.median;
    return oss.str();
}

std::string format_ratio(double base_ms, double litenp_ms) {
    if (base_ms < 0.0 || litenp_ms <= 0.0) {
        return "n/a";
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << (base_ms / litenp_ms) << "x";
    return oss.str();
}

void print_matrix_header(const std::string& title) {
    std::cout << "\n[" << title << "]\n";
    std::cout << std::left << std::setw(30) << "case"
              << std::right << std::setw(20) << "litenp b/m"
              << std::setw(20) << "Eigen b/m"
              << std::setw(20) << "libtorch b/m"
              << std::setw(12) << "E/lite"
              << std::setw(12) << "T/lite"
              << std::setw(16) << "checksum" << "\n";
}

void print_matrix_result(const std::string& name, TimingStats litenp, TimingStats eigen, TimingStats torch, double checksum) {
    g_sink += checksum;
    std::cout << std::left << std::setw(30) << name
              << std::right << std::setw(20) << format_stats(litenp)
              << std::setw(20) << format_stats(eigen)
              << std::setw(20) << format_stats(torch)
              << std::setw(12) << format_ratio(eigen.best, litenp.best)
              << std::setw(12) << format_ratio(torch.best, litenp.best)
              << std::setw(16) << std::defaultfloat << std::setprecision(6) << checksum << "\n";
}

void print_gemm_path_header() {
    std::cout << "\n[GEMM path diagnostics]\n";
    std::cout << std::left << std::setw(18) << "case"
              << std::right << std::setw(18) << "public b/m"
              << std::setw(18) << "packed b/m"
              << std::setw(18) << "6x16 b/m"
              << std::setw(18) << "4x16 b/m"
              << std::setw(18) << "8x8 b/m"
              << std::setw(18) << "CBLAS b/m"
              << std::setw(16) << "checksum" << "\n";
}

void print_gemm_path_result(
    const std::string& name,
    TimingStats public_path,
    TimingStats packed,
    TimingStats kernel6x16,
    TimingStats kernel4x16,
    TimingStats kernel8x8,
    TimingStats cblas,
    double checksum) {
    g_sink += checksum;
    std::cout << std::left << std::setw(18) << name
              << std::right << std::setw(18) << format_stats(public_path)
              << std::setw(18) << format_stats(packed)
              << std::setw(18) << format_stats(kernel6x16)
              << std::setw(18) << format_stats(kernel4x16)
              << std::setw(18) << format_stats(kernel8x8)
              << std::setw(18) << format_stats(cblas)
              << std::setw(16) << std::defaultfloat << std::setprecision(6) << checksum << "\n";
}

void print_closeout_diag_header() {
    std::cout << "\n[closeout diagnostics]\n";
    std::cout << std::left << std::setw(34) << "case"
              << std::right << std::setw(20) << "public b/m"
              << std::setw(20) << "diagnostic b/m"
              << std::setw(16) << "checksum" << "\n";
}

void print_closeout_diag_result(const std::string& name, TimingStats public_path, TimingStats diagnostic, double checksum) {
    g_sink += checksum;
    std::cout << std::left << std::setw(34) << name
              << std::right << std::setw(20) << format_stats(public_path)
              << std::setw(20) << format_stats(diagnostic)
              << std::setw(16) << std::defaultfloat << std::setprecision(6) << checksum << "\n";
}

std::string size_label(std::size_t n) {
    if (n >= (1u << 20) && n % (1u << 20) == 0) {
        return std::to_string(n / (1u << 20)) + "M";
    }
    if (n >= (1u << 10) && n % (1u << 10) == 0) {
        return std::to_string(n / (1u << 10)) + "K";
    }
    return std::to_string(n);
}

template <typename Fn, typename Check>
void print_timed(const std::string& name, Fn&& fn, Check&& check, int repeats = 5) {
    const double ms = time_ms(std::forward<Fn>(fn), repeats);
    print_result(name, ms, check());
}

template <typename Fn, typename Check>
void print_loop_timed(const std::string& name, Fn&& fn, std::size_t iterations, Check&& check, int repeats = 5) {
    const double ms = time_loop_ms(std::forward<Fn>(fn), iterations, repeats);
    print_result(name, ms, check());
}

template <typename T>
double checksum_array(const litenp::Array<T>& a) {
    if (a.size() == 0) {
        return 0.0;
    }
    return static_cast<double>(a[0]) + static_cast<double>(a[a.size() - 1]);
}

template <typename T>
double checksum_view(litenp::ArrayView<const T> v) {
    if (v.size() == 0) {
        return 0.0;
    }
    litenp::Shape first(v.ndim(), 0);
    litenp::Shape last = v.shape();
    for (auto& dim : last) {
        dim = dim == 0 ? 0 : dim - 1;
    }
    return static_cast<double>(v.at(first)) + static_cast<double>(v.at(last));
}

void bench_construction() {
    print_header("construction");
    constexpr std::size_t n = 1u << 22;

    litenp::Array<float> out;
    print_timed("default Array x1M", [&] {
                    for (int i = 0; i < 1000000; ++i) {
                        litenp::Array<float> tmp;
                        g_sink += tmp.empty() ? 1.0 : 0.0;
                    }
                }, [&] { return g_sink; }, 1);
    print_timed("Array(shape) 4M f32", [&] { out = litenp::Array<float>({n}); }, [&] { return checksum_array(out); });
    print_timed("zeros 4M f32", [&] { out = litenp::zeros<float>({n}); }, [&] { return checksum_array(out); });
    print_timed("ones 4M f32", [&] { out = litenp::ones<float>({n}); }, [&] { return checksum_array(out); });
    print_timed("full 4M f32", [&] { out = litenp::full<float>({n}, 3.0f); }, [&] { return checksum_array(out); });
    print_timed("from_vector 1M f32", [&] {
                    std::vector<float> values(1u << 20, 2.0f);
                    out = litenp::Array<float>::from_vector({1u << 20}, std::move(values));
                }, [&] { return checksum_array(out); });
    print_timed("arange 4M f32", [&] { out = litenp::Array<float>::arange(n); }, [&] { return checksum_array(out); });
    print_timed("arange start/step 4M f32", [&] { out = litenp::Array<float>::arange(n, 1.0f, 0.5f); }, [&] { return checksum_array(out); });
    print_timed("linspace 4M f32", [&] { out = litenp::linspace<float>(0.0f, 1.0f, n); }, [&] { return checksum_array(out); });
    print_timed("zeros_like 4M f32", [&] { out = litenp::zeros_like(out); }, [&] { return checksum_array(out); });
    print_timed("ones_like 4M f32", [&] { out = litenp::ones_like(out); }, [&] { return checksum_array(out); });
    print_timed("full_like 4M f32", [&] { out = litenp::full_like(out, 4.0f); }, [&] { return checksum_array(out); });
    print_timed("zeros_like view 4M f32", [&] { out = litenp::zeros_like<float>(out.view()); }, [&] { return checksum_array(out); });
    print_timed("ones_like view 4M f32", [&] { out = litenp::ones_like<float>(out.view()); }, [&] { return checksum_array(out); });
    print_timed("full_like view 4M f32", [&] { out = litenp::full_like<float>(out.view(), 5.0f); }, [&] { return checksum_array(out); });

    litenp::Array<float> eye_out;
    print_timed("eye 1024x1024 f32", [&] { eye_out = litenp::eye<float>(1024, 1024); }, [&] { return checksum_array(eye_out); });
    print_timed("identity 1024 f32", [&] { eye_out = litenp::identity<float>(1024); }, [&] { return checksum_array(eye_out); });
}

void bench_metadata_accessors() {
    print_header("metadata / accessors");
    auto owned = litenp::Array<float>::arange(1024 * 1024);
    auto view = owned.reshape({1024, 1024});
    constexpr std::size_t iters = 1000000;

    double check = 0.0;
    print_loop_timed("numel(shape) x1M", [&] {
                         check += static_cast<double>(litenp::numel(litenp::Shape{1024, 1024}));
                     }, iters, [&] { return check; });
    check = 0.0;
    print_loop_timed("Array shape/strides x1M", [&] {
                         check += static_cast<double>(owned.shape()[0] + owned.strides()[0]);
                     }, iters, [&] { return check; });
    check = 0.0;
    print_loop_timed("Array ndim/size x1M", [&] {
                         check += static_cast<double>(owned.ndim() + owned.size());
                     }, iters, [&] { return check; });
    check = 0.0;
    print_loop_timed("Array empty/contig x1M", [&] {
                         check += static_cast<double>(owned.empty() ? 1 : 0);
                         check += static_cast<double>(owned.is_contiguous() ? 1 : 0);
                     }, iters, [&] { return check; });
    check = 0.0;
    print_loop_timed("Array data/values x1M", [&] {
                         check += owned.data()[0] + owned.values()[1];
                     }, iters, [&] { return check; });
    check = 0.0;
    print_loop_timed("Array view x1M", [&] {
                         auto v = owned.view();
                         check += static_cast<double>(v.size());
                     }, iters, [&] { return check; });
    check = 0.0;
    print_loop_timed("Array at/operator() x1M", [&] {
                         check += owned.at({1}) + owned({2});
                     }, iters, [&] { return check; });
    check = 0.0;
    print_loop_timed("Array operator[] x1M", [&] {
                         check += owned[3];
                     }, iters, [&] { return check; });
    check = 0.0;
    print_loop_timed("View shape/stride x1M", [&] {
                         check += static_cast<double>(view.shape()[0] + view.strides()[0]);
                     }, iters, [&] { return check; });
    check = 0.0;
    print_loop_timed("View ndim/size/contig x1M", [&] {
                         check += static_cast<double>(view.ndim() + view.size() + (view.is_contiguous() ? 1 : 0));
                     }, iters, [&] { return check; });
    check = 0.0;
    print_loop_timed("View offset/at x1M", [&] {
                         check += static_cast<double>(view.offset({1, 2}));
                         check += view.at({1, 2});
                     }, iters, [&] { return check; });
    check = 0.0;
    print_loop_timed("View operator() x1M", [&] {
                         check += view({2, 3});
                     }, iters, [&] { return check; });
}

void bench_views() {
    print_header("views metadata");
    auto a = litenp::Array<float>::arange(1u << 20);
    auto m = a.reshape({1024, 1024});
    auto matrix_array = litenp::Array<float>({1024, 1024}, 1.0f);
    constexpr std::size_t iters = 1000000;

    double check = 0.0;
    print_loop_timed("reshape x1M", [&] {
                     auto v = a.reshape({1024, 1024});
                     check += static_cast<double>(v.shape()[0]);
                 }, iters, [&] { return check; });
    check = 0.0;
    print_loop_timed("flatten x1M", [&] {
                     auto v = m.flatten();
                     check += static_cast<double>(v.shape()[0]);
                 }, iters, [&] { return check; });
    check = 0.0;
    print_loop_timed("transpose x1M", [&] {
                     auto v = m.transpose();
                     check += static_cast<double>(v.strides()[0]);
                 }, iters, [&] { return check; });
    check = 0.0;
    print_loop_timed("free transpose x1M", [&] {
                     auto v = litenp::transpose(m);
                     check += static_cast<double>(v.strides()[0]);
                 }, iters, [&] { return check; });
    check = 0.0;
    print_loop_timed("Array transpose x1M", [&] {
                     auto v = matrix_array.transpose();
                     check += static_cast<double>(v.strides()[0]);
                 }, iters, [&] { return check; });
    check = 0.0;
    print_loop_timed("free transpose Array x1M", [&] {
                     auto v = litenp::transpose(matrix_array);
                     check += static_cast<double>(v.strides()[0]);
                 }, iters, [&] { return check; });
    check = 0.0;
    print_loop_timed("permute {1,0} x1M", [&] {
                     auto v = m.permute({1, 0});
                     check += static_cast<double>(v.shape()[0]);
                 }, iters, [&] { return check; });
    check = 0.0;
    print_loop_timed("free permute {1,0} x1M", [&] {
                     auto v = litenp::permute(m, {1, 0});
                     check += static_cast<double>(v.shape()[0]);
                 }, iters, [&] { return check; });
    check = 0.0;
    print_loop_timed("Array permute {1,0} x1M", [&] {
                     auto v = matrix_array.permute({1, 0});
                     check += static_cast<double>(v.shape()[0]);
                 }, iters, [&] { return check; });
    check = 0.0;
    print_loop_timed("free permute Array x1M", [&] {
                     auto v = litenp::permute(matrix_array, {1, 0});
                     check += static_cast<double>(v.shape()[0]);
                 }, iters, [&] { return check; });
    check = 0.0;
    print_loop_timed("slice x1M", [&] {
                     auto v = m.slice(1, 0, 512);
                     check += static_cast<double>(v.shape()[1]);
                 }, iters, [&] { return check; });
    check = 0.0;
    print_loop_timed("slice step x1M", [&] {
                     auto v = m.slice(1, 0, 1024, 2);
                     check += static_cast<double>(v.shape()[1]);
                 }, iters, [&] { return check; });
    check = 0.0;
    print_loop_timed("select x1M", [&] {
                     auto v = m.select(0, 8);
                     check += static_cast<double>(v.shape()[0]);
                 }, iters, [&] { return check; });
    check = 0.0;
    print_loop_timed("unsqueeze/squeeze x1M", [&] {
                     auto v = m.unsqueeze(1).squeeze(1);
                     check += static_cast<double>(v.shape()[1]);
                 }, iters, [&] { return check; });
    check = 0.0;
    print_loop_timed("squeeze all x1M", [&] {
                     auto v = m.unsqueeze(0).unsqueeze(2).squeeze();
                     check += static_cast<double>(v.shape()[0] + v.shape()[1]);
                 }, iters, [&] { return check; });
}

void bench_view_compare_matrix() {
    print_matrix_header("view / transpose aligned compare");
    constexpr std::size_t meta_iters = 100000;
    auto base = litenp::Array<float>::arange(1024 * 1024);
    auto matrix = base.reshape({1024, 1024});

#if defined(LITENP_HAS_EIGEN)
    using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    std::vector<float> eigen_values(1024 * 1024, 1.0f);
    Eigen::Map<RowMajorMatrixXf> eigen_matrix(eigen_values.data(), 1024, 1024);
#endif

#if defined(LITENP_HAS_TORCH)
    auto torch_base = torch::empty({1024 * 1024}, torch::kFloat32);
    auto torch_matrix = torch_base.reshape({1024, 1024});
#endif

    double check = 0.0;
    double torch_check = 0.0;

    const TimingStats reshape_ms = time_loop_stats([&] {
        auto v = base.reshape({1024, 1024});
        check += static_cast<double>(v.shape()[0] + v.strides()[0]);
    }, meta_iters, 3);
    TimingStats torch_reshape_ms;
#if defined(LITENP_HAS_TORCH)
    torch_reshape_ms = time_loop_stats([&] {
        auto v = torch_base.reshape({1024, 1024});
        torch_check += static_cast<double>(v.sizes()[0] + v.strides()[0]);
    }, meta_iters, 3);
#endif
    print_matrix_result("reshape view x100K", reshape_ms, {}, torch_reshape_ms, check + torch_check);

    check = torch_check = 0.0;
    const TimingStats transpose_ms = time_loop_stats([&] {
        auto v = matrix.transpose();
        check += static_cast<double>(v.shape()[0] + v.strides()[0]);
    }, meta_iters, 3);
    TimingStats torch_transpose_ms;
#if defined(LITENP_HAS_TORCH)
    torch_transpose_ms = time_loop_stats([&] {
        auto v = torch_matrix.transpose(0, 1);
        torch_check += static_cast<double>(v.sizes()[0] + v.strides()[0]);
    }, meta_iters, 3);
#endif
    print_matrix_result("transpose view x100K", transpose_ms, {}, torch_transpose_ms, check + torch_check);

    check = torch_check = 0.0;
    const TimingStats permute_ms = time_loop_stats([&] {
        auto v = matrix.permute({1, 0});
        check += static_cast<double>(v.shape()[0] + v.strides()[0]);
    }, meta_iters, 3);
    TimingStats torch_permute_ms;
#if defined(LITENP_HAS_TORCH)
    torch_permute_ms = time_loop_stats([&] {
        auto v = torch_matrix.permute({1, 0});
        torch_check += static_cast<double>(v.sizes()[0] + v.strides()[0]);
    }, meta_iters, 3);
#endif
    print_matrix_result("permute view x100K", permute_ms, {}, torch_permute_ms, check + torch_check);

    check = torch_check = 0.0;
    const TimingStats slice_ms = time_loop_stats([&] {
        auto v = matrix.slice(1, 0, 512);
        check += static_cast<double>(v.shape()[1] + v.strides()[1]);
    }, meta_iters, 3);
    TimingStats torch_slice_ms;
#if defined(LITENP_HAS_TORCH)
    torch_slice_ms = time_loop_stats([&] {
        auto v = torch_matrix.slice(1, 0, 512);
        torch_check += static_cast<double>(v.sizes()[1] + v.strides()[1]);
    }, meta_iters, 3);
#endif
    print_matrix_result("slice view x100K", slice_ms, {}, torch_slice_ms, check + torch_check);

    check = torch_check = 0.0;
    const TimingStats select_ms = time_loop_stats([&] {
        auto v = matrix.select(0, 8);
        check += static_cast<double>(v.shape()[0] + v.strides()[0]);
    }, meta_iters, 3);
    TimingStats torch_select_ms;
#if defined(LITENP_HAS_TORCH)
    torch_select_ms = time_loop_stats([&] {
        auto v = torch_matrix.select(0, 8);
        torch_check += static_cast<double>(v.sizes()[0] + v.strides()[0]);
    }, meta_iters, 3);
#endif
    print_matrix_result("select view x100K", select_ms, {}, torch_select_ms, check + torch_check);

#if defined(LITENP_HAS_EIGEN)
    double eigen_check = 0.0;
    const TimingStats eigen_expr_ms = time_loop_stats([&] {
        auto expr = eigen_matrix.transpose();
        eigen_check += static_cast<double>(expr.rows() + expr.cols());
    }, meta_iters, 3);
    print_matrix_result("Eigen transpose expr x100K", {}, eigen_expr_ms, {}, eigen_check);
#endif

    auto material_base = litenp::Array<float>({2048, 2048}, 1.0f);
    auto material_matrix = material_base.view();
    litenp::Array<float> material_out;
    const TimingStats transpose_copy_ms = time_stats([&] {
        material_out = litenp::as_contiguous<float>(material_matrix.transpose());
    }, 3);
    TimingStats eigen_transpose_copy_ms;
    TimingStats torch_transpose_copy_ms;
#if defined(LITENP_HAS_EIGEN)
    RowMajorMatrixXf eigen_material = RowMajorMatrixXf::Constant(2048, 2048, 1.0f);
    RowMajorMatrixXf eigen_material_out(2048, 2048);
    eigen_transpose_copy_ms = time_stats([&] {
        eigen_material_out = eigen_material.transpose();
    }, 3);
#endif
#if defined(LITENP_HAS_TORCH)
    auto torch_material = torch::full({2048, 2048}, 1.0f, torch::kFloat32);
    torch::Tensor torch_material_out;
    torch_transpose_copy_ms = time_stats([&] {
        torch_material_out = torch_material.transpose(0, 1).contiguous();
    }, 3);
#endif
    print_matrix_result("transpose materialize 2048^2", transpose_copy_ms, eigen_transpose_copy_ms, torch_transpose_copy_ms, checksum_array(material_out));
}

void bench_copy_cast() {
    print_header("copy / cast");
    auto base = litenp::Array<float>::arange(1u << 22);
    auto matrix = base.reshape({2048, 2048});
    litenp::Array<float> out_f;
    litenp::Array<double> out_d;
    litenp::Array<std::int32_t> out_i;

    print_timed("as_contiguous no-op 4M", [&] { out_f = litenp::as_contiguous<float>(base.view()); }, [&] { return checksum_array(out_f); });
    print_timed("as_contiguous transpose 4M", [&] { out_f = litenp::as_contiguous<float>(matrix.transpose()); }, [&] { return checksum_array(out_f); });
    print_timed("astype f32->f64 4M", [&] { out_d = litenp::astype<double>(base); }, [&] { return checksum_array(out_d); });
    print_timed("astype view f32->f64 4M", [&] { out_d = litenp::astype<double, float>(base.view()); }, [&] { return checksum_array(out_d); });
    print_timed("astype f32->i32 4M", [&] { out_i = litenp::astype<std::int32_t>(base); }, [&] { return checksum_array(out_i); });
}

void bench_binary() {
    print_header("binary / broadcasting");
    constexpr std::size_t n = 1u << 22;
    auto a = litenp::Array<float>({n}, 1.25f);
    auto b = litenp::Array<float>({n}, 2.75f);
    auto i32 = litenp::Array<std::int32_t>({n}, 2);
    litenp::Array<float> out({n});
    litenp::Array<float> ret;

    print_timed("add_into 4M f32", [&] { litenp::add_into<float>(a.view(), b.view(), out.view()); }, [&] { return checksum_array(out); });
    print_timed("binary_into generic add 4M", [&] { litenp::binary_into<float>(a.view(), b.view(), out.view(), litenp::detail::BinaryOp::Add); }, [&] { return checksum_array(out); });
    print_timed("subtract_into 4M f32", [&] { litenp::subtract_into<float>(a.view(), b.view(), out.view()); }, [&] { return checksum_array(out); });
    print_timed("multiply_into 4M f32", [&] { litenp::multiply_into<float>(a.view(), b.view(), out.view()); }, [&] { return checksum_array(out); });
    print_timed("divide_into 4M f32", [&] { litenp::divide_into<float>(b.view(), a.view(), out.view()); }, [&] { return checksum_array(out); });
    print_timed("minimum_into 4M f32", [&] { litenp::minimum_into<float>(a.view(), b.view(), out.view()); }, [&] { return checksum_array(out); });
    print_timed("maximum_into 4M f32", [&] { litenp::maximum_into<float>(a.view(), b.view(), out.view()); }, [&] { return checksum_array(out); });
    print_timed("scalar add_into 4M f32", [&] { litenp::add_into<float>(a.view(), 2.0f, out.view()); }, [&] { return checksum_array(out); });
    print_timed("binary_scalar_into add 4M", [&] { litenp::binary_scalar_into<float>(a.view(), 2.0f, out.view(), litenp::detail::BinaryOp::Add); }, [&] { return checksum_array(out); });
    print_timed("scalar subtract_into 4M f32", [&] { litenp::subtract_into<float>(a.view(), 2.0f, out.view()); }, [&] { return checksum_array(out); });
    print_timed("scalar multiply_into 4M f32", [&] { litenp::multiply_into<float>(a.view(), 2.0f, out.view()); }, [&] { return checksum_array(out); });
    print_timed("scalar divide_into 4M f32", [&] { litenp::divide_into<float>(a.view(), 2.0f, out.view()); }, [&] { return checksum_array(out); });
    print_timed("scalar minimum_into 4M f32", [&] { litenp::minimum_into<float>(a.view(), 2.0f, out.view()); }, [&] { return checksum_array(out); });
    print_timed("scalar maximum_into 4M f32", [&] { litenp::maximum_into<float>(a.view(), 2.0f, out.view()); }, [&] { return checksum_array(out); });
    print_timed("operator+ alloc 4M f32", [&] { ret = a + b; }, [&] { return checksum_array(ret); });
    print_timed("operator- alloc 4M f32", [&] { ret = a - b; }, [&] { return checksum_array(ret); });
    print_timed("operator* alloc 4M f32", [&] { ret = a * b; }, [&] { return checksum_array(ret); });
    print_timed("operator/ alloc 4M f32", [&] { ret = b / a; }, [&] { return checksum_array(ret); });
    print_timed("binary alloc generic add 4M", [&] { ret = litenp::binary<float>(a.view(), b.view(), litenp::detail::BinaryOp::Add); }, [&] { return checksum_array(ret); });
    print_timed("add alloc 4M f32", [&] { ret = litenp::add(a, b); }, [&] { return checksum_array(ret); });
    print_timed("subtract alloc 4M f32", [&] { ret = litenp::subtract(a, b); }, [&] { return checksum_array(ret); });
    print_timed("multiply alloc 4M f32", [&] { ret = litenp::multiply(a, b); }, [&] { return checksum_array(ret); });
    print_timed("divide alloc 4M f32", [&] { ret = litenp::divide(b, a); }, [&] { return checksum_array(ret); });
    print_timed("minimum alloc 4M f32", [&] { ret = litenp::minimum(a, b); }, [&] { return checksum_array(ret); });
    print_timed("maximum alloc 4M f32", [&] { ret = litenp::maximum(a, b); }, [&] { return checksum_array(ret); });
    print_timed("scalar operator+ alloc 4M", [&] { ret = a + 2.0f; }, [&] { return checksum_array(ret); });
    print_timed("scalar operator- alloc 4M", [&] { ret = a - 2.0f; }, [&] { return checksum_array(ret); });
    print_timed("scalar operator* alloc 4M", [&] { ret = a * 2.0f; }, [&] { return checksum_array(ret); });
    print_timed("scalar operator/ alloc 4M", [&] { ret = a / 2.0f; }, [&] { return checksum_array(ret); });
    print_timed("binary_scalar alloc add 4M", [&] { ret = litenp::binary_scalar<float>(a.view(), 2.0f, litenp::detail::BinaryOp::Add); }, [&] { return checksum_array(ret); });

    auto mat = litenp::Array<float>({2048, 2048}, 1.0f);
    auto bias = litenp::Array<float>({2048}, 0.5f);
    print_timed("broadcast add 2048x2048", [&] { ret = litenp::add<float>(mat.view(), bias.view()); }, [&] { return checksum_array(ret); }, 3);

    decltype(i32 + a) mixed;
    print_timed("mixed op+ i32+f32 4M", [&] { mixed = i32 + a; }, [&] { return checksum_array(mixed); });
    print_timed("mixed op- i32-f32 4M", [&] { mixed = i32 - a; }, [&] { return checksum_array(mixed); });
    print_timed("mixed op* i32*f32 4M", [&] { mixed = i32 * a; }, [&] { return checksum_array(mixed); });
    print_timed("mixed op/ i32/f32 4M", [&] { mixed = i32 / a; }, [&] { return checksum_array(mixed); });
    print_timed("mixed add fn i32+f32 4M", [&] { mixed = litenp::add<std::int32_t, float>(i32.view(), a.view()); }, [&] { return checksum_array(mixed); });
    print_timed("mixed subtract fn 4M", [&] { mixed = litenp::subtract<std::int32_t, float>(i32.view(), a.view()); }, [&] { return checksum_array(mixed); });
    print_timed("mixed multiply fn 4M", [&] { mixed = litenp::multiply<std::int32_t, float>(i32.view(), a.view()); }, [&] { return checksum_array(mixed); });
    print_timed("mixed divide fn 4M", [&] { mixed = litenp::divide<std::int32_t, float>(i32.view(), a.view()); }, [&] { return checksum_array(mixed); });

#if defined(LITENP_HAS_EIGEN)
    Eigen::ArrayXf ea = Eigen::ArrayXf::Constant(static_cast<Eigen::Index>(n), 1.25f);
    Eigen::ArrayXf eb = Eigen::ArrayXf::Constant(static_cast<Eigen::Index>(n), 2.75f);
    Eigen::ArrayXf eo(static_cast<Eigen::Index>(n));
    print_timed("Eigen add 4M f32", [&] { eo = ea + eb; }, [&] { return static_cast<double>(eo[0] + eo[static_cast<Eigen::Index>(n - 1)]); });
#endif

#if defined(LITENP_HAS_TORCH)
    auto ta = torch::full({static_cast<long long>(n)}, 1.25f, torch::kFloat32);
    auto tb = torch::full({static_cast<long long>(n)}, 2.75f, torch::kFloat32);
    torch::Tensor to;
    print_timed("libtorch add 4M f32", [&] { to = ta + tb; }, [&] { return to[0].item<float>() + to[-1].item<float>(); });
#endif
}

void bench_unary() {
    print_header("unary");
    constexpr std::size_t n = 1u << 22;
    auto a = litenp::Array<float>({n});
    for (std::size_t idx = 0; idx < n; ++idx) {
        a[idx] = static_cast<float>(static_cast<int>(idx % 17) - 8) / 8.0f;
    }
    auto positive = litenp::abs(a);
    litenp::Array<float> out({n});

    print_timed("negative_into 4M f32", [&] { litenp::negative_into<float>(a.view(), out.view()); }, [&] { return checksum_array(out); });
    print_timed("unary_into generic neg 4M", [&] { litenp::unary_into<float>(a.view(), out.view(), litenp::detail::UnaryOp::Neg); }, [&] { return checksum_array(out); });
    print_timed("abs_into 4M f32", [&] { litenp::abs_into<float>(a.view(), out.view()); }, [&] { return checksum_array(out); });
    print_timed("relu_into 4M f32", [&] { litenp::relu_into<float>(a.view(), out.view()); }, [&] { return checksum_array(out); });
    print_timed("sqrt_into 4M f32", [&] { litenp::sqrt_into<float>(positive.view(), out.view()); }, [&] { return checksum_array(out); });
    print_timed("exp_into 4M f32", [&] { litenp::exp_into<float>(a.view(), out.view()); }, [&] { return checksum_array(out); });
    print_timed("sigmoid_into 4M f32", [&] { litenp::sigmoid_into<float>(a.view(), out.view()); }, [&] { return checksum_array(out); });
    print_timed("negative alloc 4M f32", [&] { out = litenp::negative(a); }, [&] { return checksum_array(out); });
    print_timed("unary alloc generic neg 4M", [&] { out = litenp::unary<float>(a.view(), litenp::detail::UnaryOp::Neg); }, [&] { return checksum_array(out); });
    print_timed("negative view alloc 4M", [&] { out = litenp::negative<float>(a.view()); }, [&] { return checksum_array(out); });
    print_timed("abs alloc 4M f32", [&] { out = litenp::abs(a); }, [&] { return checksum_array(out); });
    print_timed("relu alloc 4M f32", [&] { out = litenp::relu(a); }, [&] { return checksum_array(out); });
    print_timed("sqrt alloc 4M f32", [&] { out = litenp::sqrt(positive); }, [&] { return checksum_array(out); });
    print_timed("exp alloc 4M f32", [&] { out = litenp::exp(a); }, [&] { return checksum_array(out); });
    print_timed("sigmoid alloc 4M f32", [&] { out = litenp::sigmoid(a); }, [&] { return checksum_array(out); });

#if defined(LITENP_HAS_EIGEN)
    Eigen::ArrayXf ea(static_cast<Eigen::Index>(n));
    for (Eigen::Index i = 0; i < ea.size(); ++i) {
        ea[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 8.0f;
    }
    Eigen::ArrayXf eo(static_cast<Eigen::Index>(n));
    print_timed("Eigen relu 4M f32", [&] { eo = ea.max(0.0f); }, [&] { return static_cast<double>(eo[0] + eo[static_cast<Eigen::Index>(n - 1)]); });
#endif

#if defined(LITENP_HAS_TORCH)
    auto ta = torch::empty({static_cast<long long>(n)}, torch::kFloat32);
    auto acc = ta.accessor<float, 1>();
    for (std::size_t i = 0; i < n; ++i) {
        acc[static_cast<long long>(i)] = static_cast<float>(static_cast<int>(i % 17) - 8) / 8.0f;
    }
    torch::Tensor to;
    print_timed("libtorch relu 4M f32", [&] { to = torch::relu(ta); }, [&] { return to[0].item<float>() + to[-1].item<float>(); });
#endif
}

void bench_condition() {
    print_header("condition / mask");
    constexpr std::size_t n = 1u << 22;
    auto a = litenp::Array<float>({n});
    auto b = litenp::Array<float>({n}, 0.25f);
    for (std::size_t idx = 0; idx < n; ++idx) {
        a[idx] = static_cast<float>(static_cast<int>(idx % 9) - 4);
    }
    litenp::Array<float> out({n});
    litenp::Array<std::uint8_t> mask;
    litenp::Array<float> ret;

    print_timed("compare generic greater 4M", [&] { mask = litenp::compare<float>(a.view(), b.view(), litenp::detail::CompareOp::Greater); }, [&] { return checksum_array(mask); });
    print_timed("greater mask 4M f32", [&] { mask = litenp::greater<float>(a.view(), b.view()); }, [&] { return checksum_array(mask); });
    print_timed("greater Array overload 4M", [&] { mask = litenp::greater(a, b); }, [&] { return checksum_array(mask); });
    print_timed("greater_equal mask 4M", [&] { mask = litenp::greater_equal<float>(a.view(), b.view()); }, [&] { return checksum_array(mask); });
    print_timed("less mask 4M f32", [&] { mask = litenp::less<float>(a.view(), b.view()); }, [&] { return checksum_array(mask); });
    print_timed("less_equal mask 4M f32", [&] { mask = litenp::less_equal<float>(a.view(), b.view()); }, [&] { return checksum_array(mask); });
    print_timed("equal mask 4M f32", [&] { mask = litenp::equal<float>(a.view(), b.view()); }, [&] { return checksum_array(mask); });
    print_timed("not_equal mask 4M f32", [&] { mask = litenp::not_equal<float>(a.view(), b.view()); }, [&] { return checksum_array(mask); });
    print_timed("where 4M f32", [&] { ret = litenp::where<float>(mask.view(), a.view(), b.view()); }, [&] { return checksum_array(ret); });
    print_timed("where_into 4M f32", [&] { litenp::where_into<float>(mask.view(), a.view(), b.view(), out.view()); }, [&] { return checksum_array(out); });
    print_timed("clip_into 4M f32", [&] { litenp::clip_into<float>(a.view(), -1.0f, 2.0f, out.view()); }, [&] { return checksum_array(out); });
    print_timed("clip alloc 4M f32", [&] { ret = litenp::clip(a, -1.0f, 2.0f); }, [&] { return checksum_array(ret); });
    print_timed("clip view alloc 4M f32", [&] { ret = litenp::clip<float>(a.view(), -1.0f, 2.0f); }, [&] { return checksum_array(ret); });

#if defined(LITENP_HAS_EIGEN)
    Eigen::ArrayXf ea(static_cast<Eigen::Index>(n));
    for (Eigen::Index i = 0; i < ea.size(); ++i) {
        ea[i] = static_cast<float>(static_cast<int>(i % 9) - 4);
    }
    Eigen::ArrayXf eo(static_cast<Eigen::Index>(n));
    print_timed("Eigen clip 4M f32", [&] { eo = ea.max(-1.0f).min(2.0f); }, [&] { return static_cast<double>(eo[0] + eo[static_cast<Eigen::Index>(n - 1)]); });
#endif

#if defined(LITENP_HAS_TORCH)
    auto ta = torch::empty({static_cast<long long>(n)}, torch::kFloat32);
    auto acc = ta.accessor<float, 1>();
    for (std::size_t i = 0; i < n; ++i) {
        acc[static_cast<long long>(i)] = static_cast<float>(static_cast<int>(i % 9) - 4);
    }
    torch::Tensor to;
    print_timed("libtorch clip 4M f32", [&] { to = torch::clamp(ta, -1.0f, 2.0f); }, [&] { return to[0].item<float>() + to[-1].item<float>(); });
#endif
}

void bench_reduce() {
    print_header("reductions");
    auto a = litenp::Array<float>({2048, 2048}, 1.0f);
    float scalar = 0.0f;
    double mean_scalar = 0.0;
    litenp::Array<float> out_f;
    litenp::Array<double> out_d;

    print_timed("sum all 2048x2048", [&] { scalar = litenp::sum(a); }, [&] { return scalar; });
    print_timed("sum view all 2048x2048", [&] { scalar = litenp::sum<float>(a.view()); }, [&] { return scalar; });
    print_timed("mean all 2048x2048", [&] { mean_scalar = litenp::mean(a); }, [&] { return mean_scalar; });
    print_timed("mean view all 2048x2048", [&] { mean_scalar = litenp::mean<float>(a.view()); }, [&] { return mean_scalar; });
    print_timed("max all 2048x2048", [&] { scalar = litenp::max(a); }, [&] { return scalar; });
    print_timed("max view all 2048x2048", [&] { scalar = litenp::max<float>(a.view()); }, [&] { return scalar; });
    print_timed("sum axis0 2048x2048", [&] { out_f = litenp::sum(a, 0); }, [&] { return checksum_array(out_f); });
    print_timed("sum axis1 2048x2048", [&] { out_f = litenp::sum(a, 1); }, [&] { return checksum_array(out_f); });
    print_timed("mean axis0 2048x2048", [&] { out_d = litenp::mean(a, 0); }, [&] { return checksum_array(out_d); });
    print_timed("mean axis1 2048x2048", [&] { out_d = litenp::mean(a, 1); }, [&] { return checksum_array(out_d); });
    print_timed("max axis0 2048x2048", [&] { out_f = litenp::max(a, 0); }, [&] { return checksum_array(out_f); });
    print_timed("max axis1 2048x2048", [&] { out_f = litenp::max(a, 1); }, [&] { return checksum_array(out_f); });
    print_timed("sum view axis0 2048x2048", [&] { out_f = litenp::sum<float>(a.view(), 0); }, [&] { return checksum_array(out_f); });
    print_timed("sum view axis1 2048x2048", [&] { out_f = litenp::sum<float>(a.view(), 1); }, [&] { return checksum_array(out_f); });
    print_timed("mean view axis0 2048x2048", [&] { out_d = litenp::mean<float>(a.view(), 0); }, [&] { return checksum_array(out_d); });
    print_timed("mean view axis1 2048x2048", [&] { out_d = litenp::mean<float>(a.view(), 1); }, [&] { return checksum_array(out_d); });
    print_timed("max view axis0 2048x2048", [&] { out_f = litenp::max<float>(a.view(), 0); }, [&] { return checksum_array(out_f); });
    print_timed("max view axis1 2048x2048", [&] { out_f = litenp::max<float>(a.view(), 1); }, [&] { return checksum_array(out_f); });

#if defined(LITENP_HAS_EIGEN)
    Eigen::ArrayXf ea = Eigen::ArrayXf::Constant(2048 * 2048, 1.0f);
    float eigen_sum = 0.0f;
    print_timed("Eigen sum 4M f32", [&] { eigen_sum = ea.sum(); }, [&] { return eigen_sum; });
#endif
}

void bench_combine() {
    print_header("combine");
    auto a = litenp::Array<float>({1024, 1024}, 1.0f);
    auto b = litenp::Array<float>({1024, 1024}, 2.0f);
    litenp::Array<float> out;
    auto tiny = litenp::Array<float>({64, 64}, 0.0f);

    print_timed("concatenate axis0 2x1M", [&] { out = litenp::concatenate<float>({a.view(), b.view()}, 0); }, [&] { return checksum_array(out); });
    print_timed("concatenate axis1 2x1M", [&] { out = litenp::concatenate<float>({a.view(), b.view()}, 1); }, [&] { return checksum_array(out); });
    print_timed("stack axis0 2x1M", [&] { out = litenp::stack<float>({a.view(), b.view()}, 0); }, [&] { return checksum_array(out); });
    print_timed("stack axis2 2x1M", [&] { out = litenp::stack<float>({a.view(), b.view()}, 2); }, [&] { return checksum_array(out); });
    print_timed("fill_view 64x64", [&] { litenp::fill_view<float>(tiny.view(), 5.0f); }, [&] { return checksum_array(tiny); });
}

void bench_matmul() {
    print_header("matmul");
    constexpr std::size_t m = 384;
    constexpr std::size_t k = 384;
    constexpr std::size_t n = 384;
    auto a = litenp::Array<float>({m, k}, 1.0f);
    auto b = litenp::Array<float>({k, n}, 0.5f);
    litenp::Array<float> c({m, n});

    print_timed("matmul_into 384^3 f32", [&] { litenp::matmul_into<float>(a.view(), b.view(), c.view()); }, [&] { return checksum_array(c); }, 3);
    print_timed("matmul alloc 384^3 f32", [&] { c = litenp::matmul(a, b); }, [&] { return checksum_array(c); }, 3);
    print_timed("matmul view alloc 384^3", [&] { c = litenp::matmul<float>(a.view(), b.view()); }, [&] { return checksum_array(c); }, 3);
    print_timed("matmul noncontig view 384^3", [&] { c = litenp::matmul<float>(a.transpose(), b.view()); }, [&] { return checksum_array(c); }, 3);

#if defined(LITENP_HAS_EIGEN)
    Eigen::MatrixXf ea = Eigen::MatrixXf::Constant(static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(k), 1.0f);
    Eigen::MatrixXf eb = Eigen::MatrixXf::Constant(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(n), 0.5f);
    Eigen::MatrixXf ec(static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(n));
    print_timed("Eigen matmul 384^3", [&] { ec.noalias() = ea * eb; }, [&] { return ec(0, 0) + ec(static_cast<Eigen::Index>(m - 1), static_cast<Eigen::Index>(n - 1)); }, 3);
#endif

#if defined(LITENP_HAS_TORCH)
    auto ta = torch::full({static_cast<long long>(m), static_cast<long long>(k)}, 1.0f, torch::kFloat32);
    auto tb = torch::full({static_cast<long long>(k), static_cast<long long>(n)}, 0.5f, torch::kFloat32);
    torch::Tensor tc;
    print_timed("libtorch matmul 384^3", [&] { tc = torch::matmul(ta, tb); }, [&] { return tc[0][0].item<float>() + tc[-1][-1].item<float>(); }, 3);
#endif
}

void bench_scale_matrix_1d() {
    print_matrix_header("scale matrix / 1D elementwise");
    const std::vector<std::size_t> sizes = {1u << 10, 1u << 16, 1u << 20, 1u << 22, 1u << 24};

    for (std::size_t n : sizes) {
        const int repeats = n >= (1u << 24) ? 3 : 5;
        auto a = litenp::Array<float>({n}, 1.25f);
        auto b = litenp::Array<float>({n}, 0.75f);
        litenp::Array<float> out({n});
        litenp::Array<std::uint8_t> mask;
        litenp::Array<float> selected;
        litenp::Array<float> selected_into({n});

        const TimingStats add_ms = time_stats([&] { litenp::add_into<float>(a.view(), b.view(), out.view()); }, repeats);
        TimingStats eigen_add_ms;
        TimingStats torch_add_ms;
#if defined(LITENP_HAS_EIGEN)
        Eigen::ArrayXf ea = Eigen::ArrayXf::Constant(static_cast<Eigen::Index>(n), 1.25f);
        Eigen::ArrayXf eb = Eigen::ArrayXf::Constant(static_cast<Eigen::Index>(n), 0.75f);
        Eigen::ArrayXf eo(static_cast<Eigen::Index>(n));
        eigen_add_ms = time_stats([&] { eo = ea + eb; }, repeats);
#endif
#if defined(LITENP_HAS_TORCH)
        auto ta = torch::full({static_cast<long long>(n)}, 1.25f, torch::kFloat32);
        auto tb = torch::full({static_cast<long long>(n)}, 0.75f, torch::kFloat32);
        torch::Tensor to;
        torch_add_ms = time_stats([&] { to = ta + tb; }, repeats);
#endif
        print_matrix_result("add " + size_label(n), add_ms, eigen_add_ms, torch_add_ms, checksum_array(out));

        const TimingStats cmp_ms = time_stats([&] { mask = litenp::greater<float>(a.view(), b.view()); }, repeats);
        TimingStats eigen_cmp_ms;
        TimingStats torch_cmp_ms;
#if defined(LITENP_HAS_EIGEN)
        Eigen::Array<bool, Eigen::Dynamic, 1> em(static_cast<Eigen::Index>(n));
        eigen_cmp_ms = time_stats([&] { em = ea > eb; }, repeats);
#endif
#if defined(LITENP_HAS_TORCH)
        torch::Tensor tm;
        torch_cmp_ms = time_stats([&] { tm = ta > tb; }, repeats);
#endif
        print_matrix_result("greater " + size_label(n), cmp_ms, eigen_cmp_ms, torch_cmp_ms, checksum_array(mask));

        mask = litenp::greater<float>(a.view(), b.view());
        const TimingStats where_ms = time_stats([&] { selected = litenp::where<float>(mask.view(), a.view(), b.view()); }, repeats);
        TimingStats eigen_where_alloc_ms;
        TimingStats eigen_where_reuse_ms;
        TimingStats torch_where_ms;
#if defined(LITENP_HAS_EIGEN)
        em = ea > eb;
        Eigen::ArrayXf eo_alloc;
        eigen_where_alloc_ms = time_stats([&] { eo_alloc = Eigen::ArrayXf(em.select(ea, eb)); }, repeats);
        eigen_where_reuse_ms = time_stats([&] { eo = em.select(ea, eb); }, repeats);
#endif
#if defined(LITENP_HAS_TORCH)
        auto tmask = ta > tb;
        torch_where_ms = time_stats([&] { to = torch::where(tmask, ta, tb); }, repeats);
#endif
        print_matrix_result("where " + size_label(n), where_ms, eigen_where_alloc_ms, torch_where_ms, checksum_array(selected));

        const TimingStats where_alloc_ms = time_stats([&] { selected = litenp::detail::make_uninitialized_array<float>({n}); }, repeats);
        print_matrix_result("where_alloc " + size_label(n), where_alloc_ms, TimingStats{}, TimingStats{}, 0.0);

        const TimingStats where_into_ms = time_stats([&] { litenp::where_into<float>(mask.view(), a.view(), b.view(), selected_into.view()); }, repeats);
        print_matrix_result("where_into " + size_label(n), where_into_ms, eigen_where_reuse_ms, torch_where_ms, checksum_array(selected_into));
    }
}

void bench_scale_matrix_2d() {
    print_matrix_header("scale matrix / 2D broadcast and reduce");
    const std::vector<std::size_t> sides = {256, 512, 1024, 2048};

    for (std::size_t side : sides) {
        const std::size_t n = side * side;
        const int repeats = side >= 2048 ? 3 : 5;
        auto matrix = litenp::Array<float>({side, side}, 1.0f);
        auto row = litenp::Array<float>({side}, 0.5f);
        litenp::Array<float> out;
        litenp::Array<float> out_into({side, side});

        const TimingStats broadcast_ms = time_stats([&] { out = litenp::add<float>(matrix.view(), row.view()); }, repeats);
        const TimingStats broadcast_into_ms = time_stats([&] { litenp::add_into<float>(matrix.view(), row.view(), out_into.view()); }, repeats);
        TimingStats eigen_broadcast_alloc_ms;
        TimingStats eigen_broadcast_reuse_ms;
        TimingStats torch_broadcast_ms;
#if defined(LITENP_HAS_EIGEN)
        using RowMajorArrayXXf = Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        RowMajorArrayXXf em = RowMajorArrayXXf::Constant(static_cast<Eigen::Index>(side), static_cast<Eigen::Index>(side), 1.0f);
        Eigen::Array<float, 1, Eigen::Dynamic> erow =
            Eigen::Array<float, 1, Eigen::Dynamic>::Constant(1, static_cast<Eigen::Index>(side), 0.5f);
        RowMajorArrayXXf eo(static_cast<Eigen::Index>(side), static_cast<Eigen::Index>(side));
        RowMajorArrayXXf eo_alloc;
        eigen_broadcast_alloc_ms = time_stats([&] { eo_alloc = RowMajorArrayXXf(em.rowwise() + erow); }, repeats);
        eigen_broadcast_reuse_ms = time_stats([&] { eo = em.rowwise() + erow; }, repeats);
#endif
#if defined(LITENP_HAS_TORCH)
        auto tmatrix = torch::full({static_cast<long long>(side), static_cast<long long>(side)}, 1.0f, torch::kFloat32);
        auto trow = torch::full({static_cast<long long>(side)}, 0.5f, torch::kFloat32);
        torch::Tensor tout;
        torch_broadcast_ms = time_stats([&] { tout = tmatrix + trow; }, repeats);
#endif
        print_matrix_result("broadcast " + std::to_string(side) + "^2", broadcast_ms, eigen_broadcast_alloc_ms, torch_broadcast_ms, checksum_array(out));
        print_matrix_result("broadcast_into " + std::to_string(side) + "^2", broadcast_into_ms, eigen_broadcast_reuse_ms, torch_broadcast_ms, checksum_array(out_into));

        const TimingStats axis0_ms = time_stats([&] { out = litenp::sum<float>(matrix.view(), 0); }, repeats);
        TimingStats eigen_axis0_ms;
        TimingStats torch_axis0_ms;
#if defined(LITENP_HAS_EIGEN)
        Eigen::ArrayXf ecol;
        eigen_axis0_ms = time_stats([&] { ecol = Eigen::ArrayXf(em.colwise().sum()); }, repeats);
#endif
#if defined(LITENP_HAS_TORCH)
        torch_axis0_ms = time_stats([&] { tout = tmatrix.sum(0); }, repeats);
#endif
        print_matrix_result("sum axis0 " + std::to_string(side) + "^2", axis0_ms, eigen_axis0_ms, torch_axis0_ms, checksum_array(out));

        const TimingStats axis1_ms = time_stats([&] { out = litenp::sum<float>(matrix.view(), 1); }, repeats);
        TimingStats eigen_axis1_ms;
        TimingStats torch_axis1_ms;
#if defined(LITENP_HAS_EIGEN)
        Eigen::ArrayXf ereduced;
        eigen_axis1_ms = time_stats([&] { ereduced = Eigen::ArrayXf(em.rowwise().sum()); }, repeats);
#endif
#if defined(LITENP_HAS_TORCH)
        torch_axis1_ms = time_stats([&] { tout = tmatrix.sum(1); }, repeats);
#endif
        print_matrix_result("sum axis1 " + std::to_string(side) + "^2", axis1_ms, eigen_axis1_ms, torch_axis1_ms, checksum_array(out));
        g_sink += static_cast<double>(n);
    }
}

void bench_closeout_diagnostics() {
    print_closeout_diag_header();

    {
        constexpr std::size_t n = 1u << 24;
        constexpr int repeats = 3;
        auto a = litenp::Array<float>({n}, 1.25f);
        auto b = litenp::Array<float>({n}, 0.75f);
        litenp::Array<float> out({n});
        const TimingStats public_add = time_stats([&] {
            litenp::add_into<float>(a.view(), b.view(), out.view());
        }, repeats);
        const TimingStats serial_add = time_stats([&] {
            litenp::detail::add_contiguous_serial(a.data(), b.data(), out.data(), n);
        }, repeats);
        print_closeout_diag_result("add_contiguous 16M", public_add, serial_add, checksum_array(out));
    }

    {
        constexpr std::size_t side = 2048;
        constexpr std::size_t n = side * side;
        constexpr int repeats = 3;
        auto matrix = litenp::Array<float>({side, side}, 1.0f);
        auto row = litenp::Array<float>({side}, 0.5f);
        litenp::Array<float> out({side, side});
        const TimingStats public_broadcast = time_stats([&] {
            litenp::add_into<float>(matrix.view(), row.view(), out.view());
        }, repeats);
        const TimingStats row_kernel = time_stats([&] {
            litenp::detail::add_row_broadcast_contiguous_serial(matrix.data(), row.data(), out.data(), side, side);
        }, repeats);
        print_closeout_diag_result("row_broadcast_into 2048^2", public_broadcast, row_kernel, checksum_array(out));
        g_sink += static_cast<double>(n);
    }

    {
        constexpr std::size_t side = 2048;
        constexpr int repeats = 3;
        auto matrix = litenp::Array<float>({side, side}, 1.0f);
        litenp::Array<float> out({side});
        const TimingStats public_sum = time_stats([&] {
            out = litenp::sum<float>(matrix.view(), 1);
        }, repeats);
        const TimingStats row_sum = time_stats([&] {
            litenp::detail::sum_rows_contiguous_serial(matrix.data(), out.data(), side, side);
        }, repeats);
        print_closeout_diag_result("sum_rows_axis1 2048^2", public_sum, row_sum, checksum_array(out));
    }

    {
        constexpr std::size_t n = 1u << 24;
        constexpr int repeats = 3;
        auto a = litenp::Array<float>({n}, 1.25f);
        auto b = litenp::Array<float>({n}, 0.75f);
        auto mask = litenp::greater<float>(a.view(), b.view());
        litenp::Array<float> selected;
        litenp::Array<float> out({n});
        const TimingStats public_where = time_stats([&] {
            selected = litenp::where<float>(mask.view(), a.view(), b.view());
        }, repeats);
        const TimingStats where_kernel = time_stats([&] {
            litenp::detail::where_contiguous_serial(mask.data(), a.data(), b.data(), out.data(), n);
        }, repeats);
        print_closeout_diag_result("where_kernel 16M", public_where, where_kernel, checksum_array(out));

        const TimingStats alloc_only = time_stats([&] {
            selected = litenp::detail::make_uninitialized_array<float>({n});
        }, repeats);
        print_closeout_diag_result("where_alloc_only 16M", public_where, alloc_only, checksum_array(selected));
    }
}

void bench_gemm_path_diagnostics() {
    print_gemm_path_header();
    const std::vector<std::size_t> sizes = {384, 512, 768, 1024};

    for (std::size_t side : sizes) {
        const int repeats = side >= 768 ? 2 : 3;
        auto a = litenp::Array<float>({side, side}, 1.0f);
        auto b = litenp::Array<float>({side, side}, 0.5f);
        litenp::Array<float> c({side, side});

        const TimingStats public_path = time_stats([&] {
            litenp::matmul_into<float>(a.view(), b.view(), c.view());
        }, repeats);

        TimingStats kernel4x16;
        TimingStats kernel6x16;
        TimingStats kernel8x8;
        TimingStats packed_ms;
        TimingStats cblas_ms;

#if defined(__AVX2__) && defined(__FMA__)
        if (side % 4 == 0 && side % 16 == 0) {
            packed_ms = time_stats([&] {
                litenp::detail::matmul_f32_4x16_packed_b_kernel(a.data(), b.data(), c.data(), side, side, side);
            }, repeats);
        }
        if (side % 16 == 0) {
            kernel6x16 = time_stats([&] {
                litenp::detail::matmul_f32_6x16_packed_b_kernel(a.data(), b.data(), c.data(), side, side, side);
            }, repeats);
        }
        if (side % 4 == 0 && side % 16 == 0) {
            kernel4x16 = time_stats([&] {
                litenp::detail::matmul_f32_4x16_kernel(a.data(), b.data(), c.data(), side, side, side);
            }, repeats);
        }
        if (side % 8 == 0) {
            kernel8x8 = time_stats([&] {
                litenp::detail::matmul_f32_8x8_kernel(a.data(), b.data(), c.data(), side, side, side);
            }, repeats);
        }
#endif

#if defined(LITENP_HAS_CBLAS)
        cblas_ms = time_stats([&] {
            cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                static_cast<blasint>(side),
                static_cast<blasint>(side),
                static_cast<blasint>(side),
                1.0f,
                a.data(),
                static_cast<blasint>(side),
                b.data(),
                static_cast<blasint>(side),
                0.0f,
                c.data(),
                static_cast<blasint>(side));
        }, repeats);
#endif

        print_gemm_path_result("gemm " + std::to_string(side), public_path, packed_ms, kernel6x16, kernel4x16, kernel8x8, cblas_ms, checksum_array(c));
    }
}

void bench_scale_matrix_matmul() {
    print_matrix_header("scale matrix / matmul");
    const std::vector<std::size_t> sizes = {64, 128, 256, 384, 512, 768, 1024};

    for (std::size_t side : sizes) {
        const int repeats = side >= 512 ? 2 : 3;
        auto a = litenp::Array<float>({side, side}, 1.0f);
        auto b = litenp::Array<float>({side, side}, 0.5f);
        litenp::Array<float> c({side, side});
        const TimingStats litenp_ms = time_stats([&] { litenp::matmul_into<float>(a.view(), b.view(), c.view()); }, repeats);

        TimingStats eigen_ms;
        TimingStats torch_ms;
#if defined(LITENP_HAS_EIGEN)
        Eigen::MatrixXf ea = Eigen::MatrixXf::Constant(static_cast<Eigen::Index>(side), static_cast<Eigen::Index>(side), 1.0f);
        Eigen::MatrixXf eb = Eigen::MatrixXf::Constant(static_cast<Eigen::Index>(side), static_cast<Eigen::Index>(side), 0.5f);
        Eigen::MatrixXf ec(static_cast<Eigen::Index>(side), static_cast<Eigen::Index>(side));
        eigen_ms = time_stats([&] { ec.noalias() = ea * eb; }, repeats);
#endif
#if defined(LITENP_HAS_TORCH)
        auto ta = torch::full({static_cast<long long>(side), static_cast<long long>(side)}, 1.0f, torch::kFloat32);
        auto tb = torch::full({static_cast<long long>(side), static_cast<long long>(side)}, 0.5f, torch::kFloat32);
        torch::Tensor tc;
        torch_ms = time_stats([&] { tc = torch::matmul(ta, tb); }, repeats);
#endif
        print_matrix_result("matmul " + std::to_string(side), litenp_ms, eigen_ms, torch_ms, checksum_array(c));
    }
}

void bench_scale_matrix() {
    bench_scale_matrix_1d();
    bench_scale_matrix_2d();
    bench_scale_matrix_matmul();
    bench_gemm_path_diagnostics();
    bench_closeout_diagnostics();
}

}  // namespace

int main() {
    std::cout << "litenp comprehensive benchmark (best-of timing)\n";
    bench_construction();
    bench_metadata_accessors();
    bench_views();
    bench_view_compare_matrix();
    bench_copy_cast();
    bench_binary();
    bench_unary();
    bench_condition();
    bench_reduce();
    bench_combine();
    bench_matmul();
    bench_scale_matrix();
    std::cout << "\nsink: " << g_sink << "\n";
    return 0;
}
