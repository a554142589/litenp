#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "litenp/litenp.hpp"

namespace {

template <typename T>
void expect_close(T got, T expected, double eps = 1e-6) {
    assert(std::fabs(static_cast<double>(got - expected)) <= eps);
}

template <typename Fn>
void expect_invalid_argument(Fn&& fn) {
    bool threw = false;
    try {
        fn();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

template <typename T>
void test_vectorized_compare_nan_and_tail() {
    std::vector<T> lhs(17);
    std::vector<T> rhs(17);
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        lhs[i] = static_cast<T>(static_cast<int>(i) - 8);
        rhs[i] = static_cast<T>(static_cast<int>(i % 5) - 2);
    }
    lhs[2] = std::numeric_limits<T>::quiet_NaN();
    rhs[5] = std::numeric_limits<T>::quiet_NaN();
    lhs[8] = std::numeric_limits<T>::quiet_NaN();
    rhs[8] = std::numeric_limits<T>::quiet_NaN();
    lhs[16] = static_cast<T>(1);
    rhs[16] = static_cast<T>(2);

    auto a = litenp::Array<T>::from_vector({lhs.size()}, lhs);
    auto b = litenp::Array<T>::from_vector({rhs.size()}, rhs);
    auto lt = litenp::less<T>(a.view(), b.view());
    auto le = litenp::less_equal<T>(a.view(), b.view());
    auto gt = litenp::greater<T>(a.view(), b.view());
    auto ge = litenp::greater_equal<T>(a.view(), b.view());
    auto eq = litenp::equal<T>(a.view(), b.view());
    auto ne = litenp::not_equal<T>(a.view(), b.view());

    for (std::size_t i = 0; i < lhs.size(); ++i) {
        assert(lt({i}) == static_cast<std::uint8_t>(lhs[i] < rhs[i]));
        assert(le({i}) == static_cast<std::uint8_t>(lhs[i] <= rhs[i]));
        assert(gt({i}) == static_cast<std::uint8_t>(lhs[i] > rhs[i]));
        assert(ge({i}) == static_cast<std::uint8_t>(lhs[i] >= rhs[i]));
        assert(eq({i}) == static_cast<std::uint8_t>(lhs[i] == rhs[i]));
        assert(ne({i}) == static_cast<std::uint8_t>(lhs[i] != rhs[i]));
    }
}

void test_array_shape_and_view() {
    litenp::Array<float> a({2, 3});
    assert(reinterpret_cast<std::uintptr_t>(a.data()) % 32 == 0);
    assert(a.shape() == (litenp::Shape{2, 3}));
    assert(a.strides() == (litenp::Shape{3, 1}));
    expect_close(a({0, 0}), 0.0f);
    a.values()[0] = 2.0f;
    expect_close(a({0, 0}), 2.0f);
    a({1, 2}) = 7.0f;
    expect_close(a({1, 2}), 7.0f);
    auto copied_values = a.to_vector();
    static_assert(std::is_same<decltype(copied_values), std::vector<float>>::value, "to_vector should return std::vector");
    expect_close(copied_values[0], 2.0f);
    expect_close(copied_values[5], 7.0f);

    auto flat = a.reshape({6});
    assert(flat.shape() == (litenp::Shape{6}));
    expect_close(flat({5}), 7.0f);

    auto row = a.view().slice(0, 1, 2);
    assert(row.shape() == (litenp::Shape{1, 3}));
    expect_close(row({0, 2}), 7.0f);

    auto stepped = a.view().slice(1, 0, 3, 2);
    assert(stepped.shape() == (litenp::Shape{2, 2}));
    expect_close(stepped({1, 0}), a({1, 0}));
    expect_close(stepped({1, 1}), 7.0f);

    auto selected = a.view().select(0, 1);
    assert(selected.shape() == (litenp::Shape{3}));
    expect_close(selected({2}), 7.0f);

    auto filled = litenp::ones<float>({2, 2});
    expect_close(filled({1, 1}), 1.0f);

    auto like = litenp::full_like<float>(a, 3.0f);
    assert(like.shape() == a.shape());
    expect_close(like({1, 2}), 3.0f);

    auto line = litenp::linspace<double>(0.0, 1.0, 5);
    expect_close(line({0}), 0.0);
    expect_close(line({2}), 0.5);
    expect_close(line({4}), 1.0);

    auto ident = litenp::identity<std::int32_t>(3);
    assert(reinterpret_cast<std::uintptr_t>(ident.data()) % 32 == 0);
    assert(ident({0, 0}) == 1);
    assert(ident({0, 1}) == 0);
    assert(ident({2, 2}) == 1);
}

void test_metadata_cache_invalidation() {
    auto a = litenp::zeros<float>({4});
    auto a_view = a.view();
    litenp::fill_view<float>(a_view, 1.0f);
    expect_close(litenp::sum(a), 4.0f);

    auto b = litenp::zeros<float>({10});
    auto b_tail = b.view().slice(0, 5, 10);
    litenp::fill_view<float>(b_tail, 1.0f);
    expect_close(litenp::sum(b), 5.0f);

    auto c = litenp::zeros<float>({4});
    auto c_view = c.view();
    c_view({2}) = 7.0f;
    expect_close(litenp::sum(c), 7.0f);

    auto d = litenp::zeros<float>({4});
    auto d_view = d.view();
    float* d_data = d_view.data();
    d_data[0] = 3.0f;
    expect_close(litenp::sum(d), 3.0f);

    const std::size_t n = 300000;
    std::vector<float> values(n, 1.0f);
    auto src = litenp::Array<float>::from_vector({n}, values);
    {
        auto stale = litenp::zeros<float>({n});
        expect_close(litenp::sum(stale), 0.0f);
    }
    auto copied = src;
    expect_close(litenp::sum(copied), static_cast<float>(n));
}

void test_virtual_const_index_values() {
    const auto arange = litenp::Array<float>::arange(4);
    const float& first = arange[0];
    const float& second = arange[1];
    expect_close(first, 0.0f);
    expect_close(second, 1.0f);

    const auto ident = litenp::identity<float>(2);
    const float& diag = ident[0];
    const float& off_diag = ident[1];
    expect_close(diag, 1.0f);
    expect_close(off_diag, 0.0f);
}

void test_transpose_and_permute() {
    auto a = litenp::Array<float>::from_vector({2, 3}, {1, 2, 3, 4, 5, 6});
    auto t = a.transpose();
    assert(t.shape() == (litenp::Shape{3, 2}));
    expect_close(t({0, 0}), 1.0f);
    expect_close(t({0, 1}), 4.0f);
    expect_close(t({2, 1}), 6.0f);

    auto c = litenp::as_contiguous<float>(t);
    assert(c.shape() == (litenp::Shape{3, 2}));
    expect_close(c({0, 0}), 1.0f);
    expect_close(c({0, 1}), 4.0f);
    expect_close(c({1, 0}), 2.0f);
    expect_close(c({1, 1}), 5.0f);
    expect_close(c({2, 0}), 3.0f);
    expect_close(c({2, 1}), 6.0f);

    auto avx_source = litenp::Array<float>::arange(16 * 16);
    auto avx_matrix = avx_source.reshape({16, 16});
    auto avx_copy = litenp::as_contiguous<float>(avx_matrix.transpose());
    assert(avx_copy.shape() == (litenp::Shape{16, 16}));
    for (std::size_t r = 0; r < 16; ++r) {
        for (std::size_t col = 0; col < 16; ++col) {
            expect_close(avx_copy({r, col}), static_cast<float>(col * 16 + r));
        }
    }

    auto sliced_copy = litenp::as_contiguous<float>(a.view().slice(1, 0, 3, 2));
    assert(sliced_copy.shape() == (litenp::Shape{2, 2}));
    expect_close(sliced_copy({0, 0}), 1.0f);
    expect_close(sliced_copy({0, 1}), 3.0f);
    expect_close(sliced_copy({1, 0}), 4.0f);
    expect_close(sliced_copy({1, 1}), 6.0f);

    auto p = litenp::permute(a, {1, 0});
    assert(p.shape() == (litenp::Shape{3, 2}));
    expect_close(p({0, 1}), 4.0f);
    expect_close(p({2, 0}), 3.0f);

    auto t2 = litenp::transpose(a);
    expect_close(t2({1, 0}), 2.0f);
}

void test_shape_and_dtype_helpers() {
    auto a = litenp::Array<float>::from_vector({1, 2, 1, 3}, {1, 2, 3, 4, 5, 6});
    auto squeezed = a.squeeze();
    assert(squeezed.shape() == (litenp::Shape{2, 3}));
    expect_close(squeezed({1, 2}), 6.0f);

    auto axis_squeezed = a.squeeze(2);
    assert(axis_squeezed.shape() == (litenp::Shape{1, 2, 3}));

    auto expanded = squeezed.unsqueeze(1);
    assert(expanded.shape() == (litenp::Shape{2, 1, 3}));
    expect_close(expanded({1, 0, 2}), 6.0f);

    auto flat = a.flatten();
    assert(flat.shape() == (litenp::Shape{6}));
    expect_close(flat({4}), 5.0f);

    auto ints = litenp::astype<std::int32_t>(a);
    static_assert(std::is_same<decltype(ints)::value_type, std::int32_t>::value, "astype should change dtype");
    assert(ints({0, 1, 0, 2}) == 6);

    auto transposed_ints = litenp::astype<std::int32_t, float>(
        litenp::Array<float>::from_vector({2, 3}, {1, 2, 3, 4, 5, 6}).transpose());
    assert(transposed_ints.shape() == (litenp::Shape{3, 2}));
    assert(transposed_ints({0, 1}) == 4);
    assert(transposed_ints({2, 0}) == 3);

    auto lhs = litenp::Array<std::int32_t>::from_vector({2, 3}, {1, 2, 3, 4, 5, 6});
    auto rhs = litenp::Array<float>::from_vector({3}, {0.5f, 1.5f, 2.5f});
    auto mixed = lhs + rhs;
    static_assert(std::is_same<decltype(mixed)::value_type, float>::value, "mixed op should use common_type");
    expect_close(mixed({0, 0}), 1.5f);
    expect_close(mixed({1, 2}), 8.5f);
}

void test_broadcast_and_scalar_ops() {
    auto a = litenp::Array<float>::from_vector({2, 3}, {1, 2, 3, 4, 5, 6});
    auto b = litenp::Array<float>::from_vector({3}, {10, 20, 30});
    auto c = litenp::add<float>(a.view(), b.view());

    assert(c.shape() == (litenp::Shape{2, 3}));
    expect_close(c({0, 0}), 11.0f);
    expect_close(c({0, 2}), 33.0f);
    expect_close(c({1, 1}), 25.0f);

    auto d = c * 2.0f;
    expect_close(d({1, 2}), 72.0f);

    litenp::Array<float> into({2, 3});
    litenp::add_into<float>(a.view(), b.view(), into.view());
    expect_close(into({1, 2}), 36.0f);
    litenp::multiply_into<float>(into.view(), 0.5f, into.view());
    expect_close(into({1, 2}), 18.0f);

    auto col = litenp::Array<float>::from_vector({2, 1}, {100.0f, 200.0f});
    auto col_sum = litenp::add<float>(a.view(), col.view());
    expect_close(col_sum({0, 2}), 103.0f);
    expect_close(col_sum({1, 0}), 204.0f);

    auto left_row = litenp::add<float>(b.view(), a.view());
    expect_close(left_row({0, 1}), 22.0f);
    expect_close(left_row({1, 2}), 36.0f);

    bool threw = false;
    try {
        auto bad = litenp::Array<float>({2});
        (void)litenp::add<float>(a.view(), bad.view());
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void test_empty_and_zero_dim() {
    litenp::Array<float> empty;
    assert(empty.shape() == (litenp::Shape{0}));
    assert(empty.size() == 0);
    assert(empty.view().size() == 0);
    expect_close(litenp::sum(empty), 0.0f);
    expect_invalid_argument([&]() { (void)litenp::mean(empty); });
    expect_invalid_argument([&]() { (void)litenp::max(empty); });

    litenp::Array<float> zero_rows({0, 3});
    auto row = litenp::Array<float>::from_vector({3}, {1.0f, 2.0f, 3.0f});
    auto row2d = litenp::Array<float>::from_vector({1, 3}, {4.0f, 5.0f, 6.0f});
    auto added = litenp::add<float>(zero_rows.view(), row.view());
    assert(added.shape() == (litenp::Shape{0, 3}));
    assert(added.size() == 0);
    auto added2d = litenp::add<float>(zero_rows.view(), row2d.view());
    assert(added2d.shape() == (litenp::Shape{0, 3}));
    assert(added2d.size() == 0);
    auto mask = litenp::greater<float>(zero_rows.view(), row.view());
    assert(mask.shape() == (litenp::Shape{0, 3}));
    assert(mask.size() == 0);

    auto empty_cols = litenp::Array<float>({2, 0});
    auto sums = litenp::sum(empty_cols, 1);
    assert(sums.shape() == (litenp::Shape{2}));
    expect_close(sums({0}), 0.0f);
    expect_close(sums({1}), 0.0f);
    expect_invalid_argument([&]() { (void)litenp::mean(empty_cols, 1); });
}

void test_unary_ops() {
    auto a = litenp::Array<float>::from_vector({2, 2}, {-1.0f, 0.0f, 4.0f, 9.0f});
    auto r = litenp::relu(a);
    expect_close(r({0, 0}), 0.0f);
    expect_close(r({1, 1}), 9.0f);

    auto ab = litenp::abs<float>(a.view());
    expect_close(ab({0, 0}), 1.0f);

    auto sq = litenp::sqrt<float>(ab.view());
    expect_close(sq({1, 0}), 2.0f);
    expect_close(sq({1, 1}), 3.0f);

    auto e = litenp::exp<float>(litenp::zeros<float>({2}).view());
    expect_close(e({0}), 1.0f);

    auto s = litenp::sigmoid<float>(litenp::zeros<float>({2}).view());
    expect_close(s({1}), 0.5f);

    litenp::Array<float> into({2, 2});
    litenp::negative_into<float>(a.view(), into.view());
    expect_close(into({0, 0}), 1.0f);
    expect_close(into({1, 1}), -9.0f);
}

void test_condition_ops() {
    auto a = litenp::Array<float>::from_vector({2, 3}, {-2, -1, 0, 1, 2, 3});
    auto b = litenp::Array<float>::from_vector({3}, {0, 0, 2});

    auto mn = litenp::minimum<float>(a.view(), b.view());
    expect_close(mn({0, 0}), -2.0f);
    expect_close(mn({1, 2}), 2.0f);

    auto mx = litenp::maximum<float>(a.view(), b.view());
    expect_close(mx({0, 1}), 0.0f);
    expect_close(mx({1, 2}), 3.0f);

    auto clipped = litenp::clip(a, -0.5f, 2.0f);
    expect_close(clipped({0, 0}), -0.5f);
    expect_close(clipped({1, 2}), 2.0f);

    auto mask = litenp::greater<float>(a.view(), b.view());
    assert(mask.shape() == (litenp::Shape{2, 3}));
    assert(mask({0, 2}) == 0);
    assert(mask({1, 1}) == 1);

    auto selected = litenp::where<float>(mask.view(), a.view(), b.view());
    expect_close(selected({0, 0}), 0.0f);
    expect_close(selected({1, 1}), 2.0f);
    expect_close(selected({1, 2}), 3.0f);

    auto scalar_mask = litenp::greater<float>(a.view(), litenp::Array<float>::from_vector({1}, {0.0f}).view());
    assert(scalar_mask({0, 0}) == 0);
    assert(scalar_mask({1, 2}) == 1);

    auto fallback = litenp::full<float>({1}, -9.0f);
    auto scalar_where = litenp::where<float>(scalar_mask.view(), a.view(), fallback.view());
    expect_close(scalar_where({0, 0}), -9.0f);
    expect_close(scalar_where({1, 2}), 3.0f);

    litenp::Array<float> where_out({2, 3});
    litenp::where_into<float>(mask.view(), a.view(), b.view(), where_out.view());
    expect_close(where_out({0, 0}), 0.0f);
    expect_close(where_out({1, 1}), 2.0f);

    auto row_choice = litenp::Array<float>::from_vector({3}, {-10.0f, -20.0f, -30.0f});
    litenp::where_into<float>(mask.view(), a.view(), row_choice.view(), where_out.view());
    expect_close(where_out({0, 0}), -10.0f);
    expect_close(where_out({1, 2}), 3.0f);

    auto byte_mask = litenp::Array<std::uint8_t>::from_vector({4}, {1, 0, 1, 0});
    auto byte_true = litenp::Array<std::uint8_t>::from_vector({4}, {9, 9, 9, 9});
    auto byte_false = litenp::Array<std::uint8_t>::from_vector({4}, {2, 3, 4, 5});
    litenp::where_into<std::uint8_t>(byte_mask.view(), byte_true.view(), byte_false.view(), byte_mask.view());
    assert(byte_mask({0}) == 9);
    assert(byte_mask({1}) == 3);
    assert(byte_mask({2}) == 9);
    assert(byte_mask({3}) == 5);

    const float nan = std::numeric_limits<float>::quiet_NaN();
    auto lhs_nan = litenp::Array<float>::from_vector({4}, {nan, 1.0f, nan, 2.0f});
    auto rhs_nan = litenp::Array<float>::from_vector({4}, {0.0f, 1.0f, nan, nan});
    auto eq_nan = litenp::equal<float>(lhs_nan.view(), rhs_nan.view());
    auto neq_nan = litenp::not_equal<float>(lhs_nan.view(), rhs_nan.view());
    auto lt_nan = litenp::less<float>(lhs_nan.view(), rhs_nan.view());
    assert(eq_nan({0}) == 0);
    assert(eq_nan({1}) == 1);
    assert(eq_nan({2}) == 0);
    assert(neq_nan({0}) == 1);
    assert(neq_nan({1}) == 0);
    assert(neq_nan({2}) == 1);
    assert(neq_nan({3}) == 1);
    assert(lt_nan({0}) == 0);
    assert(lt_nan({3}) == 0);

    test_vectorized_compare_nan_and_tail<float>();
    test_vectorized_compare_nan_and_tail<double>();
}

void test_combine_ops() {
    auto a = litenp::Array<float>::from_vector({2, 2}, {1, 2, 3, 4});
    auto b = litenp::Array<float>::from_vector({2, 2}, {5, 6, 7, 8});

    auto rows = litenp::concatenate<float>({a.view(), b.view()}, 0);
    assert(rows.shape() == (litenp::Shape{4, 2}));
    expect_close(rows({0, 1}), 2.0f);
    expect_close(rows({2, 0}), 5.0f);
    expect_close(rows({3, 1}), 8.0f);

    auto cols = litenp::concatenate<float>({a.view(), b.view()}, 1);
    assert(cols.shape() == (litenp::Shape{2, 4}));
    expect_close(cols({0, 2}), 5.0f);
    expect_close(cols({1, 3}), 8.0f);

    auto stacked0 = litenp::stack<float>({a.view(), b.view()}, 0);
    assert(stacked0.shape() == (litenp::Shape{2, 2, 2}));
    expect_close(stacked0({0, 1, 1}), 4.0f);
    expect_close(stacked0({1, 0, 0}), 5.0f);

    auto stacked2 = litenp::stack<float>({a.view(), b.view()}, 2);
    assert(stacked2.shape() == (litenp::Shape{2, 2, 2}));
    expect_close(stacked2({1, 0, 0}), 3.0f);
    expect_close(stacked2({1, 0, 1}), 7.0f);
}

void test_reductions() {
    auto a = litenp::Array<float>::from_vector({2, 3}, {1, 2, 3, 4, 5, 6});
    expect_close(litenp::sum(a), 21.0f);
    expect_close(litenp::mean(a), 3.5);
    expect_close(litenp::max(a), 6.0f);

    auto rows = litenp::sum(a, 1);
    assert(rows.shape() == (litenp::Shape{2}));
    expect_close(rows({0}), 6.0f);
    expect_close(rows({1}), 15.0f);

    auto cols = litenp::sum(a, 0);
    assert(cols.shape() == (litenp::Shape{3}));
    expect_close(cols({0}), 5.0f);
    expect_close(cols({2}), 9.0f);

    auto means = litenp::mean(a, 0);
    expect_close(means({1}), 3.5);

    auto maxes = litenp::max(a, 1);
    expect_close(maxes({0}), 3.0f);
    expect_close(maxes({1}), 6.0f);

    auto col_maxes = litenp::max(a, 0);
    expect_close(col_maxes({0}), 4.0f);
    expect_close(col_maxes({2}), 6.0f);
}

void test_matmul() {
    auto a = litenp::Array<float>::from_vector({2, 3}, {1, 2, 3, 4, 5, 6});
    auto b = litenp::Array<float>::from_vector({3, 2}, {7, 8, 9, 10, 11, 12});
    auto c = litenp::matmul(a, b);

    assert(c.shape() == (litenp::Shape{2, 2}));
    expect_close(c({0, 0}), 58.0f);
    expect_close(c({0, 1}), 64.0f);
    expect_close(c({1, 0}), 139.0f);
    expect_close(c({1, 1}), 154.0f);

    litenp::Array<float> into({2, 2});
    litenp::matmul_into<float>(a.view(), b.view(), into.view());
    expect_close(into({0, 0}), 58.0f);
    expect_close(into({1, 1}), 154.0f);

    auto uniform_a = litenp::full<float>({3, 4}, 2.0f);
    auto uniform_b = litenp::full<float>({4, 2}, 0.5f);
    auto uniform_c = litenp::matmul(uniform_a, uniform_b);
    assert(uniform_c.shape() == (litenp::Shape{3, 2}));
    expect_close(uniform_c({0, 0}), 4.0f);
    expect_close(uniform_c({2, 1}), 4.0f);

    auto nt = litenp::matmul<float>(a.transpose(), a.view());
    assert(nt.shape() == (litenp::Shape{3, 3}));
    expect_close(nt({0, 0}), 17.0f);
    expect_close(nt({2, 2}), 45.0f);

    auto id8 = litenp::identity<float>(8);
    std::vector<float> values8(64);
    for (std::size_t i = 0; i < values8.size(); ++i) {
        values8[i] = static_cast<float>(i + 1);
    }
    auto dense8 = litenp::Array<float>::from_vector({8, 8}, values8);
    litenp::Array<float> out8({8, 8});
    litenp::matmul_into<float>(id8.view(), dense8.view(), out8.view());
    expect_close(out8({0, 0}), 1.0f);
    expect_close(out8({3, 5}), 30.0f);
    expect_close(out8({7, 7}), 64.0f);

    std::vector<float> values4x16(64);
    for (std::size_t i = 0; i < values4x16.size(); ++i) {
        values4x16[i] = static_cast<float>(static_cast<int>(i % 19) - 4);
    }
    auto wide = litenp::Array<float>::from_vector({4, 16}, values4x16);
    auto id16 = litenp::identity<float>(16);
    litenp::Array<float> out4x16({4, 16});
    litenp::matmul_into<float>(wide.view(), id16.view(), out4x16.view());
    expect_close(out4x16({0, 0}), values4x16[0]);
    expect_close(out4x16({2, 7}), values4x16[2 * 16 + 7]);
    expect_close(out4x16({3, 15}), values4x16[3 * 16 + 15]);

#if defined(__AVX2__) && defined(__FMA__)
    assert(litenp::detail::select_gemm_backend<float>(768, 768, 768) == litenp::detail::GemmBackend::f32_6x16_packed_b);
    assert(litenp::detail::select_gemm_backend<float>(512, 512, 512) == litenp::detail::GemmBackend::f32_4x16_packed_b);

    std::vector<float> values7x16(112);
    for (std::size_t i = 0; i < values7x16.size(); ++i) {
        values7x16[i] = static_cast<float>(static_cast<int>(i % 23) - 7);
    }
    auto seven = litenp::Array<float>::from_vector({7, 16}, values7x16);
    litenp::Array<float> out7x16({7, 16});
    litenp::detail::matmul_f32_6x16_packed_b_kernel(seven.data(), id16.data(), out7x16.data(), 7, 16, 16);
    expect_close(out7x16({0, 0}), values7x16[0]);
    expect_close(out7x16({5, 9}), values7x16[5 * 16 + 9]);
    expect_close(out7x16({6, 15}), values7x16[6 * 16 + 15]);
#endif
}

void test_fast_path_regressions() {
    std::vector<float> values(80);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i + 1);
    }
    auto a = litenp::Array<float>::from_vector({2, 40}, values);
    expect_close(litenp::sum(a), 3240.0f);

    auto row_sums = litenp::sum(a, 1);
    assert(row_sums.shape() == (litenp::Shape{2}));
    expect_close(row_sums({0}), 820.0f);
    expect_close(row_sums({1}), 2420.0f);

    auto col_sums = litenp::sum(a, 0);
    assert(col_sums.shape() == (litenp::Shape{40}));
    expect_close(col_sums({0}), 42.0f);
    expect_close(col_sums({39}), 120.0f);

    std::vector<float> matrix_values(32);
    for (std::size_t i = 0; i < matrix_values.size(); ++i) {
        matrix_values[i] = static_cast<float>(i);
    }
    auto matrix = litenp::Array<float>::from_vector({4, 8}, matrix_values);
    auto row = litenp::Array<float>::from_vector({8}, {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f});
    litenp::Array<float> broadcast_out({4, 8});
    litenp::add_into<float>(matrix.view(), row.view(), broadcast_out.view());
    expect_close(broadcast_out({0, 0}), 0.5f);
    expect_close(broadcast_out({2, 5}), 26.5f);
    expect_close(broadcast_out({3, 7}), 38.5f);

    litenp::add_into<float>(row.view(), matrix.view(), broadcast_out.view());
    expect_close(broadcast_out({0, 1}), 2.5f);
    expect_close(broadcast_out({3, 6}), 36.5f);

    auto four_row_sums = litenp::sum<float>(matrix.view(), 1);
    assert(four_row_sums.shape() == (litenp::Shape{4}));
    expect_close(four_row_sums({0}), 28.0f);
    expect_close(four_row_sums({3}), 220.0f);
}

void test_aliasing_guards() {
    auto x = litenp::Array<float>::from_vector({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto scalar_alias = x.view().slice(0, 0, 1);
    litenp::add_into<float>(scalar_alias, x.view(), x.view());
    expect_close(x({0}), 2.0f);
    expect_close(x({1}), 3.0f);
    expect_close(x({2}), 4.0f);
    expect_close(x({3}), 5.0f);

    auto y = litenp::Array<float>::from_vector({5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto src = y.view().slice(0, 0, 4);
    auto dst = y.view().slice(0, 1, 5);
    litenp::negative_into<float>(src, dst);
    expect_close(y({0}), 1.0f);
    expect_close(y({1}), -1.0f);
    expect_close(y({2}), -2.0f);
    expect_close(y({3}), -3.0f);
    expect_close(y({4}), -4.0f);

    auto z = litenp::Array<float>::from_vector({5}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
    auto z_src = z.view().slice(0, 0, 4);
    auto z_dst = z.view().slice(0, 1, 5);
    litenp::clip_into<float>(z_src, -0.5f, 0.5f, z_dst);
    expect_close(z({0}), -2.0f);
    expect_close(z({1}), -0.5f);
    expect_close(z({2}), -0.5f);
    expect_close(z({3}), 0.0f);
    expect_close(z({4}), 0.5f);

    auto mask = litenp::Array<std::uint8_t>::from_vector({4}, {1, 0, 1, 0});
    auto w = litenp::Array<float>::from_vector({5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto y_vals = litenp::Array<float>::from_vector({4}, {10.0f, 20.0f, 30.0f, 40.0f});
    litenp::where_into<float>(mask.view(), w.view().slice(0, 0, 4), y_vals.view(), w.view().slice(0, 1, 5));
    expect_close(w({0}), 1.0f);
    expect_close(w({1}), 1.0f);
    expect_close(w({2}), 20.0f);
    expect_close(w({3}), 3.0f);
    expect_close(w({4}), 40.0f);

    auto y_overlap = litenp::Array<float>::from_vector({5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto x_vals = litenp::Array<float>::from_vector({4}, {10.0f, 20.0f, 30.0f, 40.0f});
    litenp::where_into<float>(mask.view(), x_vals.view(), y_overlap.view().slice(0, 0, 4), y_overlap.view().slice(0, 1, 5));
    expect_close(y_overlap({0}), 1.0f);
    expect_close(y_overlap({1}), 10.0f);
    expect_close(y_overlap({2}), 2.0f);
    expect_close(y_overlap({3}), 30.0f);
    expect_close(y_overlap({4}), 4.0f);

    auto m = litenp::Array<float>::from_vector({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto id = litenp::identity<float>(2);
    litenp::matmul_into<float>(m.view(), id.view(), m.view());
    expect_close(m({0, 0}), 1.0f);
    expect_close(m({0, 1}), 2.0f);
    expect_close(m({1, 0}), 3.0f);
    expect_close(m({1, 1}), 4.0f);
}

void test_int32() {
    auto a = litenp::Array<std::int32_t>::from_vector({4}, {1, 2, 3, 4});
    auto b = a + a;
    assert(b({0}) == 2);
    assert(b({3}) == 8);
    assert(litenp::sum(a) == 10);
}

void test_api_overload_consistency() {
    auto a = litenp::Array<float>::from_vector({3}, {1.0f, 2.0f, 4.0f});
    auto b = 2.0f + a;
    auto c = 10.0f - a;
    auto d = 3.0f * a;
    auto e = 8.0f / a;
    expect_close(b({2}), 6.0f);
    expect_close(c({0}), 9.0f);
    expect_close(d({1}), 6.0f);
    expect_close(e({2}), 2.0f);

    auto other = litenp::Array<float>::from_vector({3}, {1.0f, 3.0f, 4.0f});
    auto lt = litenp::less(a, other);
    auto le = litenp::less_equal(a, other);
    auto ge = litenp::greater_equal(a, other);
    auto eq = litenp::equal(a, other);
    auto ne = litenp::not_equal(a, other);
    assert(lt({1}) == 1);
    assert(le({0}) == 1);
    assert(ge({2}) == 1);
    assert(eq({0}) == 1);
    assert(ne({1}) == 1);
}

}  // namespace

int main() {
    test_array_shape_and_view();
    test_metadata_cache_invalidation();
    test_virtual_const_index_values();
    test_transpose_and_permute();
    test_shape_and_dtype_helpers();
    test_broadcast_and_scalar_ops();
    test_empty_and_zero_dim();
    test_unary_ops();
    test_condition_ops();
    test_combine_ops();
    test_reductions();
    test_matmul();
    test_fast_path_regressions();
    test_aliasing_guards();
    test_int32();
    test_api_overload_consistency();
    std::cout << "litenp tests passed\n";
    return 0;
}
