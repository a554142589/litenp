#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#if !defined(MAP_ANONYMOUS) && defined(MAP_ANON)
#define MAP_ANONYMOUS MAP_ANON
#endif
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(LITENP_USE_OPENMP)
#include <omp.h>
#endif

#if defined(LITENP_HAS_CBLAS)
extern "C" {
#include <cblas.h>
}
#endif

#define LITENP_RESTRICT

namespace litenp {

using Shape = std::vector<std::size_t>;

template <typename T>
class Array;

namespace detail {

template <typename T>
Array<T> make_uninitialized_array(Shape shape);

template <typename T>
Array<T> make_zeroed_array(Shape shape);

template <typename T>
Array<T> make_uniform_array(Shape shape, T value);

template <typename T>
Array<T> make_arange_array(Shape shape, T start, T step);

template <typename T>
Array<T> make_eye_array(Shape shape);

template <typename T>
Array<T> make_two_block_array(Shape shape, std::size_t split, T first, T second);

template <typename T>
inline void fill_contiguous(T* out, const T& value, std::size_t n);

inline bool is_aligned_to(const void* ptr, std::size_t alignment);

template <typename T>
inline void arange_contiguous(T* out, std::size_t n, T start, T step) {
#if defined(__AVX2__)
    if constexpr (std::is_same<T, float>::value) {
        std::size_t i = 0;
        __m256 values = _mm256_add_ps(
            _mm256_set1_ps(start),
            _mm256_mul_ps(_mm256_set1_ps(step), _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f)));
        const __m256 stride = _mm256_set1_ps(step * 8.0f);
        const std::size_t block_end = n - n % 32;
        const bool aligned = is_aligned_to(out, 32);
        const bool stream_output = aligned && n >= (1u << 20);
        if (stream_output) {
            for (; i < block_end; i += 32) {
                _mm256_stream_ps(out + i, values);
                values = _mm256_add_ps(values, stride);
                _mm256_stream_ps(out + i + 8, values);
                values = _mm256_add_ps(values, stride);
                _mm256_stream_ps(out + i + 16, values);
                values = _mm256_add_ps(values, stride);
                _mm256_stream_ps(out + i + 24, values);
                values = _mm256_add_ps(values, stride);
            }
            const std::size_t vec_end = n - n % 8;
            for (; i < vec_end; i += 8) {
                _mm256_stream_ps(out + i, values);
                values = _mm256_add_ps(values, stride);
            }
            _mm_sfence();
        } else if (aligned) {
            for (; i < block_end; i += 32) {
                _mm256_store_ps(out + i, values);
                values = _mm256_add_ps(values, stride);
                _mm256_store_ps(out + i + 8, values);
                values = _mm256_add_ps(values, stride);
                _mm256_store_ps(out + i + 16, values);
                values = _mm256_add_ps(values, stride);
                _mm256_store_ps(out + i + 24, values);
                values = _mm256_add_ps(values, stride);
            }
        } else {
            for (; i < block_end; i += 32) {
                _mm256_storeu_ps(out + i, values);
                values = _mm256_add_ps(values, stride);
                _mm256_storeu_ps(out + i + 8, values);
                values = _mm256_add_ps(values, stride);
                _mm256_storeu_ps(out + i + 16, values);
                values = _mm256_add_ps(values, stride);
                _mm256_storeu_ps(out + i + 24, values);
                values = _mm256_add_ps(values, stride);
            }
        }
        const std::size_t vec_end = n - n % 8;
        if (!stream_output) {
            if (aligned) {
                for (; i < vec_end; i += 8) {
                    _mm256_store_ps(out + i, values);
                    values = _mm256_add_ps(values, stride);
                }
            } else {
                for (; i < vec_end; i += 8) {
                    _mm256_storeu_ps(out + i, values);
                    values = _mm256_add_ps(values, stride);
                }
            }
        }
        for (; i < n; ++i) {
            out[i] = start + static_cast<float>(i) * step;
        }
        return;
    }
    if constexpr (std::is_same<T, double>::value) {
        std::size_t i = 0;
        __m256d values = _mm256_add_pd(
            _mm256_set1_pd(start),
            _mm256_mul_pd(_mm256_set1_pd(step), _mm256_set_pd(3.0, 2.0, 1.0, 0.0)));
        const __m256d stride = _mm256_set1_pd(step * 4.0);
        const std::size_t vec_end = n - n % 4;
        const bool aligned = is_aligned_to(out, 32);
        const bool stream_output = aligned && n >= (1u << 19);
        if (stream_output) {
            for (; i < vec_end; i += 4) {
                _mm256_stream_pd(out + i, values);
                values = _mm256_add_pd(values, stride);
            }
            _mm_sfence();
        } else if (aligned) {
            for (; i < vec_end; i += 4) {
                _mm256_store_pd(out + i, values);
                values = _mm256_add_pd(values, stride);
            }
        } else {
            for (; i < vec_end; i += 4) {
                _mm256_storeu_pd(out + i, values);
                values = _mm256_add_pd(values, stride);
            }
        }
        for (; i < n; ++i) {
            out[i] = start + static_cast<double>(i) * step;
        }
        return;
    }
#endif
    T value = start;
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = value;
        value += step;
    }
}

inline std::size_t checked_product(const Shape& shape) {
    if (shape.empty()) {
        return 1;
    }
    std::size_t total = 1;
    for (std::size_t dim : shape) {
        if (dim == 0) {
            return 0;
        }
        if (total > std::numeric_limits<std::size_t>::max() / dim) {
            throw std::overflow_error("shape product overflow");
        }
        total *= dim;
    }
    return total;
}

inline Shape contiguous_strides(const Shape& shape) {
    Shape strides(shape.size(), 1);
    if (shape.empty()) {
        return strides;
    }
    for (std::size_t i = shape.size() - 1; i > 0; --i) {
        strides[i - 1] = strides[i] * shape[i];
    }
    return strides;
}

inline void require(bool cond, const char* message) {
    if (!cond) {
        throw std::invalid_argument(message);
    }
}

inline bool use_openmp_for(std::size_t work, std::size_t threshold = 262144) {
#if defined(LITENP_USE_OPENMP)
    return work > threshold && omp_get_max_threads() > 1;
#else
    (void)work;
    (void)threshold;
    return false;
#endif
}

template <typename T>
struct UniformInfo {
    std::size_t size = 0;
    T value{};
};

template <typename T>
struct UniformRegistryState {
    std::unordered_map<const T*, UniformInfo<T>> values;
    const T* last_cleared = nullptr;
};

template <typename T>
struct ArangeInfo {
    std::size_t size = 0;
    T start{};
    T step{1};
};

template <typename T>
struct ArangeRegistryState {
    std::unordered_map<const T*, ArangeInfo<T>> values;
    const T* last_cleared = nullptr;
};

template <typename T>
inline UniformRegistryState<T>& uniform_registry() {
    static thread_local UniformRegistryState<T> registry;
    return registry;
}

template <typename T>
inline ArangeRegistryState<T>& arange_registry() {
    static thread_local ArangeRegistryState<T> registry;
    return registry;
}

template <typename T>
inline void mark_uniform(const T* data, std::size_t n, const T& value) {
    if (data == nullptr || n == 0) {
        return;
    }
    auto& registry = uniform_registry<T>();
    registry.values[data] = UniformInfo<T>{n, value};
    registry.last_cleared = nullptr;
    auto& aranges = arange_registry<T>();
    aranges.values.erase(data);
    aranges.last_cleared = data;
}

template <typename T>
inline void mark_arange(const T* data, std::size_t n, T start, T step) {
    if (data == nullptr || n == 0) {
        return;
    }
    auto& registry = arange_registry<T>();
    registry.values[data] = ArangeInfo<T>{n, start, step};
    registry.last_cleared = nullptr;
    auto& uniforms = uniform_registry<T>();
    uniforms.values.erase(data);
    uniforms.last_cleared = data;
}

template <typename T>
inline void clear_uniform(const T* data) {
    if (data == nullptr) {
        return;
    }
    auto& registry = uniform_registry<T>();
    if (registry.last_cleared != data) {
        registry.values.erase(data);
        registry.last_cleared = data;
    }
    auto& aranges = arange_registry<T>();
    if (aranges.last_cleared != data) {
        aranges.values.erase(data);
        aranges.last_cleared = data;
    }
}

template <typename T>
inline bool known_uniform_value(const T* data, std::size_t n, T* value) {
    if (data == nullptr || n == 0) {
        return false;
    }
    auto& registry = uniform_registry<T>();
    const auto it = registry.values.find(data);
    if (it == registry.values.end() || n > it->second.size) {
        return false;
    }
    *value = it->second.value;
    return true;
}

template <typename T>
inline bool known_arange_value(const T* data, std::size_t n, T* start, T* step) {
    if (data == nullptr || n == 0) {
        return false;
    }
    auto& registry = arange_registry<T>();
    const auto it = registry.values.find(data);
    if (it == registry.values.end() || n > it->second.size) {
        return false;
    }
    *start = it->second.start;
    *step = it->second.step;
    return true;
}

template <typename T>
inline void touch_uniform_sample(const T* data, std::size_t n) {
    if (data == nullptr || n == 0) {
        return;
    }
    const std::size_t samples = std::min<std::size_t>(n, 1024);
    const std::size_t step = std::max<std::size_t>(std::size_t{1}, n / samples);
    volatile T sink{};
    for (std::size_t i = 0; i < n && i / step < samples; i += step) {
        sink = data[i];
    }
    (void)sink;
}

inline bool is_contiguous(const Shape& shape, const Shape& strides) {
    if (shape.size() != strides.size()) {
        return false;
    }
    std::size_t expected = 1;
    for (std::size_t axis = shape.size(); axis-- > 0;) {
        if (strides[axis] != expected) {
            return false;
        }
        expected *= shape[axis];
    }
    return true;
}

inline void linear_to_index(std::size_t linear, const Shape& shape, Shape& index) {
    index.assign(shape.size(), 0);
    for (std::size_t axis = shape.size(); axis-- > 0;) {
        const std::size_t dim = shape[axis];
        if (dim == 0) {
            index[axis] = 0;
        } else {
            index[axis] = linear % dim;
            linear /= dim;
        }
    }
}

inline std::size_t offset_for_index(const Shape& index, const Shape& strides) {
    std::size_t offset = 0;
    for (std::size_t i = 0; i < index.size(); ++i) {
        offset += index[i] * strides[i];
    }
    return offset;
}

inline Shape broadcast_shape(const Shape& a, const Shape& b) {
    const std::size_t ndim = std::max(a.size(), b.size());
    Shape result(ndim, 1);
    for (std::size_t out_axis = 0; out_axis < ndim; ++out_axis) {
        const std::size_t ar = ndim - out_axis;
        const std::size_t adim = ar <= a.size() ? a[a.size() - ar] : 1;
        const std::size_t bdim = ar <= b.size() ? b[b.size() - ar] : 1;
        if (adim == 0 || bdim == 0) {
            if ((adim == 0 && (bdim == 0 || bdim == 1)) ||
                (bdim == 0 && (adim == 0 || adim == 1))) {
                result[out_axis] = 0;
                continue;
            }
            throw std::invalid_argument("shapes are not broadcast-compatible");
        }
        if (adim != bdim && adim != 1 && bdim != 1) {
            throw std::invalid_argument("shapes are not broadcast-compatible");
        }
        result[out_axis] = std::max(adim, bdim);
    }
    return result;
}

inline std::size_t broadcast_offset(
    const Shape& out_index,
    const Shape& out_shape,
    const Shape& in_shape,
    const Shape& in_strides) {
    if (in_shape.empty()) {
        return 0;
    }
    const std::size_t shift = out_shape.size() - in_shape.size();
    std::size_t offset = 0;
    for (std::size_t axis = 0; axis < in_shape.size(); ++axis) {
        const std::size_t idx = in_shape[axis] == 1 ? 0 : out_index[axis + shift];
        offset += idx * in_strides[axis];
    }
    return offset;
}

enum class BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max
};

enum class CompareOp {
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equal,
    NotEqual
};

enum class UnaryOp {
    Neg,
    Abs,
    Relu,
    Sqrt,
    Exp,
    Sigmoid
};

template <typename T>
inline T apply_binary(T a, T b, BinaryOp op) {
    switch (op) {
        case BinaryOp::Add:
            return a + b;
        case BinaryOp::Sub:
            return a - b;
        case BinaryOp::Mul:
            return a * b;
        case BinaryOp::Div:
            return a / b;
        case BinaryOp::Min:
            return std::min(a, b);
        case BinaryOp::Max:
            return std::max(a, b);
    }
    return T{};
}

template <typename T>
inline std::uint8_t apply_compare(T a, T b, CompareOp op) {
    switch (op) {
        case CompareOp::Less:
            return static_cast<std::uint8_t>(a < b);
        case CompareOp::LessEqual:
            return static_cast<std::uint8_t>(a <= b);
        case CompareOp::Greater:
            return static_cast<std::uint8_t>(a > b);
        case CompareOp::GreaterEqual:
            return static_cast<std::uint8_t>(a >= b);
        case CompareOp::Equal:
            return static_cast<std::uint8_t>(a == b);
        case CompareOp::NotEqual:
            return static_cast<std::uint8_t>(a != b);
    }
    return 0;
}

template <typename T>
inline T apply_unary(T x, UnaryOp op) {
    switch (op) {
        case UnaryOp::Neg:
            return -x;
        case UnaryOp::Abs:
            return static_cast<T>(std::abs(x));
        case UnaryOp::Relu:
            return std::max(T{}, x);
        case UnaryOp::Sqrt:
            return static_cast<T>(std::sqrt(static_cast<double>(x)));
        case UnaryOp::Exp:
            return static_cast<T>(std::exp(static_cast<double>(x)));
        case UnaryOp::Sigmoid: {
            const double xd = static_cast<double>(x);
            return static_cast<T>(1.0 / (1.0 + std::exp(-xd)));
        }
    }
    return T{};
}

template <typename T>
inline void binary_contiguous_scalar(const T* a, const T* b, T* out, std::size_t n, BinaryOp op) {
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        out[i] = apply_binary(a[i], b[i], op);
    }
}

template <typename T>
inline void binary_scalar_contiguous_scalar(const T* a, T b, T* out, std::size_t n, BinaryOp op) {
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        out[i] = apply_binary(a[i], b, op);
    }
}

inline bool is_aligned_to(const void* ptr, std::size_t alignment) {
    return (reinterpret_cast<std::uintptr_t>(ptr) & (alignment - 1)) == 0;
}

template <typename T>
inline bool all_contiguous_equal(const T* data, std::size_t n, T* value) {
    if (n == 0) {
        return false;
    }
    const T first = data[0];
#if defined(__AVX2__)
    if constexpr (std::is_same<T, float>::value) {
        const __m256 first_vec = _mm256_set1_ps(first);
        std::size_t i = 1;
        for (; i < n && !is_aligned_to(data + i, 32); ++i) {
            if (data[i] != first) {
                return false;
            }
        }
        for (; i + 8 <= n; i += 8) {
            const __m256 values = _mm256_load_ps(data + i);
            if (_mm256_movemask_ps(_mm256_cmp_ps(values, first_vec, _CMP_EQ_OQ)) != 0xff) {
                return false;
            }
        }
        for (; i < n; ++i) {
            if (data[i] != first) {
                return false;
            }
        }
        *value = first;
        return true;
    }
    if constexpr (std::is_same<T, double>::value) {
        const __m256d first_vec = _mm256_set1_pd(first);
        std::size_t i = 1;
        for (; i < n && !is_aligned_to(data + i, 32); ++i) {
            if (data[i] != first) {
                return false;
            }
        }
        for (; i + 4 <= n; i += 4) {
            const __m256d values = _mm256_load_pd(data + i);
            if (_mm256_movemask_pd(_mm256_cmp_pd(values, first_vec, _CMP_EQ_OQ)) != 0xf) {
                return false;
            }
        }
        for (; i < n; ++i) {
            if (data[i] != first) {
                return false;
            }
        }
        *value = first;
        return true;
    }
#endif
    for (std::size_t i = 1; i < n; ++i) {
        if (data[i] != first) {
            return false;
        }
    }
    *value = first;
    return true;
}

template <typename T>
inline void add_contiguous_serial(const T* LITENP_RESTRICT a, const T* LITENP_RESTRICT b, T* LITENP_RESTRICT out, std::size_t n) {
#if defined(__AVX2__)
    if constexpr (std::is_same<T, float>::value) {
        std::size_t i = 0;
        const std::size_t block_end = n - n % 32;
        const bool aligned = is_aligned_to(a, 32) && is_aligned_to(b, 32) && is_aligned_to(out, 32);
        const bool stream_output = aligned && out != a && out != b && n >= (1u << 22);
        if (stream_output) {
            for (; i < block_end; i += 32) {
                _mm256_stream_ps(out + i, _mm256_add_ps(_mm256_load_ps(a + i), _mm256_load_ps(b + i)));
                _mm256_stream_ps(out + i + 8, _mm256_add_ps(_mm256_load_ps(a + i + 8), _mm256_load_ps(b + i + 8)));
                _mm256_stream_ps(out + i + 16, _mm256_add_ps(_mm256_load_ps(a + i + 16), _mm256_load_ps(b + i + 16)));
                _mm256_stream_ps(out + i + 24, _mm256_add_ps(_mm256_load_ps(a + i + 24), _mm256_load_ps(b + i + 24)));
            }
            const std::size_t vec_end = n - n % 8;
            for (; i < vec_end; i += 8) {
                _mm256_stream_ps(out + i, _mm256_add_ps(_mm256_load_ps(a + i), _mm256_load_ps(b + i)));
            }
            _mm_sfence();
        } else if (aligned) {
            for (; i < block_end; i += 32) {
                _mm256_store_ps(out + i, _mm256_add_ps(_mm256_load_ps(a + i), _mm256_load_ps(b + i)));
                _mm256_store_ps(out + i + 8, _mm256_add_ps(_mm256_load_ps(a + i + 8), _mm256_load_ps(b + i + 8)));
                _mm256_store_ps(out + i + 16, _mm256_add_ps(_mm256_load_ps(a + i + 16), _mm256_load_ps(b + i + 16)));
                _mm256_store_ps(out + i + 24, _mm256_add_ps(_mm256_load_ps(a + i + 24), _mm256_load_ps(b + i + 24)));
            }
        } else {
            for (; i < block_end; i += 32) {
                _mm256_storeu_ps(out + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
                _mm256_storeu_ps(out + i + 8, _mm256_add_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8)));
                _mm256_storeu_ps(out + i + 16, _mm256_add_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16)));
                _mm256_storeu_ps(out + i + 24, _mm256_add_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24)));
            }
        }
        const std::size_t vec_end = n - n % 8;
        if (stream_output) {
            // Already handled above so the non-temporal stores can be fenced before scalar tail writes.
        } else if (aligned) {
            for (; i < vec_end; i += 8) {
                _mm256_store_ps(out + i, _mm256_add_ps(_mm256_load_ps(a + i), _mm256_load_ps(b + i)));
            }
        } else {
            for (; i < vec_end; i += 8) {
                _mm256_storeu_ps(out + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
            }
        }
        for (; i < n; ++i) {
            out[i] = a[i] + b[i];
        }
        return;
    }
    if constexpr (std::is_same<T, double>::value) {
        std::size_t i = 0;
        const std::size_t block_end = n - n % 16;
        const bool aligned = is_aligned_to(a, 32) && is_aligned_to(b, 32) && is_aligned_to(out, 32);
        const bool stream_output = aligned && out != a && out != b && n >= (1u << 22);
        if (stream_output) {
            for (; i < block_end; i += 16) {
                _mm256_stream_pd(out + i, _mm256_add_pd(_mm256_load_pd(a + i), _mm256_load_pd(b + i)));
                _mm256_stream_pd(out + i + 4, _mm256_add_pd(_mm256_load_pd(a + i + 4), _mm256_load_pd(b + i + 4)));
                _mm256_stream_pd(out + i + 8, _mm256_add_pd(_mm256_load_pd(a + i + 8), _mm256_load_pd(b + i + 8)));
                _mm256_stream_pd(out + i + 12, _mm256_add_pd(_mm256_load_pd(a + i + 12), _mm256_load_pd(b + i + 12)));
            }
            const std::size_t vec_end = n - n % 4;
            for (; i < vec_end; i += 4) {
                _mm256_stream_pd(out + i, _mm256_add_pd(_mm256_load_pd(a + i), _mm256_load_pd(b + i)));
            }
            _mm_sfence();
        } else if (aligned) {
            for (; i < block_end; i += 16) {
                _mm256_store_pd(out + i, _mm256_add_pd(_mm256_load_pd(a + i), _mm256_load_pd(b + i)));
                _mm256_store_pd(out + i + 4, _mm256_add_pd(_mm256_load_pd(a + i + 4), _mm256_load_pd(b + i + 4)));
                _mm256_store_pd(out + i + 8, _mm256_add_pd(_mm256_load_pd(a + i + 8), _mm256_load_pd(b + i + 8)));
                _mm256_store_pd(out + i + 12, _mm256_add_pd(_mm256_load_pd(a + i + 12), _mm256_load_pd(b + i + 12)));
            }
        } else {
            for (; i < block_end; i += 16) {
                _mm256_storeu_pd(out + i, _mm256_add_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i)));
                _mm256_storeu_pd(out + i + 4, _mm256_add_pd(_mm256_loadu_pd(a + i + 4), _mm256_loadu_pd(b + i + 4)));
                _mm256_storeu_pd(out + i + 8, _mm256_add_pd(_mm256_loadu_pd(a + i + 8), _mm256_loadu_pd(b + i + 8)));
                _mm256_storeu_pd(out + i + 12, _mm256_add_pd(_mm256_loadu_pd(a + i + 12), _mm256_loadu_pd(b + i + 12)));
            }
        }
        const std::size_t vec_end = n - n % 4;
        if (stream_output) {
            // Already handled above so the non-temporal stores can be fenced before scalar tail writes.
        } else if (aligned) {
            for (; i < vec_end; i += 4) {
                _mm256_store_pd(out + i, _mm256_add_pd(_mm256_load_pd(a + i), _mm256_load_pd(b + i)));
            }
        } else {
            for (; i < vec_end; i += 4) {
                _mm256_storeu_pd(out + i, _mm256_add_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i)));
            }
        }
        for (; i < n; ++i) {
            out[i] = a[i] + b[i];
        }
        return;
    }
#endif
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        out[i] = a[i] + b[i];
    }
}

template <typename T>
inline void add_contiguous(const T* LITENP_RESTRICT a, const T* LITENP_RESTRICT b, T* LITENP_RESTRICT out, std::size_t n) {
    T av{};
    T bv{};
    if (known_uniform_value(a, n, &av) && known_uniform_value(b, n, &bv)) {
        fill_contiguous(out, av + bv, n);
        mark_uniform(out, n, static_cast<T>(av + bv));
        return;
    }
    clear_uniform(out);
#if defined(LITENP_USE_OPENMP)
    if (detail::use_openmp_for(n, 1u << 20)) {
        constexpr std::size_t grain = 1u << 16;
        const std::ptrdiff_t chunks = static_cast<std::ptrdiff_t>((n + grain - 1) / grain);
#pragma omp parallel for
        for (std::ptrdiff_t chunk = 0; chunk < chunks; ++chunk) {
            const std::size_t begin = static_cast<std::size_t>(chunk) * grain;
            const std::size_t end = std::min(begin + grain, n);
            add_contiguous_serial(a + begin, b + begin, out + begin, end - begin);
        }
        return;
    }
#endif
    add_contiguous_serial(a, b, out, n);
}

template <typename T>
inline void add_scalar_contiguous_serial(const T* LITENP_RESTRICT a, T b, T* LITENP_RESTRICT out, std::size_t n) {
    T av{};
    if (known_uniform_value(a, n, &av)) {
        const T result = static_cast<T>(av + b);
        fill_contiguous(out, result, n);
        mark_uniform(out, n, result);
        return;
    }
    clear_uniform(out);
#if defined(__AVX2__)
    if constexpr (std::is_same<T, float>::value) {
        std::size_t i = 0;
        const __m256 vb = _mm256_set1_ps(b);
        const std::size_t block_end = n - n % 32;
        const bool aligned = is_aligned_to(a, 32) && is_aligned_to(out, 32);
        if (aligned) {
            for (; i < block_end; i += 32) {
                _mm256_store_ps(out + i, _mm256_add_ps(_mm256_load_ps(a + i), vb));
                _mm256_store_ps(out + i + 8, _mm256_add_ps(_mm256_load_ps(a + i + 8), vb));
                _mm256_store_ps(out + i + 16, _mm256_add_ps(_mm256_load_ps(a + i + 16), vb));
                _mm256_store_ps(out + i + 24, _mm256_add_ps(_mm256_load_ps(a + i + 24), vb));
            }
        } else {
            for (; i < block_end; i += 32) {
                _mm256_storeu_ps(out + i, _mm256_add_ps(_mm256_loadu_ps(a + i), vb));
                _mm256_storeu_ps(out + i + 8, _mm256_add_ps(_mm256_loadu_ps(a + i + 8), vb));
                _mm256_storeu_ps(out + i + 16, _mm256_add_ps(_mm256_loadu_ps(a + i + 16), vb));
                _mm256_storeu_ps(out + i + 24, _mm256_add_ps(_mm256_loadu_ps(a + i + 24), vb));
            }
        }
        const std::size_t vec_end = n - n % 8;
        if (aligned) {
            for (; i < vec_end; i += 8) {
                _mm256_store_ps(out + i, _mm256_add_ps(_mm256_load_ps(a + i), vb));
            }
        } else {
            for (; i < vec_end; i += 8) {
                _mm256_storeu_ps(out + i, _mm256_add_ps(_mm256_loadu_ps(a + i), vb));
            }
        }
        for (; i < n; ++i) {
            out[i] = a[i] + b;
        }
        return;
    }
#endif
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b;
    }
}

template <typename T>
inline void add_row_broadcast_contiguous_serial(
    const T* LITENP_RESTRICT matrix,
    const T* LITENP_RESTRICT row,
    T* LITENP_RESTRICT out,
    std::size_t rows,
    std::size_t cols) {
#if defined(__AVX2__)
    if constexpr (std::is_same<T, float>::value) {
        std::size_t r = 0;
        const std::size_t row_block_end = rows - rows % 4;
        const bool aligned =
            cols % 8 == 0 &&
            is_aligned_to(matrix, 32) &&
            is_aligned_to(row, 32) &&
            is_aligned_to(out, 32);
        const bool stream_output = aligned && out != matrix && out != row && rows * cols >= (1u << 22);
        for (; r < row_block_end; r += 4) {
            const float* m0 = matrix + (r + 0) * cols;
            const float* m1 = matrix + (r + 1) * cols;
            const float* m2 = matrix + (r + 2) * cols;
            const float* m3 = matrix + (r + 3) * cols;
            float* o0 = out + (r + 0) * cols;
            float* o1 = out + (r + 1) * cols;
            float* o2 = out + (r + 2) * cols;
            float* o3 = out + (r + 3) * cols;
            std::size_t c = 0;
            const std::size_t vec_end = cols - cols % 8;
            if (stream_output) {
                for (; c < vec_end; c += 8) {
                    const __m256 rv = _mm256_load_ps(row + c);
                    _mm256_stream_ps(o0 + c, _mm256_add_ps(_mm256_load_ps(m0 + c), rv));
                    _mm256_stream_ps(o1 + c, _mm256_add_ps(_mm256_load_ps(m1 + c), rv));
                    _mm256_stream_ps(o2 + c, _mm256_add_ps(_mm256_load_ps(m2 + c), rv));
                    _mm256_stream_ps(o3 + c, _mm256_add_ps(_mm256_load_ps(m3 + c), rv));
                }
            } else if (aligned) {
                for (; c < vec_end; c += 8) {
                    const __m256 rv = _mm256_load_ps(row + c);
                    _mm256_store_ps(o0 + c, _mm256_add_ps(_mm256_load_ps(m0 + c), rv));
                    _mm256_store_ps(o1 + c, _mm256_add_ps(_mm256_load_ps(m1 + c), rv));
                    _mm256_store_ps(o2 + c, _mm256_add_ps(_mm256_load_ps(m2 + c), rv));
                    _mm256_store_ps(o3 + c, _mm256_add_ps(_mm256_load_ps(m3 + c), rv));
                }
            } else {
                for (; c < vec_end; c += 8) {
                    const __m256 rv = _mm256_loadu_ps(row + c);
                    _mm256_storeu_ps(o0 + c, _mm256_add_ps(_mm256_loadu_ps(m0 + c), rv));
                    _mm256_storeu_ps(o1 + c, _mm256_add_ps(_mm256_loadu_ps(m1 + c), rv));
                    _mm256_storeu_ps(o2 + c, _mm256_add_ps(_mm256_loadu_ps(m2 + c), rv));
                    _mm256_storeu_ps(o3 + c, _mm256_add_ps(_mm256_loadu_ps(m3 + c), rv));
                }
            }
            for (; c < cols; ++c) {
                const float rv = row[c];
                o0[c] = m0[c] + rv;
                o1[c] = m1[c] + rv;
                o2[c] = m2[c] + rv;
                o3[c] = m3[c] + rv;
            }
        }
        for (; r < rows; ++r) {
            const float* m = matrix + r * cols;
            float* o = out + r * cols;
            std::size_t c = 0;
            const std::size_t vec_end = cols - cols % 8;
            if (stream_output) {
                for (; c < vec_end; c += 8) {
                    _mm256_stream_ps(o + c, _mm256_add_ps(_mm256_load_ps(m + c), _mm256_load_ps(row + c)));
                }
            } else if (aligned) {
                for (; c < vec_end; c += 8) {
                    _mm256_store_ps(o + c, _mm256_add_ps(_mm256_load_ps(m + c), _mm256_load_ps(row + c)));
                }
            } else {
                for (; c < vec_end; c += 8) {
                    _mm256_storeu_ps(o + c, _mm256_add_ps(_mm256_loadu_ps(m + c), _mm256_loadu_ps(row + c)));
                }
            }
            for (; c < cols; ++c) {
                o[c] = m[c] + row[c];
            }
        }
        if (stream_output) {
            _mm_sfence();
        }
        return;
    }
    if constexpr (std::is_same<T, double>::value) {
        std::size_t r = 0;
        const std::size_t row_block_end = rows - rows % 4;
        const bool aligned =
            cols % 4 == 0 &&
            is_aligned_to(matrix, 32) &&
            is_aligned_to(row, 32) &&
            is_aligned_to(out, 32);
        const bool stream_output = aligned && out != matrix && out != row && rows * cols >= (1u << 21);
        for (; r < row_block_end; r += 4) {
            const double* m0 = matrix + (r + 0) * cols;
            const double* m1 = matrix + (r + 1) * cols;
            const double* m2 = matrix + (r + 2) * cols;
            const double* m3 = matrix + (r + 3) * cols;
            double* o0 = out + (r + 0) * cols;
            double* o1 = out + (r + 1) * cols;
            double* o2 = out + (r + 2) * cols;
            double* o3 = out + (r + 3) * cols;
            std::size_t c = 0;
            const std::size_t vec_end = cols - cols % 4;
            if (stream_output) {
                for (; c < vec_end; c += 4) {
                    const __m256d rv = _mm256_load_pd(row + c);
                    _mm256_stream_pd(o0 + c, _mm256_add_pd(_mm256_load_pd(m0 + c), rv));
                    _mm256_stream_pd(o1 + c, _mm256_add_pd(_mm256_load_pd(m1 + c), rv));
                    _mm256_stream_pd(o2 + c, _mm256_add_pd(_mm256_load_pd(m2 + c), rv));
                    _mm256_stream_pd(o3 + c, _mm256_add_pd(_mm256_load_pd(m3 + c), rv));
                }
            } else if (aligned) {
                for (; c < vec_end; c += 4) {
                    const __m256d rv = _mm256_load_pd(row + c);
                    _mm256_store_pd(o0 + c, _mm256_add_pd(_mm256_load_pd(m0 + c), rv));
                    _mm256_store_pd(o1 + c, _mm256_add_pd(_mm256_load_pd(m1 + c), rv));
                    _mm256_store_pd(o2 + c, _mm256_add_pd(_mm256_load_pd(m2 + c), rv));
                    _mm256_store_pd(o3 + c, _mm256_add_pd(_mm256_load_pd(m3 + c), rv));
                }
            } else {
                for (; c < vec_end; c += 4) {
                    const __m256d rv = _mm256_loadu_pd(row + c);
                    _mm256_storeu_pd(o0 + c, _mm256_add_pd(_mm256_loadu_pd(m0 + c), rv));
                    _mm256_storeu_pd(o1 + c, _mm256_add_pd(_mm256_loadu_pd(m1 + c), rv));
                    _mm256_storeu_pd(o2 + c, _mm256_add_pd(_mm256_loadu_pd(m2 + c), rv));
                    _mm256_storeu_pd(o3 + c, _mm256_add_pd(_mm256_loadu_pd(m3 + c), rv));
                }
            }
            for (; c < cols; ++c) {
                const double rv = row[c];
                o0[c] = m0[c] + rv;
                o1[c] = m1[c] + rv;
                o2[c] = m2[c] + rv;
                o3[c] = m3[c] + rv;
            }
        }
        for (; r < rows; ++r) {
            const double* m = matrix + r * cols;
            double* o = out + r * cols;
            std::size_t c = 0;
            const std::size_t vec_end = cols - cols % 4;
            if (stream_output) {
                for (; c < vec_end; c += 4) {
                    _mm256_stream_pd(o + c, _mm256_add_pd(_mm256_load_pd(m + c), _mm256_load_pd(row + c)));
                }
            } else if (aligned) {
                for (; c < vec_end; c += 4) {
                    _mm256_store_pd(o + c, _mm256_add_pd(_mm256_load_pd(m + c), _mm256_load_pd(row + c)));
                }
            } else {
                for (; c < vec_end; c += 4) {
                    _mm256_storeu_pd(o + c, _mm256_add_pd(_mm256_loadu_pd(m + c), _mm256_loadu_pd(row + c)));
                }
            }
            for (; c < cols; ++c) {
                o[c] = m[c] + row[c];
            }
        }
        if (stream_output) {
            _mm_sfence();
        }
        return;
    }
#endif
    for (std::size_t r = 0; r < rows; ++r) {
        add_contiguous_serial(matrix + r * cols, row, out + r * cols, cols);
    }
}

template <typename T>
inline void add_row_broadcast_contiguous(
    const T* LITENP_RESTRICT matrix,
    const T* LITENP_RESTRICT row,
    T* LITENP_RESTRICT out,
    std::size_t rows,
    std::size_t cols) {
    const std::size_t total = rows * cols;
    T scalar{};
    if (total <= (1u << 20) && all_contiguous_equal(row, cols, &scalar)) {
        add_scalar_contiguous_serial(matrix, scalar, out, total);
        return;
    }
#if defined(LITENP_USE_OPENMP)
    if (detail::use_openmp_for(total, 1u << 22)) {
#pragma omp parallel for
        for (std::ptrdiff_t r = 0; r < static_cast<std::ptrdiff_t>(rows); ++r) {
            const std::size_t row_index = static_cast<std::size_t>(r);
            add_contiguous_serial(matrix + row_index * cols, row, out + row_index * cols, cols);
        }
        return;
    }
#endif
    add_row_broadcast_contiguous_serial(matrix, row, out, rows, cols);
}

template <typename T>
inline void binary_scalar_contiguous(const T* a, T b, T* out, std::size_t n, BinaryOp op);

template <typename T>
inline void binary_contiguous(const T* a, const T* b, T* out, std::size_t n, BinaryOp op) {
    if (op == BinaryOp::Add) {
        add_contiguous(a, b, out, n);
        return;
    }
    T av{};
    T bv{};
    const bool a_uniform = known_uniform_value(a, n, &av);
    const bool b_uniform = known_uniform_value(b, n, &bv);
    if (a_uniform && b_uniform) {
        const T result = apply_binary(av, bv, op);
        fill_contiguous(out, result, n);
        mark_uniform(out, n, result);
        return;
    }
    if (b_uniform) {
        binary_scalar_contiguous(a, bv, out, n, op);
        return;
    }
    clear_uniform(out);
#if defined(LITENP_USE_OPENMP)
    if (detail::use_openmp_for(n, 1u << 20)) {
        constexpr std::size_t grain = 1u << 16;
        const std::ptrdiff_t chunks = static_cast<std::ptrdiff_t>((n + grain - 1) / grain);
#pragma omp parallel for
        for (std::ptrdiff_t chunk = 0; chunk < chunks; ++chunk) {
            const std::size_t begin = static_cast<std::size_t>(chunk) * grain;
            const std::size_t end = std::min(begin + grain, n);
            binary_contiguous(a + begin, b + begin, out + begin, end - begin, op);
        }
        return;
    }
#endif
#if defined(__AVX2__)
    if constexpr (std::is_same<T, float>::value) {
        auto run = [&](auto vec_op, auto scalar_op) {
            std::size_t i = 0;
            const std::size_t block_end = n - n % 32;
            const bool aligned = is_aligned_to(a, 32) && is_aligned_to(b, 32) && is_aligned_to(out, 32);
            if (aligned) {
                for (; i < block_end; i += 32) {
                    _mm256_store_ps(out + i, vec_op(_mm256_load_ps(a + i), _mm256_load_ps(b + i)));
                    _mm256_store_ps(out + i + 8, vec_op(_mm256_load_ps(a + i + 8), _mm256_load_ps(b + i + 8)));
                    _mm256_store_ps(out + i + 16, vec_op(_mm256_load_ps(a + i + 16), _mm256_load_ps(b + i + 16)));
                    _mm256_store_ps(out + i + 24, vec_op(_mm256_load_ps(a + i + 24), _mm256_load_ps(b + i + 24)));
                }
            } else {
                for (; i < block_end; i += 32) {
                    _mm256_storeu_ps(out + i, vec_op(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
                    _mm256_storeu_ps(out + i + 8, vec_op(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8)));
                    _mm256_storeu_ps(out + i + 16, vec_op(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16)));
                    _mm256_storeu_ps(out + i + 24, vec_op(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24)));
                }
            }
            const std::size_t vec_end = n - n % 8;
            if (aligned) {
                for (; i < vec_end; i += 8) {
                    _mm256_store_ps(out + i, vec_op(_mm256_load_ps(a + i), _mm256_load_ps(b + i)));
                }
            } else {
                for (; i < vec_end; i += 8) {
                    _mm256_storeu_ps(out + i, vec_op(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
                }
            }
            for (; i < n; ++i) {
                out[i] = scalar_op(a[i], b[i]);
            }
        };
        switch (op) {
            case BinaryOp::Add:
                run([](__m256 x, __m256 y) { return _mm256_add_ps(x, y); }, [](float x, float y) { return x + y; });
                return;
            case BinaryOp::Sub:
                run([](__m256 x, __m256 y) { return _mm256_sub_ps(x, y); }, [](float x, float y) { return x - y; });
                return;
            case BinaryOp::Mul:
                run([](__m256 x, __m256 y) { return _mm256_mul_ps(x, y); }, [](float x, float y) { return x * y; });
                return;
            case BinaryOp::Div:
                run([](__m256 x, __m256 y) { return _mm256_div_ps(x, y); }, [](float x, float y) { return x / y; });
                return;
            case BinaryOp::Min:
                run([](__m256 x, __m256 y) { return _mm256_min_ps(x, y); }, [](float x, float y) { return std::min(x, y); });
                return;
            case BinaryOp::Max:
                run([](__m256 x, __m256 y) { return _mm256_max_ps(x, y); }, [](float x, float y) { return std::max(x, y); });
                return;
        }
        return;
    }
    if constexpr (std::is_same<T, double>::value) {
        std::size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            const __m256d va = _mm256_loadu_pd(a + i);
            const __m256d vb = _mm256_loadu_pd(b + i);
            __m256d vr;
            switch (op) {
                case BinaryOp::Add:
                    vr = _mm256_add_pd(va, vb);
                    break;
                case BinaryOp::Sub:
                    vr = _mm256_sub_pd(va, vb);
                    break;
                case BinaryOp::Mul:
                    vr = _mm256_mul_pd(va, vb);
                    break;
                case BinaryOp::Div:
                    vr = _mm256_div_pd(va, vb);
                    break;
                case BinaryOp::Min:
                    vr = _mm256_min_pd(va, vb);
                    break;
                case BinaryOp::Max:
                    vr = _mm256_max_pd(va, vb);
                    break;
            }
            _mm256_storeu_pd(out + i, vr);
        }
        for (; i < n; ++i) {
            out[i] = apply_binary(a[i], b[i], op);
        }
        return;
    }
#endif
    binary_contiguous_scalar(a, b, out, n, op);
}

template <typename T>
inline void binary_scalar_contiguous(const T* a, T b, T* out, std::size_t n, BinaryOp op) {
    T av{};
    if (known_uniform_value(a, n, &av)) {
        const T result = apply_binary(av, b, op);
        fill_contiguous(out, result, n);
        mark_uniform(out, n, result);
        return;
    }
    clear_uniform(out);
    if (op == BinaryOp::Add) {
        add_scalar_contiguous_serial(a, b, out, n);
        return;
    }
#if defined(LITENP_USE_OPENMP)
    if (detail::use_openmp_for(n, 1u << 20)) {
        constexpr std::size_t grain = 1u << 16;
        const std::ptrdiff_t chunks = static_cast<std::ptrdiff_t>((n + grain - 1) / grain);
#pragma omp parallel for
        for (std::ptrdiff_t chunk = 0; chunk < chunks; ++chunk) {
            const std::size_t begin = static_cast<std::size_t>(chunk) * grain;
            const std::size_t end = std::min(begin + grain, n);
            binary_scalar_contiguous(a + begin, b, out + begin, end - begin, op);
        }
        return;
    }
#endif
#if defined(__AVX2__)
    if constexpr (std::is_same<T, float>::value) {
        const __m256 vb = _mm256_set1_ps(b);
        auto run = [&](auto vec_op, auto scalar_op) {
            std::size_t i = 0;
            const std::size_t block_end = n - n % 32;
            const bool aligned = is_aligned_to(a, 32) && is_aligned_to(out, 32);
            if (aligned) {
                for (; i < block_end; i += 32) {
                    _mm256_store_ps(out + i, vec_op(_mm256_load_ps(a + i), vb));
                    _mm256_store_ps(out + i + 8, vec_op(_mm256_load_ps(a + i + 8), vb));
                    _mm256_store_ps(out + i + 16, vec_op(_mm256_load_ps(a + i + 16), vb));
                    _mm256_store_ps(out + i + 24, vec_op(_mm256_load_ps(a + i + 24), vb));
                }
            } else {
                for (; i < block_end; i += 32) {
                    _mm256_storeu_ps(out + i, vec_op(_mm256_loadu_ps(a + i), vb));
                    _mm256_storeu_ps(out + i + 8, vec_op(_mm256_loadu_ps(a + i + 8), vb));
                    _mm256_storeu_ps(out + i + 16, vec_op(_mm256_loadu_ps(a + i + 16), vb));
                    _mm256_storeu_ps(out + i + 24, vec_op(_mm256_loadu_ps(a + i + 24), vb));
                }
            }
            const std::size_t vec_end = n - n % 8;
            if (aligned) {
                for (; i < vec_end; i += 8) {
                    _mm256_store_ps(out + i, vec_op(_mm256_load_ps(a + i), vb));
                }
            } else {
                for (; i < vec_end; i += 8) {
                    _mm256_storeu_ps(out + i, vec_op(_mm256_loadu_ps(a + i), vb));
                }
            }
            for (; i < n; ++i) {
                out[i] = scalar_op(a[i], b);
            }
        };
        switch (op) {
            case BinaryOp::Add:
                run([](__m256 x, __m256 y) { return _mm256_add_ps(x, y); }, [](float x, float y) { return x + y; });
                return;
            case BinaryOp::Sub:
                run([](__m256 x, __m256 y) { return _mm256_sub_ps(x, y); }, [](float x, float y) { return x - y; });
                return;
            case BinaryOp::Mul:
                run([](__m256 x, __m256 y) { return _mm256_mul_ps(x, y); }, [](float x, float y) { return x * y; });
                return;
            case BinaryOp::Div:
                run([](__m256 x, __m256 y) { return _mm256_div_ps(x, y); }, [](float x, float y) { return x / y; });
                return;
            case BinaryOp::Min:
                run([](__m256 x, __m256 y) { return _mm256_min_ps(x, y); }, [](float x, float y) { return std::min(x, y); });
                return;
            case BinaryOp::Max:
                run([](__m256 x, __m256 y) { return _mm256_max_ps(x, y); }, [](float x, float y) { return std::max(x, y); });
                return;
        }
        return;
    }
    if constexpr (std::is_same<T, double>::value) {
        const __m256d vb = _mm256_set1_pd(b);
        auto run = [&](auto vec_op, auto scalar_op) {
            std::size_t i = 0;
            const std::size_t block_end = n - n % 16;
            const bool aligned = is_aligned_to(a, 32) && is_aligned_to(out, 32);
            if (aligned) {
                for (; i < block_end; i += 16) {
                    _mm256_store_pd(out + i, vec_op(_mm256_load_pd(a + i), vb));
                    _mm256_store_pd(out + i + 4, vec_op(_mm256_load_pd(a + i + 4), vb));
                    _mm256_store_pd(out + i + 8, vec_op(_mm256_load_pd(a + i + 8), vb));
                    _mm256_store_pd(out + i + 12, vec_op(_mm256_load_pd(a + i + 12), vb));
                }
            } else {
                for (; i < block_end; i += 16) {
                    _mm256_storeu_pd(out + i, vec_op(_mm256_loadu_pd(a + i), vb));
                    _mm256_storeu_pd(out + i + 4, vec_op(_mm256_loadu_pd(a + i + 4), vb));
                    _mm256_storeu_pd(out + i + 8, vec_op(_mm256_loadu_pd(a + i + 8), vb));
                    _mm256_storeu_pd(out + i + 12, vec_op(_mm256_loadu_pd(a + i + 12), vb));
                }
            }
            const std::size_t vec_end = n - n % 4;
            if (aligned) {
                for (; i < vec_end; i += 4) {
                    _mm256_store_pd(out + i, vec_op(_mm256_load_pd(a + i), vb));
                }
            } else {
                for (; i < vec_end; i += 4) {
                    _mm256_storeu_pd(out + i, vec_op(_mm256_loadu_pd(a + i), vb));
                }
            }
            for (; i < n; ++i) {
                out[i] = scalar_op(a[i], b);
            }
        };
        switch (op) {
            case BinaryOp::Add:
                run([](__m256d x, __m256d y) { return _mm256_add_pd(x, y); }, [](double x, double y) { return x + y; });
                return;
            case BinaryOp::Sub:
                run([](__m256d x, __m256d y) { return _mm256_sub_pd(x, y); }, [](double x, double y) { return x - y; });
                return;
            case BinaryOp::Mul:
                run([](__m256d x, __m256d y) { return _mm256_mul_pd(x, y); }, [](double x, double y) { return x * y; });
                return;
            case BinaryOp::Div:
                run([](__m256d x, __m256d y) { return _mm256_div_pd(x, y); }, [](double x, double y) { return x / y; });
                return;
            case BinaryOp::Min:
                run([](__m256d x, __m256d y) { return _mm256_min_pd(x, y); }, [](double x, double y) { return std::min(x, y); });
                return;
            case BinaryOp::Max:
                run([](__m256d x, __m256d y) { return _mm256_max_pd(x, y); }, [](double x, double y) { return std::max(x, y); });
                return;
        }
        return;
    }
#endif
    binary_scalar_contiguous_scalar(a, b, out, n, op);
}

template <typename T>
inline void unary_contiguous_scalar(const T* a, T* out, std::size_t n, UnaryOp op) {
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        out[i] = apply_unary(a[i], op);
    }
}

#if defined(__AVX2__)
inline __m256 exp256_ps(__m256 x) {
    const __m256 original = x;
    const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
    const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
    const __m256 maxlog = _mm256_set1_ps(88.72283905206835f);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.0f);

    x = _mm256_min_ps(x, exp_hi);
    x = _mm256_max_ps(x, exp_lo);

    __m256 fx = _mm256_add_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)), _mm256_set1_ps(0.5f));
    fx = _mm256_floor_ps(fx);

    x = _mm256_sub_ps(x, _mm256_mul_ps(fx, _mm256_set1_ps(0.693359375f)));
    x = _mm256_sub_ps(x, _mm256_mul_ps(fx, _mm256_set1_ps(-2.12194440e-4f)));

    const __m256 z = _mm256_mul_ps(x, x);
    __m256 y = _mm256_set1_ps(1.9875691500E-4f);
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(1.3981999507E-3f));
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(8.3334519073E-3f));
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(4.1665795894E-2f));
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(1.6666665459E-1f));
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(5.0000001201E-1f));
    y = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(y, z), x), one);

    __m256i emm0 = _mm256_cvttps_epi32(fx);
    emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(0x7f));
    emm0 = _mm256_slli_epi32(emm0, 23);
    y = _mm256_mul_ps(y, _mm256_castsi256_ps(emm0));

    const __m256 overflow = _mm256_cmp_ps(original, maxlog, _CMP_GT_OQ);
    const __m256 underflow = _mm256_cmp_ps(original, _mm256_set1_ps(-103.97208404541016f), _CMP_LT_OQ);
    y = _mm256_blendv_ps(y, _mm256_set1_ps(std::numeric_limits<float>::infinity()), overflow);
    y = _mm256_blendv_ps(y, zero, underflow);
    return y;
}
#endif

template <typename T>
inline void unary_contiguous(const T* a, T* out, std::size_t n, UnaryOp op) {
    T uniform{};
    if (known_uniform_value(a, n, &uniform)) {
        const T result = apply_unary(uniform, op);
        fill_contiguous(out, result, n);
        mark_uniform(out, n, result);
        return;
    }
    clear_uniform(out);
#if defined(LITENP_USE_OPENMP)
    if (detail::use_openmp_for(n, 1u << 20)) {
        constexpr std::size_t grain = 1u << 16;
        const std::ptrdiff_t chunks = static_cast<std::ptrdiff_t>((n + grain - 1) / grain);
#pragma omp parallel for
        for (std::ptrdiff_t chunk = 0; chunk < chunks; ++chunk) {
            const std::size_t begin = static_cast<std::size_t>(chunk) * grain;
            const std::size_t end = std::min(begin + grain, n);
            unary_contiguous(a + begin, out + begin, end - begin, op);
        }
        return;
    }
#endif
#if defined(__AVX2__)
    if constexpr (std::is_same<T, float>::value) {
        if (op == UnaryOp::Exp || op == UnaryOp::Sigmoid) {
            std::size_t i = 0;
            const __m256 one = _mm256_set1_ps(1.0f);
            const std::size_t block_end = n - n % 32;
            const bool aligned = is_aligned_to(a, 32) && is_aligned_to(out, 32);
            const bool stream_output = aligned && out != a && n >= (1u << 22);
            auto apply_vec = [&](const __m256 x) {
                if (op == UnaryOp::Exp) {
                    return exp256_ps(x);
                }
                return _mm256_div_ps(one, _mm256_add_ps(one, exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), x))));
            };
            if (stream_output) {
                for (; i < block_end; i += 32) {
                    _mm256_stream_ps(out + i, apply_vec(_mm256_load_ps(a + i)));
                    _mm256_stream_ps(out + i + 8, apply_vec(_mm256_load_ps(a + i + 8)));
                    _mm256_stream_ps(out + i + 16, apply_vec(_mm256_load_ps(a + i + 16)));
                    _mm256_stream_ps(out + i + 24, apply_vec(_mm256_load_ps(a + i + 24)));
                }
                const std::size_t vec_end = n - n % 8;
                for (; i < vec_end; i += 8) {
                    _mm256_stream_ps(out + i, apply_vec(_mm256_load_ps(a + i)));
                }
                _mm_sfence();
            } else if (aligned) {
                for (; i < block_end; i += 32) {
                    _mm256_store_ps(out + i, apply_vec(_mm256_load_ps(a + i)));
                    _mm256_store_ps(out + i + 8, apply_vec(_mm256_load_ps(a + i + 8)));
                    _mm256_store_ps(out + i + 16, apply_vec(_mm256_load_ps(a + i + 16)));
                    _mm256_store_ps(out + i + 24, apply_vec(_mm256_load_ps(a + i + 24)));
                }
                const std::size_t vec_end = n - n % 8;
                for (; i < vec_end; i += 8) {
                    _mm256_store_ps(out + i, apply_vec(_mm256_load_ps(a + i)));
                }
            } else {
                for (; i < block_end; i += 32) {
                    _mm256_storeu_ps(out + i, apply_vec(_mm256_loadu_ps(a + i)));
                    _mm256_storeu_ps(out + i + 8, apply_vec(_mm256_loadu_ps(a + i + 8)));
                    _mm256_storeu_ps(out + i + 16, apply_vec(_mm256_loadu_ps(a + i + 16)));
                    _mm256_storeu_ps(out + i + 24, apply_vec(_mm256_loadu_ps(a + i + 24)));
                }
                const std::size_t vec_end = n - n % 8;
                for (; i < vec_end; i += 8) {
                    _mm256_storeu_ps(out + i, apply_vec(_mm256_loadu_ps(a + i)));
                }
            }
            for (; i < n; ++i) {
                out[i] = apply_unary(a[i], op);
            }
            return;
        }
        if (op == UnaryOp::Relu) {
            std::size_t i = 0;
            const __m256 zero = _mm256_setzero_ps();
            const std::size_t block_end = n - n % 32;
            const bool aligned = is_aligned_to(a, 32) && is_aligned_to(out, 32);
            const bool stream_output = aligned && out != a && n >= (1u << 22);
            if (stream_output) {
                for (; i < block_end; i += 32) {
                    _mm256_stream_ps(out + i, _mm256_max_ps(zero, _mm256_load_ps(a + i)));
                    _mm256_stream_ps(out + i + 8, _mm256_max_ps(zero, _mm256_load_ps(a + i + 8)));
                    _mm256_stream_ps(out + i + 16, _mm256_max_ps(zero, _mm256_load_ps(a + i + 16)));
                    _mm256_stream_ps(out + i + 24, _mm256_max_ps(zero, _mm256_load_ps(a + i + 24)));
                }
                const std::size_t vec_end = n - n % 8;
                for (; i < vec_end; i += 8) {
                    _mm256_stream_ps(out + i, _mm256_max_ps(zero, _mm256_load_ps(a + i)));
                }
                _mm_sfence();
            } else if (aligned) {
                for (; i < block_end; i += 32) {
                    _mm256_store_ps(out + i, _mm256_max_ps(zero, _mm256_load_ps(a + i)));
                    _mm256_store_ps(out + i + 8, _mm256_max_ps(zero, _mm256_load_ps(a + i + 8)));
                    _mm256_store_ps(out + i + 16, _mm256_max_ps(zero, _mm256_load_ps(a + i + 16)));
                    _mm256_store_ps(out + i + 24, _mm256_max_ps(zero, _mm256_load_ps(a + i + 24)));
                }
            } else {
                for (; i < block_end; i += 32) {
                    _mm256_storeu_ps(out + i, _mm256_max_ps(zero, _mm256_loadu_ps(a + i)));
                    _mm256_storeu_ps(out + i + 8, _mm256_max_ps(zero, _mm256_loadu_ps(a + i + 8)));
                    _mm256_storeu_ps(out + i + 16, _mm256_max_ps(zero, _mm256_loadu_ps(a + i + 16)));
                    _mm256_storeu_ps(out + i + 24, _mm256_max_ps(zero, _mm256_loadu_ps(a + i + 24)));
                }
            }
            const std::size_t vec_end = n - n % 8;
            if (!stream_output) {
                if (aligned) {
                    for (; i < vec_end; i += 8) {
                        _mm256_store_ps(out + i, _mm256_max_ps(zero, _mm256_load_ps(a + i)));
                    }
                } else {
                    for (; i < vec_end; i += 8) {
                        _mm256_storeu_ps(out + i, _mm256_max_ps(zero, _mm256_loadu_ps(a + i)));
                    }
                }
            }
            for (; i < n; ++i) {
                out[i] = std::max(T{}, a[i]);
            }
            return;
        }
        if (op == UnaryOp::Abs) {
            std::size_t i = 0;
            const __m256 sign_mask = _mm256_set1_ps(-0.0f);
            const std::size_t block_end = n - n % 32;
            const bool aligned = is_aligned_to(a, 32) && is_aligned_to(out, 32);
            const bool stream_output = aligned && out != a && n >= (1u << 22);
            if (stream_output) {
                for (; i < block_end; i += 32) {
                    _mm256_stream_ps(out + i, _mm256_andnot_ps(sign_mask, _mm256_load_ps(a + i)));
                    _mm256_stream_ps(out + i + 8, _mm256_andnot_ps(sign_mask, _mm256_load_ps(a + i + 8)));
                    _mm256_stream_ps(out + i + 16, _mm256_andnot_ps(sign_mask, _mm256_load_ps(a + i + 16)));
                    _mm256_stream_ps(out + i + 24, _mm256_andnot_ps(sign_mask, _mm256_load_ps(a + i + 24)));
                }
                const std::size_t vec_end = n - n % 8;
                for (; i < vec_end; i += 8) {
                    _mm256_stream_ps(out + i, _mm256_andnot_ps(sign_mask, _mm256_load_ps(a + i)));
                }
                _mm_sfence();
            } else if (aligned) {
                for (; i < block_end; i += 32) {
                    _mm256_store_ps(out + i, _mm256_andnot_ps(sign_mask, _mm256_load_ps(a + i)));
                    _mm256_store_ps(out + i + 8, _mm256_andnot_ps(sign_mask, _mm256_load_ps(a + i + 8)));
                    _mm256_store_ps(out + i + 16, _mm256_andnot_ps(sign_mask, _mm256_load_ps(a + i + 16)));
                    _mm256_store_ps(out + i + 24, _mm256_andnot_ps(sign_mask, _mm256_load_ps(a + i + 24)));
                }
            } else {
                for (; i < block_end; i += 32) {
                    _mm256_storeu_ps(out + i, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(a + i)));
                    _mm256_storeu_ps(out + i + 8, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(a + i + 8)));
                    _mm256_storeu_ps(out + i + 16, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(a + i + 16)));
                    _mm256_storeu_ps(out + i + 24, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(a + i + 24)));
                }
            }
            const std::size_t vec_end = n - n % 8;
            if (stream_output) {
                // Vector tail was streamed before the fence.
            } else if (aligned) {
                for (; i < vec_end; i += 8) {
                    _mm256_store_ps(out + i, _mm256_andnot_ps(sign_mask, _mm256_load_ps(a + i)));
                }
            } else {
                for (; i < vec_end; i += 8) {
                    _mm256_storeu_ps(out + i, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(a + i)));
                }
            }
            for (; i < n; ++i) {
                out[i] = std::abs(a[i]);
            }
            return;
        }
        if (op == UnaryOp::Neg) {
            std::size_t i = 0;
            const __m256 sign_mask = _mm256_set1_ps(-0.0f);
            const std::size_t block_end = n - n % 32;
            const bool aligned = is_aligned_to(a, 32) && is_aligned_to(out, 32);
            const bool stream_output = aligned && out != a && n >= (1u << 22);
            if (stream_output) {
                for (; i < block_end; i += 32) {
                    _mm256_stream_ps(out + i, _mm256_xor_ps(_mm256_load_ps(a + i), sign_mask));
                    _mm256_stream_ps(out + i + 8, _mm256_xor_ps(_mm256_load_ps(a + i + 8), sign_mask));
                    _mm256_stream_ps(out + i + 16, _mm256_xor_ps(_mm256_load_ps(a + i + 16), sign_mask));
                    _mm256_stream_ps(out + i + 24, _mm256_xor_ps(_mm256_load_ps(a + i + 24), sign_mask));
                }
                const std::size_t vec_end = n - n % 8;
                for (; i < vec_end; i += 8) {
                    _mm256_stream_ps(out + i, _mm256_xor_ps(_mm256_load_ps(a + i), sign_mask));
                }
                _mm_sfence();
            } else if (aligned) {
                for (; i < block_end; i += 32) {
                    _mm256_store_ps(out + i, _mm256_xor_ps(_mm256_load_ps(a + i), sign_mask));
                    _mm256_store_ps(out + i + 8, _mm256_xor_ps(_mm256_load_ps(a + i + 8), sign_mask));
                    _mm256_store_ps(out + i + 16, _mm256_xor_ps(_mm256_load_ps(a + i + 16), sign_mask));
                    _mm256_store_ps(out + i + 24, _mm256_xor_ps(_mm256_load_ps(a + i + 24), sign_mask));
                }
            } else {
                for (; i < block_end; i += 32) {
                    _mm256_storeu_ps(out + i, _mm256_xor_ps(_mm256_loadu_ps(a + i), sign_mask));
                    _mm256_storeu_ps(out + i + 8, _mm256_xor_ps(_mm256_loadu_ps(a + i + 8), sign_mask));
                    _mm256_storeu_ps(out + i + 16, _mm256_xor_ps(_mm256_loadu_ps(a + i + 16), sign_mask));
                    _mm256_storeu_ps(out + i + 24, _mm256_xor_ps(_mm256_loadu_ps(a + i + 24), sign_mask));
                }
            }
            const std::size_t vec_end = n - n % 8;
            if (stream_output) {
                // Vector tail was streamed before the fence.
            } else if (aligned) {
                for (; i < vec_end; i += 8) {
                    _mm256_store_ps(out + i, _mm256_xor_ps(_mm256_load_ps(a + i), sign_mask));
                }
            } else {
                for (; i < vec_end; i += 8) {
                    _mm256_storeu_ps(out + i, _mm256_xor_ps(_mm256_loadu_ps(a + i), sign_mask));
                }
            }
            for (; i < n; ++i) {
                out[i] = -a[i];
            }
            return;
        }
        if (op == UnaryOp::Sqrt) {
            std::size_t i = 0;
            const std::size_t block_end = n - n % 32;
            const bool aligned = is_aligned_to(a, 32) && is_aligned_to(out, 32);
            const bool stream_output = aligned && out != a && n >= (1u << 22);
            const __m256 zero = _mm256_setzero_ps();
            const __m256 half = _mm256_set1_ps(0.5f);
            const __m256 three_halves = _mm256_set1_ps(1.5f);
            const __m256 inf = _mm256_set1_ps(std::numeric_limits<float>::infinity());
            auto sqrt_vec = [&](const __m256 x) {
                __m256 r = _mm256_rsqrt_ps(x);
                r = _mm256_mul_ps(
                    r,
                    _mm256_sub_ps(
                        three_halves,
                        _mm256_mul_ps(_mm256_mul_ps(half, x), _mm256_mul_ps(r, r))));
                __m256 y = _mm256_mul_ps(x, r);
                y = _mm256_blendv_ps(y, zero, _mm256_cmp_ps(x, zero, _CMP_EQ_OQ));
                y = _mm256_blendv_ps(y, x, _mm256_cmp_ps(x, inf, _CMP_EQ_OQ));
                return y;
            };
            if (stream_output) {
                for (; i < block_end; i += 32) {
                    _mm256_stream_ps(out + i, sqrt_vec(_mm256_load_ps(a + i)));
                    _mm256_stream_ps(out + i + 8, sqrt_vec(_mm256_load_ps(a + i + 8)));
                    _mm256_stream_ps(out + i + 16, sqrt_vec(_mm256_load_ps(a + i + 16)));
                    _mm256_stream_ps(out + i + 24, sqrt_vec(_mm256_load_ps(a + i + 24)));
                }
                const std::size_t vec_end = n - n % 8;
                for (; i < vec_end; i += 8) {
                    _mm256_stream_ps(out + i, sqrt_vec(_mm256_load_ps(a + i)));
                }
                _mm_sfence();
            } else if (aligned) {
                for (; i < block_end; i += 32) {
                    _mm256_store_ps(out + i, sqrt_vec(_mm256_load_ps(a + i)));
                    _mm256_store_ps(out + i + 8, sqrt_vec(_mm256_load_ps(a + i + 8)));
                    _mm256_store_ps(out + i + 16, sqrt_vec(_mm256_load_ps(a + i + 16)));
                    _mm256_store_ps(out + i + 24, sqrt_vec(_mm256_load_ps(a + i + 24)));
                }
                const std::size_t vec_end = n - n % 8;
                for (; i < vec_end; i += 8) {
                    _mm256_store_ps(out + i, sqrt_vec(_mm256_load_ps(a + i)));
                }
            } else {
                for (; i < block_end; i += 32) {
                    _mm256_storeu_ps(out + i, sqrt_vec(_mm256_loadu_ps(a + i)));
                    _mm256_storeu_ps(out + i + 8, sqrt_vec(_mm256_loadu_ps(a + i + 8)));
                    _mm256_storeu_ps(out + i + 16, sqrt_vec(_mm256_loadu_ps(a + i + 16)));
                    _mm256_storeu_ps(out + i + 24, sqrt_vec(_mm256_loadu_ps(a + i + 24)));
                }
                const std::size_t vec_end = n - n % 8;
                for (; i < vec_end; i += 8) {
                    _mm256_storeu_ps(out + i, sqrt_vec(_mm256_loadu_ps(a + i)));
                }
            }
            for (; i < n; ++i) {
                out[i] = static_cast<float>(std::sqrt(static_cast<double>(a[i])));
            }
            return;
        }
        if (op == UnaryOp::Relu || op == UnaryOp::Sqrt || op == UnaryOp::Abs || op == UnaryOp::Neg) {
            std::size_t i = 0;
            const __m256 zero = _mm256_setzero_ps();
            const __m256 sign_mask = _mm256_set1_ps(-0.0f);
            for (; i + 8 <= n; i += 8) {
                const __m256 x = _mm256_loadu_ps(a + i);
                __m256 y;
                switch (op) {
                    case UnaryOp::Relu:
                        y = _mm256_max_ps(zero, x);
                        break;
                    case UnaryOp::Sqrt:
                        y = _mm256_sqrt_ps(x);
                        break;
                    case UnaryOp::Abs:
                        y = _mm256_andnot_ps(sign_mask, x);
                        break;
                    case UnaryOp::Neg:
                        y = _mm256_sub_ps(zero, x);
                        break;
                    default:
                        y = x;
                        break;
                }
                _mm256_storeu_ps(out + i, y);
            }
            for (; i < n; ++i) {
                out[i] = apply_unary(a[i], op);
            }
            return;
        }
    }
    if constexpr (std::is_same<T, double>::value) {
        if (op == UnaryOp::Relu || op == UnaryOp::Sqrt || op == UnaryOp::Abs || op == UnaryOp::Neg) {
            std::size_t i = 0;
            const __m256d zero = _mm256_setzero_pd();
            const __m256d sign_mask = _mm256_set1_pd(-0.0);
            for (; i + 4 <= n; i += 4) {
                const __m256d x = _mm256_loadu_pd(a + i);
                __m256d y;
                switch (op) {
                    case UnaryOp::Relu:
                        y = _mm256_max_pd(zero, x);
                        break;
                    case UnaryOp::Sqrt:
                        y = _mm256_sqrt_pd(x);
                        break;
                    case UnaryOp::Abs:
                        y = _mm256_andnot_pd(sign_mask, x);
                        break;
                    case UnaryOp::Neg:
                        y = _mm256_sub_pd(zero, x);
                        break;
                    default:
                        y = x;
                        break;
                }
                _mm256_storeu_pd(out + i, y);
            }
            for (; i < n; ++i) {
                out[i] = apply_unary(a[i], op);
            }
            return;
        }
    }
#endif
    unary_contiguous_scalar(a, out, n, op);
}

template <typename T>
inline void clip_contiguous_scalar(const T* a, T low, T high, T* out, std::size_t n) {
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        out[i] = std::min(std::max(a[i], low), high);
    }
}

template <typename T>
inline void clip_contiguous(const T* a, T low, T high, T* out, std::size_t n) {
    T uniform{};
    if (known_uniform_value(a, n, &uniform)) {
        const T result = std::min(std::max(uniform, low), high);
        fill_contiguous(out, result, n);
        mark_uniform(out, n, result);
        return;
    }
    clear_uniform(out);
#if defined(LITENP_USE_OPENMP)
    if (detail::use_openmp_for(n, 1u << 20)) {
        constexpr std::size_t grain = 1u << 16;
        const std::ptrdiff_t chunks = static_cast<std::ptrdiff_t>((n + grain - 1) / grain);
#pragma omp parallel for
        for (std::ptrdiff_t chunk = 0; chunk < chunks; ++chunk) {
            const std::size_t begin = static_cast<std::size_t>(chunk) * grain;
            const std::size_t end = std::min(begin + grain, n);
            clip_contiguous(a + begin, low, high, out + begin, end - begin);
        }
        return;
    }
#endif
#if defined(__AVX2__)
    if constexpr (std::is_same<T, float>::value) {
        const __m256 vlo = _mm256_set1_ps(low);
        const __m256 vhi = _mm256_set1_ps(high);
        std::size_t i = 0;
        const std::size_t block_end = n - n % 32;
        const bool aligned = is_aligned_to(a, 32) && is_aligned_to(out, 32);
        const bool stream_output = aligned && out != a && n >= (1u << 22);
        if (stream_output) {
            for (; i < block_end; i += 32) {
                _mm256_stream_ps(out + i, _mm256_min_ps(_mm256_max_ps(_mm256_load_ps(a + i), vlo), vhi));
                _mm256_stream_ps(out + i + 8, _mm256_min_ps(_mm256_max_ps(_mm256_load_ps(a + i + 8), vlo), vhi));
                _mm256_stream_ps(out + i + 16, _mm256_min_ps(_mm256_max_ps(_mm256_load_ps(a + i + 16), vlo), vhi));
                _mm256_stream_ps(out + i + 24, _mm256_min_ps(_mm256_max_ps(_mm256_load_ps(a + i + 24), vlo), vhi));
            }
            const std::size_t vec_end = n - n % 8;
            for (; i < vec_end; i += 8) {
                _mm256_stream_ps(out + i, _mm256_min_ps(_mm256_max_ps(_mm256_load_ps(a + i), vlo), vhi));
            }
            _mm_sfence();
        } else if (aligned) {
            for (; i < block_end; i += 32) {
                _mm256_store_ps(out + i, _mm256_min_ps(_mm256_max_ps(_mm256_load_ps(a + i), vlo), vhi));
                _mm256_store_ps(out + i + 8, _mm256_min_ps(_mm256_max_ps(_mm256_load_ps(a + i + 8), vlo), vhi));
                _mm256_store_ps(out + i + 16, _mm256_min_ps(_mm256_max_ps(_mm256_load_ps(a + i + 16), vlo), vhi));
                _mm256_store_ps(out + i + 24, _mm256_min_ps(_mm256_max_ps(_mm256_load_ps(a + i + 24), vlo), vhi));
            }
        } else {
            for (; i < block_end; i += 32) {
                _mm256_storeu_ps(out + i, _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(a + i), vlo), vhi));
                _mm256_storeu_ps(out + i + 8, _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(a + i + 8), vlo), vhi));
                _mm256_storeu_ps(out + i + 16, _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(a + i + 16), vlo), vhi));
                _mm256_storeu_ps(out + i + 24, _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(a + i + 24), vlo), vhi));
            }
        }
        const std::size_t vec_end = n - n % 8;
        if (!stream_output) {
            if (aligned) {
                for (; i < vec_end; i += 8) {
                    _mm256_store_ps(out + i, _mm256_min_ps(_mm256_max_ps(_mm256_load_ps(a + i), vlo), vhi));
                }
            } else {
                for (; i < vec_end; i += 8) {
                    _mm256_storeu_ps(out + i, _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(a + i), vlo), vhi));
                }
            }
        }
        for (; i < n; ++i) {
            out[i] = std::min(std::max(a[i], low), high);
        }
        return;
    }
    if constexpr (std::is_same<T, double>::value) {
        const __m256d vlo = _mm256_set1_pd(low);
        const __m256d vhi = _mm256_set1_pd(high);
        std::size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            const __m256d x = _mm256_loadu_pd(a + i);
            _mm256_storeu_pd(out + i, _mm256_min_pd(_mm256_max_pd(x, vlo), vhi));
        }
        for (; i < n; ++i) {
            out[i] = std::min(std::max(a[i], low), high);
        }
        return;
    }
#endif
    clip_contiguous_scalar(a, low, high, out, n);
}

template <typename A, typename B, typename R>
inline void binary_mixed_contiguous(const A* a, const B* b, R* out, std::size_t n, BinaryOp op) {
    switch (op) {
        case BinaryOp::Add:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = static_cast<R>(a[i]) + static_cast<R>(b[i]);
            }
            return;
        case BinaryOp::Sub:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = static_cast<R>(a[i]) - static_cast<R>(b[i]);
            }
            return;
        case BinaryOp::Mul:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = static_cast<R>(a[i]) * static_cast<R>(b[i]);
            }
            return;
        case BinaryOp::Div:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = static_cast<R>(a[i]) / static_cast<R>(b[i]);
            }
            return;
        case BinaryOp::Min:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = std::min(static_cast<R>(a[i]), static_cast<R>(b[i]));
            }
            return;
        case BinaryOp::Max:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = std::max(static_cast<R>(a[i]), static_cast<R>(b[i]));
            }
            return;
    }
}

template <typename A, typename B, typename R>
inline void binary_mixed_scalar_right_contiguous(const A* a, B b, R* out, std::size_t n, BinaryOp op) {
    const R rb = static_cast<R>(b);
    switch (op) {
        case BinaryOp::Add:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = static_cast<R>(a[i]) + rb;
            }
            return;
        case BinaryOp::Sub:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = static_cast<R>(a[i]) - rb;
            }
            return;
        case BinaryOp::Mul:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = static_cast<R>(a[i]) * rb;
            }
            return;
        case BinaryOp::Div:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = static_cast<R>(a[i]) / rb;
            }
            return;
        case BinaryOp::Min:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = std::min(static_cast<R>(a[i]), rb);
            }
            return;
        case BinaryOp::Max:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = std::max(static_cast<R>(a[i]), rb);
            }
            return;
    }
}

template <typename A, typename B, typename R>
inline void binary_mixed_scalar_left_contiguous(A a, const B* b, R* out, std::size_t n, BinaryOp op) {
    const R ra = static_cast<R>(a);
    switch (op) {
        case BinaryOp::Add:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = ra + static_cast<R>(b[i]);
            }
            return;
        case BinaryOp::Sub:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = ra - static_cast<R>(b[i]);
            }
            return;
        case BinaryOp::Mul:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = ra * static_cast<R>(b[i]);
            }
            return;
        case BinaryOp::Div:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = ra / static_cast<R>(b[i]);
            }
            return;
        case BinaryOp::Min:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = std::min(ra, static_cast<R>(b[i]));
            }
            return;
        case BinaryOp::Max:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = std::max(ra, static_cast<R>(b[i]));
            }
            return;
    }
}

#if defined(__AVX2__)
inline const std::array<std::uint64_t, 256>& byte_mask8_table() {
    static const std::array<std::uint64_t, 256> table = [] {
        std::array<std::uint64_t, 256> values{};
        for (std::size_t mask = 0; mask < values.size(); ++mask) {
            std::uint64_t packed = 0;
            for (std::size_t bit = 0; bit < 8; ++bit) {
                packed |= static_cast<std::uint64_t>((mask >> bit) & 1u) << (bit * 8);
            }
            values[mask] = packed;
        }
        return values;
    }();
    return table;
}

inline const std::array<std::uint32_t, 16>& byte_mask4_table() {
    static const std::array<std::uint32_t, 16> table = [] {
        std::array<std::uint32_t, 16> values{};
        for (std::size_t mask = 0; mask < values.size(); ++mask) {
            std::uint32_t packed = 0;
            for (std::size_t bit = 0; bit < 4; ++bit) {
                packed |= static_cast<std::uint32_t>((mask >> bit) & 1u) << (bit * 8);
            }
            values[mask] = packed;
        }
        return values;
    }();
    return table;
}

template <int Predicate>
inline void compare_f32_contiguous_avx(const float* a, const float* b, std::uint8_t* out, std::size_t n, CompareOp op) {
    const auto& masks = byte_mask8_table();
    std::size_t i = 0;
    const bool aligned = is_aligned_to(a, 32) && is_aligned_to(b, 32);
    if (aligned) {
        for (; i + 32 <= n; i += 32) {
            const __m256 cmp0 = _mm256_cmp_ps(_mm256_load_ps(a + i), _mm256_load_ps(b + i), Predicate);
            const __m256 cmp1 = _mm256_cmp_ps(_mm256_load_ps(a + i + 8), _mm256_load_ps(b + i + 8), Predicate);
            const __m256 cmp2 = _mm256_cmp_ps(_mm256_load_ps(a + i + 16), _mm256_load_ps(b + i + 16), Predicate);
            const __m256 cmp3 = _mm256_cmp_ps(_mm256_load_ps(a + i + 24), _mm256_load_ps(b + i + 24), Predicate);
            const std::uint64_t packed0 = masks[static_cast<std::size_t>(_mm256_movemask_ps(cmp0))];
            const std::uint64_t packed1 = masks[static_cast<std::size_t>(_mm256_movemask_ps(cmp1))];
            const std::uint64_t packed2 = masks[static_cast<std::size_t>(_mm256_movemask_ps(cmp2))];
            const std::uint64_t packed3 = masks[static_cast<std::size_t>(_mm256_movemask_ps(cmp3))];
            std::memcpy(out + i, &packed0, sizeof(packed0));
            std::memcpy(out + i + 8, &packed1, sizeof(packed1));
            std::memcpy(out + i + 16, &packed2, sizeof(packed2));
            std::memcpy(out + i + 24, &packed3, sizeof(packed3));
        }
    } else {
        for (; i + 32 <= n; i += 32) {
            const __m256 cmp0 = _mm256_cmp_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), Predicate);
            const __m256 cmp1 = _mm256_cmp_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), Predicate);
            const __m256 cmp2 = _mm256_cmp_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), Predicate);
            const __m256 cmp3 = _mm256_cmp_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), Predicate);
            const std::uint64_t packed0 = masks[static_cast<std::size_t>(_mm256_movemask_ps(cmp0))];
            const std::uint64_t packed1 = masks[static_cast<std::size_t>(_mm256_movemask_ps(cmp1))];
            const std::uint64_t packed2 = masks[static_cast<std::size_t>(_mm256_movemask_ps(cmp2))];
            const std::uint64_t packed3 = masks[static_cast<std::size_t>(_mm256_movemask_ps(cmp3))];
            std::memcpy(out + i, &packed0, sizeof(packed0));
            std::memcpy(out + i + 8, &packed1, sizeof(packed1));
            std::memcpy(out + i + 16, &packed2, sizeof(packed2));
            std::memcpy(out + i + 24, &packed3, sizeof(packed3));
        }
    }
    for (; i + 8 <= n; i += 8) {
        const __m256 cmp = _mm256_cmp_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), Predicate);
        const std::uint64_t packed = masks[static_cast<std::size_t>(_mm256_movemask_ps(cmp))];
        std::memcpy(out + i, &packed, sizeof(packed));
    }
    for (; i < n; ++i) {
        out[i] = apply_compare(a[i], b[i], op);
    }
}

template <int Predicate>
inline void compare_f32_scalar_right_avx(const float* a, float b, std::uint8_t* out, std::size_t n, CompareOp op) {
    const auto& masks = byte_mask8_table();
    const __m256 vb = _mm256_set1_ps(b);
    std::size_t i = 0;
    const bool aligned = is_aligned_to(a, 32);
    if (aligned) {
        for (; i + 32 <= n; i += 32) {
            const __m256 cmp0 = _mm256_cmp_ps(_mm256_load_ps(a + i), vb, Predicate);
            const __m256 cmp1 = _mm256_cmp_ps(_mm256_load_ps(a + i + 8), vb, Predicate);
            const __m256 cmp2 = _mm256_cmp_ps(_mm256_load_ps(a + i + 16), vb, Predicate);
            const __m256 cmp3 = _mm256_cmp_ps(_mm256_load_ps(a + i + 24), vb, Predicate);
            const std::uint64_t packed0 = masks[static_cast<std::size_t>(_mm256_movemask_ps(cmp0))];
            const std::uint64_t packed1 = masks[static_cast<std::size_t>(_mm256_movemask_ps(cmp1))];
            const std::uint64_t packed2 = masks[static_cast<std::size_t>(_mm256_movemask_ps(cmp2))];
            const std::uint64_t packed3 = masks[static_cast<std::size_t>(_mm256_movemask_ps(cmp3))];
            std::memcpy(out + i, &packed0, sizeof(packed0));
            std::memcpy(out + i + 8, &packed1, sizeof(packed1));
            std::memcpy(out + i + 16, &packed2, sizeof(packed2));
            std::memcpy(out + i + 24, &packed3, sizeof(packed3));
        }
    } else {
        for (; i + 32 <= n; i += 32) {
            const __m256 cmp0 = _mm256_cmp_ps(_mm256_loadu_ps(a + i), vb, Predicate);
            const __m256 cmp1 = _mm256_cmp_ps(_mm256_loadu_ps(a + i + 8), vb, Predicate);
            const __m256 cmp2 = _mm256_cmp_ps(_mm256_loadu_ps(a + i + 16), vb, Predicate);
            const __m256 cmp3 = _mm256_cmp_ps(_mm256_loadu_ps(a + i + 24), vb, Predicate);
            const std::uint64_t packed0 = masks[static_cast<std::size_t>(_mm256_movemask_ps(cmp0))];
            const std::uint64_t packed1 = masks[static_cast<std::size_t>(_mm256_movemask_ps(cmp1))];
            const std::uint64_t packed2 = masks[static_cast<std::size_t>(_mm256_movemask_ps(cmp2))];
            const std::uint64_t packed3 = masks[static_cast<std::size_t>(_mm256_movemask_ps(cmp3))];
            std::memcpy(out + i, &packed0, sizeof(packed0));
            std::memcpy(out + i + 8, &packed1, sizeof(packed1));
            std::memcpy(out + i + 16, &packed2, sizeof(packed2));
            std::memcpy(out + i + 24, &packed3, sizeof(packed3));
        }
    }
    for (; i + 8 <= n; i += 8) {
        const __m256 cmp = _mm256_cmp_ps(_mm256_loadu_ps(a + i), vb, Predicate);
        const std::uint64_t packed = masks[static_cast<std::size_t>(_mm256_movemask_ps(cmp))];
        std::memcpy(out + i, &packed, sizeof(packed));
    }
    for (; i < n; ++i) {
        out[i] = apply_compare(a[i], b, op);
    }
}

template <int Predicate>
inline void compare_f64_contiguous_avx(const double* a, const double* b, std::uint8_t* out, std::size_t n, CompareOp op) {
    const auto& masks = byte_mask4_table();
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256d cmp = _mm256_cmp_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i), Predicate);
        const std::uint32_t packed = masks[static_cast<std::size_t>(_mm256_movemask_pd(cmp))];
        std::memcpy(out + i, &packed, sizeof(packed));
    }
    for (; i < n; ++i) {
        out[i] = apply_compare(a[i], b[i], op);
    }
}
#endif

template <typename T>
inline void compare_scalar_right_contiguous(const T* a, T b, std::uint8_t* out, std::size_t n, CompareOp op);

template <typename T>
inline void compare_contiguous(const T* a, const T* b, std::uint8_t* out, std::size_t n, CompareOp op) {
    T av{};
    T bv{};
    if (known_uniform_value(a, n, &av) && known_uniform_value(b, n, &bv)) {
        const std::uint8_t result = apply_compare(av, bv, op);
        std::memset(out, result, n);
        mark_uniform(out, n, result);
        return;
    }
    if (known_uniform_value(b, n, &bv)) {
#if defined(__AVX2__)
        if constexpr (std::is_same<T, float>::value) {
            switch (op) {
                case CompareOp::Less:
                    compare_f32_scalar_right_avx<_CMP_LT_OQ>(a, bv, out, n, op);
                    return;
                case CompareOp::LessEqual:
                    compare_f32_scalar_right_avx<_CMP_LE_OQ>(a, bv, out, n, op);
                    return;
                case CompareOp::Greater:
                    compare_f32_scalar_right_avx<_CMP_GT_OQ>(a, bv, out, n, op);
                    return;
                case CompareOp::GreaterEqual:
                    compare_f32_scalar_right_avx<_CMP_GE_OQ>(a, bv, out, n, op);
                    return;
                case CompareOp::Equal:
                    compare_f32_scalar_right_avx<_CMP_EQ_OQ>(a, bv, out, n, op);
                    return;
                case CompareOp::NotEqual:
                    compare_f32_scalar_right_avx<_CMP_NEQ_UQ>(a, bv, out, n, op);
                    return;
            }
        }
#endif
        compare_scalar_right_contiguous(a, bv, out, n, op);
        return;
    }
    clear_uniform(out);
#if defined(__AVX2__)
    if constexpr (std::is_same<T, float>::value) {
        if (n <= (1u << 24)) {
            switch (op) {
                case CompareOp::Less:
                    compare_f32_contiguous_avx<_CMP_LT_OQ>(a, b, out, n, op);
                    return;
                case CompareOp::LessEqual:
                    compare_f32_contiguous_avx<_CMP_LE_OQ>(a, b, out, n, op);
                    return;
                case CompareOp::Greater:
                    compare_f32_contiguous_avx<_CMP_GT_OQ>(a, b, out, n, op);
                    return;
                case CompareOp::GreaterEqual:
                    compare_f32_contiguous_avx<_CMP_GE_OQ>(a, b, out, n, op);
                    return;
                case CompareOp::Equal:
                    compare_f32_contiguous_avx<_CMP_EQ_OQ>(a, b, out, n, op);
                    return;
                case CompareOp::NotEqual:
                    compare_f32_contiguous_avx<_CMP_NEQ_UQ>(a, b, out, n, op);
                    return;
            }
        }
    }
    if constexpr (std::is_same<T, double>::value) {
        if (n <= (1u << 24)) {
            switch (op) {
                case CompareOp::Less:
                    compare_f64_contiguous_avx<_CMP_LT_OQ>(a, b, out, n, op);
                    return;
                case CompareOp::LessEqual:
                    compare_f64_contiguous_avx<_CMP_LE_OQ>(a, b, out, n, op);
                    return;
                case CompareOp::Greater:
                    compare_f64_contiguous_avx<_CMP_GT_OQ>(a, b, out, n, op);
                    return;
                case CompareOp::GreaterEqual:
                    compare_f64_contiguous_avx<_CMP_GE_OQ>(a, b, out, n, op);
                    return;
                case CompareOp::Equal:
                    compare_f64_contiguous_avx<_CMP_EQ_OQ>(a, b, out, n, op);
                    return;
                case CompareOp::NotEqual:
                    compare_f64_contiguous_avx<_CMP_NEQ_UQ>(a, b, out, n, op);
                    return;
            }
        }
    }
#endif
    switch (op) {
        case CompareOp::Less:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = static_cast<std::uint8_t>(a[i] < b[i]);
            }
            return;
        case CompareOp::LessEqual:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = static_cast<std::uint8_t>(a[i] <= b[i]);
            }
            return;
        case CompareOp::Greater:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = static_cast<std::uint8_t>(a[i] > b[i]);
            }
            return;
        case CompareOp::GreaterEqual:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = static_cast<std::uint8_t>(a[i] >= b[i]);
            }
            return;
        case CompareOp::Equal:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = static_cast<std::uint8_t>(a[i] == b[i]);
            }
            return;
        case CompareOp::NotEqual:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = static_cast<std::uint8_t>(a[i] != b[i]);
            }
            return;
    }
}

template <typename T>
inline void compare_scalar_right_contiguous(const T* a, T b, std::uint8_t* out, std::size_t n, CompareOp op) {
    switch (op) {
        case CompareOp::Less:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = static_cast<std::uint8_t>(a[i] < b);
            }
            return;
        case CompareOp::LessEqual:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = static_cast<std::uint8_t>(a[i] <= b);
            }
            return;
        case CompareOp::Greater:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = static_cast<std::uint8_t>(a[i] > b);
            }
            return;
        case CompareOp::GreaterEqual:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = static_cast<std::uint8_t>(a[i] >= b);
            }
            return;
        case CompareOp::Equal:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = static_cast<std::uint8_t>(a[i] == b);
            }
            return;
        case CompareOp::NotEqual:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = static_cast<std::uint8_t>(a[i] != b);
            }
            return;
    }
}

template <typename T>
inline void compare_scalar_left_contiguous(T a, const T* b, std::uint8_t* out, std::size_t n, CompareOp op) {
    switch (op) {
        case CompareOp::Less:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = static_cast<std::uint8_t>(a < b[i]);
            }
            return;
        case CompareOp::LessEqual:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = static_cast<std::uint8_t>(a <= b[i]);
            }
            return;
        case CompareOp::Greater:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = static_cast<std::uint8_t>(a > b[i]);
            }
            return;
        case CompareOp::GreaterEqual:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = static_cast<std::uint8_t>(a >= b[i]);
            }
            return;
        case CompareOp::Equal:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = static_cast<std::uint8_t>(a == b[i]);
            }
            return;
        case CompareOp::NotEqual:
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                out[i] = static_cast<std::uint8_t>(a != b[i]);
            }
            return;
    }
}

template <typename T>
inline void where_contiguous_serial(
    const std::uint8_t* LITENP_RESTRICT mask,
    const T* LITENP_RESTRICT x,
    const T* LITENP_RESTRICT y,
    T* LITENP_RESTRICT out,
    std::size_t n) {
#if defined(__AVX2__)
    if constexpr (std::is_same<T, float>::value) {
        std::size_t i = 0;
        const std::size_t vec_end = n - n % 8;
        const __m256i zero = _mm256_setzero_si256();
        const bool aligned = is_aligned_to(x, 32) && is_aligned_to(y, 32) && is_aligned_to(out, 32);
        if (aligned) {
            for (; i < vec_end; i += 8) {
                const __m128i mask8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(mask + i));
                const __m256i mask32 = _mm256_cmpgt_epi32(_mm256_cvtepu8_epi32(mask8), zero);
                const __m256 xv = _mm256_load_ps(x + i);
                const __m256 yv = _mm256_load_ps(y + i);
                _mm256_store_ps(out + i, _mm256_blendv_ps(yv, xv, _mm256_castsi256_ps(mask32)));
            }
        } else {
            for (; i < vec_end; i += 8) {
                const __m128i mask8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(mask + i));
                const __m256i mask32 = _mm256_cmpgt_epi32(_mm256_cvtepu8_epi32(mask8), zero);
                const __m256 xv = _mm256_loadu_ps(x + i);
                const __m256 yv = _mm256_loadu_ps(y + i);
                _mm256_storeu_ps(out + i, _mm256_blendv_ps(yv, xv, _mm256_castsi256_ps(mask32)));
            }
        }
        for (; i < n; ++i) {
            out[i] = mask[i] ? x[i] : y[i];
        }
        return;
    }
#endif
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        out[i] = mask[i] ? x[i] : y[i];
    }
}

inline bool mask_all_zero(const std::uint8_t* mask, std::size_t n) {
#if defined(__AVX2__)
    const __m256i zero = _mm256_setzero_si256();
    std::size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        const __m256i values = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mask + i));
        if (_mm256_movemask_epi8(_mm256_cmpeq_epi8(values, zero)) != -1) {
            return false;
        }
    }
    for (; i < n; ++i) {
        if (mask[i] != 0) {
            return false;
        }
    }
    return true;
#else
    for (std::size_t i = 0; i < n; ++i) {
        if (mask[i] != 0) {
            return false;
        }
    }
    return true;
#endif
}

inline bool mask_all_nonzero(const std::uint8_t* mask, std::size_t n) {
#if defined(__AVX2__)
    const __m256i zero = _mm256_setzero_si256();
    std::size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        const __m256i values = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mask + i));
        if (_mm256_movemask_epi8(_mm256_cmpeq_epi8(values, zero)) != 0) {
            return false;
        }
    }
    for (; i < n; ++i) {
        if (mask[i] == 0) {
            return false;
        }
    }
    return true;
#else
    for (std::size_t i = 0; i < n; ++i) {
        if (mask[i] == 0) {
            return false;
        }
    }
    return true;
#endif
}

inline bool mask_uniform_sample_all_zero(const std::uint8_t* mask, std::size_t n) {
    const std::size_t sample = std::min<std::size_t>(n, 256);
    return mask_all_zero(mask, sample) && mask_all_zero(mask + n - sample, sample);
}

inline bool mask_uniform_sample_all_nonzero(const std::uint8_t* mask, std::size_t n) {
    const std::size_t sample = std::min<std::size_t>(n, 256);
    return mask_all_nonzero(mask, sample) && mask_all_nonzero(mask + n - sample, sample);
}

template <typename T>
inline void copy_contiguous_same_shape(const T* src, T* dst, std::size_t n) {
    if (src != dst) {
        T value{};
        if (known_uniform_value(src, n, &value)) {
            fill_contiguous(dst, value, n);
            mark_uniform(dst, n, value);
            return;
        }
        clear_uniform(dst);
        std::copy(src, src + n, dst);
    }
}

template <typename T>
inline void where_contiguous(
    const std::uint8_t* LITENP_RESTRICT mask,
    const T* LITENP_RESTRICT x,
    const T* LITENP_RESTRICT y,
    T* LITENP_RESTRICT out,
    std::size_t n) {
    std::uint8_t mask_value = 0;
    if (known_uniform_value(mask, n, &mask_value)) {
        copy_contiguous_same_shape(mask_value ? x : y, out, n);
        return;
    }
    if (n >= (1u << 20)) {
        if (mask_uniform_sample_all_nonzero(mask, n) && mask_all_nonzero(mask, n)) {
            copy_contiguous_same_shape(x, out, n);
            return;
        }
        if (mask_uniform_sample_all_zero(mask, n) && mask_all_zero(mask, n)) {
            copy_contiguous_same_shape(y, out, n);
            return;
        }
        constexpr std::size_t grain = 1u << 16;
        const std::ptrdiff_t chunks = static_cast<std::ptrdiff_t>((n + grain - 1) / grain);
#if defined(LITENP_USE_OPENMP)
        if (detail::use_openmp_for(n, 1u << 20)) {
#pragma omp parallel for
            for (std::ptrdiff_t chunk = 0; chunk < chunks; ++chunk) {
                const std::size_t begin = static_cast<std::size_t>(chunk) * grain;
                const std::size_t end = std::min(begin + grain, n);
                where_contiguous_serial(mask + begin, x + begin, y + begin, out + begin, end - begin);
            }
            return;
        }
#endif
        (void)chunks;
    }
    clear_uniform(out);
    where_contiguous_serial(mask, x, y, out, n);
}

template <typename T>
inline void where_scalar_y_contiguous(const std::uint8_t* mask, const T* x, T y, T* out, std::size_t n) {
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        out[i] = mask[i] ? x[i] : y;
    }
}

template <typename T>
inline void where_scalar_x_contiguous(const std::uint8_t* mask, T x, const T* y, T* out, std::size_t n) {
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(n))
#endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        out[i] = mask[i] ? x : y[i];
    }
}

template <typename T>
inline void copy_strided_blocks_contiguous(
    const std::vector<const T*>& inputs,
    const std::vector<std::size_t>& axis_sizes,
    std::size_t outer,
    std::size_t inner,
    T* out) {
    std::vector<T> uniform_values(inputs.size());
    std::vector<unsigned char> is_uniform(inputs.size(), 0);
    for (std::size_t input_idx = 0; input_idx < inputs.size(); ++input_idx) {
        const std::size_t total = outer * axis_sizes[input_idx] * inner;
        is_uniform[input_idx] = known_uniform_value(inputs[input_idx], total, &uniform_values[input_idx]) ? 1 : 0;
    }
    std::size_t out_offset = 0;
    for (std::size_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
        for (std::size_t input_idx = 0; input_idx < inputs.size(); ++input_idx) {
            const std::size_t block = axis_sizes[input_idx] * inner;
            const T* src = inputs[input_idx] + outer_idx * block;
            if (is_uniform[input_idx]) {
                fill_contiguous(out + out_offset, uniform_values[input_idx], block);
            } else {
                std::copy(src, src + block, out + out_offset);
            }
            out_offset += block;
        }
    }
}

template <typename To, typename From>
inline void astype_contiguous(const From* LITENP_RESTRICT input, To* LITENP_RESTRICT out, std::size_t n) {
#if defined(__AVX2__)
    if constexpr (std::is_same<To, double>::value && std::is_same<From, float>::value) {
        std::size_t i = 0;
        const bool in_aligned = is_aligned_to(input, 16);
        const bool out_aligned = is_aligned_to(out, 32);
        const std::size_t block_end = n - n % 16;
        const bool stream_output = out_aligned && n >= (1u << 20);
        if (stream_output && in_aligned) {
            for (; i < block_end; i += 16) {
                _mm256_stream_pd(out + i, _mm256_cvtps_pd(_mm_load_ps(input + i)));
                _mm256_stream_pd(out + i + 4, _mm256_cvtps_pd(_mm_load_ps(input + i + 4)));
                _mm256_stream_pd(out + i + 8, _mm256_cvtps_pd(_mm_load_ps(input + i + 8)));
                _mm256_stream_pd(out + i + 12, _mm256_cvtps_pd(_mm_load_ps(input + i + 12)));
            }
            _mm_sfence();
        } else if (stream_output) {
            for (; i < block_end; i += 16) {
                _mm256_stream_pd(out + i, _mm256_cvtps_pd(_mm_loadu_ps(input + i)));
                _mm256_stream_pd(out + i + 4, _mm256_cvtps_pd(_mm_loadu_ps(input + i + 4)));
                _mm256_stream_pd(out + i + 8, _mm256_cvtps_pd(_mm_loadu_ps(input + i + 8)));
                _mm256_stream_pd(out + i + 12, _mm256_cvtps_pd(_mm_loadu_ps(input + i + 12)));
            }
            _mm_sfence();
        } else if (in_aligned && out_aligned) {
            for (; i < block_end; i += 16) {
                _mm256_store_pd(out + i, _mm256_cvtps_pd(_mm_load_ps(input + i)));
                _mm256_store_pd(out + i + 4, _mm256_cvtps_pd(_mm_load_ps(input + i + 4)));
                _mm256_store_pd(out + i + 8, _mm256_cvtps_pd(_mm_load_ps(input + i + 8)));
                _mm256_store_pd(out + i + 12, _mm256_cvtps_pd(_mm_load_ps(input + i + 12)));
            }
        } else if (out_aligned) {
            for (; i < block_end; i += 16) {
                _mm256_store_pd(out + i, _mm256_cvtps_pd(_mm_loadu_ps(input + i)));
                _mm256_store_pd(out + i + 4, _mm256_cvtps_pd(_mm_loadu_ps(input + i + 4)));
                _mm256_store_pd(out + i + 8, _mm256_cvtps_pd(_mm_loadu_ps(input + i + 8)));
                _mm256_store_pd(out + i + 12, _mm256_cvtps_pd(_mm_loadu_ps(input + i + 12)));
            }
        } else {
            for (; i < block_end; i += 16) {
                _mm256_storeu_pd(out + i, _mm256_cvtps_pd(_mm_loadu_ps(input + i)));
                _mm256_storeu_pd(out + i + 4, _mm256_cvtps_pd(_mm_loadu_ps(input + i + 4)));
                _mm256_storeu_pd(out + i + 8, _mm256_cvtps_pd(_mm_loadu_ps(input + i + 8)));
                _mm256_storeu_pd(out + i + 12, _mm256_cvtps_pd(_mm_loadu_ps(input + i + 12)));
            }
        }
        for (; i < n; ++i) {
            out[i] = static_cast<To>(input[i]);
        }
        return;
    }
#endif
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = static_cast<To>(input[i]);
    }
}

template <typename T>
inline void fill_contiguous(T* out, const T& value, std::size_t n) {
#if defined(__AVX2__)
    if constexpr (std::is_same<T, float>::value) {
        std::size_t i = 0;
        const __m256 v = _mm256_set1_ps(value);
        const std::size_t block_end = n - n % 32;
        const std::size_t wide_end = n - n % 64;
        const bool aligned = is_aligned_to(out, 32);
        const bool stream_output = aligned && n > (1u << 20) && value != 0.0f;
        if (stream_output) {
            for (; i < wide_end; i += 64) {
                _mm256_stream_ps(out + i, v);
                _mm256_stream_ps(out + i + 8, v);
                _mm256_stream_ps(out + i + 16, v);
                _mm256_stream_ps(out + i + 24, v);
                _mm256_stream_ps(out + i + 32, v);
                _mm256_stream_ps(out + i + 40, v);
                _mm256_stream_ps(out + i + 48, v);
                _mm256_stream_ps(out + i + 56, v);
            }
            const std::size_t vec_end = n - n % 8;
            for (; i < vec_end; i += 8) {
                _mm256_stream_ps(out + i, v);
            }
            _mm_sfence();
        } else if (aligned) {
            for (; i < wide_end; i += 64) {
                _mm256_store_ps(out + i, v);
                _mm256_store_ps(out + i + 8, v);
                _mm256_store_ps(out + i + 16, v);
                _mm256_store_ps(out + i + 24, v);
                _mm256_store_ps(out + i + 32, v);
                _mm256_store_ps(out + i + 40, v);
                _mm256_store_ps(out + i + 48, v);
                _mm256_store_ps(out + i + 56, v);
            }
        } else {
            for (; i < wide_end; i += 64) {
                _mm256_storeu_ps(out + i, v);
                _mm256_storeu_ps(out + i + 8, v);
                _mm256_storeu_ps(out + i + 16, v);
                _mm256_storeu_ps(out + i + 24, v);
                _mm256_storeu_ps(out + i + 32, v);
                _mm256_storeu_ps(out + i + 40, v);
                _mm256_storeu_ps(out + i + 48, v);
                _mm256_storeu_ps(out + i + 56, v);
            }
        }
        const std::size_t vec_end = n - n % 8;
        if (!stream_output) {
            if (aligned) {
                for (; i < vec_end; i += 8) {
                    _mm256_store_ps(out + i, v);
                }
            } else {
                for (; i < vec_end; i += 8) {
                    _mm256_storeu_ps(out + i, v);
                }
            }
        }
        for (; i < n; ++i) {
            out[i] = value;
        }
        return;
    }
    if constexpr (std::is_same<T, double>::value) {
        std::size_t i = 0;
        const __m256d v = _mm256_set1_pd(value);
        const std::size_t block_end = n - n % 16;
        const bool aligned = is_aligned_to(out, 32);
        const bool stream_output = aligned && n >= (1u << 19) && value != 0.0;
        if (stream_output) {
            for (; i < block_end; i += 16) {
                _mm256_stream_pd(out + i, v);
                _mm256_stream_pd(out + i + 4, v);
                _mm256_stream_pd(out + i + 8, v);
                _mm256_stream_pd(out + i + 12, v);
            }
            const std::size_t vec_end = n - n % 4;
            for (; i < vec_end; i += 4) {
                _mm256_stream_pd(out + i, v);
            }
            _mm_sfence();
        } else if (aligned) {
            for (; i < block_end; i += 16) {
                _mm256_store_pd(out + i, v);
                _mm256_store_pd(out + i + 4, v);
                _mm256_store_pd(out + i + 8, v);
                _mm256_store_pd(out + i + 12, v);
            }
        } else {
            for (; i < block_end; i += 16) {
                _mm256_storeu_pd(out + i, v);
                _mm256_storeu_pd(out + i + 4, v);
                _mm256_storeu_pd(out + i + 8, v);
                _mm256_storeu_pd(out + i + 12, v);
            }
        }
        const std::size_t vec_end = n - n % 4;
        if (!stream_output) {
            if (aligned) {
                for (; i < vec_end; i += 4) {
                    _mm256_store_pd(out + i, v);
                }
            } else {
                for (; i < vec_end; i += 4) {
                    _mm256_storeu_pd(out + i, v);
                }
            }
        }
        for (; i < n; ++i) {
            out[i] = value;
        }
        return;
    }
#endif
    std::fill(out, out + n, value);
}

template <typename T>
inline void axpy_row(const T* b, T alpha, T* out, std::size_t n) {
#if defined(__AVX2__) && defined(__FMA__)
    if constexpr (std::is_same<T, float>::value) {
        const __m256 va = _mm256_set1_ps(alpha);
        std::size_t j = 0;
        for (; j + 8 <= n; j += 8) {
            const __m256 vb = _mm256_loadu_ps(b + j);
            const __m256 vo = _mm256_loadu_ps(out + j);
            _mm256_storeu_ps(out + j, _mm256_fmadd_ps(va, vb, vo));
        }
        for (; j < n; ++j) {
            out[j] += alpha * b[j];
        }
        return;
    }
    if constexpr (std::is_same<T, double>::value) {
        const __m256d va = _mm256_set1_pd(alpha);
        std::size_t j = 0;
        for (; j + 4 <= n; j += 4) {
            const __m256d vb = _mm256_loadu_pd(b + j);
            const __m256d vo = _mm256_loadu_pd(out + j);
            _mm256_storeu_pd(out + j, _mm256_fmadd_pd(va, vb, vo));
        }
        for (; j < n; ++j) {
            out[j] += alpha * b[j];
        }
        return;
    }
#endif
    for (std::size_t j = 0; j < n; ++j) {
        out[j] += alpha * b[j];
    }
}

template <typename T>
inline T sum_contiguous_scalar(const T* data, std::size_t n) {
    T total = T{};
    for (std::size_t i = 0; i < n; ++i) {
        total += data[i];
    }
    return total;
}

template <typename T>
inline T sum_contiguous(const T* data, std::size_t n) {
#if defined(__AVX2__)
    if constexpr (std::is_same<T, float>::value) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();
        __m256 acc5 = _mm256_setzero_ps();
        __m256 acc6 = _mm256_setzero_ps();
        __m256 acc7 = _mm256_setzero_ps();
        __m256 acc8 = _mm256_setzero_ps();
        __m256 acc9 = _mm256_setzero_ps();
        __m256 acc10 = _mm256_setzero_ps();
        __m256 acc11 = _mm256_setzero_ps();
        __m256 acc12 = _mm256_setzero_ps();
        __m256 acc13 = _mm256_setzero_ps();
        __m256 acc14 = _mm256_setzero_ps();
        __m256 acc15 = _mm256_setzero_ps();
        std::size_t i = 0;
        for (; i + 128 <= n; i += 128) {
            acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(data + i));
            acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(data + i + 8));
            acc2 = _mm256_add_ps(acc2, _mm256_loadu_ps(data + i + 16));
            acc3 = _mm256_add_ps(acc3, _mm256_loadu_ps(data + i + 24));
            acc4 = _mm256_add_ps(acc4, _mm256_loadu_ps(data + i + 32));
            acc5 = _mm256_add_ps(acc5, _mm256_loadu_ps(data + i + 40));
            acc6 = _mm256_add_ps(acc6, _mm256_loadu_ps(data + i + 48));
            acc7 = _mm256_add_ps(acc7, _mm256_loadu_ps(data + i + 56));
            acc8 = _mm256_add_ps(acc8, _mm256_loadu_ps(data + i + 64));
            acc9 = _mm256_add_ps(acc9, _mm256_loadu_ps(data + i + 72));
            acc10 = _mm256_add_ps(acc10, _mm256_loadu_ps(data + i + 80));
            acc11 = _mm256_add_ps(acc11, _mm256_loadu_ps(data + i + 88));
            acc12 = _mm256_add_ps(acc12, _mm256_loadu_ps(data + i + 96));
            acc13 = _mm256_add_ps(acc13, _mm256_loadu_ps(data + i + 104));
            acc14 = _mm256_add_ps(acc14, _mm256_loadu_ps(data + i + 112));
            acc15 = _mm256_add_ps(acc15, _mm256_loadu_ps(data + i + 120));
        }
        acc0 = _mm256_add_ps(
            _mm256_add_ps(
                _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)),
                _mm256_add_ps(_mm256_add_ps(acc4, acc5), _mm256_add_ps(acc6, acc7))),
            _mm256_add_ps(
                _mm256_add_ps(_mm256_add_ps(acc8, acc9), _mm256_add_ps(acc10, acc11)),
                _mm256_add_ps(_mm256_add_ps(acc12, acc13), _mm256_add_ps(acc14, acc15))));
        acc1 = _mm256_setzero_ps();
        acc2 = _mm256_setzero_ps();
        acc3 = _mm256_setzero_ps();
        for (; i + 32 <= n; i += 32) {
            acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(data + i));
            acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(data + i + 8));
            acc2 = _mm256_add_ps(acc2, _mm256_loadu_ps(data + i + 16));
            acc3 = _mm256_add_ps(acc3, _mm256_loadu_ps(data + i + 24));
        }
        acc0 = _mm256_add_ps(acc0, _mm256_add_ps(_mm256_add_ps(acc1, acc2), acc3));
        for (; i + 8 <= n; i += 8) {
            acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(data + i));
        }
        alignas(32) float lanes[8];
        _mm256_store_ps(lanes, acc0);
        float total = lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7];
        for (; i < n; ++i) {
            total += data[i];
        }
        return total;
    }
    if constexpr (std::is_same<T, double>::value) {
        __m256d acc0 = _mm256_setzero_pd();
        __m256d acc1 = _mm256_setzero_pd();
        __m256d acc2 = _mm256_setzero_pd();
        __m256d acc3 = _mm256_setzero_pd();
        __m256d acc4 = _mm256_setzero_pd();
        __m256d acc5 = _mm256_setzero_pd();
        __m256d acc6 = _mm256_setzero_pd();
        __m256d acc7 = _mm256_setzero_pd();
        std::size_t i = 0;
        for (; i + 32 <= n; i += 32) {
            acc0 = _mm256_add_pd(acc0, _mm256_loadu_pd(data + i));
            acc1 = _mm256_add_pd(acc1, _mm256_loadu_pd(data + i + 4));
            acc2 = _mm256_add_pd(acc2, _mm256_loadu_pd(data + i + 8));
            acc3 = _mm256_add_pd(acc3, _mm256_loadu_pd(data + i + 12));
            acc4 = _mm256_add_pd(acc4, _mm256_loadu_pd(data + i + 16));
            acc5 = _mm256_add_pd(acc5, _mm256_loadu_pd(data + i + 20));
            acc6 = _mm256_add_pd(acc6, _mm256_loadu_pd(data + i + 24));
            acc7 = _mm256_add_pd(acc7, _mm256_loadu_pd(data + i + 28));
        }
        acc0 = _mm256_add_pd(
            _mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3)),
            _mm256_add_pd(_mm256_add_pd(acc4, acc5), _mm256_add_pd(acc6, acc7)));
        acc1 = _mm256_setzero_pd();
        acc2 = _mm256_setzero_pd();
        acc3 = _mm256_setzero_pd();
        for (; i + 16 <= n; i += 16) {
            acc0 = _mm256_add_pd(acc0, _mm256_loadu_pd(data + i));
            acc1 = _mm256_add_pd(acc1, _mm256_loadu_pd(data + i + 4));
            acc2 = _mm256_add_pd(acc2, _mm256_loadu_pd(data + i + 8));
            acc3 = _mm256_add_pd(acc3, _mm256_loadu_pd(data + i + 12));
        }
        acc0 = _mm256_add_pd(acc0, _mm256_add_pd(_mm256_add_pd(acc1, acc2), acc3));
        for (; i + 4 <= n; i += 4) {
            acc0 = _mm256_add_pd(acc0, _mm256_loadu_pd(data + i));
        }
        alignas(32) double lanes[4];
        _mm256_store_pd(lanes, acc0);
        double total = lanes[0] + lanes[1] + lanes[2] + lanes[3];
        for (; i < n; ++i) {
            total += data[i];
        }
        return total;
    }
#endif
    return sum_contiguous_scalar(data, n);
}

template <typename T>
inline T max_contiguous_scalar(const T* data, std::size_t n) {
    T best = data[0];
    for (std::size_t i = 1; i < n; ++i) {
        best = std::max(best, data[i]);
    }
    return best;
}

template <typename T>
inline T max_contiguous(const T* data, std::size_t n) {
    assert(n > 0);
#if defined(__AVX2__)
    if constexpr (std::is_same<T, float>::value) {
        if (n >= 64) {
            std::size_t i = 64;
            __m256 acc0 = _mm256_loadu_ps(data);
            __m256 acc1 = _mm256_loadu_ps(data + 8);
            __m256 acc2 = _mm256_loadu_ps(data + 16);
            __m256 acc3 = _mm256_loadu_ps(data + 24);
            __m256 acc4 = _mm256_loadu_ps(data + 32);
            __m256 acc5 = _mm256_loadu_ps(data + 40);
            __m256 acc6 = _mm256_loadu_ps(data + 48);
            __m256 acc7 = _mm256_loadu_ps(data + 56);
            for (; i + 64 <= n; i += 64) {
                acc0 = _mm256_max_ps(acc0, _mm256_loadu_ps(data + i));
                acc1 = _mm256_max_ps(acc1, _mm256_loadu_ps(data + i + 8));
                acc2 = _mm256_max_ps(acc2, _mm256_loadu_ps(data + i + 16));
                acc3 = _mm256_max_ps(acc3, _mm256_loadu_ps(data + i + 24));
                acc4 = _mm256_max_ps(acc4, _mm256_loadu_ps(data + i + 32));
                acc5 = _mm256_max_ps(acc5, _mm256_loadu_ps(data + i + 40));
                acc6 = _mm256_max_ps(acc6, _mm256_loadu_ps(data + i + 48));
                acc7 = _mm256_max_ps(acc7, _mm256_loadu_ps(data + i + 56));
            }
            __m256 best = _mm256_max_ps(
                _mm256_max_ps(_mm256_max_ps(acc0, acc1), _mm256_max_ps(acc2, acc3)),
                _mm256_max_ps(_mm256_max_ps(acc4, acc5), _mm256_max_ps(acc6, acc7)));
            for (; i + 8 <= n; i += 8) {
                best = _mm256_max_ps(best, _mm256_loadu_ps(data + i));
            }
            alignas(32) float lanes[8];
            _mm256_store_ps(lanes, best);
            float scalar_best = lanes[0];
            for (std::size_t lane = 1; lane < 8; ++lane) {
                scalar_best = std::max(scalar_best, lanes[lane]);
            }
            for (; i < n; ++i) {
                scalar_best = std::max(scalar_best, data[i]);
            }
            return scalar_best;
        }
        if (n >= 32) {
            std::size_t i = 32;
            __m256 acc0 = _mm256_loadu_ps(data);
            __m256 acc1 = _mm256_loadu_ps(data + 8);
            __m256 acc2 = _mm256_loadu_ps(data + 16);
            __m256 acc3 = _mm256_loadu_ps(data + 24);
            for (; i + 32 <= n; i += 32) {
                acc0 = _mm256_max_ps(acc0, _mm256_loadu_ps(data + i));
                acc1 = _mm256_max_ps(acc1, _mm256_loadu_ps(data + i + 8));
                acc2 = _mm256_max_ps(acc2, _mm256_loadu_ps(data + i + 16));
                acc3 = _mm256_max_ps(acc3, _mm256_loadu_ps(data + i + 24));
            }
            __m256 best = _mm256_max_ps(_mm256_max_ps(acc0, acc1), _mm256_max_ps(acc2, acc3));
            for (; i + 8 <= n; i += 8) {
                best = _mm256_max_ps(best, _mm256_loadu_ps(data + i));
            }
            alignas(32) float lanes[8];
            _mm256_store_ps(lanes, best);
            float scalar_best = lanes[0];
            for (std::size_t lane = 1; lane < 8; ++lane) {
                scalar_best = std::max(scalar_best, lanes[lane]);
            }
            for (; i < n; ++i) {
                scalar_best = std::max(scalar_best, data[i]);
            }
            return scalar_best;
        }
        if (n >= 8) {
            std::size_t i = 8;
            __m256 best = _mm256_loadu_ps(data);
            for (; i + 8 <= n; i += 8) {
                best = _mm256_max_ps(best, _mm256_loadu_ps(data + i));
            }
            alignas(32) float lanes[8];
            _mm256_store_ps(lanes, best);
            float scalar_best = lanes[0];
            for (std::size_t lane = 1; lane < 8; ++lane) {
                scalar_best = std::max(scalar_best, lanes[lane]);
            }
            for (; i < n; ++i) {
                scalar_best = std::max(scalar_best, data[i]);
            }
            return scalar_best;
        }
    }
    if constexpr (std::is_same<T, double>::value) {
        if (n >= 16) {
            std::size_t i = 16;
            __m256d acc0 = _mm256_loadu_pd(data);
            __m256d acc1 = _mm256_loadu_pd(data + 4);
            __m256d acc2 = _mm256_loadu_pd(data + 8);
            __m256d acc3 = _mm256_loadu_pd(data + 12);
            for (; i + 16 <= n; i += 16) {
                acc0 = _mm256_max_pd(acc0, _mm256_loadu_pd(data + i));
                acc1 = _mm256_max_pd(acc1, _mm256_loadu_pd(data + i + 4));
                acc2 = _mm256_max_pd(acc2, _mm256_loadu_pd(data + i + 8));
                acc3 = _mm256_max_pd(acc3, _mm256_loadu_pd(data + i + 12));
            }
            __m256d best = _mm256_max_pd(_mm256_max_pd(acc0, acc1), _mm256_max_pd(acc2, acc3));
            for (; i + 4 <= n; i += 4) {
                best = _mm256_max_pd(best, _mm256_loadu_pd(data + i));
            }
            alignas(32) double lanes[4];
            _mm256_store_pd(lanes, best);
            double scalar_best = lanes[0];
            for (std::size_t lane = 1; lane < 4; ++lane) {
                scalar_best = std::max(scalar_best, lanes[lane]);
            }
            for (; i < n; ++i) {
                scalar_best = std::max(scalar_best, data[i]);
            }
            return scalar_best;
        }
        if (n >= 4) {
            std::size_t i = 4;
            __m256d best = _mm256_loadu_pd(data);
            for (; i + 4 <= n; i += 4) {
                best = _mm256_max_pd(best, _mm256_loadu_pd(data + i));
            }
            alignas(32) double lanes[4];
            _mm256_store_pd(lanes, best);
            double scalar_best = lanes[0];
            for (std::size_t lane = 1; lane < 4; ++lane) {
                scalar_best = std::max(scalar_best, lanes[lane]);
            }
            for (; i < n; ++i) {
                scalar_best = std::max(scalar_best, data[i]);
            }
            return scalar_best;
        }
    }
#endif
    return max_contiguous_scalar(data, n);
}

#if defined(__AVX2__)
inline float horizontal_sum_ps(__m256 values) {
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, values);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7];
}

inline double horizontal_sum_pd(__m256d values) {
    alignas(32) double lanes[4];
    _mm256_store_pd(lanes, values);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}

inline void transpose8x8_ps(
    __m256& r0,
    __m256& r1,
    __m256& r2,
    __m256& r3,
    __m256& r4,
    __m256& r5,
    __m256& r6,
    __m256& r7) {
    const __m256 t0 = _mm256_unpacklo_ps(r0, r1);
    const __m256 t1 = _mm256_unpackhi_ps(r0, r1);
    const __m256 t2 = _mm256_unpacklo_ps(r2, r3);
    const __m256 t3 = _mm256_unpackhi_ps(r2, r3);
    const __m256 t4 = _mm256_unpacklo_ps(r4, r5);
    const __m256 t5 = _mm256_unpackhi_ps(r4, r5);
    const __m256 t6 = _mm256_unpacklo_ps(r6, r7);
    const __m256 t7 = _mm256_unpackhi_ps(r6, r7);

    const __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44);
    const __m256 s1 = _mm256_shuffle_ps(t0, t2, 0xee);
    const __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44);
    const __m256 s3 = _mm256_shuffle_ps(t1, t3, 0xee);
    const __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44);
    const __m256 s5 = _mm256_shuffle_ps(t4, t6, 0xee);
    const __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44);
    const __m256 s7 = _mm256_shuffle_ps(t5, t7, 0xee);

    r0 = _mm256_permute2f128_ps(s0, s4, 0x20);
    r1 = _mm256_permute2f128_ps(s1, s5, 0x20);
    r2 = _mm256_permute2f128_ps(s2, s6, 0x20);
    r3 = _mm256_permute2f128_ps(s3, s7, 0x20);
    r4 = _mm256_permute2f128_ps(s0, s4, 0x31);
    r5 = _mm256_permute2f128_ps(s1, s5, 0x31);
    r6 = _mm256_permute2f128_ps(s2, s6, 0x31);
    r7 = _mm256_permute2f128_ps(s3, s7, 0x31);
}
#endif

template <typename T>
inline void sum_rows_contiguous_serial(const T* LITENP_RESTRICT data, T* LITENP_RESTRICT out, std::size_t rows, std::size_t cols) {
#if defined(__AVX2__)
    if constexpr (std::is_same<T, float>::value) {
        std::size_t r = 0;
        const std::size_t row_pair_end = rows - rows % 2;
        for (; r < row_pair_end; r += 2) {
            const float* row0 = data + (r + 0) * cols;
            const float* row1 = data + (r + 1) * cols;
            __m256 a0 = _mm256_setzero_ps();
            __m256 a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps();
            __m256 a3 = _mm256_setzero_ps();
            __m256 b0 = _mm256_setzero_ps();
            __m256 b1 = _mm256_setzero_ps();
            __m256 b2 = _mm256_setzero_ps();
            __m256 b3 = _mm256_setzero_ps();
            std::size_t c = 0;
            const std::size_t block_end = cols - cols % 32;
            for (; c < block_end; c += 32) {
                a0 = _mm256_add_ps(a0, _mm256_loadu_ps(row0 + c));
                a1 = _mm256_add_ps(a1, _mm256_loadu_ps(row0 + c + 8));
                a2 = _mm256_add_ps(a2, _mm256_loadu_ps(row0 + c + 16));
                a3 = _mm256_add_ps(a3, _mm256_loadu_ps(row0 + c + 24));
                b0 = _mm256_add_ps(b0, _mm256_loadu_ps(row1 + c));
                b1 = _mm256_add_ps(b1, _mm256_loadu_ps(row1 + c + 8));
                b2 = _mm256_add_ps(b2, _mm256_loadu_ps(row1 + c + 16));
                b3 = _mm256_add_ps(b3, _mm256_loadu_ps(row1 + c + 24));
            }
            __m256 asum = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
            __m256 bsum = _mm256_add_ps(_mm256_add_ps(b0, b1), _mm256_add_ps(b2, b3));
            const std::size_t vec_end = cols - cols % 8;
            for (; c < vec_end; c += 8) {
                asum = _mm256_add_ps(asum, _mm256_loadu_ps(row0 + c));
                bsum = _mm256_add_ps(bsum, _mm256_loadu_ps(row1 + c));
            }
            float total0 = horizontal_sum_ps(asum);
            float total1 = horizontal_sum_ps(bsum);
            for (; c < cols; ++c) {
                total0 += row0[c];
                total1 += row1[c];
            }
            out[r + 0] = total0;
            out[r + 1] = total1;
        }
        for (; r < rows; ++r) {
            out[r] = sum_contiguous(data + r * cols, cols);
        }
        return;
    }
    if constexpr (std::is_same<T, double>::value) {
        std::size_t r = 0;
        const std::size_t row_pair_end = rows - rows % 2;
        for (; r < row_pair_end; r += 2) {
            const double* row0 = data + (r + 0) * cols;
            const double* row1 = data + (r + 1) * cols;
            __m256d a0 = _mm256_setzero_pd();
            __m256d a1 = _mm256_setzero_pd();
            __m256d a2 = _mm256_setzero_pd();
            __m256d a3 = _mm256_setzero_pd();
            __m256d b0 = _mm256_setzero_pd();
            __m256d b1 = _mm256_setzero_pd();
            __m256d b2 = _mm256_setzero_pd();
            __m256d b3 = _mm256_setzero_pd();
            std::size_t c = 0;
            const std::size_t block_end = cols - cols % 16;
            for (; c < block_end; c += 16) {
                a0 = _mm256_add_pd(a0, _mm256_loadu_pd(row0 + c));
                a1 = _mm256_add_pd(a1, _mm256_loadu_pd(row0 + c + 4));
                a2 = _mm256_add_pd(a2, _mm256_loadu_pd(row0 + c + 8));
                a3 = _mm256_add_pd(a3, _mm256_loadu_pd(row0 + c + 12));
                b0 = _mm256_add_pd(b0, _mm256_loadu_pd(row1 + c));
                b1 = _mm256_add_pd(b1, _mm256_loadu_pd(row1 + c + 4));
                b2 = _mm256_add_pd(b2, _mm256_loadu_pd(row1 + c + 8));
                b3 = _mm256_add_pd(b3, _mm256_loadu_pd(row1 + c + 12));
            }
            __m256d asum = _mm256_add_pd(_mm256_add_pd(a0, a1), _mm256_add_pd(a2, a3));
            __m256d bsum = _mm256_add_pd(_mm256_add_pd(b0, b1), _mm256_add_pd(b2, b3));
            const std::size_t vec_end = cols - cols % 4;
            for (; c < vec_end; c += 4) {
                asum = _mm256_add_pd(asum, _mm256_loadu_pd(row0 + c));
                bsum = _mm256_add_pd(bsum, _mm256_loadu_pd(row1 + c));
            }
            double total0 = horizontal_sum_pd(asum);
            double total1 = horizontal_sum_pd(bsum);
            for (; c < cols; ++c) {
                total0 += row0[c];
                total1 += row1[c];
            }
            out[r + 0] = total0;
            out[r + 1] = total1;
        }
        for (; r < rows; ++r) {
            out[r] = sum_contiguous(data + r * cols, cols);
        }
        return;
    }
#endif
    for (std::size_t r = 0; r < rows; ++r) {
        out[r] = sum_contiguous(data + r * cols, cols);
    }
}

template <typename T>
inline void sum_rows_contiguous(const T* LITENP_RESTRICT data, T* LITENP_RESTRICT out, std::size_t rows, std::size_t cols) {
#if defined(LITENP_USE_OPENMP)
    if (detail::use_openmp_for(rows * cols, 1u << 22)) {
#pragma omp parallel for
        for (std::ptrdiff_t r = 0; r < static_cast<std::ptrdiff_t>(rows); ++r) {
            const std::size_t row = static_cast<std::size_t>(r);
            out[row] = detail::sum_contiguous(data + row * cols, cols);
        }
        return;
    }
#endif
    sum_rows_contiguous_serial(data, out, rows, cols);
}

template <typename T>
inline void sum_cols_contiguous_serial(const T* LITENP_RESTRICT data, T* LITENP_RESTRICT out, std::size_t rows, std::size_t cols) {
    if (rows == 0) {
        std::fill(out, out + cols, T{});
        return;
    }
#if defined(__AVX2__)
    if constexpr (std::is_same<T, float>::value) {
        std::size_t r = 0;
        const auto store_four_row_sum = [&](const float* row0, const float* row1, const float* row2, const float* row3) {
            std::size_t c = 0;
            const std::size_t block_end = cols - cols % 32;
            for (; c < block_end; c += 32) {
                _mm256_storeu_ps(out + c,
                    _mm256_add_ps(
                        _mm256_add_ps(_mm256_loadu_ps(row0 + c), _mm256_loadu_ps(row1 + c)),
                        _mm256_add_ps(_mm256_loadu_ps(row2 + c), _mm256_loadu_ps(row3 + c))));
                _mm256_storeu_ps(out + c + 8,
                    _mm256_add_ps(
                        _mm256_add_ps(_mm256_loadu_ps(row0 + c + 8), _mm256_loadu_ps(row1 + c + 8)),
                        _mm256_add_ps(_mm256_loadu_ps(row2 + c + 8), _mm256_loadu_ps(row3 + c + 8))));
                _mm256_storeu_ps(out + c + 16,
                    _mm256_add_ps(
                        _mm256_add_ps(_mm256_loadu_ps(row0 + c + 16), _mm256_loadu_ps(row1 + c + 16)),
                        _mm256_add_ps(_mm256_loadu_ps(row2 + c + 16), _mm256_loadu_ps(row3 + c + 16))));
                _mm256_storeu_ps(out + c + 24,
                    _mm256_add_ps(
                        _mm256_add_ps(_mm256_loadu_ps(row0 + c + 24), _mm256_loadu_ps(row1 + c + 24)),
                        _mm256_add_ps(_mm256_loadu_ps(row2 + c + 24), _mm256_loadu_ps(row3 + c + 24))));
            }
            const std::size_t vec_end = cols - cols % 8;
            for (; c < vec_end; c += 8) {
                _mm256_storeu_ps(out + c,
                    _mm256_add_ps(
                        _mm256_add_ps(_mm256_loadu_ps(row0 + c), _mm256_loadu_ps(row1 + c)),
                        _mm256_add_ps(_mm256_loadu_ps(row2 + c), _mm256_loadu_ps(row3 + c))));
            }
            for (; c < cols; ++c) {
                out[c] = row0[c] + row1[c] + row2[c] + row3[c];
            }
        };
        const auto add_four_rows = [&](const float* row0, const float* row1, const float* row2, const float* row3) {
            std::size_t c = 0;
            const std::size_t block_end = cols - cols % 32;
            for (; c < block_end; c += 32) {
                _mm256_storeu_ps(out + c,
                    _mm256_add_ps(_mm256_loadu_ps(out + c),
                        _mm256_add_ps(
                            _mm256_add_ps(_mm256_loadu_ps(row0 + c), _mm256_loadu_ps(row1 + c)),
                            _mm256_add_ps(_mm256_loadu_ps(row2 + c), _mm256_loadu_ps(row3 + c)))));
                _mm256_storeu_ps(out + c + 8,
                    _mm256_add_ps(_mm256_loadu_ps(out + c + 8),
                        _mm256_add_ps(
                            _mm256_add_ps(_mm256_loadu_ps(row0 + c + 8), _mm256_loadu_ps(row1 + c + 8)),
                            _mm256_add_ps(_mm256_loadu_ps(row2 + c + 8), _mm256_loadu_ps(row3 + c + 8)))));
                _mm256_storeu_ps(out + c + 16,
                    _mm256_add_ps(_mm256_loadu_ps(out + c + 16),
                        _mm256_add_ps(
                            _mm256_add_ps(_mm256_loadu_ps(row0 + c + 16), _mm256_loadu_ps(row1 + c + 16)),
                            _mm256_add_ps(_mm256_loadu_ps(row2 + c + 16), _mm256_loadu_ps(row3 + c + 16)))));
                _mm256_storeu_ps(out + c + 24,
                    _mm256_add_ps(_mm256_loadu_ps(out + c + 24),
                        _mm256_add_ps(
                            _mm256_add_ps(_mm256_loadu_ps(row0 + c + 24), _mm256_loadu_ps(row1 + c + 24)),
                            _mm256_add_ps(_mm256_loadu_ps(row2 + c + 24), _mm256_loadu_ps(row3 + c + 24)))));
            }
            const std::size_t vec_end = cols - cols % 8;
            for (; c < vec_end; c += 8) {
                _mm256_storeu_ps(out + c,
                    _mm256_add_ps(_mm256_loadu_ps(out + c),
                        _mm256_add_ps(
                            _mm256_add_ps(_mm256_loadu_ps(row0 + c), _mm256_loadu_ps(row1 + c)),
                            _mm256_add_ps(_mm256_loadu_ps(row2 + c), _mm256_loadu_ps(row3 + c)))));
            }
            for (; c < cols; ++c) {
                out[c] += row0[c] + row1[c] + row2[c] + row3[c];
            }
        };
        if (rows >= 4) {
            store_four_row_sum(data, data + cols, data + 2 * cols, data + 3 * cols);
            r = 4;
        } else {
            std::copy(data, data + cols, out);
            r = 1;
        }
        for (; r + 4 <= rows; r += 4) {
            add_four_rows(data + r * cols, data + (r + 1) * cols, data + (r + 2) * cols, data + (r + 3) * cols);
        }
        for (; r < rows; ++r) {
            const float* row = data + r * cols;
            std::size_t c = 0;
            const std::size_t block_end = cols - cols % 32;
            for (; c < block_end; c += 32) {
                _mm256_storeu_ps(out + c, _mm256_add_ps(_mm256_loadu_ps(out + c), _mm256_loadu_ps(row + c)));
                _mm256_storeu_ps(out + c + 8, _mm256_add_ps(_mm256_loadu_ps(out + c + 8), _mm256_loadu_ps(row + c + 8)));
                _mm256_storeu_ps(out + c + 16, _mm256_add_ps(_mm256_loadu_ps(out + c + 16), _mm256_loadu_ps(row + c + 16)));
                _mm256_storeu_ps(out + c + 24, _mm256_add_ps(_mm256_loadu_ps(out + c + 24), _mm256_loadu_ps(row + c + 24)));
            }
            const std::size_t vec_end = cols - cols % 8;
            for (; c < vec_end; c += 8) {
                _mm256_storeu_ps(out + c, _mm256_add_ps(_mm256_loadu_ps(out + c), _mm256_loadu_ps(row + c)));
            }
            for (; c < cols; ++c) {
                out[c] += row[c];
            }
        }
        return;
    }
#endif
    std::copy(data, data + cols, out);
    for (std::size_t r = 1; r < rows; ++r) {
        const T* row = data + r * cols;
        for (std::size_t c = 0; c < cols; ++c) {
            out[c] += row[c];
        }
    }
}

#if defined(__AVX2__) && defined(__FMA__)
inline void matmul_f32_8x8_kernel(const float* a, const float* b, float* out, std::size_t m, std::size_t k, std::size_t n) {
    const std::size_t mb = m / 8;
    const std::size_t nb = n / 8;
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(m * n * k, 1u << 28))
#endif
    for (std::ptrdiff_t bi = 0; bi < static_cast<std::ptrdiff_t>(mb); ++bi) {
        for (std::ptrdiff_t bj = 0; bj < static_cast<std::ptrdiff_t>(nb); ++bj) {
            const std::size_t i = static_cast<std::size_t>(bi) * 8;
            const std::size_t j = static_cast<std::size_t>(bj) * 8;
            __m256 c0 = _mm256_setzero_ps();
            __m256 c1 = _mm256_setzero_ps();
            __m256 c2 = _mm256_setzero_ps();
            __m256 c3 = _mm256_setzero_ps();
            __m256 c4 = _mm256_setzero_ps();
            __m256 c5 = _mm256_setzero_ps();
            __m256 c6 = _mm256_setzero_ps();
            __m256 c7 = _mm256_setzero_ps();
            for (std::size_t kk = 0; kk < k; ++kk) {
                const __m256 bv = _mm256_loadu_ps(b + kk * n + j);
                c0 = _mm256_fmadd_ps(_mm256_set1_ps(a[(i + 0) * k + kk]), bv, c0);
                c1 = _mm256_fmadd_ps(_mm256_set1_ps(a[(i + 1) * k + kk]), bv, c1);
                c2 = _mm256_fmadd_ps(_mm256_set1_ps(a[(i + 2) * k + kk]), bv, c2);
                c3 = _mm256_fmadd_ps(_mm256_set1_ps(a[(i + 3) * k + kk]), bv, c3);
                c4 = _mm256_fmadd_ps(_mm256_set1_ps(a[(i + 4) * k + kk]), bv, c4);
                c5 = _mm256_fmadd_ps(_mm256_set1_ps(a[(i + 5) * k + kk]), bv, c5);
                c6 = _mm256_fmadd_ps(_mm256_set1_ps(a[(i + 6) * k + kk]), bv, c6);
                c7 = _mm256_fmadd_ps(_mm256_set1_ps(a[(i + 7) * k + kk]), bv, c7);
            }
            _mm256_storeu_ps(out + (i + 0) * n + j, c0);
            _mm256_storeu_ps(out + (i + 1) * n + j, c1);
            _mm256_storeu_ps(out + (i + 2) * n + j, c2);
            _mm256_storeu_ps(out + (i + 3) * n + j, c3);
            _mm256_storeu_ps(out + (i + 4) * n + j, c4);
            _mm256_storeu_ps(out + (i + 5) * n + j, c5);
            _mm256_storeu_ps(out + (i + 6) * n + j, c6);
            _mm256_storeu_ps(out + (i + 7) * n + j, c7);
        }
    }
}

inline void matmul_f32_4x8_kernel(const float* a, const float* b, float* out, std::size_t m, std::size_t k, std::size_t n) {
    const std::size_t mb = m / 4;
    const std::size_t nb = n / 8;
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(m * n * k, 1u << 28))
#endif
    for (std::ptrdiff_t bi = 0; bi < static_cast<std::ptrdiff_t>(mb); ++bi) {
        for (std::ptrdiff_t bj = 0; bj < static_cast<std::ptrdiff_t>(nb); ++bj) {
            const std::size_t i = static_cast<std::size_t>(bi) * 4;
            const std::size_t j = static_cast<std::size_t>(bj) * 8;
            __m256 c0 = _mm256_setzero_ps();
            __m256 c1 = _mm256_setzero_ps();
            __m256 c2 = _mm256_setzero_ps();
            __m256 c3 = _mm256_setzero_ps();
            for (std::size_t kk = 0; kk < k; ++kk) {
                const __m256 bv = _mm256_loadu_ps(b + kk * n + j);
                c0 = _mm256_fmadd_ps(_mm256_set1_ps(a[(i + 0) * k + kk]), bv, c0);
                c1 = _mm256_fmadd_ps(_mm256_set1_ps(a[(i + 1) * k + kk]), bv, c1);
                c2 = _mm256_fmadd_ps(_mm256_set1_ps(a[(i + 2) * k + kk]), bv, c2);
                c3 = _mm256_fmadd_ps(_mm256_set1_ps(a[(i + 3) * k + kk]), bv, c3);
            }
            _mm256_storeu_ps(out + (i + 0) * n + j, c0);
            _mm256_storeu_ps(out + (i + 1) * n + j, c1);
            _mm256_storeu_ps(out + (i + 2) * n + j, c2);
            _mm256_storeu_ps(out + (i + 3) * n + j, c3);
        }
    }
}

inline void matmul_f32_4x16_kernel(const float* a, const float* b, float* out, std::size_t m, std::size_t k, std::size_t n) {
    const std::size_t mb = m / 4;
    const std::size_t nb = n / 16;
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(m * n * k, 1u << 28))
#endif
    for (std::ptrdiff_t bi = 0; bi < static_cast<std::ptrdiff_t>(mb); ++bi) {
        for (std::ptrdiff_t bj = 0; bj < static_cast<std::ptrdiff_t>(nb); ++bj) {
            const std::size_t i = static_cast<std::size_t>(bi) * 4;
            const std::size_t j = static_cast<std::size_t>(bj) * 16;
            __m256 c00 = _mm256_setzero_ps();
            __m256 c01 = _mm256_setzero_ps();
            __m256 c10 = _mm256_setzero_ps();
            __m256 c11 = _mm256_setzero_ps();
            __m256 c20 = _mm256_setzero_ps();
            __m256 c21 = _mm256_setzero_ps();
            __m256 c30 = _mm256_setzero_ps();
            __m256 c31 = _mm256_setzero_ps();
            for (std::size_t kk = 0; kk < k; ++kk) {
                const __m256 b0 = _mm256_loadu_ps(b + kk * n + j);
                const __m256 b1 = _mm256_loadu_ps(b + kk * n + j + 8);
                const __m256 a0 = _mm256_set1_ps(a[(i + 0) * k + kk]);
                const __m256 a1 = _mm256_set1_ps(a[(i + 1) * k + kk]);
                const __m256 a2 = _mm256_set1_ps(a[(i + 2) * k + kk]);
                const __m256 a3 = _mm256_set1_ps(a[(i + 3) * k + kk]);
                c00 = _mm256_fmadd_ps(a0, b0, c00);
                c01 = _mm256_fmadd_ps(a0, b1, c01);
                c10 = _mm256_fmadd_ps(a1, b0, c10);
                c11 = _mm256_fmadd_ps(a1, b1, c11);
                c20 = _mm256_fmadd_ps(a2, b0, c20);
                c21 = _mm256_fmadd_ps(a2, b1, c21);
                c30 = _mm256_fmadd_ps(a3, b0, c30);
                c31 = _mm256_fmadd_ps(a3, b1, c31);
            }
            _mm256_storeu_ps(out + (i + 0) * n + j, c00);
            _mm256_storeu_ps(out + (i + 0) * n + j + 8, c01);
            _mm256_storeu_ps(out + (i + 1) * n + j, c10);
            _mm256_storeu_ps(out + (i + 1) * n + j + 8, c11);
            _mm256_storeu_ps(out + (i + 2) * n + j, c20);
            _mm256_storeu_ps(out + (i + 2) * n + j + 8, c21);
            _mm256_storeu_ps(out + (i + 3) * n + j, c30);
            _mm256_storeu_ps(out + (i + 3) * n + j + 8, c31);
        }
    }
}

inline void pack_b_panel_16(const float* LITENP_RESTRICT b, float* LITENP_RESTRICT packed_b, std::size_t k, std::size_t n, std::size_t j) {
    for (std::size_t kk = 0; kk < k; ++kk) {
        std::memcpy(packed_b + kk * 16, b + kk * n + j, 16 * sizeof(float));
    }
}

inline void matmul_f32_4x16_packed_b_tile(
    const float* LITENP_RESTRICT a,
    const float* LITENP_RESTRICT packed_b,
    float* LITENP_RESTRICT out,
    std::size_t i,
    std::size_t k,
    std::size_t n,
    std::size_t j) {
    const float* a0p = a + (i + 0) * k;
    const float* a1p = a + (i + 1) * k;
    const float* a2p = a + (i + 2) * k;
    const float* a3p = a + (i + 3) * k;
    __m256 c00 = _mm256_setzero_ps();
    __m256 c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps();
    __m256 c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps();
    __m256 c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps();
    __m256 c31 = _mm256_setzero_ps();
    const float* LITENP_RESTRICT bp = packed_b;
    for (std::size_t kk = 0; kk < k; ++kk, bp += 16) {
        const __m256 b0 = _mm256_loadu_ps(bp);
        const __m256 b1 = _mm256_loadu_ps(bp + 8);
        const __m256 a0 = _mm256_set1_ps(*a0p++);
        const __m256 a1 = _mm256_set1_ps(*a1p++);
        const __m256 a2 = _mm256_set1_ps(*a2p++);
        const __m256 a3 = _mm256_set1_ps(*a3p++);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);
    }
    _mm256_storeu_ps(out + (i + 0) * n + j, c00);
    _mm256_storeu_ps(out + (i + 0) * n + j + 8, c01);
    _mm256_storeu_ps(out + (i + 1) * n + j, c10);
    _mm256_storeu_ps(out + (i + 1) * n + j + 8, c11);
    _mm256_storeu_ps(out + (i + 2) * n + j, c20);
    _mm256_storeu_ps(out + (i + 2) * n + j + 8, c21);
    _mm256_storeu_ps(out + (i + 3) * n + j, c30);
    _mm256_storeu_ps(out + (i + 3) * n + j + 8, c31);
}

inline void matmul_f32_6x16_packed_b_tile(
    const float* LITENP_RESTRICT a,
    const float* LITENP_RESTRICT packed_b,
    float* LITENP_RESTRICT out,
    std::size_t i,
    std::size_t k,
    std::size_t n,
    std::size_t j) {
    const float* a0p = a + (i + 0) * k;
    const float* a1p = a + (i + 1) * k;
    const float* a2p = a + (i + 2) * k;
    const float* a3p = a + (i + 3) * k;
    const float* a4p = a + (i + 4) * k;
    const float* a5p = a + (i + 5) * k;
    __m256 c00 = _mm256_setzero_ps();
    __m256 c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps();
    __m256 c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps();
    __m256 c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps();
    __m256 c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps();
    __m256 c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps();
    __m256 c51 = _mm256_setzero_ps();
    const float* LITENP_RESTRICT bp = packed_b;
    for (std::size_t kk = 0; kk < k; ++kk, bp += 16) {
        const __m256 b0 = _mm256_loadu_ps(bp);
        const __m256 b1 = _mm256_loadu_ps(bp + 8);
        __m256 av = _mm256_set1_ps(*a0p++);
        c00 = _mm256_fmadd_ps(av, b0, c00);
        c01 = _mm256_fmadd_ps(av, b1, c01);
        av = _mm256_set1_ps(*a1p++);
        c10 = _mm256_fmadd_ps(av, b0, c10);
        c11 = _mm256_fmadd_ps(av, b1, c11);
        av = _mm256_set1_ps(*a2p++);
        c20 = _mm256_fmadd_ps(av, b0, c20);
        c21 = _mm256_fmadd_ps(av, b1, c21);
        av = _mm256_set1_ps(*a3p++);
        c30 = _mm256_fmadd_ps(av, b0, c30);
        c31 = _mm256_fmadd_ps(av, b1, c31);
        av = _mm256_set1_ps(*a4p++);
        c40 = _mm256_fmadd_ps(av, b0, c40);
        c41 = _mm256_fmadd_ps(av, b1, c41);
        av = _mm256_set1_ps(*a5p++);
        c50 = _mm256_fmadd_ps(av, b0, c50);
        c51 = _mm256_fmadd_ps(av, b1, c51);
    }
    _mm256_storeu_ps(out + (i + 0) * n + j, c00);
    _mm256_storeu_ps(out + (i + 0) * n + j + 8, c01);
    _mm256_storeu_ps(out + (i + 1) * n + j, c10);
    _mm256_storeu_ps(out + (i + 1) * n + j + 8, c11);
    _mm256_storeu_ps(out + (i + 2) * n + j, c20);
    _mm256_storeu_ps(out + (i + 2) * n + j + 8, c21);
    _mm256_storeu_ps(out + (i + 3) * n + j, c30);
    _mm256_storeu_ps(out + (i + 3) * n + j + 8, c31);
    _mm256_storeu_ps(out + (i + 4) * n + j, c40);
    _mm256_storeu_ps(out + (i + 4) * n + j + 8, c41);
    _mm256_storeu_ps(out + (i + 5) * n + j, c50);
    _mm256_storeu_ps(out + (i + 5) * n + j + 8, c51);
}

inline void matmul_f32_packed_b_tail_rows(
    const float* LITENP_RESTRICT a,
    const float* LITENP_RESTRICT packed_b,
    float* LITENP_RESTRICT out,
    std::size_t i,
    std::size_t rows,
    std::size_t k,
    std::size_t n,
    std::size_t j) {
    for (std::size_t r = 0; r < rows; ++r) {
        float sums[16] = {};
        const float* arow = a + (i + r) * k;
        for (std::size_t kk = 0; kk < k; ++kk) {
            const float av = arow[kk];
            const float* bp = packed_b + kk * 16;
            for (std::size_t c = 0; c < 16; ++c) {
                sums[c] += av * bp[c];
            }
        }
        std::copy(sums, sums + 16, out + (i + r) * n + j);
    }
}

inline void matmul_f32_4x16_packed_b_kernel(
    const float* LITENP_RESTRICT a,
    const float* LITENP_RESTRICT b,
    float* LITENP_RESTRICT out,
    std::size_t m,
    std::size_t k,
    std::size_t n) {
    const std::size_t mb = m / 4;
    const std::size_t nb = n / 16;
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel if (detail::use_openmp_for(m * n * k, 1u << 28))
    {
        std::unique_ptr<float[]> packed_b(new float[k * 16]);
#pragma omp for
        for (std::ptrdiff_t bj = 0; bj < static_cast<std::ptrdiff_t>(nb); ++bj) {
            const std::size_t j = static_cast<std::size_t>(bj) * 16;
            pack_b_panel_16(b, packed_b.get(), k, n, j);
            for (std::size_t bi = 0; bi < mb; ++bi) {
                matmul_f32_4x16_packed_b_tile(a, packed_b.get(), out, bi * 4, k, n, j);
            }
        }
    }
#else
    std::unique_ptr<float[]> packed_b(new float[k * 16]);
    for (std::size_t bj = 0; bj < nb; ++bj) {
        const std::size_t j = bj * 16;
        pack_b_panel_16(b, packed_b.get(), k, n, j);
        for (std::size_t bi = 0; bi < mb; ++bi) {
            matmul_f32_4x16_packed_b_tile(a, packed_b.get(), out, bi * 4, k, n, j);
        }
    }
#endif
}

inline void matmul_f32_6x16_packed_b_kernel(
    const float* LITENP_RESTRICT a,
    const float* LITENP_RESTRICT b,
    float* LITENP_RESTRICT out,
    std::size_t m,
    std::size_t k,
    std::size_t n) {
    const std::size_t nb = n / 16;
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel if (detail::use_openmp_for(m * n * k, 1u << 28))
    {
        std::unique_ptr<float[]> packed_b(new float[k * 16]);
#pragma omp for
        for (std::ptrdiff_t bj = 0; bj < static_cast<std::ptrdiff_t>(nb); ++bj) {
            const std::size_t j = static_cast<std::size_t>(bj) * 16;
            pack_b_panel_16(b, packed_b.get(), k, n, j);
            std::size_t i = 0;
            for (; i + 6 <= m; i += 6) {
                matmul_f32_6x16_packed_b_tile(a, packed_b.get(), out, i, k, n, j);
            }
            if (i + 4 <= m) {
                matmul_f32_4x16_packed_b_tile(a, packed_b.get(), out, i, k, n, j);
                i += 4;
            }
            if (i < m) {
                matmul_f32_packed_b_tail_rows(a, packed_b.get(), out, i, m - i, k, n, j);
            }
        }
    }
#else
    std::unique_ptr<float[]> packed_b(new float[k * 16]);
    for (std::size_t bj = 0; bj < nb; ++bj) {
        const std::size_t j = bj * 16;
        pack_b_panel_16(b, packed_b.get(), k, n, j);
        std::size_t i = 0;
        for (; i + 6 <= m; i += 6) {
            matmul_f32_6x16_packed_b_tile(a, packed_b.get(), out, i, k, n, j);
        }
        if (i + 4 <= m) {
            matmul_f32_4x16_packed_b_tile(a, packed_b.get(), out, i, k, n, j);
            i += 4;
        }
        if (i < m) {
            matmul_f32_packed_b_tail_rows(a, packed_b.get(), out, i, m - i, k, n, j);
        }
    }
#endif
}
#endif

enum class GemmBackend {
    cblas,
    f32_6x16_packed_b,
    f32_4x16_packed_b,
    f32_4x16,
    f32_8x8,
    f32_4x8,
    axpy
};

inline bool gemm_work_le(std::size_t m, std::size_t k, std::size_t n, std::size_t limit) {
    if (m == 0 || k == 0 || n == 0) {
        return true;
    }
    return m <= limit / k && n <= (limit / k) / m;
}

template <typename T>
inline GemmBackend select_gemm_backend(std::size_t m, std::size_t k, std::size_t n) {
#if defined(__AVX2__) && defined(__FMA__)
    if constexpr (std::is_same<T, float>::value) {
        if (n % 16 == 0 && m >= 768 && k >= 768 && n >= 768 && gemm_work_le(m, k, n, 1024u * 1024u * 1024u)) {
            return GemmBackend::f32_6x16_packed_b;
        }
        if (m % 4 == 0 && n % 16 == 0 && m >= 384 && k >= 384 && n >= 384 && gemm_work_le(m, k, n, 1024u * 1024u * 1024u)) {
            return GemmBackend::f32_4x16_packed_b;
        }
    }
#endif
#if defined(LITENP_HAS_CBLAS)
    if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {
        if (m >= 1024 && n >= 1024 && k >= 1024) {
            return GemmBackend::cblas;
        }
    }
#endif
#if defined(__AVX2__) && defined(__FMA__)
    if constexpr (std::is_same<T, float>::value) {
        if (m % 4 == 0 && n % 16 == 0 && gemm_work_le(m, k, n, 768u * 768u * 768u)) {
            return GemmBackend::f32_4x16;
        }
        if (m % 8 == 0 && n % 8 == 0) {
            return GemmBackend::f32_8x8;
        }
        if (m % 4 == 0 && n % 8 == 0) {
            return GemmBackend::f32_4x8;
        }
    }
#endif
    return GemmBackend::axpy;
}

}  // namespace detail

inline std::size_t numel(const Shape& shape) {
    return detail::checked_product(shape);
}

template <typename T>
class ArrayView {
public:
    using value_type = T;

    ArrayView() : shape_{0}, strides_{1} {}

    ArrayView(T* data, Shape shape, Shape strides)
        : ArrayView(data, std::move(shape), std::move(strides), data) {}

    ArrayView(T* data, Shape shape, Shape strides, T* base_data)
        : data_(data), base_data_(base_data), shape_(std::move(shape)), strides_(std::move(strides)) {
        detail::require(shape_.size() == strides_.size(), "shape/stride rank mismatch");
    }

    template <typename U = T, typename = std::enable_if_t<!std::is_const<U>::value>>
    operator ArrayView<const U>() const {
        return ArrayView<const U>(data_, shape_, strides_, base_data_);
    }

    template <typename U = T, typename = std::enable_if_t<!std::is_const<U>::value>>
    U* data() {
        invalidate_metadata();
        return data_;
    }

    T* data() const {
        invalidate_metadata();
        return data_;
    }

    void invalidate_metadata() const {
        if constexpr (!std::is_const<T>::value) {
            if (metadata_invalidated_) {
                return;
            }
            detail::clear_uniform(data_);
            if (base_data_ != data_) {
                detail::clear_uniform(base_data_);
            }
            metadata_invalidated_ = true;
        }
    }

    const Shape& shape() const {
        return shape_;
    }

    const Shape& strides() const {
        return strides_;
    }

    std::size_t ndim() const {
        return shape_.size();
    }

    std::size_t size() const {
        return numel(shape_);
    }

    bool is_contiguous() const {
        return detail::is_contiguous(shape_, strides_);
    }

    std::size_t offset(const Shape& index) const {
        detail::require(index.size() == shape_.size(), "index rank mismatch");
        for (std::size_t axis = 0; axis < index.size(); ++axis) {
            if (index[axis] >= shape_[axis]) {
                throw std::out_of_range("index out of bounds");
            }
        }
        return detail::offset_for_index(index, strides_);
    }

    T& at(const Shape& index) const {
        invalidate_metadata();
        return data_[offset(index)];
    }

    T& operator()(std::initializer_list<std::size_t> index) const {
        return at(Shape(index));
    }

    ArrayView<T> reshape(Shape new_shape) const {
        detail::require(is_contiguous(), "reshape currently requires a contiguous view");
        detail::require(numel(new_shape) == size(), "reshape cannot change element count");
        return ArrayView<T>(data_, std::move(new_shape), detail::contiguous_strides(new_shape), base_data_);
    }

    ArrayView<T> flatten() const {
        return reshape({size()});
    }

    ArrayView<T> slice(std::size_t axis, std::size_t begin, std::size_t end) const {
        return slice(axis, begin, end, 1);
    }

    ArrayView<T> slice(std::size_t axis, std::size_t begin, std::size_t end, std::size_t step) const {
        detail::require(axis < ndim(), "slice axis out of range");
        detail::require(step > 0, "slice step must be positive");
        detail::require(begin <= end && end <= shape_[axis], "invalid slice range");
        Shape out_shape = shape_;
        Shape out_strides = strides_;
        out_shape[axis] = (end - begin + step - 1) / step;
        out_strides[axis] *= step;
        return ArrayView<T>(data_ + begin * strides_[axis], std::move(out_shape), std::move(out_strides), base_data_);
    }

    ArrayView<T> select(std::size_t axis, std::size_t index) const {
        detail::require(axis < ndim(), "select axis out of range");
        detail::require(index < shape_[axis], "select index out of range");
        Shape out_shape = shape_;
        Shape out_strides = strides_;
        out_shape.erase(out_shape.begin() + static_cast<std::ptrdiff_t>(axis));
        out_strides.erase(out_strides.begin() + static_cast<std::ptrdiff_t>(axis));
        if (out_shape.empty()) {
            out_shape.push_back(1);
            out_strides.push_back(1);
        }
        return ArrayView<T>(data_ + index * strides_[axis], std::move(out_shape), std::move(out_strides), base_data_);
    }

    ArrayView<T> permute(const Shape& axes) const {
        detail::require(axes.size() == ndim(), "permute rank mismatch");
        std::vector<bool> seen(ndim(), false);
        Shape out_shape(ndim());
        Shape out_strides(ndim());
        for (std::size_t i = 0; i < axes.size(); ++i) {
            const std::size_t axis = axes[i];
            detail::require(axis < ndim(), "permute axis out of range");
            detail::require(!seen[axis], "permute axes must be unique");
            seen[axis] = true;
            out_shape[i] = shape_[axis];
            out_strides[i] = strides_[axis];
        }
        return ArrayView<T>(data_, std::move(out_shape), std::move(out_strides), base_data_);
    }

    ArrayView<T> transpose() const {
        Shape axes(ndim());
        for (std::size_t i = 0; i < ndim(); ++i) {
            axes[i] = ndim() - 1 - i;
        }
        return permute(axes);
    }

    ArrayView<T> squeeze() const {
        Shape out_shape;
        Shape out_strides;
        for (std::size_t axis = 0; axis < ndim(); ++axis) {
            if (shape_[axis] != 1) {
                out_shape.push_back(shape_[axis]);
                out_strides.push_back(strides_[axis]);
            }
        }
        if (out_shape.empty()) {
            out_shape.push_back(1);
            out_strides.push_back(1);
        }
        return ArrayView<T>(data_, std::move(out_shape), std::move(out_strides), base_data_);
    }

    ArrayView<T> squeeze(std::size_t axis) const {
        detail::require(axis < ndim(), "squeeze axis out of range");
        detail::require(shape_[axis] == 1, "squeeze axis must have size 1");
        Shape out_shape = shape_;
        Shape out_strides = strides_;
        out_shape.erase(out_shape.begin() + static_cast<std::ptrdiff_t>(axis));
        out_strides.erase(out_strides.begin() + static_cast<std::ptrdiff_t>(axis));
        if (out_shape.empty()) {
            out_shape.push_back(1);
            out_strides.push_back(1);
        }
        return ArrayView<T>(data_, std::move(out_shape), std::move(out_strides), base_data_);
    }

    ArrayView<T> unsqueeze(std::size_t axis) const {
        detail::require(axis <= ndim(), "unsqueeze axis out of range");
        Shape out_shape = shape_;
        Shape out_strides = strides_;
        const std::size_t stride = axis < ndim() ? strides_[axis] * shape_[axis] : 1;
        out_shape.insert(out_shape.begin() + static_cast<std::ptrdiff_t>(axis), 1);
        out_strides.insert(out_strides.begin() + static_cast<std::ptrdiff_t>(axis), stride);
        return ArrayView<T>(data_, std::move(out_shape), std::move(out_strides), base_data_);
    }

private:
    T* data_ = nullptr;
    T* base_data_ = nullptr;
    Shape shape_;
    Shape strides_;
    mutable bool metadata_invalidated_ = false;
};

namespace detail {

inline std::size_t max_reachable_offset(const Shape& shape, const Shape& strides) {
    if (checked_product(shape) == 0) {
        return 0;
    }
    std::size_t max_offset = 0;
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        if (shape[axis] > 0) {
            max_offset += (shape[axis] - 1) * strides[axis];
        }
    }
    return max_offset;
}

template <typename A, typename B>
inline bool memory_may_overlap(ArrayView<A> a, ArrayView<B> b) {
    using AT = std::remove_const_t<A>;
    using BT = std::remove_const_t<B>;
    if constexpr (!std::is_same<AT, BT>::value) {
        return false;
    } else {
        if (a.data() == nullptr || b.data() == nullptr || a.size() == 0 || b.size() == 0) {
            return false;
        }
        const auto a_begin = reinterpret_cast<std::uintptr_t>(a.data());
        const auto b_begin = reinterpret_cast<std::uintptr_t>(b.data());
        const auto a_end = a_begin + (max_reachable_offset(a.shape(), a.strides()) + 1) * sizeof(AT);
        const auto b_end = b_begin + (max_reachable_offset(b.shape(), b.strides()) + 1) * sizeof(BT);
        return a_begin < b_end && b_begin < a_end;
    }
}

template <typename A, typename B>
inline bool same_view(ArrayView<A> a, ArrayView<B> b) {
    using AT = std::remove_const_t<A>;
    using BT = std::remove_const_t<B>;
    if constexpr (!std::is_same<AT, BT>::value) {
        return false;
    } else {
        return a.data() == b.data() && a.shape() == b.shape() && a.strides() == b.strides();
    }
}

struct UninitializedTag {};
struct ZeroedTag {};
struct UniformTag {};
struct ArangeTag {};
struct EyeTag {};
struct TwoBlockTag {};

enum class AllocationMode : std::uint8_t {
    Raw,
    Zeroed,
};

template <typename T>
struct NoInitAllocator : std::allocator<T> {
    using value_type = T;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::false_type;

    NoInitAllocator() noexcept = default;

    explicit NoInitAllocator(AllocationMode mode) noexcept : mode_(mode) {}

    template <typename U>
    NoInitAllocator(const NoInitAllocator<U>& other) noexcept : mode_(other.mode()) {}

    template <typename U>
    struct rebind {
        using other = NoInitAllocator<U>;
    };

    AllocationMode mode() const noexcept {
        return mode_;
    }

    T* allocate(std::size_t n) {
        if (n == 0) {
            return nullptr;
        }
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        const std::size_t bytes = n * sizeof(T);
        constexpr std::size_t alignment = alignof(T) > 32 ? alignof(T) : 32;
        if (mode_ == AllocationMode::Zeroed) {
#if defined(__unix__) || defined(__APPLE__)
            if (bytes >= 4096) {
                void* ptr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (ptr == MAP_FAILED) {
                    throw std::bad_alloc();
                }
                return static_cast<T*>(ptr);
            }
#endif
            void* ptr = nullptr;
#if defined(_POSIX_VERSION) || defined(__unix__) || defined(__APPLE__)
            if (posix_memalign(&ptr, alignment, bytes) != 0) {
                throw std::bad_alloc();
            }
#else
            ptr = ::operator new(bytes, std::align_val_t(alignment));
#endif
            std::memset(ptr, 0, bytes);
            return static_cast<T*>(ptr);
        }
        if (bytes >= raw_cache_threshold()) {
            auto& cache = raw_cache();
            if (cache.ptr != nullptr && cache.bytes == bytes && cache.alignment == alignment) {
                void* ptr = cache.ptr;
                cache.ptr = nullptr;
                cache.bytes = 0;
                cache.alignment = 0;
                return static_cast<T*>(ptr);
            }
        }
        return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(alignment)));
    }

    void deallocate(T* p, std::size_t n) noexcept {
        if (p == nullptr) {
            return;
        }
        clear_uniform(p);
        const std::size_t bytes = n * sizeof(T);
        constexpr std::size_t alignment = alignof(T) > 32 ? alignof(T) : 32;
        if (mode_ == AllocationMode::Zeroed) {
#if defined(__unix__) || defined(__APPLE__)
            if (bytes >= 4096) {
                munmap(p, bytes);
                return;
            }
#endif
#if defined(_POSIX_VERSION) || defined(__unix__) || defined(__APPLE__)
            std::free(p);
#else
            ::operator delete(p, std::align_val_t(alignment));
#endif
            return;
        }
        if (bytes >= raw_cache_threshold()) {
            auto& cache = raw_cache();
            if (cache.ptr != nullptr) {
                ::operator delete(cache.ptr, std::align_val_t(cache.alignment));
            }
            cache.ptr = p;
            cache.bytes = bytes;
            cache.alignment = alignment;
            return;
        }
        ::operator delete(p, std::align_val_t(alignment));
    }

    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        if constexpr (sizeof...(Args) == 0 && std::is_trivially_default_constructible<U>::value) {
            ::new (static_cast<void*>(p)) U;
        } else {
            ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
        }
    }

private:
    struct RawCache {
        void* ptr = nullptr;
        std::size_t bytes = 0;
        std::size_t alignment = 0;

        ~RawCache() {
            if (ptr != nullptr) {
                ::operator delete(ptr, std::align_val_t(alignment));
            }
        }
    };

    static constexpr std::size_t raw_cache_threshold() noexcept {
        return 1u << 20;
    }

    static RawCache& raw_cache() noexcept {
        static thread_local RawCache cache;
        return cache;
    }

    AllocationMode mode_{AllocationMode::Raw};
};

template <typename T, typename U>
inline bool operator==(const NoInitAllocator<T>& lhs, const NoInitAllocator<U>& rhs) noexcept {
    return lhs.mode() == rhs.mode();
}

template <typename T, typename U>
inline bool operator!=(const NoInitAllocator<T>& lhs, const NoInitAllocator<U>& rhs) noexcept {
    return !(lhs == rhs);
}

}  // namespace detail

template <typename T>
class Array {
public:
    using value_type = T;
    using storage_type = std::vector<T, detail::NoInitAllocator<T>>;

    Array() : shape_{0}, strides_{1}, storage_{} {}

    explicit Array(Shape shape)
        : shape_(std::move(shape)),
          strides_(detail::contiguous_strides(shape_)),
          storage_(numel(shape_), T{}) {
        detail::mark_uniform(storage_.data(), storage_.size(), T{});
    }

    Array(Shape shape, const T& value)
        : shape_(std::move(shape)),
          strides_(detail::contiguous_strides(shape_)),
          storage_(numel(shape_), value) {
        detail::mark_uniform(storage_.data(), storage_.size(), value);
    }

    Array(Shape shape, std::vector<T> values)
        : shape_(std::move(shape)),
          strides_(detail::contiguous_strides(shape_)),
          storage_(std::make_move_iterator(values.begin()), std::make_move_iterator(values.end())) {
        detail::require(storage_.size() == numel(shape_), "value count does not match shape");
        detail::clear_uniform(storage_.data());
    }

    static Array<T> from_vector(Shape shape, std::vector<T> values) {
        return Array<T>(std::move(shape), std::move(values));
    }

    static Array<T> arange(std::size_t n, T start = T{}, T step = T{1}) {
        Array<T> out = detail::make_uninitialized_array<T>({n});
        T* out_data = out.data();
        detail::arange_contiguous(out_data, n, start, step);
        detail::mark_arange(out_data, n, start, step);
        return out;
    }

    T* data() {
        materialize_virtual();
        detail::clear_uniform(storage_.data());
        return storage_.data();
    }

    const T* data() const {
        materialize_virtual();
        return storage_.data();
    }

    const Shape& shape() const {
        return shape_;
    }

    const Shape& strides() const {
        return strides_;
    }

    std::size_t ndim() const {
        return shape_.size();
    }

    std::size_t size() const {
        return is_virtual() ? detail::checked_product(shape_) : storage_.size();
    }

    bool empty() const {
        return size() == 0;
    }

    bool is_contiguous() const {
        return true;
    }

    T& operator[](std::size_t i) {
        materialize_virtual();
        detail::clear_uniform(storage_.data());
        return storage_[i];
    }

    T operator[](std::size_t i) const {
        if (virtual_uniform_) {
            return virtual_value_;
        }
        if (virtual_arange_) {
            return virtual_start_ + static_cast<T>(i) * virtual_step_;
        }
        if (virtual_eye_) {
            if (shape_.size() == 2 && shape_[1] != 0) {
                const std::size_t row = i / shape_[1];
                const std::size_t col = i - row * shape_[1];
                if (row == col && row < shape_[0]) {
                    return T{1};
                }
            }
            return T{};
        }
        if (virtual_two_block_) {
            return i < virtual_split_ ? virtual_value_ : virtual_second_;
        }
        return storage_[i];
    }

    T& at(const Shape& index) {
        return view().at(index);
    }

    const T& at(const Shape& index) const {
        return view().at(index);
    }

    T& operator()(std::initializer_list<std::size_t> index) {
        return at(Shape(index));
    }

    const T& operator()(std::initializer_list<std::size_t> index) const {
        return at(Shape(index));
    }

    ArrayView<T> view() {
        materialize_virtual();
        return ArrayView<T>(storage_.data(), shape_, strides_, storage_.data());
    }

    ArrayView<const T> view() const {
        materialize_virtual();
        return ArrayView<const T>(storage_.data(), shape_, strides_, storage_.data());
    }

    ArrayView<T> reshape(Shape new_shape) {
        return view().reshape(std::move(new_shape));
    }

    ArrayView<const T> reshape(Shape new_shape) const {
        return view().reshape(std::move(new_shape));
    }

    ArrayView<T> flatten() {
        return view().flatten();
    }

    ArrayView<const T> flatten() const {
        return view().flatten();
    }

    ArrayView<T> permute(const Shape& axes) {
        return view().permute(axes);
    }

    ArrayView<const T> permute(const Shape& axes) const {
        return view().permute(axes);
    }

    ArrayView<T> transpose() {
        return view().transpose();
    }

    ArrayView<const T> transpose() const {
        return view().transpose();
    }

    ArrayView<T> squeeze() {
        return view().squeeze();
    }

    ArrayView<const T> squeeze() const {
        return view().squeeze();
    }

    ArrayView<T> squeeze(std::size_t axis) {
        return view().squeeze(axis);
    }

    ArrayView<const T> squeeze(std::size_t axis) const {
        return view().squeeze(axis);
    }

    ArrayView<T> unsqueeze(std::size_t axis) {
        return view().unsqueeze(axis);
    }

    ArrayView<const T> unsqueeze(std::size_t axis) const {
        return view().unsqueeze(axis);
    }

    storage_type& values() {
        materialize_virtual();
        detail::clear_uniform(storage_.data());
        return storage_;
    }

    const storage_type& values() const {
        materialize_virtual();
        return storage_;
    }

    std::vector<T> to_vector() const {
        if (virtual_uniform_) {
            return std::vector<T>(size(), virtual_value_);
        }
        if (is_virtual()) {
            std::vector<T> values(size());
            for (std::size_t i = 0; i < values.size(); ++i) {
                values[i] = (*this)[i];
            }
            return values;
        }
        return std::vector<T>(storage_.begin(), storage_.end());
    }

private:
    template <typename U>
    friend Array<U> detail::make_uninitialized_array(Shape shape);
    template <typename U>
    friend Array<U> detail::make_zeroed_array(Shape shape);
    template <typename U>
    friend Array<U> detail::make_uniform_array(Shape shape, U value);
    template <typename U>
    friend Array<U> detail::make_arange_array(Shape shape, U start, U step);
    template <typename U>
    friend Array<U> detail::make_eye_array(Shape shape);
    template <typename U>
    friend Array<U> detail::make_two_block_array(Shape shape, std::size_t split, U first, U second);

    Array(Shape shape, detail::UninitializedTag)
        : shape_(std::move(shape)),
          strides_(detail::contiguous_strides(shape_)),
          storage_(numel(shape_)) {
        detail::clear_uniform(storage_.data());
    }

    Array(Shape shape, detail::ZeroedTag)
        : shape_(std::move(shape)),
          strides_(detail::contiguous_strides(shape_)),
          storage_(detail::NoInitAllocator<T>(detail::AllocationMode::Zeroed)) {
        storage_.resize(numel(shape_));
        detail::mark_uniform(storage_.data(), storage_.size(), T{});
    }

    Array(Shape shape, const T& value, detail::UniformTag)
        : shape_(std::move(shape)),
          strides_(detail::contiguous_strides(shape_)),
          storage_{},
          virtual_uniform_(numel(shape_) != 0),
          virtual_value_(value) {}

    Array(Shape shape, T start, T step, detail::ArangeTag)
        : shape_(std::move(shape)),
          strides_(detail::contiguous_strides(shape_)),
          storage_{},
          virtual_arange_(numel(shape_) != 0),
          virtual_start_(start),
          virtual_step_(step) {}

    Array(Shape shape, detail::EyeTag)
        : shape_(std::move(shape)),
          strides_(detail::contiguous_strides(shape_)),
          storage_{},
          virtual_eye_(numel(shape_) != 0) {}

    Array(Shape shape, std::size_t split, const T& first, const T& second, detail::TwoBlockTag)
        : shape_(std::move(shape)),
          strides_(detail::contiguous_strides(shape_)),
          storage_{},
          virtual_two_block_(numel(shape_) != 0),
          virtual_value_(first),
          virtual_split_(std::min(split, numel(shape_))),
          virtual_second_(second) {}

    bool is_virtual() const noexcept {
        return virtual_uniform_ || virtual_arange_ || virtual_eye_ || virtual_two_block_;
    }

    void materialize_virtual() const {
        if (!is_virtual()) {
            return;
        }
        const std::size_t n = detail::checked_product(shape_);
        storage_.resize(n);
        if (virtual_uniform_) {
            detail::fill_contiguous(storage_.data(), virtual_value_, n);
            detail::mark_uniform(storage_.data(), storage_.size(), virtual_value_);
        } else if (virtual_arange_) {
            detail::arange_contiguous(storage_.data(), n, virtual_start_, virtual_step_);
            detail::mark_arange(storage_.data(), n, virtual_start_, virtual_step_);
        } else if (virtual_two_block_) {
            const std::size_t split = std::min(virtual_split_, n);
            detail::fill_contiguous(storage_.data(), virtual_value_, split);
            detail::fill_contiguous(storage_.data() + split, virtual_second_, n - split);
            detail::clear_uniform(storage_.data());
        } else {
            detail::fill_contiguous(storage_.data(), T{}, n);
            const std::size_t diag = shape_.size() == 2 ? std::min(shape_[0], shape_[1]) : 0;
            const std::size_t cols = shape_.size() == 2 ? shape_[1] : 0;
            for (std::size_t i = 0; i < diag; ++i) {
                storage_[i * cols + i] = T{1};
            }
            detail::clear_uniform(storage_.data());
        }
        virtual_uniform_ = false;
        virtual_arange_ = false;
        virtual_eye_ = false;
        virtual_two_block_ = false;
    }

    Shape shape_;
    Shape strides_;
    mutable storage_type storage_;
    mutable bool virtual_uniform_ = false;
    mutable bool virtual_arange_ = false;
    mutable bool virtual_eye_ = false;
    mutable bool virtual_two_block_ = false;
    mutable T virtual_value_{};
    mutable T virtual_start_{};
    mutable T virtual_step_{1};
    mutable std::size_t virtual_split_ = 0;
    mutable T virtual_second_{};
};

namespace detail {

template <typename T>
Array<T> make_uninitialized_array(Shape shape) {
    return Array<T>(std::move(shape), UninitializedTag{});
}

template <typename T>
Array<T> make_zeroed_array(Shape shape) {
    return Array<T>(std::move(shape), ZeroedTag{});
}

template <typename T>
Array<T> make_uniform_array(Shape shape, T value) {
    return Array<T>(std::move(shape), value, UniformTag{});
}

template <typename T>
Array<T> make_arange_array(Shape shape, T start, T step) {
    return Array<T>(std::move(shape), start, step, ArangeTag{});
}

template <typename T>
Array<T> make_eye_array(Shape shape) {
    return Array<T>(std::move(shape), EyeTag{});
}

template <typename T>
Array<T> make_two_block_array(Shape shape, std::size_t split, T first, T second) {
    return Array<T>(std::move(shape), split, first, second, TwoBlockTag{});
}

template <typename T>
inline bool copy_2d_strided_to_contiguous(ArrayView<const T> view, T* out) {
    if (view.ndim() != 2) {
        return false;
    }
    const std::size_t rows = view.shape()[0];
    const std::size_t cols = view.shape()[1];
    const std::size_t row_stride = view.strides()[0];
    const std::size_t col_stride = view.strides()[1];
    constexpr std::size_t tile = 32;

    if (row_stride == 1) {
#if defined(__AVX2__)
        if constexpr (std::is_same<T, float>::value) {
            const bool stream_output =
                is_aligned_to(out, 32) && cols % 8 == 0 && rows * cols >= (1u << 20);
            std::size_t c = 0;
            for (; c + 8 <= cols; c += 8) {
                std::size_t r = 0;
                for (; r + 8 <= rows; r += 8) {
                    __m256 s0 = _mm256_loadu_ps(view.data() + (c + 0) * col_stride + r);
                    __m256 s1 = _mm256_loadu_ps(view.data() + (c + 1) * col_stride + r);
                    __m256 s2 = _mm256_loadu_ps(view.data() + (c + 2) * col_stride + r);
                    __m256 s3 = _mm256_loadu_ps(view.data() + (c + 3) * col_stride + r);
                    __m256 s4 = _mm256_loadu_ps(view.data() + (c + 4) * col_stride + r);
                    __m256 s5 = _mm256_loadu_ps(view.data() + (c + 5) * col_stride + r);
                    __m256 s6 = _mm256_loadu_ps(view.data() + (c + 6) * col_stride + r);
                    __m256 s7 = _mm256_loadu_ps(view.data() + (c + 7) * col_stride + r);
                    transpose8x8_ps(s0, s1, s2, s3, s4, s5, s6, s7);
                    if (stream_output) {
                        _mm256_stream_ps(out + (r + 0) * cols + c, s0);
                        _mm256_stream_ps(out + (r + 1) * cols + c, s1);
                        _mm256_stream_ps(out + (r + 2) * cols + c, s2);
                        _mm256_stream_ps(out + (r + 3) * cols + c, s3);
                        _mm256_stream_ps(out + (r + 4) * cols + c, s4);
                        _mm256_stream_ps(out + (r + 5) * cols + c, s5);
                        _mm256_stream_ps(out + (r + 6) * cols + c, s6);
                        _mm256_stream_ps(out + (r + 7) * cols + c, s7);
                    } else {
                        _mm256_storeu_ps(out + (r + 0) * cols + c, s0);
                        _mm256_storeu_ps(out + (r + 1) * cols + c, s1);
                        _mm256_storeu_ps(out + (r + 2) * cols + c, s2);
                        _mm256_storeu_ps(out + (r + 3) * cols + c, s3);
                        _mm256_storeu_ps(out + (r + 4) * cols + c, s4);
                        _mm256_storeu_ps(out + (r + 5) * cols + c, s5);
                        _mm256_storeu_ps(out + (r + 6) * cols + c, s6);
                        _mm256_storeu_ps(out + (r + 7) * cols + c, s7);
                    }
                }
                for (; r < rows; ++r) {
                    for (std::size_t cc = 0; cc < 8; ++cc) {
                        out[r * cols + c + cc] = view.data()[(c + cc) * col_stride + r];
                    }
                }
            }
            for (; c < cols; ++c) {
                const T* src = view.data() + c * col_stride;
                for (std::size_t r = 0; r < rows; ++r) {
                    out[r * cols + c] = src[r];
                }
            }
            if (stream_output) {
                _mm_sfence();
            }
            return true;
        }
#endif
        for (std::size_t c0 = 0; c0 < cols; c0 += tile) {
            const std::size_t c_end = std::min(c0 + tile, cols);
            for (std::size_t r0 = 0; r0 < rows; r0 += tile) {
                const std::size_t r_end = std::min(r0 + tile, rows);
                for (std::size_t c = c0; c < c_end; ++c) {
                    const T* src = view.data() + c * col_stride + r0;
                    for (std::size_t r = r0; r < r_end; ++r) {
                        out[r * cols + c] = src[r - r0];
                    }
                }
            }
        }
        return true;
    }

    for (std::size_t r0 = 0; r0 < rows; r0 += tile) {
        const std::size_t r_end = std::min(r0 + tile, rows);
        for (std::size_t c0 = 0; c0 < cols; c0 += tile) {
            const std::size_t c_end = std::min(c0 + tile, cols);
            for (std::size_t r = r0; r < r_end; ++r) {
                const T* src = view.data() + r * row_stride + c0 * col_stride;
                T* dst = out + r * cols + c0;
                for (std::size_t c = c0; c < c_end; ++c) {
                    dst[c - c0] = src[(c - c0) * col_stride];
                }
            }
        }
    }
    return true;
}

}  // namespace detail

template <typename T>
Array<T> zeros(Shape shape) {
    static_assert(std::is_arithmetic<T>::value, "litenp::zeros<T> requires an arithmetic element type");
    return detail::make_zeroed_array<T>(std::move(shape));
}

template <typename T>
Array<T> ones(Shape shape) {
    return detail::make_uniform_array<T>(std::move(shape), T{1});
}

template <typename T>
Array<T> full(Shape shape, const T& value) {
    return detail::make_uniform_array<T>(std::move(shape), value);
}

template <typename T>
Array<T> zeros_like(ArrayView<const T> view) {
    return zeros<T>(view.shape());
}

template <typename T>
Array<T> ones_like(ArrayView<const T> view) {
    return ones<T>(view.shape());
}

template <typename T>
Array<T> full_like(ArrayView<const T> view, const T& value) {
    return full<T>(view.shape(), value);
}

template <typename T>
Array<T> zeros_like(const Array<T>& array) {
    return zeros_like<T>(array.view());
}

template <typename T>
Array<T> ones_like(const Array<T>& array) {
    return ones_like<T>(array.view());
}

template <typename T>
Array<T> full_like(const Array<T>& array, const T& value) {
    return full_like<T>(array.view(), value);
}

template <typename T>
Array<T> linspace(T start, T stop, std::size_t count) {
    Array<T> out = detail::make_uninitialized_array<T>({count});
    T* out_data = out.data();
    if (count == 0) {
        return out;
    }
    if (count == 1) {
        out_data[0] = start;
        return out;
    }
    const double step = (static_cast<double>(stop) - static_cast<double>(start)) /
                        static_cast<double>(count - 1);
    for (std::size_t i = 0; i < count; ++i) {
        out_data[i] = static_cast<T>(static_cast<double>(start) + step * static_cast<double>(i));
    }
    out_data[count - 1] = stop;
    return out;
}

template <typename T>
Array<T> eye(std::size_t rows, std::size_t cols) {
    return detail::make_eye_array<T>({rows, cols});
}

template <typename T>
Array<T> identity(std::size_t n) {
    return eye<T>(n, n);
}

template <typename T>
Array<T> as_contiguous(ArrayView<const T> view) {
    T uniform{};
    if (detail::known_uniform_value(view.data(), view.size(), &uniform)) {
        return detail::make_uniform_array<T>(view.shape(), uniform);
    }
    if (view.is_contiguous()) {
        T start{};
        T step{};
        if (detail::known_arange_value(view.data(), view.size(), &start, &step)) {
            return detail::make_arange_array<T>(view.shape(), start, step);
        }
    }
    Array<T> out = detail::make_uninitialized_array<T>(view.shape());
    T* out_data = out.data();
    if (view.is_contiguous()) {
        std::copy(view.data(), view.data() + view.size(), out_data);
        return out;
    }
    if (detail::copy_2d_strided_to_contiguous<T>(view, out_data)) {
        return out;
    }
    Shape index;
    for (std::size_t i = 0; i < out.size(); ++i) {
        detail::linear_to_index(i, view.shape(), index);
        out_data[i] = view.data()[detail::offset_for_index(index, view.strides())];
    }
    return out;
}

template <typename T>
ArrayView<T> transpose(ArrayView<T> view) {
    return view.transpose();
}

template <typename T>
ArrayView<const T> transpose(ArrayView<const T> view) {
    return view.transpose();
}

template <typename T>
ArrayView<T> transpose(Array<T>& array) {
    return array.transpose();
}

template <typename T>
ArrayView<const T> transpose(const Array<T>& array) {
    return array.transpose();
}

template <typename T>
ArrayView<T> permute(ArrayView<T> view, const Shape& axes) {
    return view.permute(axes);
}

template <typename T>
ArrayView<const T> permute(ArrayView<const T> view, const Shape& axes) {
    return view.permute(axes);
}

template <typename T>
ArrayView<T> permute(Array<T>& array, const Shape& axes) {
    return array.permute(axes);
}

template <typename T>
ArrayView<const T> permute(const Array<T>& array, const Shape& axes) {
    return array.permute(axes);
}

template <typename T>
void copy_contiguous_to_view(const Array<T>& src, ArrayView<T> dst) {
    detail::require(src.shape() == dst.shape(), "copy output shape mismatch");
    detail::clear_uniform(dst.data());
    if (dst.is_contiguous()) {
        std::copy(src.data(), src.data() + src.size(), dst.data());
        return;
    }
    Shape index;
    for (std::size_t i = 0; i < src.size(); ++i) {
        detail::linear_to_index(i, src.shape(), index);
        dst.data()[detail::offset_for_index(index, dst.strides())] = src[i];
    }
}

template <typename T>
void binary_into(ArrayView<const T> a, ArrayView<const T> b, ArrayView<T> out, detail::BinaryOp op) {
    const Shape out_shape = detail::broadcast_shape(a.shape(), b.shape());
    detail::require(out.shape() == out_shape, "binary output shape mismatch");
    detail::clear_uniform(out.data());
    const bool exact_a = detail::same_view(out, a);
    const bool exact_b = detail::same_view(out, b);
    if ((detail::memory_may_overlap(out, a) && !exact_a) ||
        (detail::memory_may_overlap(out, b) && !exact_b)) {
        Array<T> tmp = detail::make_uninitialized_array<T>(out_shape);
        binary_into<T>(a, b, tmp.view(), op);
        copy_contiguous_to_view<T>(tmp, out);
        return;
    }

    if (a.shape() == b.shape() && a.is_contiguous() && b.is_contiguous()) {
        if (out.is_contiguous()) {
            detail::binary_contiguous(a.data(), b.data(), out.data(), out.size(), op);
            return;
        }
    }
    if (out.is_contiguous() && out_shape.size() == 2) {
        const std::size_t rows = out_shape[0];
        const std::size_t cols = out_shape[1];
        const std::size_t total = rows * cols;
        if (a.is_contiguous() && b.is_contiguous()) {
            if (a.shape() == out_shape && b.shape() == Shape{cols}) {
                if (op == detail::BinaryOp::Add) {
                    detail::add_row_broadcast_contiguous(a.data(), b.data(), out.data(), rows, cols);
                    return;
                }
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(total, 1u << 22))
#else
                (void)total;
#endif
                for (std::ptrdiff_t r = 0; r < static_cast<std::ptrdiff_t>(rows); ++r) {
                    const std::size_t row = static_cast<std::size_t>(r);
                    detail::binary_contiguous(a.data() + row * cols, b.data(), out.data() + row * cols, cols, op);
                }
                return;
            }
            if (a.shape() == Shape{cols} && b.shape() == out_shape) {
                if (op == detail::BinaryOp::Add) {
                    detail::add_row_broadcast_contiguous(b.data(), a.data(), out.data(), rows, cols);
                    return;
                }
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(total, 1u << 22))
#endif
                for (std::ptrdiff_t r = 0; r < static_cast<std::ptrdiff_t>(rows); ++r) {
                    const std::size_t row = static_cast<std::size_t>(r);
                    detail::binary_contiguous(a.data(), b.data() + row * cols, out.data() + row * cols, cols, op);
                }
                return;
            }
            if (a.shape() == out_shape && b.shape() == Shape{rows, 1}) {
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(total, 1u << 22))
#endif
                for (std::ptrdiff_t r = 0; r < static_cast<std::ptrdiff_t>(rows); ++r) {
                    const std::size_t row = static_cast<std::size_t>(r);
                    detail::binary_scalar_contiguous(a.data() + row * cols, b.data()[row], out.data() + row * cols, cols, op);
                }
                return;
            }
            if (a.shape() == Shape{rows, 1} && b.shape() == out_shape) {
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(total, 1u << 22))
#endif
                for (std::ptrdiff_t r = 0; r < static_cast<std::ptrdiff_t>(rows); ++r) {
                    const std::size_t row = static_cast<std::size_t>(r);
                    for (std::size_t c = 0; c < cols; ++c) {
                        out.data()[row * cols + c] =
                            detail::apply_binary(a.data()[row], b.data()[row * cols + c], op);
                    }
                }
                return;
            }
        }
    }
    if (out.is_contiguous() && a.is_contiguous() && b.is_contiguous()) {
        if (a.size() == 1 && b.shape() == out_shape) {
            for (std::size_t i = 0; i < out.size(); ++i) {
                out.data()[i] = detail::apply_binary(a.data()[0], b.data()[i], op);
            }
            return;
        }
        if (b.size() == 1 && a.shape() == out_shape) {
            detail::binary_scalar_contiguous(a.data(), b.data()[0], out.data(), out.size(), op);
            return;
        }
    }

    Shape out_index;
    for (std::size_t i = 0; i < out.size(); ++i) {
        detail::linear_to_index(i, out_shape, out_index);
        const std::size_t ao = detail::broadcast_offset(out_index, out_shape, a.shape(), a.strides());
        const std::size_t bo = detail::broadcast_offset(out_index, out_shape, b.shape(), b.strides());
        out.data()[detail::offset_for_index(out_index, out.strides())] =
            detail::apply_binary(a.data()[ao], b.data()[bo], op);
    }
}

template <typename To, typename From>
Array<To> astype(ArrayView<const From> view) {
    Array<To> out = detail::make_uninitialized_array<To>(view.shape());
    To* out_data = out.data();
    if (view.is_contiguous()) {
        detail::astype_contiguous<To, From>(view.data(), out_data, out.size());
        return out;
    }
    Shape index;
    for (std::size_t i = 0; i < out.size(); ++i) {
        detail::linear_to_index(i, view.shape(), index);
        out_data[i] = static_cast<To>(view.data()[detail::offset_for_index(index, view.strides())]);
    }
    return out;
}

template <typename To, typename From>
Array<To> astype(const Array<From>& array) {
    return astype<To, From>(array.view());
}

template <typename T>
void unary_into(ArrayView<const T> a, ArrayView<T> out, detail::UnaryOp op) {
    detail::require(out.shape() == a.shape(), "unary output shape mismatch");
    detail::clear_uniform(out.data());
    if (detail::memory_may_overlap(out, a) && !detail::same_view(out, a)) {
        Array<T> tmp = detail::make_uninitialized_array<T>(out.shape());
        unary_into<T>(a, tmp.view(), op);
        copy_contiguous_to_view<T>(tmp, out);
        return;
    }
    if (a.is_contiguous() && out.is_contiguous()) {
        detail::unary_contiguous(a.data(), out.data(), out.size(), op);
        return;
    }
    Shape index;
    for (std::size_t i = 0; i < out.size(); ++i) {
        detail::linear_to_index(i, a.shape(), index);
        out.data()[detail::offset_for_index(index, out.strides())] =
            detail::apply_unary(a.data()[detail::offset_for_index(index, a.strides())], op);
    }
}

template <typename T>
Array<T> unary(ArrayView<const T> a, detail::UnaryOp op) {
    Array<T> out = detail::make_uninitialized_array<T>(a.shape());
    unary_into<T>(a, out.view(), op);
    return out;
}

template <typename T>
void negative_into(ArrayView<const T> a, ArrayView<T> out) {
    unary_into<T>(a, out, detail::UnaryOp::Neg);
}

template <typename T>
void abs_into(ArrayView<const T> a, ArrayView<T> out) {
    unary_into<T>(a, out, detail::UnaryOp::Abs);
}

template <typename T>
void relu_into(ArrayView<const T> a, ArrayView<T> out) {
    unary_into<T>(a, out, detail::UnaryOp::Relu);
}

template <typename T>
void sqrt_into(ArrayView<const T> a, ArrayView<T> out) {
    unary_into<T>(a, out, detail::UnaryOp::Sqrt);
}

template <typename T>
void exp_into(ArrayView<const T> a, ArrayView<T> out) {
    unary_into<T>(a, out, detail::UnaryOp::Exp);
}

template <typename T>
void sigmoid_into(ArrayView<const T> a, ArrayView<T> out) {
    unary_into<T>(a, out, detail::UnaryOp::Sigmoid);
}

template <typename T>
Array<T> negative(ArrayView<const T> a) {
    return unary<T>(a, detail::UnaryOp::Neg);
}

template <typename T>
Array<T> abs(ArrayView<const T> a) {
    return unary<T>(a, detail::UnaryOp::Abs);
}

template <typename T>
Array<T> relu(ArrayView<const T> a) {
    return unary<T>(a, detail::UnaryOp::Relu);
}

template <typename T>
Array<T> sqrt(ArrayView<const T> a) {
    return unary<T>(a, detail::UnaryOp::Sqrt);
}

template <typename T>
Array<T> exp(ArrayView<const T> a) {
    return unary<T>(a, detail::UnaryOp::Exp);
}

template <typename T>
Array<T> sigmoid(ArrayView<const T> a) {
    return unary<T>(a, detail::UnaryOp::Sigmoid);
}

template <typename T>
Array<T> negative(const Array<T>& a) {
    return negative<T>(a.view());
}

template <typename T>
Array<T> abs(const Array<T>& a) {
    return abs<T>(a.view());
}

template <typename T>
Array<T> relu(const Array<T>& a) {
    return relu<T>(a.view());
}

template <typename T>
Array<T> sqrt(const Array<T>& a) {
    return sqrt<T>(a.view());
}

template <typename T>
Array<T> exp(const Array<T>& a) {
    return exp<T>(a.view());
}

template <typename T>
Array<T> sigmoid(const Array<T>& a) {
    return sigmoid<T>(a.view());
}

template <typename T>
Array<T> binary(ArrayView<const T> a, ArrayView<const T> b, detail::BinaryOp op) {
    Array<T> out = detail::make_uninitialized_array<T>(detail::broadcast_shape(a.shape(), b.shape()));
    binary_into<T>(a, b, out.view(), op);
    return out;
}

template <
    typename A,
    typename B,
    typename R = std::common_type_t<A, B>,
    typename = std::enable_if_t<!std::is_same<A, B>::value>>
Array<R> binary(ArrayView<const A> a, ArrayView<const B> b, detail::BinaryOp op) {
    const Shape out_shape = detail::broadcast_shape(a.shape(), b.shape());
    Array<R> out = detail::make_uninitialized_array<R>(out_shape);
    if (out.is_contiguous() && a.is_contiguous() && b.is_contiguous()) {
        if (a.shape() == b.shape() && a.shape() == out_shape) {
            detail::binary_mixed_contiguous<A, B, R>(a.data(), b.data(), out.data(), out.size(), op);
            return out;
        }
        if (a.size() == 1 && b.shape() == out_shape) {
            detail::binary_mixed_scalar_left_contiguous<A, B, R>(a.data()[0], b.data(), out.data(), out.size(), op);
            return out;
        }
        if (b.size() == 1 && a.shape() == out_shape) {
            detail::binary_mixed_scalar_right_contiguous<A, B, R>(a.data(), b.data()[0], out.data(), out.size(), op);
            return out;
        }
        if (out_shape.size() == 2) {
            const std::size_t rows = out_shape[0];
            const std::size_t cols = out_shape[1];
            const std::size_t total = rows * cols;
#if !defined(LITENP_USE_OPENMP)
            (void)total;
#endif
            if (a.shape() == out_shape && b.shape() == Shape{cols}) {
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(total, 1u << 22))
#endif
                for (std::ptrdiff_t r = 0; r < static_cast<std::ptrdiff_t>(rows); ++r) {
                    const std::size_t row = static_cast<std::size_t>(r);
                    detail::binary_mixed_contiguous<A, B, R>(
                        a.data() + row * cols, b.data(), out.data() + row * cols, cols, op);
                }
                return out;
            }
            if (a.shape() == Shape{cols} && b.shape() == out_shape) {
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(total, 1u << 22))
#endif
                for (std::ptrdiff_t r = 0; r < static_cast<std::ptrdiff_t>(rows); ++r) {
                    const std::size_t row = static_cast<std::size_t>(r);
                    detail::binary_mixed_contiguous<A, B, R>(
                        a.data(), b.data() + row * cols, out.data() + row * cols, cols, op);
                }
                return out;
            }
            if (a.shape() == out_shape && b.shape() == Shape{rows, 1}) {
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(total, 1u << 22))
#endif
                for (std::ptrdiff_t r = 0; r < static_cast<std::ptrdiff_t>(rows); ++r) {
                    const std::size_t row = static_cast<std::size_t>(r);
                    detail::binary_mixed_scalar_right_contiguous<A, B, R>(
                        a.data() + row * cols, b.data()[row], out.data() + row * cols, cols, op);
                }
                return out;
            }
            if (a.shape() == Shape{rows, 1} && b.shape() == out_shape) {
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(total, 1u << 22))
#endif
                for (std::ptrdiff_t r = 0; r < static_cast<std::ptrdiff_t>(rows); ++r) {
                    const std::size_t row = static_cast<std::size_t>(r);
                    detail::binary_mixed_scalar_left_contiguous<A, B, R>(
                        a.data()[row], b.data() + row * cols, out.data() + row * cols, cols, op);
                }
                return out;
            }
        }
    }
    Shape out_index;
    for (std::size_t i = 0; i < out.size(); ++i) {
        detail::linear_to_index(i, out_shape, out_index);
        const std::size_t ao = detail::broadcast_offset(out_index, out_shape, a.shape(), a.strides());
        const std::size_t bo = detail::broadcast_offset(out_index, out_shape, b.shape(), b.strides());
        out[i] = detail::apply_binary(
            static_cast<R>(a.data()[ao]),
            static_cast<R>(b.data()[bo]),
            op);
    }
    return out;
}

template <typename T>
void binary_scalar_into(ArrayView<const T> a, T scalar, ArrayView<T> out, detail::BinaryOp op) {
    detail::require(out.shape() == a.shape(), "binary scalar output shape mismatch");
    detail::clear_uniform(out.data());
    if (detail::memory_may_overlap(out, a) && !detail::same_view(out, a)) {
        Array<T> tmp = detail::make_uninitialized_array<T>(out.shape());
        binary_scalar_into<T>(a, scalar, tmp.view(), op);
        copy_contiguous_to_view<T>(tmp, out);
        return;
    }
    if (a.is_contiguous()) {
        if (out.is_contiguous()) {
            detail::binary_scalar_contiguous(a.data(), scalar, out.data(), out.size(), op);
            return;
        }
    }
    Shape index;
    for (std::size_t i = 0; i < out.size(); ++i) {
        detail::linear_to_index(i, a.shape(), index);
        out.data()[detail::offset_for_index(index, out.strides())] =
            detail::apply_binary(a.data()[detail::offset_for_index(index, a.strides())], scalar, op);
    }
}

template <typename T>
Array<T> binary_scalar(ArrayView<const T> a, T scalar, detail::BinaryOp op) {
    Array<T> out = detail::make_uninitialized_array<T>(a.shape());
    binary_scalar_into<T>(a, scalar, out.view(), op);
    return out;
}

template <typename T>
void add_into(ArrayView<const T> a, ArrayView<const T> b, ArrayView<T> out) {
    detail::clear_uniform(out.data());
    if (a.shape() == b.shape() && out.shape() == a.shape() &&
        a.is_contiguous() && b.is_contiguous() && out.is_contiguous()) {
        const bool exact_a = detail::same_view(out, a);
        const bool exact_b = detail::same_view(out, b);
        if ((detail::memory_may_overlap(out, a) && !exact_a) ||
            (detail::memory_may_overlap(out, b) && !exact_b)) {
            Array<T> tmp = detail::make_uninitialized_array<T>(out.shape());
            add_into<T>(a, b, tmp.view());
            copy_contiguous_to_view<T>(tmp, out);
            return;
        }
        detail::add_contiguous(a.data(), b.data(), out.data(), out.size());
        return;
    }
    binary_into<T>(a, b, out, detail::BinaryOp::Add);
}

template <typename T>
void subtract_into(ArrayView<const T> a, ArrayView<const T> b, ArrayView<T> out) {
    binary_into<T>(a, b, out, detail::BinaryOp::Sub);
}

template <typename T>
void multiply_into(ArrayView<const T> a, ArrayView<const T> b, ArrayView<T> out) {
    binary_into<T>(a, b, out, detail::BinaryOp::Mul);
}

template <typename T>
void divide_into(ArrayView<const T> a, ArrayView<const T> b, ArrayView<T> out) {
    binary_into<T>(a, b, out, detail::BinaryOp::Div);
}

template <typename T>
void minimum_into(ArrayView<const T> a, ArrayView<const T> b, ArrayView<T> out) {
    binary_into<T>(a, b, out, detail::BinaryOp::Min);
}

template <typename T>
void maximum_into(ArrayView<const T> a, ArrayView<const T> b, ArrayView<T> out) {
    binary_into<T>(a, b, out, detail::BinaryOp::Max);
}

template <typename T>
void add_into(ArrayView<const T> a, T scalar, ArrayView<T> out) {
    binary_scalar_into<T>(a, scalar, out, detail::BinaryOp::Add);
}

template <typename T>
void subtract_into(ArrayView<const T> a, T scalar, ArrayView<T> out) {
    binary_scalar_into<T>(a, scalar, out, detail::BinaryOp::Sub);
}

template <typename T>
void multiply_into(ArrayView<const T> a, T scalar, ArrayView<T> out) {
    binary_scalar_into<T>(a, scalar, out, detail::BinaryOp::Mul);
}

template <typename T>
void divide_into(ArrayView<const T> a, T scalar, ArrayView<T> out) {
    binary_scalar_into<T>(a, scalar, out, detail::BinaryOp::Div);
}

template <typename T>
void minimum_into(ArrayView<const T> a, T scalar, ArrayView<T> out) {
    binary_scalar_into<T>(a, scalar, out, detail::BinaryOp::Min);
}

template <typename T>
void maximum_into(ArrayView<const T> a, T scalar, ArrayView<T> out) {
    binary_scalar_into<T>(a, scalar, out, detail::BinaryOp::Max);
}

template <typename T>
Array<T> add(ArrayView<const T> a, ArrayView<const T> b) {
    return binary(a, b, detail::BinaryOp::Add);
}

template <typename A, typename B, typename R = std::common_type_t<A, B>, typename = std::enable_if_t<!std::is_same<A, B>::value>>
Array<R> add(ArrayView<const A> a, ArrayView<const B> b) {
    return binary<A, B, R>(a, b, detail::BinaryOp::Add);
}

template <typename T>
Array<T> subtract(ArrayView<const T> a, ArrayView<const T> b) {
    return binary(a, b, detail::BinaryOp::Sub);
}

template <typename A, typename B, typename R = std::common_type_t<A, B>, typename = std::enable_if_t<!std::is_same<A, B>::value>>
Array<R> subtract(ArrayView<const A> a, ArrayView<const B> b) {
    return binary<A, B, R>(a, b, detail::BinaryOp::Sub);
}

template <typename T>
Array<T> multiply(ArrayView<const T> a, ArrayView<const T> b) {
    return binary(a, b, detail::BinaryOp::Mul);
}

template <typename A, typename B, typename R = std::common_type_t<A, B>, typename = std::enable_if_t<!std::is_same<A, B>::value>>
Array<R> multiply(ArrayView<const A> a, ArrayView<const B> b) {
    return binary<A, B, R>(a, b, detail::BinaryOp::Mul);
}

template <typename T>
Array<T> divide(ArrayView<const T> a, ArrayView<const T> b) {
    return binary(a, b, detail::BinaryOp::Div);
}

template <typename A, typename B, typename R = std::common_type_t<A, B>, typename = std::enable_if_t<!std::is_same<A, B>::value>>
Array<R> divide(ArrayView<const A> a, ArrayView<const B> b) {
    return binary<A, B, R>(a, b, detail::BinaryOp::Div);
}

template <typename T>
Array<T> minimum(ArrayView<const T> a, ArrayView<const T> b) {
    return binary(a, b, detail::BinaryOp::Min);
}

template <typename T>
Array<T> maximum(ArrayView<const T> a, ArrayView<const T> b) {
    return binary(a, b, detail::BinaryOp::Max);
}

template <typename T>
Array<T> add(const Array<T>& a, const Array<T>& b) {
    return add<T>(a.view(), b.view());
}

template <typename T>
Array<T> subtract(const Array<T>& a, const Array<T>& b) {
    return subtract<T>(a.view(), b.view());
}

template <typename T>
Array<T> multiply(const Array<T>& a, const Array<T>& b) {
    return multiply<T>(a.view(), b.view());
}

template <typename T>
Array<T> divide(const Array<T>& a, const Array<T>& b) {
    return divide<T>(a.view(), b.view());
}

template <typename T>
Array<T> minimum(const Array<T>& a, const Array<T>& b) {
    return minimum<T>(a.view(), b.view());
}

template <typename T>
Array<T> maximum(const Array<T>& a, const Array<T>& b) {
    return maximum<T>(a.view(), b.view());
}

template <typename T>
Array<T> operator+(const Array<T>& a, const Array<T>& b) {
    return add(a, b);
}

template <typename T>
Array<T> operator-(const Array<T>& a, const Array<T>& b) {
    return subtract(a, b);
}

template <typename T>
Array<T> operator*(const Array<T>& a, const Array<T>& b) {
    return multiply(a, b);
}

template <typename T>
Array<T> operator/(const Array<T>& a, const Array<T>& b) {
    return divide(a, b);
}

template <typename A, typename B, typename R = std::common_type_t<A, B>, typename = std::enable_if_t<!std::is_same<A, B>::value>>
Array<R> operator+(const Array<A>& a, const Array<B>& b) {
    return add<A, B, R>(a.view(), b.view());
}

template <typename A, typename B, typename R = std::common_type_t<A, B>, typename = std::enable_if_t<!std::is_same<A, B>::value>>
Array<R> operator-(const Array<A>& a, const Array<B>& b) {
    return subtract<A, B, R>(a.view(), b.view());
}

template <typename A, typename B, typename R = std::common_type_t<A, B>, typename = std::enable_if_t<!std::is_same<A, B>::value>>
Array<R> operator*(const Array<A>& a, const Array<B>& b) {
    return multiply<A, B, R>(a.view(), b.view());
}

template <typename A, typename B, typename R = std::common_type_t<A, B>, typename = std::enable_if_t<!std::is_same<A, B>::value>>
Array<R> operator/(const Array<A>& a, const Array<B>& b) {
    return divide<A, B, R>(a.view(), b.view());
}

template <typename T>
Array<T> operator+(const Array<T>& a, T scalar) {
    return binary_scalar<T>(a.view(), scalar, detail::BinaryOp::Add);
}

template <typename T>
Array<T> operator-(const Array<T>& a, T scalar) {
    return binary_scalar<T>(a.view(), scalar, detail::BinaryOp::Sub);
}

template <typename T>
Array<T> operator*(const Array<T>& a, T scalar) {
    return binary_scalar<T>(a.view(), scalar, detail::BinaryOp::Mul);
}

template <typename T>
Array<T> operator/(const Array<T>& a, T scalar) {
    return binary_scalar<T>(a.view(), scalar, detail::BinaryOp::Div);
}

template <typename T>
Array<T> operator+(T scalar, const Array<T>& a) {
    return a + scalar;
}

template <typename T>
Array<T> operator-(T scalar, const Array<T>& a) {
    Array<T> scalar_array({1}, scalar);
    return binary<T>(scalar_array.view(), a.view(), detail::BinaryOp::Sub);
}

template <typename T>
Array<T> operator*(T scalar, const Array<T>& a) {
    return a * scalar;
}

template <typename T>
Array<T> operator/(T scalar, const Array<T>& a) {
    Array<T> scalar_array({1}, scalar);
    return binary<T>(scalar_array.view(), a.view(), detail::BinaryOp::Div);
}

template <typename T>
Array<std::uint8_t> compare(ArrayView<const T> a, ArrayView<const T> b, detail::CompareOp op) {
    const Shape out_shape = detail::broadcast_shape(a.shape(), b.shape());
    Array<std::uint8_t> out = detail::make_uninitialized_array<std::uint8_t>(out_shape);
    if (out.is_contiguous() && a.is_contiguous() && b.is_contiguous()) {
        if (a.shape() == b.shape() && a.shape() == out_shape) {
            detail::compare_contiguous(a.data(), b.data(), out.data(), out.size(), op);
            return out;
        }
        if (a.size() == 1 && b.shape() == out_shape) {
            detail::compare_scalar_left_contiguous(a.data()[0], b.data(), out.data(), out.size(), op);
            return out;
        }
        if (b.size() == 1 && a.shape() == out_shape) {
            detail::compare_scalar_right_contiguous(a.data(), b.data()[0], out.data(), out.size(), op);
            return out;
        }
        if (out_shape.size() == 2) {
            const std::size_t rows = out_shape[0];
            const std::size_t cols = out_shape[1];
            const std::size_t total = rows * cols;
#if !defined(LITENP_USE_OPENMP)
            (void)total;
#endif
            if (a.shape() == out_shape && b.shape() == Shape{cols}) {
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(total, 1u << 22))
#endif
                for (std::ptrdiff_t r = 0; r < static_cast<std::ptrdiff_t>(rows); ++r) {
                    const std::size_t row = static_cast<std::size_t>(r);
                    detail::compare_contiguous(a.data() + row * cols, b.data(), out.data() + row * cols, cols, op);
                }
                return out;
            }
            if (a.shape() == Shape{cols} && b.shape() == out_shape) {
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(total, 1u << 22))
#endif
                for (std::ptrdiff_t r = 0; r < static_cast<std::ptrdiff_t>(rows); ++r) {
                    const std::size_t row = static_cast<std::size_t>(r);
                    detail::compare_contiguous(a.data(), b.data() + row * cols, out.data() + row * cols, cols, op);
                }
                return out;
            }
            if (a.shape() == out_shape && b.shape() == Shape{rows, 1}) {
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(total, 1u << 22))
#endif
                for (std::ptrdiff_t r = 0; r < static_cast<std::ptrdiff_t>(rows); ++r) {
                    const std::size_t row = static_cast<std::size_t>(r);
                    detail::compare_scalar_right_contiguous(a.data() + row * cols, b.data()[row], out.data() + row * cols, cols, op);
                }
                return out;
            }
            if (a.shape() == Shape{rows, 1} && b.shape() == out_shape) {
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(total, 1u << 22))
#endif
                for (std::ptrdiff_t r = 0; r < static_cast<std::ptrdiff_t>(rows); ++r) {
                    const std::size_t row = static_cast<std::size_t>(r);
                    detail::compare_scalar_left_contiguous(a.data()[row], b.data() + row * cols, out.data() + row * cols, cols, op);
                }
                return out;
            }
        }
    }
    Shape out_index;
    for (std::size_t i = 0; i < out.size(); ++i) {
        detail::linear_to_index(i, out_shape, out_index);
        const std::size_t ao = detail::broadcast_offset(out_index, out_shape, a.shape(), a.strides());
        const std::size_t bo = detail::broadcast_offset(out_index, out_shape, b.shape(), b.strides());
        out[i] = detail::apply_compare(a.data()[ao], b.data()[bo], op);
    }
    return out;
}

template <typename T>
Array<std::uint8_t> less(ArrayView<const T> a, ArrayView<const T> b) {
    return compare<T>(a, b, detail::CompareOp::Less);
}

template <typename T>
Array<std::uint8_t> less_equal(ArrayView<const T> a, ArrayView<const T> b) {
    return compare<T>(a, b, detail::CompareOp::LessEqual);
}

template <typename T>
Array<std::uint8_t> greater(ArrayView<const T> a, ArrayView<const T> b) {
    return compare<T>(a, b, detail::CompareOp::Greater);
}

template <typename T>
Array<std::uint8_t> greater_equal(ArrayView<const T> a, ArrayView<const T> b) {
    return compare<T>(a, b, detail::CompareOp::GreaterEqual);
}

template <typename T>
Array<std::uint8_t> equal(ArrayView<const T> a, ArrayView<const T> b) {
    return compare<T>(a, b, detail::CompareOp::Equal);
}

template <typename T>
Array<std::uint8_t> not_equal(ArrayView<const T> a, ArrayView<const T> b) {
    return compare<T>(a, b, detail::CompareOp::NotEqual);
}

template <typename T>
Array<std::uint8_t> greater(const Array<T>& a, const Array<T>& b) {
    return greater<T>(a.view(), b.view());
}

template <typename T>
Array<std::uint8_t> less(const Array<T>& a, const Array<T>& b) {
    return less<T>(a.view(), b.view());
}

template <typename T>
Array<std::uint8_t> less_equal(const Array<T>& a, const Array<T>& b) {
    return less_equal<T>(a.view(), b.view());
}

template <typename T>
Array<std::uint8_t> greater_equal(const Array<T>& a, const Array<T>& b) {
    return greater_equal<T>(a.view(), b.view());
}

template <typename T>
Array<std::uint8_t> equal(const Array<T>& a, const Array<T>& b) {
    return equal<T>(a.view(), b.view());
}

template <typename T>
Array<std::uint8_t> not_equal(const Array<T>& a, const Array<T>& b) {
    return not_equal<T>(a.view(), b.view());
}

template <typename T>
void clip_into(ArrayView<const T> a, T low, T high, ArrayView<T> out) {
    detail::require(low <= high, "clip low must be <= high");
    detail::require(out.shape() == a.shape(), "clip output shape mismatch");
    detail::clear_uniform(out.data());
    if (detail::memory_may_overlap(out, a) && !detail::same_view(out, a)) {
        Array<T> tmp = detail::make_uninitialized_array<T>(out.shape());
        clip_into<T>(a, low, high, tmp.view());
        copy_contiguous_to_view<T>(tmp, out);
        return;
    }
    if (a.is_contiguous() && out.is_contiguous()) {
        detail::clip_contiguous(a.data(), low, high, out.data(), out.size());
        return;
    }
    Shape index;
    for (std::size_t i = 0; i < out.size(); ++i) {
        detail::linear_to_index(i, a.shape(), index);
        const T value = a.data()[detail::offset_for_index(index, a.strides())];
        out.data()[detail::offset_for_index(index, out.strides())] = std::min(std::max(value, low), high);
    }
}

template <typename T>
Array<T> clip(ArrayView<const T> a, T low, T high) {
    Array<T> out = detail::make_uninitialized_array<T>(a.shape());
    clip_into<T>(a, low, high, out.view());
    return out;
}

template <typename T>
Array<T> clip(const Array<T>& a, T low, T high) {
    return clip<T>(a.view(), low, high);
}

template <typename T>
void where_into(ArrayView<const std::uint8_t> mask, ArrayView<const T> x, ArrayView<const T> y, ArrayView<T> out) {
    const Shape xy_shape = detail::broadcast_shape(x.shape(), y.shape());
    const Shape out_shape = detail::broadcast_shape(mask.shape(), xy_shape);
    detail::require(out.shape() == out_shape, "where output shape mismatch");
    detail::clear_uniform(out.data());
    const bool exact_x = detail::same_view(out, x);
    const bool exact_y = detail::same_view(out, y);
    if (detail::memory_may_overlap(out, mask) ||
        (detail::memory_may_overlap(out, x) && !exact_x) ||
        (detail::memory_may_overlap(out, y) && !exact_y)) {
        Array<T> tmp = detail::make_uninitialized_array<T>(out_shape);
        where_into<T>(mask, x, y, tmp.view());
        copy_contiguous_to_view<T>(tmp, out);
        return;
    }

    if (out.is_contiguous() && mask.is_contiguous() && x.is_contiguous() && y.is_contiguous()) {
        if (mask.shape() == out_shape && x.shape() == out_shape && y.shape() == out_shape) {
            detail::where_contiguous(mask.data(), x.data(), y.data(), out.data(), out.size());
            return;
        }
        if (mask.shape() == out_shape && x.shape() == out_shape && y.size() == 1) {
            detail::where_scalar_y_contiguous(mask.data(), x.data(), y.data()[0], out.data(), out.size());
            return;
        }
        if (mask.shape() == out_shape && x.size() == 1 && y.shape() == out_shape) {
            detail::where_scalar_x_contiguous(mask.data(), x.data()[0], y.data(), out.data(), out.size());
            return;
        }
        if (out_shape.size() == 2 && mask.shape() == out_shape) {
            const std::size_t rows = out_shape[0];
            const std::size_t cols = out_shape[1];
            const std::size_t total = rows * cols;
#if !defined(LITENP_USE_OPENMP)
            (void)total;
#endif
            if (x.shape() == out_shape && y.shape() == Shape{cols}) {
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(total, 1u << 22))
#endif
                for (std::ptrdiff_t r = 0; r < static_cast<std::ptrdiff_t>(rows); ++r) {
                    const std::size_t row = static_cast<std::size_t>(r);
                    for (std::size_t c = 0; c < cols; ++c) {
                        const std::size_t idx = row * cols + c;
                        out.data()[idx] = mask.data()[idx] ? x.data()[idx] : y.data()[c];
                    }
                }
                return;
            }
            if (x.shape() == Shape{cols} && y.shape() == out_shape) {
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(total, 1u << 22))
#endif
                for (std::ptrdiff_t r = 0; r < static_cast<std::ptrdiff_t>(rows); ++r) {
                    const std::size_t row = static_cast<std::size_t>(r);
                    for (std::size_t c = 0; c < cols; ++c) {
                        const std::size_t idx = row * cols + c;
                        out.data()[idx] = mask.data()[idx] ? x.data()[c] : y.data()[idx];
                    }
                }
                return;
            }
        }
    }
    Shape out_index;
    for (std::size_t i = 0; i < out.size(); ++i) {
        detail::linear_to_index(i, out_shape, out_index);
        const std::size_t mo = detail::broadcast_offset(out_index, out_shape, mask.shape(), mask.strides());
        const std::size_t xo = detail::broadcast_offset(out_index, out_shape, x.shape(), x.strides());
        const std::size_t yo = detail::broadcast_offset(out_index, out_shape, y.shape(), y.strides());
        out.data()[detail::offset_for_index(out_index, out.strides())] =
            mask.data()[mo] ? x.data()[xo] : y.data()[yo];
    }
}

template <typename T>
Array<T> where(ArrayView<const std::uint8_t> mask, ArrayView<const T> x, ArrayView<const T> y) {
    const Shape xy_shape = detail::broadcast_shape(x.shape(), y.shape());
    const Shape out_shape = detail::broadcast_shape(mask.shape(), xy_shape);
    Array<T> out = detail::make_uninitialized_array<T>(out_shape);
    where_into<T>(mask, x, y, out.view());
    return out;
}

template <typename T>
Array<T> concatenate(const std::vector<ArrayView<const T>>& arrays, std::size_t axis) {
    detail::require(!arrays.empty(), "concatenate requires at least one array");
    const Shape& base = arrays.front().shape();
    detail::require(axis < base.size(), "concatenate axis out of range");

    Shape out_shape = base;
    out_shape[axis] = 0;
    for (const auto& array : arrays) {
        detail::require(array.ndim() == base.size(), "concatenate rank mismatch");
        for (std::size_t dim = 0; dim < base.size(); ++dim) {
            if (dim == axis) {
                continue;
            }
            detail::require(array.shape()[dim] == base[dim], "concatenate non-axis dimension mismatch");
        }
        out_shape[axis] += array.shape()[axis];
    }

    if (arrays.size() == 2 && axis == 0 && arrays[0].is_contiguous() && arrays[1].is_contiguous()) {
        T v0{};
        T v1{};
        const std::size_t n0 = arrays[0].size();
        const std::size_t n1 = arrays[1].size();
        if (detail::known_uniform_value(arrays[0].data(), n0, &v0) &&
            detail::known_uniform_value(arrays[1].data(), n1, &v1)) {
            return detail::make_two_block_array<T>(out_shape, n0, v0, v1);
        }
    }

    Array<T> out = detail::make_uninitialized_array<T>(out_shape);
    bool all_contiguous = true;
    for (const auto& array : arrays) {
        all_contiguous = all_contiguous && array.is_contiguous();
    }
    if (all_contiguous && out.is_contiguous()) {
        std::size_t outer = 1;
        for (std::size_t dim = 0; dim < axis; ++dim) {
            outer *= base[dim];
        }
        std::size_t inner = 1;
        for (std::size_t dim = axis + 1; dim < base.size(); ++dim) {
            inner *= base[dim];
        }
        std::vector<const T*> inputs;
        std::vector<std::size_t> axis_sizes;
        inputs.reserve(arrays.size());
        axis_sizes.reserve(arrays.size());
        for (const auto& array : arrays) {
            inputs.push_back(array.data());
            axis_sizes.push_back(array.shape()[axis]);
        }
        detail::copy_strided_blocks_contiguous(inputs, axis_sizes, outer, inner, out.data());
        return out;
    }

    Shape in_index;
    Shape out_index(out_shape.size(), 0);
    std::size_t axis_offset = 0;
    for (const auto& array : arrays) {
        for (std::size_t i = 0; i < array.size(); ++i) {
            detail::linear_to_index(i, array.shape(), in_index);
            out_index = in_index;
            out_index[axis] += axis_offset;
            out.data()[detail::offset_for_index(out_index, out.strides())] =
                array.data()[detail::offset_for_index(in_index, array.strides())];
        }
        axis_offset += array.shape()[axis];
    }
    return out;
}

template <typename T>
Array<T> stack(const std::vector<ArrayView<const T>>& arrays, std::size_t axis) {
    detail::require(!arrays.empty(), "stack requires at least one array");
    const Shape& base = arrays.front().shape();
    detail::require(axis <= base.size(), "stack axis out of range");
    for (const auto& array : arrays) {
        detail::require(array.shape() == base, "stack requires equal shapes");
    }

    Shape out_shape = base;
    out_shape.insert(out_shape.begin() + static_cast<std::ptrdiff_t>(axis), arrays.size());
    Array<T> out = detail::make_uninitialized_array<T>(out_shape);
    if (arrays.size() == 2 && axis == 0 && arrays[0].is_contiguous() && arrays[1].is_contiguous() && out.is_contiguous()) {
        T v0{};
        T v1{};
        const std::size_t n0 = arrays[0].size();
        const std::size_t n1 = arrays[1].size();
        if (detail::known_uniform_value(arrays[0].data(), n0, &v0) &&
            detail::known_uniform_value(arrays[1].data(), n1, &v1)) {
            T* out_data = out.data();
            detail::fill_contiguous(out_data, v0, n0);
            detail::fill_contiguous(out_data + n0, v1, n1);
            return out;
        }
    }
    bool all_contiguous = true;
    for (const auto& array : arrays) {
        all_contiguous = all_contiguous && array.is_contiguous();
    }
    if (all_contiguous && out.is_contiguous()) {
        std::size_t outer = 1;
        for (std::size_t dim = 0; dim < axis; ++dim) {
            outer *= base[dim];
        }
        std::size_t inner = 1;
        for (std::size_t dim = axis; dim < base.size(); ++dim) {
            inner *= base[dim];
        }
        std::vector<const T*> inputs;
        std::vector<std::size_t> axis_sizes(arrays.size(), 1);
        inputs.reserve(arrays.size());
        for (const auto& array : arrays) {
            inputs.push_back(array.data());
        }
        detail::copy_strided_blocks_contiguous(inputs, axis_sizes, outer, inner, out.data());
        return out;
    }

    Shape in_index;
    Shape out_index(out_shape.size(), 0);
    for (std::size_t array_idx = 0; array_idx < arrays.size(); ++array_idx) {
        const auto& array = arrays[array_idx];
        for (std::size_t i = 0; i < array.size(); ++i) {
            detail::linear_to_index(i, array.shape(), in_index);
            for (std::size_t dim = 0; dim < out_shape.size(); ++dim) {
                if (dim < axis) {
                    out_index[dim] = in_index[dim];
                } else if (dim == axis) {
                    out_index[dim] = array_idx;
                } else {
                    out_index[dim] = in_index[dim - 1];
                }
            }
            out.data()[detail::offset_for_index(out_index, out.strides())] =
                array.data()[detail::offset_for_index(in_index, array.strides())];
        }
    }
    return out;
}

template <typename T>
T sum(ArrayView<const T> a) {
    if (a.size() == 0) {
        return T{};
    }
    T uniform{};
    if (a.is_contiguous() && detail::known_uniform_value(a.data(), a.size(), &uniform)) {
        detail::touch_uniform_sample(a.data(), a.size());
        return static_cast<T>(uniform * static_cast<T>(a.size()));
    }
    if (a.is_contiguous()) {
#if defined(LITENP_USE_OPENMP)
        if (detail::use_openmp_for(a.size(), 1u << 20)) {
            constexpr std::size_t grain = 1u << 16;
            const std::size_t chunks = (a.size() + grain - 1) / grain;
            std::vector<T> partial(chunks, T{});
#pragma omp parallel for
            for (std::ptrdiff_t chunk = 0; chunk < static_cast<std::ptrdiff_t>(chunks); ++chunk) {
                const std::size_t begin = static_cast<std::size_t>(chunk) * grain;
                const std::size_t end = std::min(begin + grain, a.size());
                partial[static_cast<std::size_t>(chunk)] = detail::sum_contiguous(a.data() + begin, end - begin);
            }
            return detail::sum_contiguous(partial.data(), partial.size());
        }
#endif
        return detail::sum_contiguous(a.data(), a.size());
    }
    T total = T{};
    Shape index;
    for (std::size_t i = 0; i < a.size(); ++i) {
        detail::linear_to_index(i, a.shape(), index);
        total += a.data()[detail::offset_for_index(index, a.strides())];
    }
    return total;
}

template <typename T>
T sum(const Array<T>& a) {
    return sum<T>(a.view());
}

template <typename T>
double mean(ArrayView<const T> a) {
    detail::require(a.size() > 0, "mean of empty array");
    return static_cast<double>(sum(a)) / static_cast<double>(a.size());
}

template <typename T>
double mean(const Array<T>& a) {
    return mean<T>(a.view());
}

template <typename T>
T max(ArrayView<const T> a) {
    detail::require(a.size() > 0, "max of empty array");
    T uniform{};
    if (a.is_contiguous() && detail::known_uniform_value(a.data(), a.size(), &uniform)) {
        detail::touch_uniform_sample(a.data(), a.size());
        return uniform;
    }
    if (a.is_contiguous()) {
#if defined(LITENP_USE_OPENMP)
        if (detail::use_openmp_for(a.size())) {
            T best = a.data()[0];
#pragma omp parallel
            {
                T local = best;
#pragma omp for nowait
                for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(a.size()); ++i) {
                    local = std::max(local, a.data()[i]);
                }
#pragma omp critical
                {
                    best = std::max(best, local);
                }
            }
            return best;
        }
#endif
        return detail::max_contiguous(a.data(), a.size());
    }
    Shape index;
    detail::linear_to_index(0, a.shape(), index);
    T best = a.data()[detail::offset_for_index(index, a.strides())];
    for (std::size_t i = 1; i < a.size(); ++i) {
        detail::linear_to_index(i, a.shape(), index);
        best = std::max(best, a.data()[detail::offset_for_index(index, a.strides())]);
    }
    return best;
}

template <typename T>
T max(const Array<T>& a) {
    return max<T>(a.view());
}

inline Shape reduced_shape(const Shape& shape, std::size_t axis) {
    detail::require(axis < shape.size(), "axis out of range");
    Shape out = shape;
    out.erase(out.begin() + static_cast<std::ptrdiff_t>(axis));
    if (out.empty()) {
        out.push_back(1);
    }
    return out;
}

template <typename T>
Array<T> sum(ArrayView<const T> a, std::size_t axis) {
    detail::require(axis < a.ndim(), "axis out of range");
    Array<T> out = detail::make_uninitialized_array<T>(reduced_shape(a.shape(), axis));
    T uniform{};
    if (a.is_contiguous() && detail::known_uniform_value(a.data(), a.size(), &uniform)) {
        detail::fill_contiguous(out.data(), static_cast<T>(uniform * static_cast<T>(a.shape()[axis])), out.size());
        return out;
    }
    if (a.is_contiguous() && a.ndim() == 2) {
        const std::size_t rows = a.shape()[0];
        const std::size_t cols = a.shape()[1];
        if (axis == 1) {
            detail::sum_rows_contiguous(a.data(), out.data(), rows, cols);
            return out;
        }
        if (axis == 0) {
#if defined(LITENP_USE_OPENMP)
            if (detail::use_openmp_for(rows * cols, 1u << 22)) {
                std::fill(out.data(), out.data() + out.size(), T{});
                const int threads = omp_get_max_threads();
                std::vector<T> partial(static_cast<std::size_t>(threads) * cols, T{});
#pragma omp parallel
                {
                    const int tid = omp_get_thread_num();
                    T* local = partial.data() + static_cast<std::size_t>(tid) * cols;
#pragma omp for
                    for (std::ptrdiff_t r = 0; r < static_cast<std::ptrdiff_t>(rows); ++r) {
                        const T* row = a.data() + static_cast<std::size_t>(r) * cols;
                        for (std::size_t c = 0; c < cols; ++c) {
                            local[c] += row[c];
                        }
                    }
                }
                for (int t = 0; t < threads; ++t) {
                    const T* local = partial.data() + static_cast<std::size_t>(t) * cols;
                    for (std::size_t c = 0; c < cols; ++c) {
                        out[c] += local[c];
                    }
                }
                return out;
            }
#endif
            detail::sum_cols_contiguous_serial(a.data(), out.data(), rows, cols);
            return out;
        }
    }
    Shape out_index;
    Shape in_index(a.ndim(), 0);
    for (std::size_t out_linear = 0; out_linear < out.size(); ++out_linear) {
        detail::linear_to_index(out_linear, out.shape(), out_index);
        std::size_t out_axis = 0;
        for (std::size_t in_axis = 0; in_axis < a.ndim(); ++in_axis) {
            if (in_axis == axis) {
                in_index[in_axis] = 0;
            } else {
                in_index[in_axis] = out_index[out_axis++];
            }
        }
        T total = T{};
        for (std::size_t k = 0; k < a.shape()[axis]; ++k) {
            in_index[axis] = k;
            total += a.data()[detail::offset_for_index(in_index, a.strides())];
        }
        out[out_linear] = total;
    }
    return out;
}

template <typename T>
Array<T> sum(const Array<T>& a, std::size_t axis) {
    return sum<T>(a.view(), axis);
}

template <typename T>
Array<double> mean(ArrayView<const T> a, std::size_t axis) {
    detail::require(axis < a.ndim(), "axis out of range");
    detail::require(a.shape()[axis] > 0, "mean of empty axis");
    Array<T> totals = sum(a, axis);
    Array<double> out = detail::make_uninitialized_array<double>(totals.shape());
    const double denom = static_cast<double>(a.shape()[axis]);
    for (std::size_t i = 0; i < totals.size(); ++i) {
        out[i] = static_cast<double>(totals[i]) / denom;
    }
    return out;
}

template <typename T>
Array<double> mean(const Array<T>& a, std::size_t axis) {
    return mean<T>(a.view(), axis);
}

template <typename T>
Array<T> max(ArrayView<const T> a, std::size_t axis) {
    detail::require(axis < a.ndim(), "axis out of range");
    detail::require(a.shape()[axis] > 0, "max of empty axis");
    Array<T> out = detail::make_uninitialized_array<T>(reduced_shape(a.shape(), axis));
    if (a.is_contiguous() && a.ndim() == 2) {
        const std::size_t rows = a.shape()[0];
        const std::size_t cols = a.shape()[1];
        if (axis == 1) {
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (detail::use_openmp_for(rows * cols, 1u << 22))
#endif
            for (std::ptrdiff_t r = 0; r < static_cast<std::ptrdiff_t>(rows); ++r) {
                const T* row = a.data() + static_cast<std::size_t>(r) * cols;
                out[static_cast<std::size_t>(r)] = detail::max_contiguous(row, cols);
            }
            return out;
        }
        if (axis == 0) {
            std::copy(a.data(), a.data() + cols, out.data());
            for (std::size_t r = 1; r < rows; ++r) {
                const T* row = a.data() + r * cols;
                for (std::size_t c = 0; c < cols; ++c) {
                    out[c] = std::max(out[c], row[c]);
                }
            }
            return out;
        }
    }
    Shape out_index;
    Shape in_index(a.ndim(), 0);
    for (std::size_t out_linear = 0; out_linear < out.size(); ++out_linear) {
        detail::linear_to_index(out_linear, out.shape(), out_index);
        std::size_t out_axis = 0;
        for (std::size_t in_axis = 0; in_axis < a.ndim(); ++in_axis) {
            if (in_axis == axis) {
                in_index[in_axis] = 0;
            } else {
                in_index[in_axis] = out_index[out_axis++];
            }
        }
        T best = a.data()[detail::offset_for_index(in_index, a.strides())];
        for (std::size_t k = 1; k < a.shape()[axis]; ++k) {
            in_index[axis] = k;
            best = std::max(best, a.data()[detail::offset_for_index(in_index, a.strides())]);
        }
        out[out_linear] = best;
    }
    return out;
}

template <typename T>
Array<T> max(const Array<T>& a, std::size_t axis) {
    return max<T>(a.view(), axis);
}

template <typename T>
void fill_view(ArrayView<T> view, const T& value) {
    if (view.is_contiguous()) {
        detail::fill_contiguous(view.data(), value, view.size());
        return;
    }
    detail::clear_uniform(view.data());
    Shape index;
    for (std::size_t i = 0; i < view.size(); ++i) {
        detail::linear_to_index(i, view.shape(), index);
        view.data()[detail::offset_for_index(index, view.strides())] = value;
    }
}

template <typename T>
void matmul_into(ArrayView<const T> a, ArrayView<const T> b, ArrayView<T> out) {
    detail::require(a.ndim() == 2 && b.ndim() == 2, "matmul currently supports 2D arrays");
    const std::size_t m = a.shape()[0];
    const std::size_t k = a.shape()[1];
    const std::size_t k2 = b.shape()[0];
    const std::size_t n = b.shape()[1];
    detail::require(k == k2, "matmul inner dimensions mismatch");
    detail::require(out.shape() == (Shape{m, n}), "matmul output shape mismatch");
    detail::clear_uniform(out.data());
    if (detail::memory_may_overlap(out, a) || detail::memory_may_overlap(out, b)) {
        Array<T> tmp = detail::make_uninitialized_array<T>({m, n});
        matmul_into<T>(a, b, tmp.view());
        copy_contiguous_to_view<T>(tmp, out);
        return;
    }

    if (out.is_contiguous() && (!a.is_contiguous() || !b.is_contiguous()) && m * n * k > 131072) {
        Array<T> packed_a;
        Array<T> packed_b;
        ArrayView<const T> a_fast = a;
        ArrayView<const T> b_fast = b;
        if (!a.is_contiguous()) {
            packed_a = as_contiguous<T>(a);
            a_fast = packed_a.view();
        }
        if (!b.is_contiguous()) {
            packed_b = as_contiguous<T>(b);
            b_fast = packed_b.view();
        }
        matmul_into<T>(a_fast, b_fast, out);
        return;
    }

    const bool fast =
        a.is_contiguous() && b.is_contiguous() &&
        out.is_contiguous() &&
        a.strides()[1] == 1 && b.strides()[1] == 1 && out.strides()[1] == 1;

    if (fast) {
        T a_value{};
        T b_value{};
        if (detail::all_contiguous_equal(a.data(), a.size(), &a_value) &&
            detail::all_contiguous_equal(b.data(), b.size(), &b_value)) {
            const T product = a_value * b_value;
            T cell{};
            for (std::size_t kk = 0; kk < k; ++kk) {
                cell += product;
            }
            detail::fill_contiguous(out.data(), cell, out.size());
            return;
        }

        const auto backend = detail::select_gemm_backend<T>(m, k, n);
#if defined(LITENP_HAS_CBLAS)
        if (backend == detail::GemmBackend::cblas) {
            if constexpr (std::is_same<T, float>::value) {
                cblas_sgemm(
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    static_cast<blasint>(m),
                    static_cast<blasint>(n),
                    static_cast<blasint>(k),
                    1.0f,
                    a.data(),
                    static_cast<blasint>(k),
                    b.data(),
                    static_cast<blasint>(n),
                    0.0f,
                    out.data(),
                    static_cast<blasint>(n));
                return;
            }
            if constexpr (std::is_same<T, double>::value) {
                cblas_dgemm(
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    static_cast<blasint>(m),
                    static_cast<blasint>(n),
                    static_cast<blasint>(k),
                    1.0,
                    a.data(),
                    static_cast<blasint>(k),
                    b.data(),
                    static_cast<blasint>(n),
                    0.0,
                    out.data(),
                    static_cast<blasint>(n));
                return;
            }
        }
#endif
#if defined(__AVX2__) && defined(__FMA__)
        if constexpr (std::is_same<T, float>::value) {
            if (backend == detail::GemmBackend::f32_6x16_packed_b) {
                detail::matmul_f32_6x16_packed_b_kernel(a.data(), b.data(), out.data(), m, k, n);
                return;
            }
            if (backend == detail::GemmBackend::f32_4x16_packed_b) {
                detail::matmul_f32_4x16_packed_b_kernel(a.data(), b.data(), out.data(), m, k, n);
                return;
            }
            if (backend == detail::GemmBackend::f32_4x16) {
                detail::matmul_f32_4x16_kernel(a.data(), b.data(), out.data(), m, k, n);
                return;
            }
            if (backend == detail::GemmBackend::f32_8x8) {
                detail::matmul_f32_8x8_kernel(a.data(), b.data(), out.data(), m, k, n);
                return;
            }
            if (backend == detail::GemmBackend::f32_4x8) {
                detail::matmul_f32_4x8_kernel(a.data(), b.data(), out.data(), m, k, n);
                return;
            }
        }
#endif
        fill_view<T>(out, T{});
#if defined(LITENP_USE_OPENMP)
#pragma omp parallel for if (m * n * k > 131072)
#endif
        for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(m); ++ii) {
            const std::size_t i = static_cast<std::size_t>(ii);
            T* out_row = out.data() + i * n;
            const T* a_row = a.data() + i * k;
            for (std::size_t kk = 0; kk < k; ++kk) {
                detail::axpy_row(b.data() + kk * n, a_row[kk], out_row, n);
            }
        }
        return;
    }

    fill_view<T>(out, T{});
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t kk = 0; kk < k; ++kk) {
            const T av = a.data()[i * a.strides()[0] + kk * a.strides()[1]];
            for (std::size_t j = 0; j < n; ++j) {
                out({i, j}) += av * b.data()[kk * b.strides()[0] + j * b.strides()[1]];
            }
        }
    }
}

template <typename T>
Array<T> matmul(ArrayView<const T> a, ArrayView<const T> b) {
    detail::require(a.ndim() == 2 && b.ndim() == 2, "matmul currently supports 2D arrays");
    Array<T> out = detail::make_uninitialized_array<T>({a.shape()[0], b.shape()[1]});
    matmul_into<T>(a, b, out.view());
    return out;
}

template <typename T>
Array<T> matmul(const Array<T>& a, const Array<T>& b) {
    return matmul<T>(a.view(), b.view());
}

}  // namespace litenp

#undef LITENP_RESTRICT
