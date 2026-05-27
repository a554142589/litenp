#include <iostream>

#include "litenp/litenp.hpp"

int main() {
    auto a = litenp::Array<float>::from_vector({2, 3}, {1, 2, 3, 4, 5, 6});
    auto bias = litenp::Array<float>::from_vector({3}, {10, 20, 30});

    auto shifted = a + bias;
    auto activated = litenp::relu(shifted - 15.0f);
    auto row_sum = litenp::sum(activated, 1);

    auto weights = litenp::Array<float>::from_vector({3, 2}, {1, 0, 0, 1, 1, 1});
    auto projected = litenp::matmul(a, weights);

    std::cout << "row_sum: " << row_sum({0}) << ", " << row_sum({1}) << "\n";
    std::cout << "projected(1,1): " << projected({1, 1}) << "\n";
    return (projected({1, 1}) == 11.0f) ? 0 : 1;
}
