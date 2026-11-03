#pragma once
#include <cstdint>
#include <cstddef>
#include <algorithm>

namespace EdgeML {

/**
 * @brief Fixed-point INT8 vector dot-product with INT32 accumulator.
 * Zero-allocation, deterministic execution time.
 */
inline int32_t dot_product_int8(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    size_t length) 
{
    int32_t sum0 = 0;
    int32_t sum1 = 0;
    int32_t sum2 = 0;
    int32_t sum3 = 0;

    size_t i = 0;
    // 4-way loop unrolling for pipelined architectures
    for (; i + 3 < length; i += 4) {
        sum0 += static_cast<int32_t>(a[i])     * static_cast<int32_t>(b[i]);
        sum1 += static_cast<int32_t>(a[i + 1]) * static_cast<int32_t>(b[i + 1]);
        sum2 += static_cast<int32_t>(a[i + 2]) * static_cast<int32_t>(b[i + 2]);
        sum3 += static_cast<int32_t>(a[i + 3]) * static_cast<int32_t>(b[i + 3]);
    }

    int32_t total = (sum0 + sum1) + (sum2 + sum3);

    // Handle remaining elements
    for (; i < length; ++i) {
        total += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    }

    return total;
}

/**
 * @brief Requantize INT32 accumulator back to INT8 using fixed-point multiplier & bit shift.
 */
inline int8_t requantize(int32_t acc, int32_t multiplier, int32_t shift, int32_t zero_point) {
    int64_t scaled = (static_cast<int64_t>(acc) * multiplier) >> shift;
    int32_t result = static_cast<int32_t>(scaled) + zero_point;
    return static_cast<int8_t>(std::clamp(result, -128, 127));
}

} // namespace EdgeML
