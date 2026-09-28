#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace {

constexpr std::size_t kGroupSize = 128;
constexpr std::size_t kPackedBytes = kGroupSize / 4;

int32_t sum_lanes(__m256i lanes) {
    const __m128i halves = _mm_add_epi32(_mm256_castsi256_si128(lanes),
                                          _mm256_extracti128_si256(lanes, 1));
    const __m128i pairs = _mm_hadd_epi32(halves, halves);
    const __m128i total = _mm_hadd_epi32(pairs, pairs);
    return _mm_cvtsi128_si32(total);
}

int32_t dot_group(const uint8_t* packed, const int8_t* activations) {
    const __m256i packed_lanes =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed));
    const __m256i code_mask = _mm256_set1_epi8(3);
    const __m256i ones = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256i coded_pairs = _mm256_setzero_si256();
    __m256i activation_pairs = _mm256_setzero_si256();

    for (int lane = 0; lane < 4; ++lane) {
        const __m256i codes = _mm256_and_si256(
            _mm256_srli_epi16(packed_lanes, 6 - 2 * lane), code_mask);
        const __m256i values = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(activations + lane * 32));
        coded_pairs = _mm256_add_epi16(
            coded_pairs, _mm256_maddubs_epi16(codes, values));
        activation_pairs = _mm256_add_epi16(
            activation_pairs, _mm256_maddubs_epi16(ones, values));
    }

    const __m256i signed_pairs = _mm256_sub_epi16(coded_pairs, activation_pairs);
    return sum_lanes(_mm256_madd_epi16(signed_pairs, ones16));
}

}

extern "C" int bitnet_group_scale_matvec_avx2(
    const uint8_t* packed, const float* weight_scales, const int8_t* activations,
    const float* activation_scales, std::size_t rows, std::size_t groups,
    float* output) {
    if (!packed || !weight_scales || !activations || !activation_scales ||
        !output || rows == 0 || groups == 0) {
        return 1;
    }
    for (std::size_t row = 0; row < rows; ++row) {
        float value = 0.0f;
        for (std::size_t group = 0; group < groups; ++group) {
            const std::size_t offset = row * groups + group;
            const int32_t partial = dot_group(
                packed + offset * kPackedBytes,
                activations + group * kGroupSize);
            value += static_cast<float>(partial) * weight_scales[offset] *
                     activation_scales[group];
        }
        output[row] = value;
    }
    return 0;
}