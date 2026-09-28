#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" int bitnet_group_scale_matmul_avx2(
    const uint8_t* packed, const float* weight_scales, const int8_t* activations,
    const float* activation_scales, std::size_t tokens, std::size_t rows,
    std::size_t groups, float* output);

int main() {
    constexpr int tokens = 2;
    constexpr int rows = 4;
    constexpr int groups = 2;
    constexpr int group_width = 128;
    constexpr int width = groups * group_width;
    ggml_init_params params = { 1024 * 1024, nullptr, true };
    ggml_context* context = ggml_init(params);
    if (!context) {
        return 1;
    }
    ggml_tensor* matrix = ggml_new_tensor_2d(context, GGML_TYPE_F32, group_width, group_width);
    ggml_tensor* signs = ggml_new_tensor_1d(context, GGML_TYPE_F32, width);
    ggml_tensor* input = ggml_new_tensor_2d(context, GGML_TYPE_F32, width, tokens);
    ggml_tensor* signed_input = ggml_mul(context, input, signs);
    ggml_tensor* grouped = ggml_reshape_2d(context, signed_input, group_width, groups * tokens);
    ggml_tensor* rotated_groups = ggml_mul_mat(context, matrix, grouped);
    ggml_mul_mat_set_hint(rotated_groups, GGML_HINT_SRC0_IS_HADAMARD);
    ggml_tensor* rotated = ggml_reshape_2d(context, rotated_groups, width, tokens);
    ggml_set_input(matrix);
    ggml_set_input(signs);
    ggml_set_input(input);
    ggml_set_output(rotated);
    ggml_cgraph* graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, rotated);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (!backend || !allocator || !ggml_gallocr_alloc_graph(allocator, graph)) {
        return 2;
    }
    std::vector<float> unused_matrix(group_width * group_width, 0.0f);
    std::array<float, width> explicit_signs;
    std::array<float, tokens * width> inputs;
    for (int index = 0; index < width; ++index) {
        explicit_signs[index] = index % 3 == 0 ? -1.0f : 1.0f;
        for (int token = 0; token < tokens; ++token) {
            inputs[token * width + index] = (index % 11 - 5) * (token == 0 ? 0.25f : -0.5f);
        }
    }
    ggml_backend_tensor_set(matrix, unused_matrix.data(), 0, unused_matrix.size() * sizeof(float));
    ggml_backend_tensor_set(signs, explicit_signs.data(), 0, sizeof(explicit_signs));
    ggml_backend_tensor_set(input, inputs.data(), 0, sizeof(inputs));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        return 3;
    }
    std::array<float, tokens * width> transformed;
    ggml_backend_tensor_get(rotated, transformed.data(), 0, sizeof(transformed));
    float max_transform_error = 0.0f;
    for (int token = 0; token < tokens; ++token) {
        for (int group = 0; group < groups; ++group) {
            for (int column = 0; column < group_width; ++column) {
                double expected = 0.0;
                for (int index = 0; index < group_width; ++index) {
                    const int offset = group * group_width + index;
                    const int parity = __builtin_popcount(static_cast<unsigned>(index & column)) & 1;
                    expected += inputs[token * width + offset] * explicit_signs[offset] *
                                (parity ? -1 : 1);
                }
                expected /= std::sqrt(group_width);
                const float error = std::abs(transformed[token * width + group * group_width + column] - expected);
                if (!std::isfinite(error) || error > 1e-4f) {
                    return 4;
                }
                max_transform_error = std::max(max_transform_error, error);
            }
        }
    }

    std::array<int8_t, tokens * width> a8;
    std::array<float, tokens * groups> a8_scales;
    for (int token = 0; token < tokens; ++token) {
        for (int group = 0; group < groups; ++group) {
            const int offset = token * width + group * group_width;
            float max_abs = 0.0f;
            for (int index = 0; index < group_width; ++index) {
                max_abs = std::max(max_abs, std::abs(transformed[offset + index]));
            }
            const float scale = max_abs == 0.0f ? 1.0f : max_abs / 127.0f;
            a8_scales[token * groups + group] = scale;
            for (int index = 0; index < group_width; ++index) {
                a8[offset + index] = static_cast<int8_t>(std::clamp(
                    std::nearbyint(transformed[offset + index] / scale), -127.0f, 127.0f));
            }
        }
    }

    std::array<uint8_t, rows * groups * group_width / 4> packed{};
    std::array<int8_t, rows * width> reference_codes;
    for (int row = 0; row < rows; ++row) {
        for (int group = 0; group < groups; ++group) {
            for (int index = 0; index < group_width; ++index) {
                const int code = (row + group + index) % 3 - 1;
                reference_codes[row * width + group * group_width + index] = code;
                const int packed_index = (row * groups + group) * 32 + index % 32;
                packed[packed_index] |= (code + 1) << (6 - 2 * (index / 32));
            }
        }
    }
    const std::array<float, rows * groups> weight_scales = {
        0.25f, 1.0f, 0.5f, 1.5f, 2.0f, 0.125f, 0.75f, 3.0f
    };
    std::array<float, tokens * rows> actual;
    if (bitnet_group_scale_matmul_avx2(
        packed.data(), weight_scales.data(), a8.data(), a8_scales.data(),
        tokens, rows, groups, actual.data()) != 0) {
        return 5;
    }
    float max_output_error = 0.0f;
    for (int token = 0; token < tokens; ++token) {
        for (int row = 0; row < rows; ++row) {
            double expected = 0.0;
            for (int group = 0; group < groups; ++group) {
                int32_t partial = 0;
                for (int index = 0; index < group_width; ++index) {
                    partial += reference_codes[row * width + group * group_width + index] *
                               a8[token * width + group * group_width + index];
                }
                expected += partial * weight_scales[row * groups + group] * a8_scales[token * groups + group];
            }
            const float error = std::abs(actual[token * rows + row] - expected);
            if (!std::isfinite(error) || error > 0.005f) {
                return 6;
            }
            max_output_error = std::max(max_output_error, error);
        }
    }
    std::printf(
        "{\"tokens\":%d,\"groups\":%d,\"max_transform_error\":%.8f,"
        "\"max_output_error\":%.8f}\n",
        tokens, groups, max_transform_error, max_output_error);
    ggml_gallocr_free(allocator);
    ggml_backend_free(backend);
    ggml_free(context);
    return 0;
}