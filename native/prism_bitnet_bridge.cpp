#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" int bitnet_group_scale_matmul_avx2(
    const uint8_t* packed, const float* weight_scales, const int8_t* activations,
    const float* activation_scales, std::size_t tokens, std::size_t rows,
    std::size_t groups, float* output);

namespace {

constexpr int tokens = 2;
constexpr int rows = 4;
constexpr int groups = 2;
constexpr int group_width = 128;
constexpr int width = groups * group_width;

struct bridge_data {
    std::array<uint8_t, rows * groups * group_width / 4> packed{};
    std::array<float, rows * groups> weight_scales = {
        0.25f, 1.0f, 0.5f, 1.5f, 2.0f, 0.125f, 0.75f, 3.0f
    };
    std::array<int8_t, tokens * width> activations{};
    std::array<float, tokens * groups> activation_scales{};
    int calls = 0;
    int status = 0;
};

void grouped_bitnet_op(
    ggml_tensor* destination, const ggml_tensor* output_shape,
    const ggml_tensor* rotated, int thread, int thread_count, void* userdata) {
    if (thread != 0) {
        return;
    }
    auto* state = static_cast<bridge_data*>(userdata);
    ++state->calls;
    if (thread_count < 1 || destination->type != GGML_TYPE_F32 ||
        destination->ne[0] != rows || destination->ne[1] != tokens ||
        output_shape->type != GGML_TYPE_F32 || rotated->type != GGML_TYPE_F32 ||
        rotated->ne[0] != width || rotated->ne[1] != tokens ||
        !destination->data || !rotated->data) {
        state->status = 5;
        return;
    }
    const float* values = static_cast<const float*>(rotated->data);
    for (int token = 0; token < tokens; ++token) {
        for (int group = 0; group < groups; ++group) {
            const int offset = token * width + group * group_width;
            float max_abs = 0.0f;
            for (int index = 0; index < group_width; ++index) {
                if (!std::isfinite(values[offset + index])) {
                    state->status = 5;
                    return;
                }
                max_abs = std::max(max_abs, std::abs(values[offset + index]));
            }
            const float scale = max_abs == 0.0f ? 1.0f : max_abs / 127.0f;
            state->activation_scales[token * groups + group] = scale;
            for (int index = 0; index < group_width; ++index) {
                state->activations[offset + index] = static_cast<int8_t>(std::clamp(
                    std::nearbyint(values[offset + index] / scale), -127.0f, 127.0f));
            }
        }
    }
    state->status = bitnet_group_scale_matmul_avx2(
        state->packed.data(), state->weight_scales.data(), state->activations.data(),
        state->activation_scales.data(), tokens, rows, groups,
        static_cast<float*>(destination->data));
}

}

int main(int argc, char** argv) {
    const bool grouped_values = argc == 2 && std::strcmp(argv[1], "--grouped-v") == 0;
    if (argc > 2 || (argc == 2 && !grouped_values)) {
        return 11;
    }
    bridge_data bridge;
    std::array<int8_t, rows * width> reference_codes;
    for (int row = 0; row < rows; ++row) {
        for (int group = 0; group < groups; ++group) {
            for (int index = 0; index < group_width; ++index) {
                const int code = (row + group + index) % 3 - 1;
                reference_codes[row * width + group * group_width + index] = code;
                const int packed_index = (row * groups + group) * 32 + index % 32;
                bridge.packed[packed_index] |= (code + 1) << (6 - 2 * (index / 32));
            }
        }
    }
    ggml_init_params params = { 1024 * 1024, nullptr, true };
    ggml_context* context = ggml_init(params);
    if (!context) {
        return 1;
    }
    ggml_tensor* matrix = ggml_new_tensor_2d(context, GGML_TYPE_F32, group_width, group_width);
    ggml_tensor* signs = ggml_new_tensor_1d(context, GGML_TYPE_F32, width);
    ggml_tensor* input = ggml_new_tensor_2d(context, GGML_TYPE_F32, width, tokens);
    ggml_tensor* ordered_input = input;
    if (grouped_values) {
        ggml_tensor* tiled = ggml_reshape_4d(context, input, 64, 2, 2, tokens);
        ggml_tensor* permuted = ggml_permute(context, tiled, 0, 2, 1, 3);
        ordered_input = ggml_reshape_2d(context, ggml_cont(context, permuted), width, tokens);
    }
    ggml_tensor* signed_input = ggml_mul(context, ordered_input, signs);
    ggml_tensor* grouped = ggml_reshape_2d(context, signed_input, group_width, groups * tokens);
    ggml_tensor* rotated_groups = ggml_mul_mat(context, matrix, grouped);
    ggml_mul_mat_set_hint(rotated_groups, GGML_HINT_SRC0_IS_HADAMARD);
    ggml_tensor* rotated = ggml_reshape_2d(context, rotated_groups, width, tokens);
    ggml_tensor* output_shape = ggml_new_tensor_2d(context, GGML_TYPE_F32, rows, tokens);
    ggml_tensor* output = ggml_map_custom2(
        context, output_shape, rotated, grouped_bitnet_op, 1, &bridge);
    ggml_set_input(matrix);
    ggml_set_input(signs);
    ggml_set_input(input);
    ggml_set_input(output_shape);
    ggml_set_output(rotated);
    ggml_set_output(output);
    ggml_cgraph* graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);

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
    std::array<float, tokens * rows> empty_output{};
    ggml_backend_tensor_set(output_shape, empty_output.data(), 0, sizeof(empty_output));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        return 3;
    }
    if (bridge.calls != 1 || bridge.status != 0) {
        std::fprintf(stderr, "Prism custom op calls=%d status=%d\n", bridge.calls, bridge.status);
        return 5;
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
                    const int source_index = grouped_values
                        ? (((offset / 64) % 2) * 2 + (offset / 128)) * 64 + offset % 64
                        : offset;
                    const int parity = __builtin_popcount(static_cast<unsigned>(index & column)) & 1;
                    expected += inputs[token * width + source_index] * explicit_signs[offset] *
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

    std::array<float, tokens * rows> actual;
    ggml_backend_tensor_get(output, actual.data(), 0, sizeof(actual));
    float max_output_error = 0.0f;
    for (int token = 0; token < tokens; ++token) {
        for (int row = 0; row < rows; ++row) {
            double expected = 0.0;
            for (int group = 0; group < groups; ++group) {
                int32_t partial = 0;
                for (int index = 0; index < group_width; ++index) {
                    partial += reference_codes[row * width + group * group_width + index] *
                               bridge.activations[token * width + group * group_width + index];
                }
                expected += partial * bridge.weight_scales[row * groups + group] *
                            bridge.activation_scales[token * groups + group];
            }
            const float error = std::abs(actual[token * rows + row] - expected);
            if (!std::isfinite(error) || error > 0.005f) {
                return 6;
            }
            max_output_error = std::max(max_output_error, error);
        }
    }
    const auto first_output = actual;
    for (float& value : inputs) {
        value *= -2.0f;
    }
    ggml_backend_tensor_set(input, inputs.data(), 0, sizeof(inputs));
    ggml_backend_tensor_set(signs, explicit_signs.data(), 0, sizeof(explicit_signs));
    std::array<float, tokens * width> observed_input;
    ggml_backend_tensor_get(input, observed_input.data(), 0, sizeof(observed_input));
    std::array<float, width> observed_signs;
    ggml_backend_tensor_get(signs, observed_signs.data(), 0, sizeof(observed_signs));
    if (observed_input != inputs || observed_signs != explicit_signs) {
        std::fprintf(stderr, "repeat graph leaves not restored before evaluation\n");
        return 10;
    }
    bridge.calls = 0;
    bridge.status = 0;
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS ||
        bridge.calls != 1 || bridge.status != 0) {
        return 7;
    }
    std::array<float, tokens * width> repeated_transform;
    ggml_backend_tensor_get(rotated, repeated_transform.data(), 0, sizeof(repeated_transform));
    for (std::size_t index = 0; index < transformed.size(); ++index) {
        const float error = std::abs(repeated_transform[index] + 2.0f * transformed[index]);
        if (!std::isfinite(error) || error > 1e-4f) {
            std::fprintf(stderr, "repeat FWHT mismatch index=%zu first=%.8f second=%.8f error=%.8f\n",
                         index, transformed[index], repeated_transform[index], error);
            return 9;
        }
    }
    ggml_backend_tensor_get(output, actual.data(), 0, sizeof(actual));
    float repeat_error = 0.0f;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float error = std::abs(actual[index] + 2.0f * first_output[index]);
        if (!std::isfinite(error) || error > 0.005f) {
            std::fprintf(stderr, "repeat mismatch index=%zu first=%.8f second=%.8f error=%.8f\n",
                         index, first_output[index], actual[index], error);
            return 8;
        }
        repeat_error = std::max(repeat_error, error);
    }
    std::printf(
        "{\"tokens\":%d,\"groups\":%d,\"gdn_v_grouped\":%s,\"graph_op\":\"map_custom2\","
        "\"graph_evaluations\":2,\"callback_calls\":%d,\"max_transform_error\":%.8f,"
        "\"max_output_error\":%.8f,\"repeat_scale_error\":%.8f}\n",
        tokens, groups, grouped_values ? "true" : "false",
        1 + bridge.calls, max_transform_error, max_output_error, repeat_error);
    ggml_gallocr_free(allocator);
    ggml_backend_free(backend);
    ggml_free(context);
    return 0;
}