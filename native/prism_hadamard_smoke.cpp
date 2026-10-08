#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

static int qk_norm_control() {
    constexpr int width = 128;
    constexpr int rows = 5;
    constexpr float epsilon = 1e-6f;
    ggml_init_params params = { 1024 * 1024, nullptr, true };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);
    if (!context) {
        return 1;
    }
    auto* input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, width, rows);
    auto* output = ggml_l2_norm(context.get(), input, epsilon);
    ggml_set_input(input);
    ggml_set_output(output);
    auto* graph = ggml_new_graph(context.get());
    ggml_build_forward_expand(graph, output);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
        ggml_gallocr_new(ggml_backend_cpu_buffer_type()), ggml_gallocr_free);
    if (!backend || !allocator || !ggml_backend_supports_op(backend.get(), output) ||
            !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
        return 2;
    }
    ggml_backend_cpu_set_n_threads(backend.get(), 1);
    const std::array<float, rows> amplitudes = { 0.0f, 1e-8f, 1e-4f, 0.01f, 1.0f };
    std::array<float, width * rows> values;
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < width; ++column) {
            values[row * width + column] = (column % 7 - 3) * amplitudes[row];
        }
    }
    ggml_backend_tensor_set(input, values.data(), 0, sizeof(values));
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        return 3;
    }
    std::array<float, width * rows> actual;
    ggml_backend_tensor_get(output, actual.data(), 0, sizeof(actual));
    if (!std::all_of(actual.begin(), actual.end(), [](float value) { return std::isfinite(value); })) {
        return 4;
    }
    std::printf("{\"width\":%d,\"rows\":%d,\"epsilon\":%.9g,\"outputs\":[", width, rows, epsilon);
    for (std::size_t index = 0; index < actual.size(); ++index) {
        std::printf("%s%.9g", index ? "," : "", actual[index]);
    }
    std::puts("]}");
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--qk-norm-control") == 0) {
        return qk_norm_control();
    }
    if (argc != 1) {
        return 5;
    }
    constexpr int width = 128;
    constexpr int tokens = 2;
    ggml_init_params params = { 1024 * 1024, nullptr, true };
    ggml_context* context = ggml_init(params);
    if (!context) {
        return 1;
    }
    ggml_tensor* matrix = ggml_new_tensor_2d(context, GGML_TYPE_F32, width, width);
    ggml_tensor* signs = ggml_new_tensor_1d(context, GGML_TYPE_F32, width);
    ggml_tensor* input = ggml_new_tensor_2d(context, GGML_TYPE_F32, width, tokens);
    ggml_tensor* signed_input = ggml_mul(context, input, signs);
    ggml_tensor* output = ggml_mul_mat(context, matrix, signed_input);
    ggml_mul_mat_set_hint(output, GGML_HINT_SRC0_IS_HADAMARD);
    ggml_set_input(matrix);
    ggml_set_input(signs);
    ggml_set_input(input);
    ggml_set_output(output);
    ggml_cgraph* graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (!backend || !allocator || !ggml_gallocr_alloc_graph(allocator, graph)) {
        return 2;
    }
    std::vector<float> ignored_matrix(width * width, 0.0f);
    std::array<float, width> explicit_signs;
    std::array<float, width * tokens> inputs;
    for (int index = 0; index < width; ++index) {
        explicit_signs[index] = index % 3 == 0 ? -1.0f : 1.0f;
        for (int token = 0; token < tokens; ++token) {
            inputs[token * width + index] = (index % 9 - 4) * (token == 0 ? 0.25f : -0.5f);
        }
    }
    ggml_backend_tensor_set(matrix, ignored_matrix.data(), 0, ignored_matrix.size() * sizeof(float));
    ggml_backend_tensor_set(signs, explicit_signs.data(), 0, sizeof(explicit_signs));
    ggml_backend_tensor_set(input, inputs.data(), 0, sizeof(inputs));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        return 3;
    }

    std::array<float, width * tokens> actual;
    ggml_backend_tensor_get(output, actual.data(), 0, sizeof(actual));
    float max_error = 0.0f;
    for (int token = 0; token < tokens; ++token) {
        for (int column = 0; column < width; ++column) {
            double expected = 0.0;
            for (int index = 0; index < width; ++index) {
                const int parity = __builtin_popcount(static_cast<unsigned>(index & column)) & 1;
                expected += inputs[token * width + index] * explicit_signs[index] *
                            (parity ? -1 : 1);
            }
            expected /= std::sqrt(width);
            const float error = std::abs(actual[token * width + column] - expected);
            if (!std::isfinite(error) || error > 1e-4f) {
                std::fprintf(stderr, "Prism FWHT mismatch at token %d, column %d\n", token, column);
                return 4;
            }
            max_error = std::max(max_error, error);
        }
    }
    std::printf("{\"tokens\":%d,\"width\":%d,\"max_abs_error\":%.8f}\n",
                tokens, width, max_error);
    ggml_gallocr_free(allocator);
    ggml_backend_free(backend);
    ggml_free(context);
    return 0;
}