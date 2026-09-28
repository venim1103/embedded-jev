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

int main(int argc, char** argv) {
    const bool batched = argc == 2 && std::strcmp(argv[1], "--batch") == 0;
    const bool grouped = batched || (argc == 2 && std::strcmp(argv[1], "--groups") == 0);
    if (argc > 2 || (argc == 2 && !grouped)) {
        return 5;
    }
    const int tokens = batched ? 2 : 1;
    ggml_init_params params = { 1024 * 1024, nullptr, true };
    ggml_context* context = ggml_init(params);
    if (!context) {
        return 1;
    }
    ggml_tensor* weights = ggml_new_tensor_2d(context, GGML_TYPE_I2_S, 128, 4);
    ggml_tensor* inputs = ggml_new_tensor_2d(context, GGML_TYPE_F32, 128, tokens);
    ggml_tensor* output = ggml_mul_mat(context, weights, inputs);
    ggml_set_input(weights);
    ggml_set_input(inputs);
    ggml_set_output(output);
    ggml_cgraph* graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (!backend || !allocator || !ggml_gallocr_alloc_graph(allocator, graph)) {
        return 2;
    }

    std::vector<uint8_t> packed(ggml_nbytes(weights), 0);
    std::memset(packed.data(), 0x55, 32);
    std::memset(packed.data() + 32, 0xAA, 32);
    std::memset(packed.data() + 64, 0x00, 32);
    std::memset(packed.data() + 96, 0x55, 32);
    if (grouped) {
        std::memset(packed.data() + 96, 0xAA, 32);
    }
    const float weight_scale = 1.0f;
    std::memcpy(packed.data() + 128, &weight_scale, sizeof(weight_scale));
    std::array<float, 256> activations;
    activations.fill(1.0f);
    if (batched) {
        std::fill(activations.begin() + 128, activations.end(), -2.0f);
    }
    ggml_backend_tensor_set(weights, packed.data(), 0, packed.size());
    ggml_backend_tensor_set(inputs, activations.data(), 0, tokens * 128 * sizeof(float));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        return 3;
    }

    std::array<float, 8> results;
    ggml_backend_tensor_get(output, results.data(), 0, tokens * 4 * sizeof(float));
    const std::array<float, 4> expected = {0.0f, 128.0f, -128.0f, grouped ? 128.0f : 0.0f};
    for (int token = 0; token < tokens; ++token) {
        for (std::size_t row = 0; row < expected.size(); ++row) {
            const float value = results[token * 4 + row];
            std::printf("%.6f\n", value);
            if (!std::isfinite(value) ||
                std::abs(value - expected[row] * (token == 0 ? 1.0f : -2.0f)) > 1e-4f) {
                return 4;
            }
        }
    }
    if (grouped) {
        std::memset(packed.data(), 0xAA, 32);
        std::memset(packed.data() + 32, 0x00, 32);
        std::memset(packed.data() + 64, 0x55, 32);
        std::memset(packed.data() + 96, 0x00, 32);
        std::memcpy(packed.data() + 128, &weight_scale, sizeof(weight_scale));
        ggml_backend_tensor_set(weights, packed.data(), 0, packed.size());
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            return 3;
        }
        std::array<float, 8> other;
        ggml_backend_tensor_get(output, other.data(), 0, tokens * 4 * sizeof(float));
        const std::array<std::array<float, 2>, 4> scales = {{{1.0f, 0.25f},
            {0.5f, 3.0f}, {2.0f, 0.25f}, {1.0f, 0.5f}}};
        const std::array<float, 4> expected_grouped = {32.0f, -320.0f, -256.0f, 64.0f};
        for (int token = 0; token < tokens; ++token) {
            for (std::size_t row = 0; row < expected_grouped.size(); ++row) {
                const float value = results[token * 4 + row] * scales[row][0]
                    + other[token * 4 + row] * scales[row][1];
                std::printf("grouped %.6f\n", value);
                if (!std::isfinite(value) ||
                    std::abs(value - expected_grouped[row] * (token == 0 ? 1.0f : -2.0f)) > 1e-4f) {
                    return 6;
                }
            }
        }
    }
    ggml_gallocr_free(allocator);
    ggml_backend_free(backend);
    ggml_free(context);
    return 0;
}