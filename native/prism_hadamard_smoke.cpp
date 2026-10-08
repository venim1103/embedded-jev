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

static int gdn_control(int tokens, bool initialized, bool raw_gates) {
    constexpr int width = 128;
    constexpr int key_heads = 2;
    constexpr int value_heads = 4;
    ggml_init_params params = { 1024 * 1024, nullptr, true };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);
    if (!context) {
        return 1;
    }
    auto* query = ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, width, key_heads, tokens, 1);
    auto* key = ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, width, key_heads, tokens, 1);
    auto* value = ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, width, value_heads, tokens, 1);
    auto* decay = ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, 1, value_heads, tokens, 1);
    auto* beta = ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, 1, value_heads, tokens, 1);
    auto* state = ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, width, width, value_heads, 1);
    auto* dt_bias = raw_gates ? ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, value_heads) : nullptr;
    auto* decay_multiplier = raw_gates ? ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, value_heads) : nullptr;
    for (auto* input : { query, key, value, decay, beta, state }) {
        ggml_set_input(input);
    }
    auto* output = ggml_gated_delta_net(context.get(),
        ggml_l2_norm(context.get(), query, 1e-6f), ggml_l2_norm(context.get(), key, 1e-6f),
        value, decay, beta, state, 1);
    if (raw_gates) {
        ggml_set_input(dt_bias);
        ggml_set_input(decay_multiplier);
        ggml_gated_delta_net_set_raw_gates(output, dt_bias, decay_multiplier);
    }
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
    std::vector<float> queries(width * key_heads * tokens);
    std::vector<float> keys(queries.size());
    std::vector<float> values(width * value_heads * tokens);
    std::vector<float> decays(value_heads * tokens);
    std::vector<float> betas(decays.size());
    std::vector<float> initial(width * width * value_heads);
    const std::array<float, 5> raw_alpha = { -25.0f, -3.0f, 0.0f, 20.0f, 25.0f };
    for (int token = 0; token < tokens; ++token) {
        for (int head = 0; head < key_heads; ++head) {
            for (int column = 0; column < width; ++column) {
                const int index = (token * key_heads + head) * width + column;
                queries[index] = ((column * 7 + token * 3 + head * 11) % 29 - 14) / 64.0f;
                keys[index] = ((column * 3 + token * 7 + head * 5) % 31 - 15) / 64.0f;
            }
        }
        for (int head = 0; head < value_heads; ++head) {
            decays[token * value_heads + head] = raw_gates ? raw_alpha[(token + 2 * head) % 5] :
                -0.125f - ((token + 2 * head) % 5) / 16.0f;
            betas[token * value_heads + head] = raw_gates ? ((token + head) % 7 - 3) / 4.0f :
                0.25f + ((token + head) % 7) / 16.0f;
            for (int column = 0; column < width; ++column) {
                values[(token * value_heads + head) * width + column] =
                    ((column * 5 + token * 2 + head * 13) % 23 - 11) / 32.0f;
            }
        }
    }
    if (initialized) {
        for (int head = 0; head < value_heads; ++head) {
            for (int value_index = 0; value_index < width; ++value_index) {
                for (int key_index = 0; key_index < width; ++key_index) {
                    initial[(head * width + value_index) * width + key_index] =
                        ((key_index * 3 + value_index * 5 + head * 7) % 19 - 9) / 4096.0f;
                }
            }
        }
    }
    const auto upload = [](ggml_tensor* tensor, const std::vector<float>& data) {
        ggml_backend_tensor_set(tensor, data.data(), 0, data.size() * sizeof(float));
    };
    upload(query, queries);
    upload(key, keys);
    upload(value, values);
    upload(decay, decays);
    upload(beta, betas);
    upload(state, initial);
    if (raw_gates) {
        std::vector<float> biases(value_heads);
        std::vector<float> multipliers(value_heads);
        for (int head = 0; head < value_heads; ++head) {
            biases[head] = (head - 2) / 8.0f;
            multipliers[head] = -(head + 1) / 4.0f;
        }
        upload(dt_bias, biases);
        upload(decay_multiplier, multipliers);
    }
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        return 3;
    }
    const std::size_t output_count = values.size() + initial.size();
    if (ggml_nbytes(output) != output_count * sizeof(float)) {
        return 4;
    }
    std::vector<float> actual(output_count);
    std::vector<float> unchanged(initial.size());
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    ggml_backend_tensor_get(state, unchanged.data(), 0, unchanged.size() * sizeof(float));
    if (unchanged != initial || !std::all_of(actual.begin(), actual.end(), [](float value) { return std::isfinite(value); })) {
        return 4;
    }
    std::printf("{\"width\":%d,\"key_heads\":%d,\"value_heads\":%d,\"tokens\":%d,"
                "\"initialized\":%s,\"raw_gates\":%s,\"broadcast\":\"tiled\",\"state_layout\":\"value_by_key\",\"outputs\":[",
                width, key_heads, value_heads, tokens, initialized ? "true" : "false", raw_gates ? "true" : "false");
    for (std::size_t index = 0; index < values.size(); ++index) {
        std::printf("%s%.9g", index ? "," : "", actual[index]);
    }
    std::printf("],\"state\":[");
    for (std::size_t index = values.size(); index < actual.size(); ++index) {
        std::printf("%s%.9g", index == values.size() ? "" : ",", actual[index]);
    }
    std::puts("]}");
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--qk-norm-control") == 0) {
        return qk_norm_control();
    }
    if ((argc == 4 || argc == 5) && std::strcmp(argv[1], "--gdn-control") == 0) {
        const int tokens = std::strcmp(argv[2], "1") == 0 ? 1 : std::strcmp(argv[2], "7") == 0 ? 7 :
                           std::strcmp(argv[2], "64") == 0 ? 64 : std::strcmp(argv[2], "80") == 0 ? 80 : 0;
        if (!tokens || (std::strcmp(argv[3], "zero") != 0 && std::strcmp(argv[3], "nonzero") != 0) ||
            (argc == 5 && std::strcmp(argv[4], "raw") != 0)) {
            return 5;
        }
        return gdn_control(tokens, std::strcmp(argv[3], "nonzero") == 0, argc == 5);
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