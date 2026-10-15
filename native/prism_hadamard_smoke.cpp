#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
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

static int attention_control(int tokens, int first_token) {
    constexpr int width = 256;
    constexpr int query_heads = 4;
    constexpr int cache_heads = 2;
    constexpr int cache_tokens = 128;
    ggml_init_params params = { 1024 * 1024, nullptr, true };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);
    if (!context) {
        return 1;
    }
    auto* query = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, width, tokens, query_heads);
    auto* key = ggml_new_tensor_3d(context.get(), GGML_TYPE_F16, width, cache_tokens, cache_heads);
    auto* value = ggml_new_tensor_3d(context.get(), GGML_TYPE_F16, cache_tokens, width, cache_heads);
    auto* mask = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, cache_tokens, tokens);
    for (auto* input : { query, key, value, mask }) {
        ggml_set_input(input);
        ggml_set_output(input);
    }
    auto* scores = ggml_mul_mat(context.get(), key, query);
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    auto* probabilities = ggml_soft_max_ext(context.get(), scores, mask, 1.0f / 16.0f, 0.0f);
    auto* weighted = ggml_mul_mat(context.get(), value, probabilities);
    auto* output = ggml_cont(context.get(), ggml_permute(context.get(), weighted, 0, 2, 1, 3));
    auto* graph = ggml_new_graph(context.get());
    for (auto* tensor : { scores, probabilities, output }) {
        ggml_set_output(tensor);
        ggml_build_forward_expand(graph, tensor);
    }
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
        ggml_gallocr_new(ggml_backend_cpu_buffer_type()), ggml_gallocr_free);
    if (!backend || !allocator || !ggml_backend_supports_op(backend.get(), scores) ||
            !ggml_backend_supports_op(backend.get(), probabilities) ||
            !ggml_backend_supports_op(backend.get(), weighted) ||
            !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
        return 2;
    }
    ggml_backend_cpu_set_n_threads(backend.get(), 1);
    std::vector<float> queries(width * tokens * query_heads);
    std::vector<ggml_fp16_t> keys(width * cache_tokens * cache_heads);
    std::vector<ggml_fp16_t> values(keys.size());
    std::vector<float> masks(cache_tokens * tokens);
    for (int head = 0; head < query_heads; ++head) {
        for (int token = 0; token < tokens; ++token) {
            for (int column = 0; column < width; ++column) {
                queries[(head * tokens + token) * width + column] =
                    ((column * 7 + (first_token + token) * 3 + head * 11) % 29 - 14) / 8.0f +
                    ((column + first_token + token + head) % 5 - 2) * 0.0003f;
            }
        }
    }
    for (int head = 0; head < cache_heads; ++head) {
        for (int token = 0; token < cache_tokens; ++token) {
            for (int column = 0; column < width; ++column) {
                keys[(head * cache_tokens + token) * width + column] =
                    ggml_fp32_to_fp16(((column * 3 + token * 7 + head * 5) % 31 - 15) / 32.0f);
                values[(head * width + column) * cache_tokens + token] =
                    ggml_fp32_to_fp16(((column * 5 + token * 2 + head * 13) % 23 - 11) / 32.0f);
            }
        }
    }
    for (int token = 0; token < tokens; ++token) {
        for (int cached = 0; cached < cache_tokens; ++cached) {
            masks[token * cache_tokens + cached] = cached <= first_token + token ? 0.0f : -INFINITY;
        }
    }
    ggml_backend_tensor_set(query, queries.data(), 0, queries.size() * sizeof(float));
    ggml_backend_tensor_set(key, keys.data(), 0, keys.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(value, values.data(), 0, values.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(mask, masks.data(), 0, masks.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        return 3;
    }
    std::vector<float> actual(width * tokens * query_heads);
    std::vector<float> actual_scores(cache_tokens * tokens * query_heads);
    std::vector<float> actual_probabilities(actual_scores.size());
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    ggml_backend_tensor_get(scores, actual_scores.data(), 0, actual_scores.size() * sizeof(float));
    ggml_backend_tensor_get(probabilities, actual_probabilities.data(), 0, actual_probabilities.size() * sizeof(float));
    for (const auto* data : { &actual, &actual_scores, &actual_probabilities }) {
        if (!std::all_of(data->begin(), data->end(), [](float entry) { return std::isfinite(entry); })) {
            return 4;
        }
    }
    std::vector<float> unchanged_query(queries.size());
    std::vector<float> unchanged_mask(masks.size());
    std::vector<ggml_fp16_t> unchanged_key(keys.size());
    std::vector<ggml_fp16_t> unchanged_value(values.size());
    ggml_backend_tensor_get(query, unchanged_query.data(), 0, queries.size() * sizeof(float));
    ggml_backend_tensor_get(mask, unchanged_mask.data(), 0, masks.size() * sizeof(float));
    ggml_backend_tensor_get(key, unchanged_key.data(), 0, keys.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_get(value, unchanged_value.data(), 0, values.size() * sizeof(ggml_fp16_t));
    if (std::memcmp(queries.data(), unchanged_query.data(), queries.size() * sizeof(float)) != 0 ||
            std::memcmp(masks.data(), unchanged_mask.data(), masks.size() * sizeof(float)) != 0 ||
            std::memcmp(keys.data(), unchanged_key.data(), keys.size() * sizeof(ggml_fp16_t)) != 0 ||
            std::memcmp(values.data(), unchanged_value.data(), values.size() * sizeof(ggml_fp16_t)) != 0) {
        return 4;
    }
    std::printf("{\"width\":%d,\"tokens\":%d,\"first_token\":%d,\"query_heads\":%d,\"cache_heads\":%d,\"cache_tokens\":%d,"
                "\"cache_type\":\"f16\",\"broadcast\":\"grouped\",\"flash_attention\":false,\"inputs_unchanged\":true",
                width, tokens, first_token, query_heads, cache_heads, cache_tokens);
    const auto emit = [](const char* name, const std::vector<float>& data) {
        std::printf(",\"%s\":[", name);
        for (std::size_t index = 0; index < data.size(); ++index) {
            std::printf("%s%.9g", index ? "," : "", data[index]);
        }
        std::printf("]");
    };
    emit("scores", actual_scores);
    emit("probabilities", actual_probabilities);
    emit("outputs", actual);
    std::puts("}");
    return 0;
}

static int gdn_control(int tokens, bool initialized, bool raw_gates,
                       int key_heads = 2, int value_heads = 4,
                       std::FILE* source = nullptr, const char* output_path = nullptr) {
    constexpr int width = 128;
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
    const auto read = [source](std::vector<float>& data) {
        return std::fread(data.data(), sizeof(float), data.size(), source) == data.size() &&
            std::all_of(data.begin(), data.end(), [](float value) { return std::isfinite(value); });
    };
    if (source) {
        for (auto* data : { &queries, &keys, &values, &decays, &betas, &initial }) {
            if (!read(*data)) {
                return 6;
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
        if (source && (!read(biases) || !read(multipliers) || std::fgetc(source) != EOF ||
                std::ferror(source) || !std::all_of(multipliers.begin(), multipliers.end(),
                    [](float value) { return value < 0; }))) {
            return 6;
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
    if (output_path) {
        std::FILE* destination = std::fopen(output_path, "wbx");
        if (!destination) {
            return 6;
        }
        const bool written = std::fwrite(actual.data(), sizeof(float), actual.size(), destination) == actual.size();
        const bool closed = std::fclose(destination) == 0;
        if (!written || !closed) {
            std::remove(output_path);
            return 6;
        }
        std::printf("{\"width\":%d,\"key_heads\":%d,\"value_heads\":%d,\"tokens\":%d,"
                    "\"raw_gates\":true,\"broadcast\":\"tiled\",\"state_layout\":\"value_by_key\","
                    "\"output_floats\":%zu,\"state_floats\":%zu}\n",
                    width, key_heads, value_heads, tokens, values.size(), initial.size());
        return 0;
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
    if ((argc == 3 || argc == 4) && std::strcmp(argv[1], "--attention-control") == 0) {
        const int tokens = std::strcmp(argv[2], "1") == 0 ? 1 : std::strcmp(argv[2], "7") == 0 ? 7 :
                           std::strcmp(argv[2], "80") == 0 ? 80 : 0;
        int first_token = 0;
        if (argc == 4) {
            const char* end = argv[3] + std::strlen(argv[3]);
            const auto parsed = std::from_chars(argv[3], end, first_token);
            if (parsed.ec != std::errc() || parsed.ptr != end) {
                return 5;
            }
        }
        return tokens && first_token >= 0 && first_token < 80 && first_token + tokens <= 80 ?
            attention_control(tokens, first_token) : 5;
    }
    if (argc == 4 && std::strcmp(argv[1], "--gdn-file-control") == 0) {
        std::unique_ptr<std::FILE, decltype(&std::fclose)> source(std::fopen(argv[2], "rb"), std::fclose);
        std::array<std::uint32_t, 5> header;
        if (!source || std::fread(header.data(), sizeof(header), 1, source.get()) != 1 ||
                header[0] != 0x3144474a || header[1] < 1 || header[1] > 128 || header[2] != 128 ||
                header[3] < 1 || header[3] > 32 || header[4] < 1 || header[4] > 32 ||
                header[4] % header[3] != 0) {
            return 6;
        }
        return gdn_control(header[1], false, true, header[3], header[4], source.get(), argv[3]);
    }
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