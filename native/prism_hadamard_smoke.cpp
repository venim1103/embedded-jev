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
#include <utility>
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

static int kv_write_control(int tokens, int first_token, bool permuted) {
    constexpr int width = 256;
    constexpr int heads = 2;
    constexpr int cache_tokens = 128;
    constexpr int channels = width * heads;
    ggml_init_params params = { 1024 * 1024, nullptr, true };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);
    if (!context) {
        return 1;
    }
    auto* key = ggml_new_tensor_2d(context.get(), GGML_TYPE_F16, channels, cache_tokens);
    auto* value = ggml_new_tensor_2d(context.get(), GGML_TYPE_F16, 1, channels * cache_tokens);
    auto* key_source = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, channels, tokens);
    auto* value_source = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 1, channels * tokens);
    auto* key_indices = ggml_new_tensor_1d(context.get(), GGML_TYPE_I64, tokens);
    auto* value_indices = ggml_new_tensor_1d(context.get(), GGML_TYPE_I64, channels * tokens);
    for (auto* tensor : { key, value, key_source, value_source, key_indices, value_indices }) {
        ggml_set_input(tensor);
        ggml_set_output(tensor);
    }
    auto* key_output = ggml_set_rows(context.get(), key, key_source, key_indices);
    auto* value_output = ggml_set_rows(context.get(), value, value_source, value_indices);
    auto* graph = ggml_new_graph(context.get());
    for (auto* tensor : { key_output, value_output }) {
        ggml_set_output(tensor);
        ggml_build_forward_expand(graph, tensor);
    }
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
        ggml_gallocr_new(ggml_backend_cpu_buffer_type()), ggml_gallocr_free);
    if (!backend || !allocator || !ggml_backend_supports_op(backend.get(), key_output) ||
            !ggml_backend_supports_op(backend.get(), value_output) ||
            !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
        return 2;
    }
    ggml_backend_cpu_set_n_threads(backend.get(), 1);
    std::vector<ggml_fp16_t> keys(cache_tokens * channels);
    std::vector<ggml_fp16_t> values(keys.size());
    for (int slot = 0; slot < cache_tokens; ++slot) {
        for (int channel = 0; channel < channels; ++channel) {
            keys[slot * channels + channel] = ggml_fp32_to_fp16(((channel * 3 + slot * 7) % 31 - 15) / 32.0f);
            values[channel * cache_tokens + slot] = ggml_fp32_to_fp16(((channel * 5 + slot * 2) % 23 - 11) / 32.0f);
        }
    }
    std::vector<float> updates_key(tokens * channels);
    std::vector<float> updates_value(updates_key.size());
    std::vector<std::int64_t> indices_key(tokens);
    std::vector<std::int64_t> indices_value(tokens * channels);
    for (int token = 0; token < tokens; ++token) {
        const int absolute = first_token + token;
        const int slot = permuted ? (absolute * 37) % cache_tokens : absolute;
        indices_key[token] = slot;
        for (int head = 0; head < heads; ++head) {
            for (int column = 0; column < width; ++column) {
                const int channel = head * width + column;
                const int index = token * channels + channel;
                updates_key[index] = ((column * 7 + absolute * 3 + head * 11) % 29 - 14) / 103.0f;
                updates_value[index] = ((column * 5 + absolute * 2 + head * 13) % 23 - 11) / 97.0f;
                indices_value[index] = channel * cache_tokens + slot;
            }
        }
    }
    ggml_backend_tensor_set(key, keys.data(), 0, keys.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(value, values.data(), 0, values.size() * sizeof(ggml_fp16_t));
    for (const auto& entry : { std::make_pair(key_source, &updates_key), std::make_pair(value_source, &updates_value) }) {
        ggml_backend_tensor_set(entry.first, entry.second->data(), 0, entry.second->size() * sizeof(float));
    }
    for (const auto& entry : { std::make_pair(key_indices, &indices_key), std::make_pair(value_indices, &indices_value) }) {
        ggml_backend_tensor_set(entry.first, entry.second->data(), 0, entry.second->size() * sizeof(std::int64_t));
    }
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        return 3;
    }
    ggml_backend_tensor_get(key_output, keys.data(), 0, keys.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_get(value_output, values.data(), 0, values.size() * sizeof(ggml_fp16_t));
    for (const auto& entry : { std::make_pair(key_source, &updates_key), std::make_pair(value_source, &updates_value) }) {
        std::vector<float> unchanged(entry.second->size());
        ggml_backend_tensor_get(entry.first, unchanged.data(), 0, unchanged.size() * sizeof(float));
        if (std::memcmp(unchanged.data(), entry.second->data(), unchanged.size() * sizeof(float)) != 0) {
            return 4;
        }
    }
    for (const auto& entry : { std::make_pair(key_indices, &indices_key), std::make_pair(value_indices, &indices_value) }) {
        std::vector<std::int64_t> unchanged(entry.second->size());
        ggml_backend_tensor_get(entry.first, unchanged.data(), 0, unchanged.size() * sizeof(std::int64_t));
        if (std::memcmp(unchanged.data(), entry.second->data(), unchanged.size() * sizeof(std::int64_t)) != 0) {
            return 4;
        }
    }
    for (const auto* data : { &keys, &values }) {
        if (!std::all_of(data->begin(), data->end(), [](ggml_fp16_t entry) { return std::isfinite(ggml_fp16_to_fp32(entry)); })) {
            return 4;
        }
    }
    std::printf("{\"width\":%d,\"cache_heads\":%d,\"cache_tokens\":%d,\"tokens\":%d,\"first_token\":%d,"
                "\"row_mode\":\"%s\",\"cache_type\":\"f16\",\"operation\":\"set_rows\",\"source_and_indices_unchanged\":true",
                width, heads, cache_tokens, tokens, first_token, permuted ? "permuted" : "contiguous");
    const auto emit = [](const char* name, const std::vector<ggml_fp16_t>& data) {
        std::printf(",\"%s\":[", name);
        for (std::size_t index = 0; index < data.size(); ++index) {
            std::printf("%s%.9g", index ? "," : "", ggml_fp16_to_fp32(data[index]));
        }
        std::printf("]");
    };
    emit("key", keys);
    emit("value_transposed", values);
    std::puts("}");
    return 0;
}

static int rope_control(int tokens, int first_token, bool distinct_axes) {
    constexpr int width = 256;
    constexpr int heads = 4;
    constexpr int rotary_dimensions = 64;
    int sections[GGML_MROPE_SECTIONS] = { 11, 11, 10, 0 };
    ggml_init_params params = { 1024 * 1024, nullptr, true };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);
    if (!context) {
        return 1;
    }
    auto* input = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, width, heads, tokens);
    auto* positions = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, tokens * 4);
    for (auto* tensor : { input, positions }) {
        ggml_set_input(tensor);
        ggml_set_output(tensor);
    }
    auto* output = ggml_rope_multi(context.get(), input, positions, nullptr, rotary_dimensions, sections,
                                   GGML_ROPE_TYPE_IMROPE, 32768, 10000000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
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
    std::vector<float> values(width * heads * tokens);
    std::vector<std::int32_t> position_ids(tokens * 4);
    for (int token = 0; token < tokens; ++token) {
        const int absolute = first_token + token;
        position_ids[token] = absolute;
        position_ids[tokens + token] = distinct_axes ? (absolute * 2 + 3) % 80 : absolute;
        position_ids[tokens * 2 + token] = distinct_axes ? (absolute * 5 + 7) % 80 : absolute;
        position_ids[tokens * 3 + token] = distinct_axes ? (absolute * 11 + 5) % 80 : absolute;
        for (int head = 0; head < heads; ++head) {
            for (int column = 0; column < width; ++column) {
                values[(token * heads + head) * width + column] =
                    ((column * 7 + absolute * 3 + head * 11) % 29 - 14) / 8.0f;
            }
        }
    }
    ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
    ggml_backend_tensor_set(positions, position_ids.data(), 0, position_ids.size() * sizeof(std::int32_t));
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        return 3;
    }
    std::vector<float> actual(values.size());
    std::vector<float> unchanged(values.size());
    std::vector<std::int32_t> unchanged_positions(position_ids.size());
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    ggml_backend_tensor_get(input, unchanged.data(), 0, unchanged.size() * sizeof(float));
    ggml_backend_tensor_get(positions, unchanged_positions.data(), 0, unchanged_positions.size() * sizeof(std::int32_t));
    if (!std::all_of(actual.begin(), actual.end(), [](float value) { return std::isfinite(value); }) ||
            std::memcmp(values.data(), unchanged.data(), values.size() * sizeof(float)) != 0 ||
            std::memcmp(position_ids.data(), unchanged_positions.data(), position_ids.size() * sizeof(std::int32_t)) != 0) {
        return 4;
    }
    std::printf("{\"width\":%d,\"heads\":%d,\"tokens\":%d,\"first_token\":%d,\"rotary_dimensions\":%d,"
                "\"mode\":\"imrope\",\"position_mode\":\"%s\",\"sections\":[11,11,10,0],\"frequency_base\":10000000,"
                "\"inputs_unchanged\":true,\"outputs\":[",
                width, heads, tokens, first_token, rotary_dimensions, distinct_axes ? "axes" : "text");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        std::printf("%s%.9g", index ? "," : "", actual[index]);
    }
    std::puts("]}");
    return 0;
}

static int cached_attention_control(const std::vector<int>& partition, bool reset) {
    constexpr int width = 256, query_heads = 4, cache_heads = 2, channels = width * cache_heads;
    constexpr int cache_tokens = 128, total_tokens = 80;
    ggml_init_params params = { 1024 * 1024, nullptr, true };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> cache_context(ggml_init(params), ggml_free);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    if (!cache_context || !backend) {
        return 1;
    }
    auto* key = ggml_new_tensor_2d(cache_context.get(), GGML_TYPE_F16, channels, cache_tokens);
    auto* value = ggml_new_tensor_2d(cache_context.get(), GGML_TYPE_F16, 1, channels * cache_tokens);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> cache_buffer(
        ggml_backend_alloc_ctx_tensors(cache_context.get(), backend.get()), ggml_backend_buffer_free);
    if (!cache_buffer) {
        return 2;
    }
    ggml_backend_cpu_set_n_threads(backend.get(), 1);
    const auto* key_address = key->data;
    const auto* value_address = value->data;
    std::vector<ggml_fp16_t> initial_key(cache_tokens * channels), initial_value(initial_key.size());
    for (int slot = 0; slot < cache_tokens; ++slot) {
        for (int channel = 0; channel < channels; ++channel) {
            initial_key[slot * channels + channel] = ggml_fp32_to_fp16(((channel * 3 + slot * 7) % 31 - 15) / 32.0f);
            initial_value[channel * cache_tokens + slot] = ggml_fp32_to_fp16(((channel * 5 + slot * 2) % 23 - 11) / 32.0f);
        }
    }
    ggml_backend_tensor_set(key, initial_key.data(), 0, initial_key.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(value, initial_value.data(), 0, initial_value.size() * sizeof(ggml_fp16_t));
    auto expected_key = initial_key, expected_value = initial_value;
    std::vector<float> all_scores(query_heads * total_tokens * cache_tokens), all_probabilities(all_scores.size());
    std::vector<float> all_outputs(total_tokens * query_heads * width);
    const auto matches = [](ggml_tensor* tensor, const auto& expected) {
        const auto bytes = expected.size() * sizeof(expected[0]);
        if (ggml_nbytes(tensor) != bytes) {
            return false;
        }
        std::vector<std::uint8_t> actual(bytes);
        ggml_backend_tensor_get(tensor, actual.data(), 0, bytes);
        return std::memcmp(actual.data(), expected.data(), bytes) == 0;
    };
    int first = 0;
    for (const int tokens : partition) {
        if (reset && first) {
            expected_key = initial_key;
            expected_value = initial_value;
            ggml_backend_tensor_set(key, initial_key.data(), 0, initial_key.size() * sizeof(ggml_fp16_t));
            ggml_backend_tensor_set(value, initial_value.data(), 0, initial_value.size() * sizeof(ggml_fp16_t));
        }
        const auto step = [&]() -> int {
            std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);
            if (!context) {
                return 1;
            }
            auto* key_source = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, channels, tokens);
            auto* value_source = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 1, channels * tokens);
            auto* key_indices = ggml_new_tensor_1d(context.get(), GGML_TYPE_I64, tokens);
            auto* value_indices = ggml_new_tensor_1d(context.get(), GGML_TYPE_I64, channels * tokens);
            auto* query = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, width, tokens, query_heads);
            auto* mask = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, cache_tokens, tokens);
            for (auto* input : { key_source, value_source, key_indices, value_indices, query, mask }) {
                ggml_set_input(input);
                ggml_set_output(input);
            }
            auto* key_written = ggml_set_rows(context.get(), key, key_source, key_indices);
            auto* value_written = ggml_set_rows(context.get(), value, value_source, value_indices);
            auto* key_view = ggml_view_3d(context.get(), key_written, width, cache_tokens, cache_heads,
                                         channels * sizeof(ggml_fp16_t), width * sizeof(ggml_fp16_t), 0);
            auto* value_view = ggml_view_3d(context.get(), value_written, cache_tokens, width, cache_heads,
                                           cache_tokens * sizeof(ggml_fp16_t), cache_tokens * width * sizeof(ggml_fp16_t), 0);
            auto* scores = ggml_mul_mat(context.get(), key_view, query);
            ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
            auto* probabilities = ggml_soft_max_ext(context.get(), scores, mask, 1.0f / 16.0f, 0.0f);
            auto* weighted = ggml_mul_mat(context.get(), value_view, probabilities);
            auto* output = ggml_cont(context.get(), ggml_permute(context.get(), weighted, 0, 2, 1, 3));
            auto* graph = ggml_new_graph(context.get());
            for (auto* tensor : { scores, probabilities, output }) {
                ggml_set_output(tensor);
                ggml_build_forward_expand(graph, tensor);
            }
            std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
                ggml_gallocr_new(ggml_backend_cpu_buffer_type()), ggml_gallocr_free);
            if (!allocator || !ggml_backend_supports_op(backend.get(), key_written) ||
                    !ggml_backend_supports_op(backend.get(), value_written) ||
                    !ggml_backend_supports_op(backend.get(), scores) ||
                    !ggml_backend_supports_op(backend.get(), probabilities) ||
                    !ggml_backend_supports_op(backend.get(), weighted) || !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
                return 2;
            }
            std::vector<float> updates_key(tokens * channels), updates_value(updates_key.size());
            std::vector<std::int64_t> indices_key(tokens), indices_value(updates_key.size());
            std::vector<float> queries(query_heads * tokens * width), masks(tokens * cache_tokens);
            for (int token = 0; token < tokens; ++token) {
                const int slot = first + token;
                indices_key[token] = slot;
                for (int head = 0; head < cache_heads; ++head) {
                    for (int column = 0; column < width; ++column) {
                        const int channel = head * width + column, index = token * channels + channel;
                        updates_key[index] = ((column * 3 + slot * 7 + head * 5) % 31 - 15) / 32.0f;
                        updates_value[index] = ((column * 5 + slot * 2 + head * 13) % 23 - 11) / 32.0f;
                        indices_value[index] = channel * cache_tokens + slot;
                        expected_key[slot * channels + channel] = ggml_fp32_to_fp16(updates_key[index]);
                        expected_value[channel * cache_tokens + slot] = ggml_fp32_to_fp16(updates_value[index]);
                    }
                }
                for (int head = 0; head < query_heads; ++head) {
                    for (int column = 0; column < width; ++column) {
                        queries[(head * tokens + token) * width + column] =
                            ((column * 7 + slot * 3 + head * 11) % 29 - 14) / 8.0f +
                            ((column + slot + head) % 5 - 2) * 0.0003f;
                    }
                }
                for (int cached = 0; cached < cache_tokens; ++cached) {
                    masks[token * cache_tokens + cached] = cached <= slot ? 0.0f : -INFINITY;
                }
            }
            for (const auto& entry : { std::make_pair(key_source, &updates_key), std::make_pair(value_source, &updates_value),
                                       std::make_pair(query, &queries), std::make_pair(mask, &masks) }) {
                ggml_backend_tensor_set(entry.first, entry.second->data(), 0, entry.second->size() * sizeof(float));
            }
            for (const auto& entry : { std::make_pair(key_indices, &indices_key), std::make_pair(value_indices, &indices_value) }) {
                ggml_backend_tensor_set(entry.first, entry.second->data(), 0, entry.second->size() * sizeof(std::int64_t));
            }
            if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
                return 3;
            }
            for (const auto& entry : { std::make_pair(key_source, &updates_key), std::make_pair(value_source, &updates_value),
                                       std::make_pair(query, &queries), std::make_pair(mask, &masks) }) {
                if (!matches(entry.first, *entry.second)) {
                    return 4;
                }
            }
            if (!matches(key_indices, indices_key) || !matches(value_indices, indices_value)) {
                return 4;
            }
            std::vector<float> step_scores(query_heads * tokens * cache_tokens), step_probabilities(step_scores.size());
            ggml_backend_tensor_get(scores, step_scores.data(), 0, step_scores.size() * sizeof(float));
            ggml_backend_tensor_get(probabilities, step_probabilities.data(), 0, step_probabilities.size() * sizeof(float));
            ggml_backend_tensor_get(output, all_outputs.data() + first * query_heads * width, 0, tokens * query_heads * width * sizeof(float));
            for (int head = 0; head < query_heads; ++head) {
                std::copy_n(step_scores.data() + head * tokens * cache_tokens, tokens * cache_tokens,
                            all_scores.data() + (head * total_tokens + first) * cache_tokens);
                std::copy_n(step_probabilities.data() + head * tokens * cache_tokens, tokens * cache_tokens,
                            all_probabilities.data() + (head * total_tokens + first) * cache_tokens);
            }
            return 0;
        };
        const int status = step();
        if (status) {
            return status;
        }
        if (key->data != key_address || value->data != value_address || key->buffer != cache_buffer.get() ||
                value->buffer != cache_buffer.get() || !matches(key, expected_key) || !matches(value, expected_value)) {
            return 4;
        }
        first += tokens;
    }
    if (first != total_tokens) {
        return 4;
    }
    for (const auto* data : { &all_scores, &all_probabilities, &all_outputs }) {
        if (!std::all_of(data->begin(), data->end(), [](float entry) { return std::isfinite(entry); })) {
            return 4;
        }
    }
    std::vector<ggml_fp16_t> actual_key(expected_key.size()), actual_value(expected_value.size());
    ggml_backend_tensor_get(key, actual_key.data(), 0, actual_key.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_get(value, actual_value.data(), 0, actual_value.size() * sizeof(ggml_fp16_t));
    std::printf("{\"tokens\":80,\"width\":256,\"query_heads\":4,\"cache_heads\":2,\"cache_tokens\":128,"
                "\"cache_type\":\"f16\",\"flash_attention\":false,\"steps\":%zu,\"reset\":%s,"
                "\"cache_addresses_stable\":true,\"cache_exact_after_graph_free\":true,\"sources_and_indices_unchanged\":true,"
                "\"initial_cache_uploads\":2,\"host_cache_restore_calls\":%zu",
                partition.size(), reset ? "true" : "false", reset ? (partition.size() - 1) * 2 : 0);
    for (const auto& entry : { std::make_pair("key_bits", &actual_key), std::make_pair("value_transposed_bits", &actual_value) }) {
        std::printf(",\"%s\":[", entry.first);
        for (std::size_t index = 0; index < entry.second->size(); ++index) {
            std::printf("%s%u", index ? "," : "", static_cast<unsigned>((*entry.second)[index]));
        }
        std::printf("]");
    }
    for (const auto& entry : { std::make_pair("scores", &all_scores), std::make_pair("probabilities", &all_probabilities),
                               std::make_pair("outputs", &all_outputs) }) {
        std::printf(",\"%s\":[", entry.first);
        for (std::size_t index = 0; index < entry.second->size(); ++index) {
            std::printf("%s%.9g", index ? "," : "", (*entry.second)[index]);
        }
        std::printf("]");
    }
    std::puts("}");
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

static int cached_gdn_control(const std::vector<int>& partition, bool reset) {
    constexpr int width = 128, key_heads = 2, value_heads = 4, total_tokens = 80;
    constexpr int state_count = width * width * value_heads;
    ggml_init_params params = { 1024 * 1024, nullptr, true };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> cache_context(ggml_init(params), ggml_free);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    if (!cache_context || !backend) {
        return 1;
    }
    auto* state = ggml_new_tensor_4d(cache_context.get(), GGML_TYPE_F32, width, width, value_heads, 1);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> cache_buffer(
        ggml_backend_alloc_ctx_tensors(cache_context.get(), backend.get()), ggml_backend_buffer_free);
    if (!cache_buffer) {
        return 2;
    }
    ggml_backend_cpu_set_n_threads(backend.get(), 1);
    const auto* state_address = state->data;
    std::vector<float> initial(state_count), expected(state_count), actual(state_count);
    for (int head = 0; head < value_heads; ++head) {
        for (int value_index = 0; value_index < width; ++value_index) {
            for (int key_index = 0; key_index < width; ++key_index) {
                initial[(head * width + value_index) * width + key_index] =
                    ((key_index * 3 + value_index * 5 + head * 7) % 19 - 9) / 4096.0f;
            }
        }
    }
    ggml_backend_tensor_set(state, initial.data(), 0, initial.size() * sizeof(float));
    std::vector<float> all_outputs(total_tokens * value_heads * width);
    const std::array<float, 5> raw_alpha = { -25.0f, -3.0f, 0.0f, 20.0f, 25.0f };
    int first = 0;
    for (const int tokens : partition) {
        if (reset && first) {
            ggml_backend_tensor_set(state, initial.data(), 0, initial.size() * sizeof(float));
        }
        const auto step = [&]() -> int {
            std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);
            if (!context) {
                return 1;
            }
            auto* query = ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, width, key_heads, tokens, 1);
            auto* key = ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, width, key_heads, tokens, 1);
            auto* value = ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, width, value_heads, tokens, 1);
            auto* alpha = ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, 1, value_heads, tokens, 1);
            auto* beta = ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, 1, value_heads, tokens, 1);
            auto* bias = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, value_heads);
            auto* multiplier = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, value_heads);
            for (auto* input : { query, key, value, alpha, beta, bias, multiplier }) {
                ggml_set_input(input);
                ggml_set_output(input);
            }
            auto* output = ggml_gated_delta_net(context.get(), ggml_l2_norm(context.get(), query, 1e-6f),
                ggml_l2_norm(context.get(), key, 1e-6f), value, alpha, beta, state, 1);
            ggml_gated_delta_net_set_raw_gates(output, bias, multiplier);
            auto* final_state = ggml_view_4d(context.get(), output, width, width, value_heads, 1,
                width * sizeof(float), width * width * sizeof(float), state_count * sizeof(float),
                tokens * value_heads * width * sizeof(float));
            auto* copied = ggml_cpy(context.get(), final_state, state);
            ggml_set_output(output);
            ggml_set_output(copied);
            auto* graph = ggml_new_graph(context.get());
            ggml_build_forward_expand(graph, copied);
            std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
                ggml_gallocr_new(ggml_backend_cpu_buffer_type()), ggml_gallocr_free);
            if (!allocator || !ggml_backend_supports_op(backend.get(), output) ||
                    !ggml_backend_supports_op(backend.get(), copied) || !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
                return 2;
            }
            std::vector<float> queries(tokens * key_heads * width), keys(queries.size());
            std::vector<float> values(tokens * value_heads * width), alphas(tokens * value_heads), betas(alphas.size());
            std::vector<float> biases(value_heads), multipliers(value_heads);
            for (int token = 0; token < tokens; ++token) {
                const int absolute = first + token;
                for (int head = 0; head < key_heads; ++head) {
                    for (int column = 0; column < width; ++column) {
                        const int index = (token * key_heads + head) * width + column;
                        queries[index] = ((column * 7 + absolute * 3 + head * 11) % 29 - 14) / 64.0f;
                        keys[index] = ((column * 3 + absolute * 7 + head * 5) % 31 - 15) / 64.0f;
                    }
                }
                for (int head = 0; head < value_heads; ++head) {
                    alphas[token * value_heads + head] = raw_alpha[(absolute + 2 * head) % 5];
                    betas[token * value_heads + head] = ((absolute + head) % 7 - 3) / 4.0f;
                    for (int column = 0; column < width; ++column) {
                        values[(token * value_heads + head) * width + column] =
                            ((column * 5 + absolute * 2 + head * 13) % 23 - 11) / 32.0f;
                    }
                }
            }
            for (int head = 0; head < value_heads; ++head) {
                biases[head] = (head - 2) / 8.0f;
                multipliers[head] = -(head + 1) / 4.0f;
            }
            const auto sources = { std::make_pair(query, &queries), std::make_pair(key, &keys),
                std::make_pair(value, &values), std::make_pair(alpha, &alphas), std::make_pair(beta, &betas),
                std::make_pair(bias, &biases), std::make_pair(multiplier, &multipliers) };
            for (const auto& entry : sources) {
                ggml_backend_tensor_set(entry.first, entry.second->data(), 0, entry.second->size() * sizeof(float));
            }
            if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
                return 3;
            }
            for (const auto& entry : sources) {
                std::vector<float> unchanged(entry.second->size());
                ggml_backend_tensor_get(entry.first, unchanged.data(), 0, unchanged.size() * sizeof(float));
                if (std::memcmp(unchanged.data(), entry.second->data(), unchanged.size() * sizeof(float)) != 0) {
                    return 4;
                }
            }
            const auto output_bytes = values.size() * sizeof(float);
            if (ggml_nbytes(output) != output_bytes + state_count * sizeof(float)) {
                return 4;
            }
            ggml_backend_tensor_get(output, all_outputs.data() + first * value_heads * width, 0, output_bytes);
            ggml_backend_tensor_get(output, expected.data(), output_bytes, expected.size() * sizeof(float));
            return 0;
        };
        const int status = step();
        if (status) {
            return status;
        }
        ggml_backend_tensor_get(state, actual.data(), 0, actual.size() * sizeof(float));
        if (state->data != state_address || state->buffer != cache_buffer.get() ||
                std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) != 0 ||
                !std::all_of(actual.begin(), actual.end(), [](float value) { return std::isfinite(value); })) {
            return 4;
        }
        first += tokens;
    }
    if (first != total_tokens || !std::all_of(all_outputs.begin(), all_outputs.end(),
            [](float value) { return std::isfinite(value); })) {
        return 4;
    }
    std::printf("{\"width\":%d,\"key_heads\":%d,\"value_heads\":%d,\"tokens\":%d,\"chunks\":%zu,"
                "\"raw_gates\":true,\"initialized\":true,\"reset\":%s,\"initial_state_uploads\":1,"
                "\"host_state_restores\":%zu,\"native_state_copies\":%zu,"
                "\"state_owner_and_bytes_stable_after_graph_teardown\":true,\"sources_unchanged\":true,\"outputs\":[",
                width, key_heads, value_heads, total_tokens, partition.size(), reset ? "true" : "false",
                reset ? partition.size() - 1 : 0, partition.size());
    for (std::size_t index = 0; index < all_outputs.size(); ++index) {
        std::printf("%s%.9g", index ? "," : "", all_outputs[index]);
    }
    std::printf("],\"state\":[");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        std::printf("%s%.9g", index ? "," : "", actual[index]);
    }
    std::puts("]}");
    return 0;
}

static int cached_conv_control(const std::vector<int>& partition, bool reset) {
    constexpr int channels = 8, total_tokens = 80;
    ggml_init_params params = { 1024 * 1024, nullptr, true };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> cache_context(ggml_init(params), ggml_free);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    if (!cache_context || !backend) {
        return 1;
    }
    auto* history = ggml_new_tensor_2d(cache_context.get(), GGML_TYPE_F32, 3, channels);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> cache_buffer(
        ggml_backend_alloc_ctx_tensors(cache_context.get(), backend.get()), ggml_backend_buffer_free);
    if (!cache_buffer) {
        return 2;
    }
    ggml_backend_cpu_set_n_threads(backend.get(), 1);
    const auto* history_address = history->data;
    std::vector<float> initial(channels * 3), expected(initial.size()), actual(initial.size());
    for (int channel = 0; channel < channels; ++channel) {
        for (int index = 0; index < 3; ++index) {
            initial[channel * 3 + index] = ((channel * 5 + index * 3) % 19 - 9) / 32.0f;
        }
    }
    expected = initial;
    ggml_backend_tensor_set(history, initial.data(), 0, initial.size() * sizeof(float));
    std::vector<float> raw_outputs(total_tokens * channels), silu_outputs(raw_outputs.size());
    int first = 0;
    for (const int tokens : partition) {
        if (reset && first) {
            ggml_backend_tensor_set(history, initial.data(), 0, initial.size() * sizeof(float));
            expected = initial;
        }
        const auto step = [&]() -> int {
            std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);
            if (!context) {
                return 1;
            }
            auto* input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, tokens, channels);
            auto* weight = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 4, channels);
            for (auto* source : { input, weight }) {
                ggml_set_input(source);
                ggml_set_output(source);
            }
            auto* combined = ggml_concat(context.get(), history, input, 0);
            auto* output = ggml_ssm_conv(context.get(), combined, weight);
            auto* activated = ggml_silu(context.get(), output);
            auto* tail = ggml_view_2d(context.get(), combined, 3, channels,
                (tokens + 3) * sizeof(float), tokens * sizeof(float));
            auto* copied = ggml_cpy(context.get(), tail, history);
            auto* graph = ggml_new_graph(context.get());
            for (auto* result : { output, activated, copied }) {
                ggml_set_output(result);
                ggml_build_forward_expand(graph, result);
            }
            std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
                ggml_gallocr_new(ggml_backend_cpu_buffer_type()), ggml_gallocr_free);
            if (!allocator || !ggml_backend_supports_op(backend.get(), combined) ||
                    !ggml_backend_supports_op(backend.get(), output) || !ggml_backend_supports_op(backend.get(), activated) ||
                    !ggml_backend_supports_op(backend.get(), copied) || !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
                return 2;
            }
            std::vector<float> features(tokens * channels), weights(4 * channels);
            for (int channel = 0; channel < channels; ++channel) {
                for (int token = 0; token < tokens; ++token) {
                    features[channel * tokens + token] = ((channel * 11 + (first + token) * 7) % 31 - 15) / 64.0f;
                }
                for (int tap = 0; tap < 4; ++tap) {
                    weights[channel * 4 + tap] = ((channel * 3 + tap * 5) % 11 - 5) / 16.0f;
                }
            }
            const auto sources = { std::make_pair(input, &features), std::make_pair(weight, &weights) };
            for (const auto& entry : sources) {
                ggml_backend_tensor_set(entry.first, entry.second->data(), 0, entry.second->size() * sizeof(float));
            }
            if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
                return 3;
            }
            for (const auto& entry : sources) {
                std::vector<float> unchanged(entry.second->size());
                ggml_backend_tensor_get(entry.first, unchanged.data(), 0, unchanged.size() * sizeof(float));
                if (std::memcmp(unchanged.data(), entry.second->data(), unchanged.size() * sizeof(float)) != 0) {
                    return 4;
                }
            }
            const auto output_bytes = tokens * channels * sizeof(float);
            if (ggml_nbytes(output) != output_bytes || ggml_nbytes(activated) != output_bytes) {
                return 4;
            }
            ggml_backend_tensor_get(output, raw_outputs.data() + first * channels, 0, output_bytes);
            ggml_backend_tensor_get(activated, silu_outputs.data() + first * channels, 0, output_bytes);
            for (int channel = 0; channel < channels; ++channel) {
                for (int index = 0; index < 3; ++index) {
                    const int source = tokens + index;
                    expected[channel * 3 + index] = source < 3 ? expected[channel * 3 + source] :
                        features[channel * tokens + source - 3];
                }
            }
            return 0;
        };
        const int status = step();
        if (status) {
            return status;
        }
        ggml_backend_tensor_get(history, actual.data(), 0, actual.size() * sizeof(float));
        if (history->data != history_address || history->buffer != cache_buffer.get() ||
                std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) != 0) {
            return 4;
        }
        first += tokens;
    }
    if (first != total_tokens || !std::all_of(raw_outputs.begin(), raw_outputs.end(),
            [](float value) { return std::isfinite(value); }) || !std::all_of(silu_outputs.begin(), silu_outputs.end(),
            [](float value) { return std::isfinite(value); })) {
        return 4;
    }
    std::printf("{\"channels\":%d,\"tokens\":%d,\"chunks\":%zu,\"reset\":%s,\"initial_history_uploads\":1,"
                "\"host_history_restores\":%zu,\"native_history_copies\":%zu,"
                "\"history_owner_and_bytes_stable_after_graph_teardown\":true,\"sources_unchanged\":true,\"raw\":[",
                channels, total_tokens, partition.size(), reset ? "true" : "false",
                reset ? partition.size() - 1 : 0, partition.size());
    for (std::size_t index = 0; index < raw_outputs.size(); ++index) {
        std::printf("%s%.9g", index ? "," : "", raw_outputs[index]);
    }
    std::printf("],\"silu\":[");
    for (std::size_t index = 0; index < silu_outputs.size(); ++index) {
        std::printf("%s%.9g", index ? "," : "", silu_outputs[index]);
    }
    std::printf("],\"history\":[");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        std::printf("%s%.9g", index ? "," : "", actual[index]);
    }
    std::puts("]}");
    return 0;
}

int main(int argc, char** argv) {
    if ((argc == 3 || argc == 4) && (std::strcmp(argv[1], "--cached-attention-control") == 0 ||
            std::strcmp(argv[1], "--cached-gdn-control") == 0 || std::strcmp(argv[1], "--cached-conv-control") == 0)) {
        std::vector<int> partition;
        if (std::strcmp(argv[2], "80") == 0) {
            partition = {80};
        } else if (std::strcmp(argv[2], "31+49") == 0) {
            partition = {31, 49};
        } else if (std::strcmp(argv[2], "40+40") == 0) {
            partition = {40, 40};
        } else if (std::strcmp(argv[2], "1+63+16") == 0) {
            partition = {1, 63, 16};
        } else if (std::strcmp(argv[2], "singletons") == 0) {
            partition.assign(80, 1);
        } else if (std::strcmp(argv[2], "64+singletons") == 0) {
            partition.assign(17, 1);
            partition[0] = 64;
        } else {
            return 5;
        }
        if (argc == 4 && std::strcmp(argv[3], "carry") != 0 && std::strcmp(argv[3], "reset") != 0) {
            return 5;
        }
        const bool reset = argc == 4 && std::strcmp(argv[3], "reset") == 0;
        if (std::strcmp(argv[1], "--cached-conv-control") == 0) {
            return cached_conv_control(partition, reset);
        }
        return std::strcmp(argv[1], "--cached-gdn-control") == 0 ?
            cached_gdn_control(partition, reset) : cached_attention_control(partition, reset);
    }
    if ((argc == 3 || argc == 4 || argc == 5) && std::strcmp(argv[1], "--kv-write-control") == 0) {
        const int tokens = std::strcmp(argv[2], "1") == 0 ? 1 : std::strcmp(argv[2], "7") == 0 ? 7 :
                           std::strcmp(argv[2], "80") == 0 ? 80 : 0;
        int first_token = 0;
        if (argc >= 4) {
            const char* end = argv[3] + std::strlen(argv[3]);
            const auto parsed = std::from_chars(argv[3], end, first_token);
            if (parsed.ec != std::errc() || parsed.ptr != end) {
                return 5;
            }
        }
        if (argc == 5 && std::strcmp(argv[4], "contiguous") != 0 && std::strcmp(argv[4], "permuted") != 0) {
            return 5;
        }
        return tokens && first_token >= 0 && first_token < 80 && first_token + tokens <= 80 ?
            kv_write_control(tokens, first_token, argc == 5 && std::strcmp(argv[4], "permuted") == 0) : 5;
    }
    if ((argc == 3 || argc == 4 || argc == 5) && std::strcmp(argv[1], "--rope-control") == 0) {
        const int tokens = std::strcmp(argv[2], "1") == 0 ? 1 : std::strcmp(argv[2], "7") == 0 ? 7 :
                           std::strcmp(argv[2], "80") == 0 ? 80 : 0;
        int first_token = 0;
        if (argc >= 4) {
            const char* end = argv[3] + std::strlen(argv[3]);
            const auto parsed = std::from_chars(argv[3], end, first_token);
            if (parsed.ec != std::errc() || parsed.ptr != end) {
                return 5;
            }
        }
        if (argc == 5 && std::strcmp(argv[4], "text") != 0 && std::strcmp(argv[4], "axes") != 0) {
            return 5;
        }
        return tokens && first_token >= 0 && first_token < 80 && first_token + tokens <= 80 ?
            rope_control(tokens, first_token, argc == 5 && std::strcmp(argv[4], "axes") == 0) : 5;
    }
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