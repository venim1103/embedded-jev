#include "prism_bitnet_runtime.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "llama.h"
#ifdef JEV_TEST_REAL_LOADER
#include "llama-model-loader.h"
#endif
#ifdef JEV_TEST_FULL_RUNTIME
#include "llama-model.h"
#endif

#include <algorithm>
#include <atomic>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <future>
#include <limits>
#include <memory>
#include <regex>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

extern "C" int bitnet_group_scale_prepare_a8(
    const float*, std::size_t, std::size_t, int8_t*, float*);
extern "C" int bitnet_group_scale_matmul_avx2(
    const uint8_t*, const float*, const int8_t*, const float*,
    std::size_t, std::size_t, std::size_t, float*);

extern "C" int prism_pq2_tensor_matmul(
    const uint8_t*, const float*, std::size_t, std::size_t, std::size_t, float*);

static bool run_concurrent_batch(
    ggml_tensor* weight, const uint8_t* packed, const float* scales, std::size_t tokens, void* policy = nullptr) {
    const std::size_t groups = weight->ne[0] / 128;
    const std::size_t rows = weight->ne[1];
    ggml_init_params params = { 1024 * 1024, nullptr, true };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);
    auto* input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, groups * 128, tokens);
    auto* result = ggml_mul_mat(context.get(), weight, input);
    ggml_set_input(input);
    ggml_set_output(result);
    auto* graph = ggml_new_graph(context.get());
    ggml_build_forward_expand(graph, result);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    ggml_backend_cpu_set_n_threads(backend.get(), 1);
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
        ggml_gallocr_new(ggml_backend_cpu_buffer_type()), ggml_gallocr_free);
    if (!ggml_backend_supports_op(backend.get(), result) || !allocator ||
        !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
        return false;
    }
    std::vector<float> values(tokens * groups * 128);
    std::vector<int8_t> activations(values.size());
    std::vector<float> activation_scales(tokens * groups);
    std::vector<float> expected(tokens * rows);
    std::vector<float> output(expected.size());
    for (std::size_t evaluation = 0; evaluation < 4; ++evaluation) {
        for (std::size_t index = 0; index < values.size(); ++index) {
            values[index] = static_cast<float>(static_cast<int>((index + evaluation * 7) % 43) - 21) / 17.0f;
        }
        if (bitnet_group_scale_prepare_a8(values.data(), tokens, groups, activations.data(), activation_scales.data()) != 0 ||
            bitnet_group_scale_matmul_avx2(packed, scales, activations.data(), activation_scales.data(),
                tokens, rows, groups, expected.data()) != 0) {
            return false;
        }
        if (policy && evaluation == 0) {
            std::size_t calls = 0;
            std::size_t repacks = 0;
            if (prism_bitnet_cpu_tensor_status_v2(policy, weight, &calls, &repacks) != 0) {
                return false;
            }
            const auto previous_calls = calls;
            const float saved = values[0];
            values[0] = std::numeric_limits<float>::quiet_NaN();
            ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
            if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS ||
                prism_bitnet_cpu_tensor_status_v2(policy, weight, &calls, &repacks) != 2 ||
                calls != previous_calls || repacks != 1) {
                return false;
            }
            ggml_backend_tensor_get(result, output.data(), 0, output.size() * sizeof(float));
            if (!std::all_of(output.begin(), output.end(), [](float value) { return std::isnan(value); })) {
                return false;
            }
            values[0] = saved;
        }
        ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
            return false;
        }
        ggml_backend_tensor_get(result, output.data(), 0, output.size() * sizeof(float));
        if (output != expected) {
            return false;
        }
    }
    return true;
}

static int test_weight_execution(ggml_backend_dev_t device, ggml_backend_buffer_type_t buffer_type) {
    constexpr std::size_t rows = 5;
    constexpr std::size_t groups = 3;
    ggml_init_params params = { 1024 * 1024, nullptr, true };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);
    ggml_tensor* weight = ggml_new_tensor_2d(context.get(), GGML_TYPE_PQ2_0, groups * 128, rows);
    ggml_set_name(weight, "blk.3.ffn_down.weight");
    ggml_tensor* probe_input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, groups * 128, 512);
    ggml_tensor* probe = ggml_mul_mat(context.get(), weight, probe_input);
    weight->buffer = ggml_backend_buft_alloc_buffer(buffer_type, 0);
    if (ggml_backend_dev_supports_op(device, probe)) {
        std::fputs("implicit loader selection must remain disabled\n", stderr);
        return 5;
    }
    ggml_set_name(weight, "blk.2.ffn_down.weight");
    if (ggml_backend_dev_supports_op(device, probe)) {
        return 5;
    }
    ggml_set_name(weight, "blk.3.ffn_down.weight");
    ggml_backend_buffer_free(weight->buffer);
    weight->buffer = nullptr;
    const std::size_t bytes = rows * groups * 34;
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> storage(
        ggml_backend_buft_alloc_buffer(buffer_type, bytes), ggml_backend_buffer_free);
    if (!storage || ggml_backend_tensor_alloc(storage.get(), weight,
            ggml_backend_buffer_get_base(storage.get())) != GGML_STATUS_SUCCESS) {
        return 6;
    }
    ggml_backend_buffer_set_usage(storage.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<uint8_t> blocks(bytes, 0);
    std::vector<uint8_t> packed(rows * groups * 32, 0);
    std::vector<float> scales(rows * groups);
    for (std::size_t block = 0; block < rows * groups; ++block) {
        const ggml_fp16_t bits = ggml_fp32_to_fp16(static_cast<float>(block) * 0.03125f);
        blocks[block * 34] = bits & 255;
        blocks[block * 34 + 1] = bits >> 8;
        scales[block] = ggml_fp16_to_fp32(bits);
        for (std::size_t column = 0; column < 128; ++column) {
            const uint8_t code = (block + column) % 3;
            blocks[block * 34 + 2 + column / 4] |= code << (2 * (column % 4));
            packed[block * 32 + column % 32] |= code << (6 - 2 * (column / 32));
        }
    }
    std::size_t calls = 0;
    std::size_t repacks = 0;
    std::size_t last_input_tokens = 99;
    if (prism_bitnet_cpu_tensor_last_input_tokens_v1(nullptr, &last_input_tokens) != 1 || last_input_tokens != 0 ||
            prism_bitnet_cpu_tensor_last_input_tokens_v1(weight, nullptr) != 1 ||
            prism_bitnet_cpu_tensor_last_input_tokens_v1(probe_input, &last_input_tokens) != 1 || last_input_tokens != 0) {
        return 7;
    }
    if (prism_bitnet_cpu_tensor_status_v1(weight, &calls, &repacks) != 8 || calls != 0 || repacks != 0 ||
            prism_bitnet_cpu_tensor_last_input_tokens_v1(weight, &last_input_tokens) != 8 || last_input_tokens != 0) {
        return 7;
    }
    ggml_backend_tensor_set(weight, blocks.data(), 0, bytes / 2);
    if (prism_bitnet_cpu_tensor_status_v1(weight, &calls, &repacks) != 8 || repacks != 0 ||
            prism_bitnet_cpu_tensor_last_input_tokens_v1(weight, &last_input_tokens) != 8 || last_input_tokens != 0) {
        return 7;
    }
    ggml_backend_tensor_set(weight, blocks.data() + bytes / 2, bytes / 2, bytes - bytes / 2);
    if (prism_bitnet_cpu_tensor_status_v1(weight, &calls, &repacks) != 0 || calls != 0 || repacks != 1 ||
            prism_bitnet_cpu_tensor_last_input_tokens_v1(weight, &last_input_tokens) != 0 || last_input_tokens != 0) {
        return 7;
    }
    std::size_t resident_bytes = 0;
    std::size_t auxiliary_bytes = 99;
    if (prism_bitnet_cpu_tensor_storage_bytes_v1(weight, &resident_bytes, &auxiliary_bytes) != 0 ||
            resident_bytes != bytes || auxiliary_bytes != 0) {
        return 7;
    }
    std::vector<uint8_t> readback(bytes);
    ggml_backend_tensor_get(weight, readback.data(), 0, bytes);
    if (readback != blocks) {
        return 7;
    }
    for (std::size_t offset = 0; offset < bytes; ++offset) {
        uint8_t value = 0;
        ggml_backend_tensor_get(weight, &value, offset, 1);
        if (value != blocks[offset]) {
            return 7;
        }
    }
    const jev_bitnet_projection_spec_v2 specification {
        JEV_BITNET_PROJECTION_ABI_V2, "blk.0.ffn_down.weight", rows, groups * 128, blocks.data(), bytes
    };
    void* owned_handle = nullptr;
    if (prism_bitnet_registered_projection_create_v2(&specification, blocks.data(), bytes, 2, &owned_handle) != 0 || !owned_handle) {
        return 7;
    }
    std::vector<float> owned_inputs(2 * groups * 128, 0.375f);
    std::vector<int8_t> owned_activations(owned_inputs.size());
    std::vector<float> owned_scales(2 * groups);
    std::vector<float> owned_expected(2 * rows);
    std::vector<float> owned_output(2 * rows);
    const bool owned_ok = bitnet_group_scale_prepare_a8(owned_inputs.data(), 2, groups,
            owned_activations.data(), owned_scales.data()) == 0 &&
        bitnet_group_scale_matmul_avx2(packed.data(), scales.data(), owned_activations.data(), owned_scales.data(),
            2, rows, groups, owned_expected.data()) == 0 &&
        prism_bitnet_registered_projection_compute(owned_handle, owned_inputs.data(), owned_output.data(), &calls, &repacks) == 0 &&
        calls == 1 && repacks == 1 && owned_output == owned_expected;
    prism_bitnet_registered_projection_free(owned_handle);
    if (!owned_ok) {
        return 7;
    }
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    ggml_backend_cpu_set_n_threads(backend.get(), 1);
    if (ggml_backend_dev_supports_op(device, probe)) {
        return 8;
    }
    for (std::size_t tokens : { 1U, 2U, 128U }) {
        ggml_tensor* input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, groups * 128, tokens);
        ggml_tensor* result = ggml_mul_mat(context.get(), weight, input);
        if (!ggml_backend_supports_op(backend.get(), result)) {
            return 8;
        }
        ggml_set_input(input);
        ggml_set_output(result);
        ggml_cgraph* graph = ggml_new_graph(context.get());
        ggml_build_forward_expand(graph, result);
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
            ggml_gallocr_new(ggml_backend_cpu_buffer_type()), ggml_gallocr_free);
        if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
            return 8;
        }
        std::vector<float> values(tokens * groups * 128);
        for (std::size_t index = 0; index < values.size(); ++index) {
            values[index] = static_cast<float>(static_cast<int>(index % 73) - 36) / 19.0f;
        }
        std::vector<int8_t> activations(values.size());
        std::vector<float> activation_scales(tokens * groups);
        std::vector<float> expected(tokens * rows);
        std::vector<float> output(expected.size());
        if (bitnet_group_scale_prepare_a8(values.data(), tokens, groups, activations.data(), activation_scales.data()) != 0 ||
            bitnet_group_scale_matmul_avx2(packed.data(), scales.data(), activations.data(), activation_scales.data(),
                tokens, rows, groups, expected.data()) != 0) {
            return 9;
        }
        ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS ||
                prism_bitnet_cpu_tensor_status_v1(weight, &calls, &repacks) != 0 || repacks != 1 ||
                prism_bitnet_cpu_tensor_last_input_tokens_v1(weight, &last_input_tokens) != 0 || last_input_tokens != tokens) {
            return 10;
        }
        ggml_backend_tensor_get(result, output.data(), 0, output.size() * sizeof(float));
        if (output != expected) {
            return 10;
        }
        if (tokens == 2) {
            const auto previous_calls = calls;
            const float saved = values[0];
            values[0] = std::numeric_limits<float>::quiet_NaN();
            ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
            if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS ||
                    prism_bitnet_cpu_tensor_status_v1(weight, &calls, &repacks) != 2 || calls != previous_calls ||
                    prism_bitnet_cpu_tensor_last_input_tokens_v1(weight, &last_input_tokens) != 2 || last_input_tokens != tokens) {
                return 12;
            }
            ggml_backend_tensor_get(result, output.data(), 0, output.size() * sizeof(float));
            if (!std::all_of(output.begin(), output.end(), [](float value) { return std::isnan(value); })) {
                return 12;
            }
            values[0] = saved;
        }
    }
    std::vector<float> ordinary_inputs(2 * groups * 128, 1.0f);
    std::vector<float> ordinary_expected(2 * rows);
    if (prism_pq2_tensor_matmul(blocks.data(), ordinary_inputs.data(), 2, rows, groups, ordinary_expected.data()) != 0) {
        return 23;
    }
    std::atomic<bool> failed { false };
    std::promise<void> ready;
    const auto start = ready.get_future().share();
    std::vector<std::thread> workers;
    for (std::size_t tokens : { 1U, 2U }) {
        workers.emplace_back([&, tokens] {
            start.wait();
            if (!run_concurrent_batch(weight, packed.data(), scales.data(), tokens)) {
                failed = true;
            }
        });
    }
    workers.emplace_back([&] {
        start.wait();
        std::vector<float> output(ordinary_expected.size());
        for (std::size_t evaluation = 0; evaluation < 4; ++evaluation) {
            if (prism_pq2_tensor_matmul(blocks.data(), ordinary_inputs.data(), 2, rows, groups, output.data()) != 0 ||
                output != ordinary_expected) {
                failed = true;
            }
        }
    });
    ready.set_value();
    for (auto& worker : workers) {
        worker.join();
    }
    if (failed || prism_bitnet_cpu_tensor_status_v1(weight, &calls, &repacks) != 0 || calls != 11 || repacks != 1) {
        return 23;
    }
    ggml_backend_tensor_set(weight, blocks.data(), 0, bytes);
    if (prism_bitnet_cpu_tensor_status_v1(weight, &calls, &repacks) != 8 || calls != 11 || repacks != 1) {
        return 11;
    }
    for (int invalid : { 0, 1 }) {
        ggml_tensor* other = ggml_new_tensor_2d(context.get(), GGML_TYPE_PQ2_0, groups * 128, rows);
        ggml_set_name(other, "blk.3.ffn_down.weight");
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> other_storage(
            ggml_backend_buft_alloc_buffer(buffer_type, bytes), ggml_backend_buffer_free);
        if (!other_storage || ggml_backend_tensor_alloc(other_storage.get(), other,
                ggml_backend_buffer_get_base(other_storage.get())) != GGML_STATUS_SUCCESS) {
            return 13;
        }
        auto corrupt = blocks;
        if (invalid == 0) {
            corrupt[2] |= 3;
        } else {
            corrupt[0] = 0;
            corrupt[1] = 124;
        }
        ggml_backend_tensor_set(other, corrupt.data(), 0, bytes);
        if (prism_bitnet_cpu_tensor_status_v1(other, &calls, &repacks) != 5 || calls != 0 || repacks != 0) {
            return 13;
        }
    }
    return 0;
}

static int test_model_policy_execution_v2(bool late_refusal = false) {
    constexpr std::size_t rows = 8192;
    constexpr std::size_t groups = 2;
    constexpr std::size_t columns = groups * 128;
    const char* name = "blk.0.ffn_up.weight";
    std::vector<uint8_t> blocks(rows * groups * 34, 0);
    std::vector<uint8_t> packed(rows * groups * 32, 0);
    std::vector<float> scales(rows * groups);
    for (std::size_t block = 0; block < rows * groups; ++block) {
        const auto bits = ggml_fp32_to_fp16(static_cast<float>(block % 31 + 1) / 32.0f);
        blocks[block * 34] = bits & 255;
        blocks[block * 34 + 1] = bits >> 8;
        scales[block] = ggml_fp16_to_fp32(bits);
        for (std::size_t column = 0; column < 128; ++column) {
            const uint8_t code = (block + column) % 3;
            blocks[block * 34 + 2 + column / 4] |= code << (2 * (column % 4));
            packed[block * 32 + column % 32] |= code << (6 - 2 * (column / 32));
        }
    }
    ggml_init_params params = { 1024 * 1024, nullptr, true };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);
    if (!context) {
        return 36;
    }
    std::vector<uint8_t> dense(columns * 32 * 2, 0);
    std::vector<float> norm(columns, 1.0f);
    const jev_bitnet_model_tensor_v2 tensors[] = {
        { "token_embd.weight", GGML_TYPE_BF16, { columns, 32, 1, 1 } },
        { "output.weight", GGML_TYPE_BF16, { columns, 32, 1, 1 } },
        { "output_norm.weight", GGML_TYPE_F32, { columns, 1, 1, 1 } },
        { name, GGML_TYPE_PQ2_0, { columns, rows, 1, 1 } },
    };
    auto metadata = std::unique_ptr<gguf_context, decltype(&gguf_free)>(gguf_init_empty(), gguf_free);
    const char* digest = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const char* revision = "1111111111111111111111111111111111111111";
    for (const auto& entry : {
            std::make_pair("general.architecture", "qwen35"),
            std::make_pair("jev.model.profile_id", "synthetic-native-control"),
            std::make_pair("jev.model.profile_sha256", digest),
            std::make_pair("jev.model.source_revision", revision),
            std::make_pair("jev.model.converter_revision", JEV_PRISM_SOURCE_REVISION),
            std::make_pair("jev.bitnet.source_tensor_sha256", digest),
            std::make_pair("jev.bitnet.model", "profile-qwen35-single-projection-v2"),
            std::make_pair("jev.bitnet.execution", "group128-a8-fp32-nearest-even-identity-v1") }) {
        gguf_set_val_str(metadata.get(), entry.first, entry.second);
    }
    gguf_set_val_u32(metadata.get(), "qwen35.block_count", 1);
    gguf_set_val_u32(metadata.get(), "qwen35.embedding_length", columns);
    gguf_set_val_u32(metadata.get(), "qwen35.feed_forward_length", rows);
    const char* integer_names[] = {
        "qwen35.attention.head_count", "qwen35.attention.head_count_kv", "qwen35.attention.key_length",
        "qwen35.attention.value_length", "qwen35.rope.dimension_count", "qwen35.ssm.conv_kernel",
        "qwen35.ssm.inner_size", "qwen35.ssm.state_size", "qwen35.ssm.time_step_rank",
        "qwen35.ssm.group_count", "qwen35.full_attention_interval"
    };
    const uint32_t integer_values[] = { 4, 1, 64, 64, 16, 4, 256, 64, 4, 1, 1 };
    const float epsilon = 1e-6f;
    const float theta = 10000.0f;
    const int32_t sections[] = { 3, 3, 2, 0 };
    std::vector<jev_bitnet_model_metadata_v2> expected_metadata;
    for (std::size_t index = 0; index < 11; ++index) {
        gguf_set_val_u32(metadata.get(), integer_names[index], integer_values[index]);
        expected_metadata.push_back({ integer_names[index], 1, 1, &integer_values[index] });
    }
    gguf_set_val_f32(metadata.get(), "qwen35.attention.layer_norm_rms_epsilon", epsilon);
    gguf_set_val_f32(metadata.get(), "qwen35.rope.freq_base", theta);
    gguf_set_arr_data(metadata.get(), "qwen35.rope.dimension_sections", GGUF_TYPE_INT32, sections, 4);
    expected_metadata.push_back({ "qwen35.attention.layer_norm_rms_epsilon", 2, 1, &epsilon });
    expected_metadata.push_back({ "qwen35.rope.freq_base", 2, 1, &theta });
    expected_metadata.push_back({ "qwen35.rope.dimension_sections", 3, 4, sections });
    for (const auto& entry : tensors) {
        auto* tensor = ggml_new_tensor_2d(context.get(), static_cast<ggml_type>(entry.type), entry.ne[0], entry.ne[1]);
        ggml_set_name(tensor, entry.name);
        tensor->data = entry.type == GGML_TYPE_PQ2_0 ? static_cast<void*>(blocks.data()) :
            entry.type == GGML_TYPE_F32 ? static_cast<void*>(norm.data()) : static_cast<void*>(dense.data());
        gguf_add_tensor(metadata.get(), tensor);
    }
    struct temporary_file {
        char path[40] = "/tmp/jev-policy-v2-XXXXXX";
        int descriptor = mkstemp(path);
        ~temporary_file() {
            if (descriptor >= 0) {
                close(descriptor);
                std::remove(path);
            }
        }
    } file;
    if (file.descriptor < 0 || !gguf_write_to_file(metadata.get(), file.path, false)) {
        return 36;
    }
    const jev_bitnet_model_spec_v2 specification {
        2, "synthetic-native-control", digest, revision, digest, 1, columns, rows, 32, tensors, 4,
        expected_metadata.data(), expected_metadata.size(),
        { 2, name, rows, columns, blocks.data(), blocks.size() }
    };
    void* raw_policy = nullptr;
    const llama_model_tensor_buft_override* overrides = nullptr;
    const int status = prism_bitnet_cpu_model_override_from_gguf_v2(
        file.path, &specification, JEV_PRISM_SOURCE_REVISION, 2, &raw_policy, &overrides);
    if (late_refusal) {
        return status == 7 && !raw_policy && !overrides ? 0 : 36;
    }
    if (status != 0 || !raw_policy || !overrides) {
        return 36;
    }
    std::unique_ptr<void, decltype(&prism_bitnet_cpu_model_policy_free_v2)> policy(raw_policy, prism_bitnet_cpu_model_policy_free_v2);
    auto* weight = ggml_new_tensor_2d(context.get(), GGML_TYPE_PQ2_0, columns, rows);
    ggml_set_name(weight, name);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> storage(
        ggml_backend_buft_alloc_buffer(overrides[0].buft, blocks.size()), ggml_backend_buffer_free);
    if (!storage) {
        return 36;
    }
    for (const char* wrong_name : { "blk.3.ffn_down.weight", "blk.0.ffn_gate.weight" }) {
        auto* refused = ggml_new_tensor_2d(context.get(), GGML_TYPE_PQ2_0, columns, rows);
        ggml_set_name(refused, wrong_name);
        if (ggml_backend_tensor_alloc(storage.get(), refused, ggml_backend_buffer_get_base(storage.get())) != GGML_STATUS_FAILED) {
            return 36;
        }
    }
    ggml_set_name(weight, name);
    if (ggml_backend_tensor_alloc(storage.get(), weight, ggml_backend_buffer_get_base(storage.get())) != GGML_STATUS_SUCCESS) {
        return 36;
    }
    ggml_backend_buffer_set_usage(storage.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::size_t calls = 99;
    std::size_t repacks = 99;
    if (prism_bitnet_cpu_tensor_status_v1(weight, &calls, &repacks) != 1 || calls != 0 || repacks != 0 ||
        prism_bitnet_cpu_tensor_status_v2(policy.get(), weight, &calls, &repacks) != 8) {
        return 36;
    }
    ggml_backend_tensor_set(weight, blocks.data(), 0, 17);
    ggml_backend_tensor_set(weight, blocks.data() + 17, 17, blocks.size() - 17);
    std::vector<uint8_t> readback(blocks.size());
    ggml_backend_tensor_get(weight, readback.data(), 0, readback.size());
    if (readback != blocks || prism_bitnet_cpu_tensor_status_v2(policy.get(), weight, &calls, &repacks) != 0 || repacks != 1) {
        return 36;
    }
    for (std::size_t tokens : { 1U, 2U, 128U }) {
        if (!run_concurrent_batch(weight, packed.data(), scales.data(), tokens, policy.get())) {
            return 36;
        }
    }
    if (prism_bitnet_cpu_tensor_status_v2(policy.get(), weight, &calls, &repacks) != 0 || calls != 12 || repacks != 1) {
        return 36;
    }
    auto* other = ggml_new_tensor_2d(context.get(), GGML_TYPE_PQ2_0, columns, rows);
    ggml_set_name(other, name);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> other_storage(
        ggml_backend_buft_alloc_buffer(overrides[0].buft, blocks.size()), ggml_backend_buffer_free);
    if (!other_storage || ggml_backend_tensor_alloc(other_storage.get(), other,
            ggml_backend_buffer_get_base(other_storage.get())) != GGML_STATUS_SUCCESS) {
        return 36;
    }
    auto changed = blocks;
    changed[2] ^= 1;
    ggml_backend_tensor_set(other, changed.data(), 0, changed.size());
    if (run_concurrent_batch(other, packed.data(), scales.data(), 1) ||
        prism_bitnet_cpu_tensor_status_v2(policy.get(), other, &calls, &repacks) != 8 || calls != 0 || repacks != 0) {
        return 36;
    }
    return 0;
}

#ifdef JEV_TEST_REAL_LOADER
static int test_real_loader(const char* path, const char* untagged_path) {
    const llama_model_tensor_buft_override* overrides = nullptr;
    if (prism_bitnet_cpu_loader_override_from_gguf_v1(
            path, JEV_PRISM_SOURCE_REVISION, JEV_BITNET_RUNTIME_ABI_V1, &overrides) != 0) {
        return 15;
    }
    auto device = ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
    const buft_list_t candidates = { { device, overrides[0].buft }, { device, ggml_backend_cpu_buffer_type() } };
    std::vector<std::string> splits;
    const llama_hparams hparams = {};
    const LLM_TN_IMPL name(LLM_ARCH_QWEN35, LLM_TENSOR_FFN_DOWN, "weight", 3, -1);
    llama_model_loader loader(nullptr, nullptr, nullptr, path, splits, nullptr,
        LLAMA_LOAD_MODE_NONE, false, false, false, nullptr, overrides);
    const auto* metadata = loader.require_tensor_meta("blk.3.ffn_down.weight");
    const std::size_t rows = metadata->ne[1];
    const std::size_t groups = metadata->ne[0] / 128;
    auto* weight = loader.create_tensor(hparams, &candidates, &candidates, &candidates, &candidates,
        name, { static_cast<int64_t>(groups * 128), static_cast<int64_t>(rows) }, 0);
    if (!weight || loader.ctx_map.size() != 1 || !loader.ctx_map.count(overrides[0].buft)) {
        return 16;
    }
    auto* weight_context = loader.ctx_map.at(overrides[0].buft).get();
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> storage(
        ggml_backend_alloc_ctx_tensors_from_buft(weight_context, overrides[0].buft), ggml_backend_buffer_free);
    if (!storage) {
        return 17;
    }
    ggml_backend_buffer_set_usage(storage.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    loader.done_getting_tensors();
    loader.init_mappings(false);
    llama_buf_map buffers = { { 0, storage.get() } };
    if (!loader.load_all_data(weight_context, buffers, nullptr, nullptr, nullptr)) {
        return 17;
    }
    std::size_t calls = 0;
    std::size_t repacks = 0;
    if (prism_bitnet_cpu_tensor_status_v1(weight, &calls, &repacks) != 0 || calls != 0 || repacks != 1) {
        return 18;
    }
    std::vector<uint8_t> blocks(ggml_nbytes(weight));
    std::vector<uint8_t> packed(rows * groups * 32, 0);
    std::vector<float> scales(rows * groups);
    ggml_backend_tensor_get(weight, blocks.data(), 0, blocks.size());
    for (std::size_t block = 0; block < rows * groups; ++block) {
        scales[block] = ggml_fp16_to_fp32(blocks[block * 34] | (blocks[block * 34 + 1] << 8));
        for (std::size_t column = 0; column < 128; ++column) {
            const uint8_t code = (blocks[block * 34 + 2 + column / 4] >> (2 * (column % 4))) & 3;
            packed[block * 32 + column % 32] |= code << (6 - 2 * (column / 32));
        }
    }
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    ggml_backend_cpu_set_n_threads(backend.get(), 1);
    for (std::size_t tokens : { 1U, 2U }) {
        ggml_init_params params = { 1024 * 1024, nullptr, true };
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);
        auto* input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, groups * 128, tokens);
        auto* result = ggml_mul_mat(context.get(), weight, input);
        if (!ggml_backend_supports_op(backend.get(), result)) {
            return 19;
        }
        ggml_set_input(input);
        ggml_set_output(result);
        auto* graph = ggml_new_graph(context.get());
        ggml_build_forward_expand(graph, result);
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
            ggml_gallocr_new(ggml_backend_cpu_buffer_type()), ggml_gallocr_free);
        if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
            return 19;
        }
        std::vector<float> values(tokens * groups * 128);
        for (std::size_t index = 0; index < values.size(); ++index) {
            values[index] = static_cast<float>(static_cast<int>((index + tokens * 7) % 43) - 21) / 17.0f;
        }
        std::vector<int8_t> activations(values.size());
        std::vector<float> activation_scales(tokens * groups);
        std::vector<float> expected(tokens * rows);
        std::vector<float> output(expected.size());
        if (bitnet_group_scale_prepare_a8(values.data(), tokens, groups, activations.data(), activation_scales.data()) != 0 ||
            bitnet_group_scale_matmul_avx2(packed.data(), scales.data(), activations.data(), activation_scales.data(),
                tokens, rows, groups, expected.data()) != 0) {
            return 20;
        }
        ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS ||
            prism_bitnet_cpu_tensor_status_v1(weight, &calls, &repacks) != 0 || calls != tokens || repacks != 1) {
            return 20;
        }
        ggml_backend_tensor_get(result, output.data(), 0, output.size() * sizeof(float));
        if (output != expected) {
            return 20;
        }
    }
    llama_model_loader ordinary(nullptr, nullptr, nullptr, untagged_path, splits, nullptr,
        LLAMA_LOAD_MODE_NONE, false, false, false, nullptr, nullptr);
    const auto* ordinary_metadata = ordinary.require_tensor_meta("blk.3.ffn_down.weight");
    auto* ordinary_weight = ordinary.create_tensor(hparams, &candidates, &candidates, &candidates, &candidates,
        name, { ordinary_metadata->ne[0], ordinary_metadata->ne[1] }, 0);
    if (!ordinary_weight || ordinary.ctx_map.count(overrides[0].buft) ||
        !ordinary.ctx_map.count(ggml_backend_cpu_buffer_type())) {
        return 21;
    }
    std::puts("{\"real_loader_selected\":true,\"untagged_pq2_unselected\":true,\"kernel_calls\":2,\"weight_repacks\":1}");
    return 0;
}
#endif

static int test_loader_override(const char* path) {
    const llama_model_tensor_buft_override* overrides = nullptr;
    const int status = prism_bitnet_cpu_loader_override_from_gguf_v1(
        path, JEV_PRISM_SOURCE_REVISION, JEV_BITNET_RUNTIME_ABI_V1, &overrides);
    if (status != 0) {
        if (overrides) {
            return 14;
        }
        std::printf("{\"status\":%d,\"override_ready\":false}\n", status);
        return 0;
    }
    if (!overrides || !overrides[0].pattern || !overrides[0].buft ||
        overrides[1].pattern || overrides[1].buft) {
        return 14;
    }
    const std::regex pattern(overrides[0].pattern);
    if (!std::regex_search("blk.3.ffn_down.weight", pattern)) {
        return 14;
    }
    for (const char* name : { "blk.2.ffn_down.weight", "blk.3.ffn_downXweight",
                              "prefixblk.3.ffn_down.weight", "blk.3.ffn_down.weight.extra" }) {
        if (std::regex_search(name, pattern)) {
            return 14;
        }
    }
    llama_model_params params = {};
    params.tensor_buft_overrides = overrides;
    std::printf("{\"status\":0,\"override_ready\":%s}\n",
        params.tensor_buft_overrides == overrides ? "true" : "false");
    return 0;
}

#ifdef JEV_TEST_FULL_RUNTIME
struct prefill_trace {
    struct record {
        std::string name;
        std::size_t tokens;
        std::size_t width;
    };
    std::filesystem::path directory;
    std::vector<record> records;
    std::size_t bytes = 0;
    bool failed = false;

    bool save(const std::string& name, ggml_tensor* tensor) {
        if (tensor->type != GGML_TYPE_F32 || tensor->ne[0] < 1 || tensor->ne[0] > 12288 ||
            tensor->ne[1] < 1 || tensor->ne[1] > 128 || tensor->ne[2] != 1 || tensor->ne[3] != 1 ||
            tensor->nb[0] != sizeof(float) || records.size() >= 36 ||
            std::any_of(records.begin(), records.end(), [&](const record& entry) { return entry.name == name; })) {
            return false;
        }
        const auto width = static_cast<std::size_t>(tensor->ne[0]);
        const auto tokens = static_cast<std::size_t>(tensor->ne[1]);
        const auto size = tokens * width * sizeof(float);
        if (size > 96 * 1024 * 1024 - bytes) {
            return false;
        }
        std::vector<float> values(tokens * width);
        for (std::size_t token = 0; token < tokens; ++token) {
            ggml_backend_tensor_get(tensor, values.data() + token * width, token * tensor->nb[1], width * sizeof(float));
        }
        if (!std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); })) {
            return false;
        }
        std::unique_ptr<FILE, decltype(&std::fclose)> output(
            std::fopen((directory / (name + ".f32")).c_str(), "wbx"), std::fclose);
        if (!output || std::fwrite(values.data(), sizeof(float), values.size(), output.get()) != values.size() ||
            std::fflush(output.get()) != 0) {
            return false;
        }
        records.push_back({ name, tokens, width });
        bytes += size;
        return true;
    }

    static bool callback(ggml_tensor* tensor, bool ask, void* data) {
        auto& trace = *static_cast<prefill_trace*>(data);
        try {
            const std::string name = ggml_get_name(tensor);
            static const std::regex layer_name("l_out-([0-9]|[12][0-9]|3[01])");
            const bool projection = tensor->op == GGML_OP_MUL_MAT && tensor->src[0] &&
                std::strcmp(ggml_get_name(tensor->src[0]), "blk.3.ffn_down.weight") == 0;
            const bool selected = projection || name == "model.input_embed" ||
                name == "result_norm" || std::regex_match(name, layer_name);
            if (ask || !selected) {
                return selected || !ask;
            }
            const bool saved = projection ?
                trace.save("ffn_input", tensor->src[1]) && trace.save("ffn_output", tensor) :
                trace.save(name == "result_norm" ? "final_norm" : name, tensor);
            trace.failed = trace.failed || !saved;
            return saved;
        } catch (const std::exception&) {
            trace.failed = true;
            return false;
        }
    }

    bool finish(std::size_t layers) const {
        if (failed || records.size() != layers + 4) {
            return false;
        }
        std::unique_ptr<FILE, decltype(&std::fclose)> output(
            std::fopen((directory / "manifest.json").c_str(), "wx"), std::fclose);
        if (!output) {
            return false;
        }
        std::fputs("{\"format\":\"jev-prefill-f32-trace-v1\",\"dtype\":\"float32\",\"byte_order\":\"little\",\"tensors\":[", output.get());
        for (std::size_t index = 0; index < records.size(); ++index) {
            const auto& entry = records[index];
            std::fprintf(output.get(), "%s{\"name\":\"%s\",\"shape\":[%zu,%zu]}",
                index ? "," : "", entry.name.c_str(), entry.tokens, entry.width);
        }
        std::fputs("]}\n", output.get());
        return std::fflush(output.get()) == 0 && std::ferror(output.get()) == 0;
    }
};

static int test_prefill_control(
    const char* path, bool use_bitnet = false, const char* mode = "full", const char* prompt = nullptr,
    const char* projection_path = nullptr, const char* trace_path = nullptr) {
    const bool chunked = std::strcmp(mode, "chunked") == 0;
    const bool reset = std::strcmp(mode, "reset") == 0;
    const bool reordered = std::strcmp(mode, "reordered") == 0;
    const bool parallel = std::strcmp(mode, "parallel") == 0;
    const bool maximum_batch = std::strcmp(mode, "maximum") == 0;
    const bool kernel_error = use_bitnet && std::strcmp(mode, "kernel-error") == 0;
    const bool kernel_recovery = use_bitnet && std::strcmp(mode, "kernel-recovery") == 0;
    if (!chunked && !reset && !reordered && !parallel && !maximum_batch && !kernel_error && !kernel_recovery &&
            std::strcmp(mode, "full") != 0) {
        return 32;
    }
    if (prompt && (std::strlen(prompt) == 0 || std::strlen(prompt) > 4096 ||
            (std::strcmp(mode, "full") != 0 && !(projection_path && (kernel_error || kernel_recovery))))) {
        return 34;
    }
    prefill_trace trace;
    if (trace_path) {
        const uint32_t endian = 1;
        std::error_code error;
        trace.directory = trace_path;
        if (std::strcmp(mode, "full") != 0 || *reinterpret_cast<const uint8_t*>(&endian) != 1 ||
            !std::filesystem::create_directory(trace.directory, error) || error) {
            return 35;
        }
    }
    ggml_backend_buffer_type_t buffer_type = nullptr;
    if (prism_bitnet_cpu_runtime_init_v1(
            JEV_PRISM_SOURCE_REVISION, JEV_BITNET_RUNTIME_ABI_V1, &buffer_type) != 0) {
        return 27;
    }
    struct backend_guard {
        backend_guard() { llama_backend_init(); }
        ~backend_guard() { llama_backend_free(); }
    } backend;
    auto params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    params.check_tensors = true;
    const llama_model_tensor_buft_override overrides[] = {
        { "^blk\\.3\\.ffn_down\\.weight$", buffer_type }, { nullptr, nullptr },
    };
    if (use_bitnet) {
        if (projection_path) {
            if (prism_bitnet_cpu_model_override_from_gguf_v1(
                    path, projection_path, JEV_PRISM_SOURCE_REVISION, JEV_BITNET_RUNTIME_ABI_V1,
                    &params.tensor_buft_overrides) != 0) {
                return 27;
            }
        } else {
            std::error_code error;
            const auto file_size = std::filesystem::file_size(path, error);
            if (error || file_size > 1024 * 1024) {
                return 27;
            }
            params.tensor_buft_overrides = overrides;
        }
    }
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(path, params), llama_model_free);
    if (!model) {
        return 27;
    }
    const ggml_tensor* weight = use_bitnet ? model->get_tensor("blk.3.ffn_down.weight") : nullptr;
    std::size_t dispatch_calls = 0;
    std::size_t weight_repacks = 0;
    std::size_t last_input_tokens = 0;
        if (use_bitnet && (!weight || model->hparams.n_layer() != (projection_path ? 32 : 4) ||
            weight->ne[0] != (projection_path ? 12288 : 256) || weight->ne[1] != (projection_path ? 4096 : 32) ||
            prism_bitnet_cpu_tensor_status_v1(weight, &dispatch_calls, &weight_repacks) != 0 ||
            dispatch_calls != 0 || weight_repacks != 1 ||
            prism_bitnet_cpu_tensor_last_input_tokens_v1(weight, &last_input_tokens) != 0 || last_input_tokens != 0)) {
        return 30;
    }
    const auto* vocab = llama_model_get_vocab(model.get());
    const int32_t vocab_size = llama_vocab_n_tokens(vocab);
    if (vocab_size < 24) {
        return 27;
    }
    std::vector<llama_token> tokens = { 3, 5, 7 };
    if (maximum_batch) {
        tokens.assign(128, 2);
    }
    std::vector<llama_token> option_tokens = { 11, 17, 23 };
    if (prompt) {
        const int32_t count = -llama_tokenize(vocab, prompt, std::strlen(prompt), nullptr, 0, false, true);
        if (count <= 0 || count > 128) {
            return 34;
        }
        tokens.resize(count);
        if (llama_tokenize(vocab, prompt, std::strlen(prompt), tokens.data(), count, false, true) != count) {
            return 34;
        }
        std::size_t index = 0;
        for (const char* label : { "A", "B", "C" }) {
            if (llama_tokenize(vocab, label, 1, &option_tokens[index], 1, false, true) != 1 ||
                    option_tokens[index] < 0 || option_tokens[index] >= vocab_size) {
                return 34;
            }
            const std::string extended = std::string(prompt) + label;
            std::vector<llama_token> extended_tokens(count + 1);
            if (llama_tokenize(vocab, extended.data(), extended.size(), extended_tokens.data(),
                    extended_tokens.size(), false, true) != count + 1 ||
                    !std::equal(tokens.begin(), tokens.end(), extended_tokens.begin()) ||
                    extended_tokens.back() != option_tokens[index]) {
                return 34;
            }
            ++index;
        }
        if (option_tokens[0] == option_tokens[1] || option_tokens[0] == option_tokens[2] ||
                option_tokens[1] == option_tokens[2]) {
            return 34;
        }
    }
    auto context_params = llama_context_default_params();
    context_params.n_ctx = 128;
    context_params.n_batch = 128;
    context_params.n_ubatch = 128;
    context_params.n_seq_max = 1;
    context_params.n_threads = 1;
    context_params.n_threads_batch = 1;
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    context_params.type_k = GGML_TYPE_F16;
    context_params.type_v = GGML_TYPE_F16;
    if (trace_path) {
        context_params.cb_eval = prefill_trace::callback;
        context_params.cb_eval_user_data = &trace;
    }
    std::unique_ptr<llama_context, decltype(&llama_free)> context(
        llama_init_from_model(model.get(), context_params), llama_free);
    if (!context) {
        return 28;
    }
    const auto memory = llama_get_memory(context.get());
    std::size_t prefill_calls = 0;
    std::size_t rejected_prefill_calls = 0;
    int kernel_error_status = 0;
    if (kernel_error || kernel_recovery) {
        struct rounding_guard {
            int saved = std::fegetround();
            ~rounding_guard() { std::fesetround(saved); }
        } rounding;
        if (rounding.saved != FE_TONEAREST || std::fesetround(FE_UPWARD) != 0) {
            return 33;
        }
        std::vector<llama_token> rejected = tokens;
        const int decode_status = llama_decode(context.get(), llama_batch_get_one(rejected.data(), rejected.size()));
        if (std::fesetround(rounding.saved) != 0) {
            return 33;
        }
        kernel_error_status = prism_bitnet_cpu_tensor_status_v1(weight, &dispatch_calls, &weight_repacks);
        const float* failed_logits = llama_get_logits_ith(context.get(), -1);
        if (decode_status != 0 || kernel_error_status <= 0 || dispatch_calls != 0 || weight_repacks != 1 ||
            prism_bitnet_cpu_tensor_last_input_tokens_v1(weight, &last_input_tokens) != kernel_error_status ||
            last_input_tokens != rejected.size() ||
                !failed_logits || std::all_of(failed_logits, failed_logits + vocab_size,
                    [](float value) { return std::isfinite(value); })) {
            return 33;
        }
        ++rejected_prefill_calls;
        if (kernel_error) {
            return 29;
        }
        llama_memory_clear(memory, true);
        if (llama_memory_seq_pos_min(memory, 0) != -1 || llama_memory_seq_pos_max(memory, 0) != -1) {
            return 31;
        }
    }
    std::unique_ptr<llama_context, decltype(&llama_free)> peer(nullptr, llama_free);
    if (parallel) {
        peer.reset(llama_init_from_model(model.get(), context_params));
        if (!peer) {
            return 28;
        }
        std::vector<llama_token> prefix = { 4, 9, 19 };
        if (llama_decode(peer.get(), llama_batch_get_one(prefix.data(), prefix.size())) != 0) {
            return 29;
        }
        ++prefill_calls;
    }
    if (reset) {
        std::vector<llama_token> warmup = { 4, 9, 19 };
        if (llama_decode(context.get(), llama_batch_get_one(warmup.data(), warmup.size())) != 0) {
            return 29;
        }
        ++prefill_calls;
        llama_memory_clear(memory, true);
        if (llama_memory_seq_pos_min(memory, 0) != -1 || llama_memory_seq_pos_max(memory, 0) != -1) {
            return 31;
        }
    }
    std::vector<llama_token> peer_tokens = tokens;
    std::future<int> peer_result;
    if (parallel) {
        peer_result = std::async(std::launch::async, [&] {
            return llama_decode(peer.get(), llama_batch_get_one(peer_tokens.data(), peer_tokens.size()));
        });
    }
    if (chunked) {
        if (llama_decode(context.get(), llama_batch_get_one(tokens.data(), 1)) != 0) {
            return 29;
        }
        ++prefill_calls;
    }
    const std::size_t offset = chunked ? 1 : 0;
    if (llama_decode(context.get(), llama_batch_get_one(tokens.data() + offset, tokens.size() - offset)) != 0) {
        return 29;
    }
    ++prefill_calls;
    if (parallel) {
        if (peer_result.get() != 0) {
            return 29;
        }
        ++prefill_calls;
        const auto peer_memory = llama_get_memory(peer.get());
        if (llama_memory_seq_pos_min(peer_memory, 0) != 5 || llama_memory_seq_pos_max(peer_memory, 0) != 5) {
            return 31;
        }
    }
    const auto position_min = llama_memory_seq_pos_min(memory, 0);
    const auto position_max = llama_memory_seq_pos_max(memory, 0);
    const auto expected_position = static_cast<llama_pos>(tokens.size() - 1);
    if (position_min != expected_position || position_max != expected_position) {
        return 31;
    }
    const float* logits = llama_get_logits_ith(context.get(), -1);
    if (!logits || !std::all_of(logits, logits + vocab_size, [](float value) { return std::isfinite(value); })) {
        return 29;
    }
    const float* peer_logits = parallel ? llama_get_logits_ith(peer.get(), -1) : nullptr;
    if (parallel && (!peer_logits ||
            !std::all_of(peer_logits, peer_logits + vocab_size, [](float value) { return std::isfinite(value); }))) {
        return 29;
    }
    if (use_bitnet && (prism_bitnet_cpu_tensor_status_v1(weight, &dispatch_calls, &weight_repacks) != 0 ||
            dispatch_calls != prefill_calls || weight_repacks != 1 ||
            prism_bitnet_cpu_tensor_last_input_tokens_v1(weight, &last_input_tokens) != 0)) {
        return 30;
    }
    if (trace_path && !trace.finish(model->hparams.n_layer())) {
        return 35;
    }
    struct control_option {
        const char* id;
        const char* label;
        llama_token token;
    };
    std::vector<control_option> options = {
        { "inspect", "A", option_tokens[0] }, { "edit", "B", option_tokens[1] }, { "ask", "C", option_tokens[2] },
    };
    if (reordered) {
        std::rotate(options.begin(), options.begin() + 2, options.end());
    }
    std::vector<double> selected;
    for (const auto& option : options) {
        selected.push_back(logits[option.token]);
    }
    const double maximum = *std::max_element(selected.begin(), selected.end());
    std::vector<double> scores;
    double denominator = 0;
    for (const auto value : selected) {
        scores.push_back(std::exp(value - maximum));
        denominator += scores.back();
    }
    std::printf("{\"prefilled_tokens\":%zu,\"generated_answer_tokens\":0,\"input_tokens\":[",
        tokens.size());
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        std::printf("%s%d", index ? "," : "", tokens[index]);
    }
    std::fputs("],\"option_ids\":[", stdout);
    for (std::size_t index = 0; index < options.size(); ++index) {
        std::printf("%s\"%s\"", index ? "," : "", options[index].id);
    }
    std::fputs("],\"option_token_ids\":[", stdout);
    for (std::size_t index = 0; index < options.size(); ++index) {
        std::printf("%s%d", index ? "," : "", options[index].token);
    }
    std::fputs("],\"option_labels\":[", stdout);
    for (std::size_t index = 0; index < options.size(); ++index) {
        std::printf("%s\"%s\"", index ? "," : "", options[index].label);
    }
    std::fputs("],\"logits\":[", stdout);
    for (std::size_t index = 0; index < selected.size(); ++index) {
        std::printf("%s%.17g", index ? "," : "", selected[index]);
    }
    std::fputs("],\"conditional_scores\":[", stdout);
    for (std::size_t index = 0; index < scores.size(); ++index) {
        std::printf("%s%.17g", index ? "," : "", scores[index] / denominator);
    }
    std::fputs("],\"peer_logits\":[", stdout);
    if (parallel) {
        for (std::size_t index = 0; index < options.size(); ++index) {
            std::printf("%s%.17g", index ? "," : "", static_cast<double>(peer_logits[options[index].token]));
        }
    }
    std::printf("],\"backend\":\"%s\",\"mode\":\"%s\",\"prefill_calls\":%zu,"
            "\"bitnet_dispatch_calls\":%zu,\"weight_repacks\":%zu,\"memory_position_min\":%d,\"memory_position_max\":%d,"
            "\"rejected_prefill_calls\":%zu,\"kernel_error_status\":%d,\"bitnet_last_input_tokens\":%zu}\n",
        use_bitnet ? "bitnet" : "dense", mode, prefill_calls, dispatch_calls, weight_repacks,
        position_min, position_max, rejected_prefill_calls, kernel_error_status, last_input_tokens);
    return 0;
}

static int test_vocab_only(const char* path, const char* prompt) {
    ggml_backend_buffer_type_t buffer_type = nullptr;
    if (std::strlen(prompt) > 4096 || prism_bitnet_cpu_runtime_init_v1(
            JEV_PRISM_SOURCE_REVISION, JEV_BITNET_RUNTIME_ABI_V1, &buffer_type) != 0) {
        return 24;
    }
    llama_backend_init();
    auto params = llama_model_default_params();
    params.vocab_only = true;
    params.n_gpu_layers = 0;
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(path, params), llama_model_free);
    if (!model) {
        llama_backend_free();
        return 24;
    }
    const auto* vocab = llama_model_get_vocab(model.get());
    const int32_t count = -llama_tokenize(vocab, prompt, std::strlen(prompt), nullptr, 0, false, true);
    if (count <= 0 || count > 4096) {
        return 25;
    }
    std::vector<llama_token> tokens(count);
    if (llama_tokenize(vocab, prompt, std::strlen(prompt), tokens.data(), count, false, true) != count) {
        return 25;
    }
    std::vector<llama_token> labels;
    for (const char* label : { "A", "B", "C" }) {
        llama_token token = -1;
        if (llama_tokenize(vocab, label, 1, &token, 1, false, true) != 1) {
            return 26;
        }
        labels.push_back(token);
    }
    std::fputs("{\"vocab_only\":true,\"generated_answer_tokens\":0,\"tokens\":[", stdout);
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        std::printf("%s%d", index ? "," : "", tokens[index]);
    }
    std::fputs("],\"labels\":[", stdout);
    for (std::size_t index = 0; index < labels.size(); ++index) {
        std::printf("%s%d", index ? "," : "", labels[index]);
    }
    std::puts("]}");
    model.reset();
    llama_backend_free();
    return 0;
}
#endif

int main(int argc, char** argv) {
#ifdef JEV_TEST_FULL_RUNTIME
    if (argc == 6 && std::strcmp(argv[1], "--prefill-model-bitnet-trace-control") == 0) {
        return test_prefill_control(argv[2], true, "full", argv[4], argv[3], argv[5]);
    }
    if (argc == 4 && (std::strcmp(argv[1], "--prefill-bitnet-trace-control") == 0 ||
            std::strcmp(argv[1], "--prefill-trace-control") == 0)) {
        return test_prefill_control(argv[2], std::strcmp(argv[1], "--prefill-bitnet-trace-control") == 0,
            "full", nullptr, nullptr, argv[3]);
    }
    if ((argc == 5 || argc == 6) && std::strcmp(argv[1], "--prefill-model-bitnet-control") == 0) {
        return test_prefill_control(argv[2], true, argc == 6 ? argv[5] : "full", argv[4], argv[3]);
    }
    if (argc == 4 && std::strcmp(argv[1], "--prefill-text-control") == 0) {
        return test_prefill_control(argv[2], false, "full", argv[3]);
    }
    if ((argc == 3 || argc == 4) && std::strcmp(argv[1], "--prefill-bitnet-control") == 0) {
        return test_prefill_control(argv[2], true, argc == 4 ? argv[3] : "full");
    }
    if ((argc == 3 || argc == 4) && std::strcmp(argv[1], "--prefill-control") == 0) {
        return test_prefill_control(argv[2], false, argc == 4 ? argv[3] : "full");
    }
    if (argc == 4 && std::strcmp(argv[1], "--vocab-only") == 0) {
        return test_vocab_only(argv[2], argv[3]);
    }
#endif
#ifdef JEV_TEST_REAL_LOADER
    if (argc == 4 && std::strcmp(argv[1], "--real-loader") == 0) {
        try {
            const int status = test_real_loader(argv[2], argv[3]);
            if (status != 0) {
                std::fprintf(stderr, "real loader control stage %d\n", status);
            }
            return status;
        } catch (const std::exception& error) {
            std::fprintf(stderr, "real loader exception: %s\n", error.what());
            return 22;
        }
    }
#endif
    if (argc == 3 && std::strcmp(argv[1], "--override") == 0) {
        return test_loader_override(argv[2]);
    }
    const bool late = argc == 2 && std::strcmp(argv[1], "--late") == 0;
    if (argc != 1 && !late) {
        std::fputs("unrecognized native control arguments\n", stderr);
        return 2;
    }
    ggml_backend_reg_t registry = ggml_backend_cpu_reg();
    ggml_backend_dev_t device = ggml_backend_reg_dev_get(registry, 0);
    auto discovery = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
        ggml_backend_reg_get_proc_address(registry, "ggml_backend_dev_get_extra_bufts"));
    if (!discovery) {
        return 1;
    }
    if (late) {
        discovery(device);
    }
    ggml_backend_buffer_type_t buffer_type = nullptr;
    if (prism_bitnet_cpu_runtime_init_v1("wrong", JEV_BITNET_RUNTIME_ABI_V1, &buffer_type) != 1 || buffer_type) {
        return 2;
    }
    if (prism_bitnet_cpu_runtime_init_v1(JEV_PRISM_SOURCE_REVISION, 2, &buffer_type) != 1 || buffer_type) {
        return 2;
    }
    const int status = prism_bitnet_cpu_runtime_init_v1(
        JEV_PRISM_SOURCE_REVISION, JEV_BITNET_RUNTIME_ABI_V1, &buffer_type);
    if (late) {
        if (status != 7 || buffer_type || test_model_policy_execution_v2(true) != 0) {
            return 3;
        }
        std::puts("{\"late_init_refused\":true}");
        return 0;
    }
    if (status != 0 || !buffer_type) {
        return 3;
    }
    int matches = 0;
    for (auto* candidate = discovery(device); candidate && *candidate; ++candidate) {
        if (*candidate == buffer_type) {
            ++matches;
        }
    }
    ggml_backend_buffer_type_t repeated = nullptr;
    if (matches != 1 || prism_bitnet_cpu_runtime_init_v1(
            JEV_PRISM_SOURCE_REVISION, JEV_BITNET_RUNTIME_ABI_V1, &repeated) != 0 || repeated != buffer_type ||
        std::strcmp(ggml_backend_buft_name(buffer_type), "JEV_BITNET_LOADER_V1") != 0 ||
        !ggml_backend_dev_supports_buft(device, buffer_type)) {
        return 4;
    }
    const int execution_status = test_weight_execution(device, buffer_type);
    if (execution_status != 0) {
        std::fprintf(stderr, "loader control stage %d\n", execution_status);
        return execution_status;
    }
    const int policy_status = test_model_policy_execution_v2();
    if (policy_status != 0) {
        std::fprintf(stderr, "v2 policy control stage %d\n", policy_status);
        return policy_status;
    }
    std::puts("{\"discovery_matches\":1,\"idempotent_init\":true,\"kernel_calls\":11,\"weight_repacks\":1,\"concurrent_graphs\":true}");
    return 0;
}