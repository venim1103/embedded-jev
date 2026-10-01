#include "prism_bitnet_runtime.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

extern "C" int bitnet_group_scale_prepare_a8(
    const float*, std::size_t, std::size_t, int8_t*, float*);
extern "C" int bitnet_group_scale_matmul_avx2(
    const uint8_t*, const float*, const int8_t*, const float*,
    std::size_t, std::size_t, std::size_t, float*);

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
    if (prism_bitnet_cpu_tensor_status_v1(weight, &calls, &repacks) != 8 || calls != 0 || repacks != 0) {
        return 7;
    }
    ggml_backend_tensor_set(weight, blocks.data(), 0, bytes / 2);
    if (prism_bitnet_cpu_tensor_status_v1(weight, &calls, &repacks) != 8 || repacks != 0) {
        return 7;
    }
    ggml_backend_tensor_set(weight, blocks.data() + bytes / 2, bytes / 2, bytes - bytes / 2);
    if (prism_bitnet_cpu_tensor_status_v1(weight, &calls, &repacks) != 0 || calls != 0 || repacks != 1) {
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
            prism_bitnet_cpu_tensor_status_v1(weight, &calls, &repacks) != 0 || repacks != 1) {
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
                prism_bitnet_cpu_tensor_status_v1(weight, &calls, &repacks) != 2 || calls != previous_calls) {
                return 12;
            }
            ggml_backend_tensor_get(result, output.data(), 0, output.size() * sizeof(float));
            if (!std::all_of(output.begin(), output.end(), [](float value) { return std::isnan(value); })) {
                return 12;
            }
            values[0] = saved;
        }
    }
    ggml_backend_tensor_set(weight, blocks.data(), 0, bytes);
    if (prism_bitnet_cpu_tensor_status_v1(weight, &calls, &repacks) != 8 || calls != 3 || repacks != 1) {
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

int main(int argc, char** argv) {
    const bool late = argc == 2 && std::strcmp(argv[1], "--late") == 0;
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
        if (status != 7 || buffer_type) {
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
    std::puts("{\"discovery_matches\":1,\"idempotent_init\":true,\"kernel_calls\":3,\"weight_repacks\":1}");
    return 0;
}