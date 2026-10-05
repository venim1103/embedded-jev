#include "prism_bitnet_runtime.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "llama.h"
#ifdef JEV_TEST_REAL_LOADER
#include "llama-model-loader.h"
#endif
#ifdef JEV_TEST_FULL_RUNTIME
#include "llama-model.h"
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <future>
#include <limits>
#include <memory>
#include <regex>
#include <thread>
#include <vector>

extern "C" int bitnet_group_scale_prepare_a8(
    const float*, std::size_t, std::size_t, int8_t*, float*);
extern "C" int bitnet_group_scale_matmul_avx2(
    const uint8_t*, const float*, const int8_t*, const float*,
    std::size_t, std::size_t, std::size_t, float*);

extern "C" int prism_pq2_tensor_matmul(
    const uint8_t*, const float*, std::size_t, std::size_t, std::size_t, float*);

static bool run_concurrent_batch(
    ggml_tensor* weight, const uint8_t* packed, const float* scales, std::size_t tokens) {
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
static int test_prefill_control(const char* path, bool use_bitnet = false, const char* mode = "full") {
    const bool chunked = std::strcmp(mode, "chunked") == 0;
    const bool reset = std::strcmp(mode, "reset") == 0;
    const bool reordered = std::strcmp(mode, "reordered") == 0;
    if (!chunked && !reset && !reordered && std::strcmp(mode, "full") != 0) {
        return 32;
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
        std::error_code error;
        const auto file_size = std::filesystem::file_size(path, error);
        if (error || file_size > 1024 * 1024) {
            return 27;
        }
        params.tensor_buft_overrides = overrides;
    }
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(path, params), llama_model_free);
    if (!model) {
        return 27;
    }
    const ggml_tensor* weight = use_bitnet ? model->get_tensor("blk.3.ffn_down.weight") : nullptr;
    std::size_t dispatch_calls = 0;
    std::size_t weight_repacks = 0;
    if (use_bitnet && (!weight || model->hparams.n_layer() != 4 ||
            weight->ne[0] != 256 || weight->ne[1] != 32 ||
            prism_bitnet_cpu_tensor_status_v1(weight, &dispatch_calls, &weight_repacks) != 0 ||
            dispatch_calls != 0 || weight_repacks != 1)) {
        return 30;
    }
    const auto* vocab = llama_model_get_vocab(model.get());
    const int32_t vocab_size = llama_vocab_n_tokens(vocab);
    if (vocab_size < 24) {
        return 27;
    }
    auto context_params = llama_context_default_params();
    context_params.n_ctx = 128;
    context_params.n_batch = 128;
    context_params.n_ubatch = 128;
    context_params.n_seq_max = 1;
    context_params.n_threads = 1;
    context_params.n_threads_batch = 1;
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    std::unique_ptr<llama_context, decltype(&llama_free)> context(
        llama_init_from_model(model.get(), context_params), llama_free);
    if (!context) {
        return 28;
    }
    const auto memory = llama_get_memory(context.get());
    std::size_t prefill_calls = 0;
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
    std::vector<llama_token> tokens = { 3, 5, 7 };
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
    const auto position_min = llama_memory_seq_pos_min(memory, 0);
    const auto position_max = llama_memory_seq_pos_max(memory, 0);
    if (position_min != 2 || position_max != 2) {
        return 31;
    }
    const float* logits = llama_get_logits_ith(context.get(), -1);
    if (!logits || !std::all_of(logits, logits + vocab_size, [](float value) { return std::isfinite(value); })) {
        return 29;
    }
    if (use_bitnet && (prism_bitnet_cpu_tensor_status_v1(weight, &dispatch_calls, &weight_repacks) != 0 ||
            dispatch_calls != prefill_calls || weight_repacks != 1)) {
        return 30;
    }
    struct control_option {
        const char* id;
        llama_token token;
    };
    std::vector<control_option> options = { { "inspect", 11 }, { "edit", 17 }, { "ask", 23 } };
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
    std::printf("{\"prefilled_tokens\":%zu,\"generated_answer_tokens\":0,\"option_ids\":[",
        tokens.size());
    for (std::size_t index = 0; index < options.size(); ++index) {
        std::printf("%s\"%s\"", index ? "," : "", options[index].id);
    }
    std::fputs("],\"option_token_ids\":[", stdout);
    for (std::size_t index = 0; index < options.size(); ++index) {
        std::printf("%s%d", index ? "," : "", options[index].token);
    }
    std::fputs("],\"logits\":[", stdout);
    for (std::size_t index = 0; index < selected.size(); ++index) {
        std::printf("%s%.17g", index ? "," : "", selected[index]);
    }
    std::fputs("],\"conditional_scores\":[", stdout);
    for (std::size_t index = 0; index < scores.size(); ++index) {
        std::printf("%s%.17g", index ? "," : "", scores[index] / denominator);
    }
    std::printf("],\"backend\":\"%s\",\"mode\":\"%s\",\"prefill_calls\":%zu,"
            "\"bitnet_dispatch_calls\":%zu,\"weight_repacks\":%zu,\"memory_position_min\":%d,\"memory_position_max\":%d}\n",
        use_bitnet ? "bitnet" : "dense", mode, prefill_calls, dispatch_calls, weight_repacks,
        position_min, position_max);
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
    std::puts("{\"discovery_matches\":1,\"idempotent_init\":true,\"kernel_calls\":11,\"weight_repacks\":1,\"concurrent_graphs\":true}");
    return 0;
}