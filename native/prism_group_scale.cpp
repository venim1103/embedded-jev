#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "traits.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <dlfcn.h>
#include <memory>
#include <mutex>
#include <vector>

extern "C" int bitnet_group_scale_matmul_avx2(
    const uint8_t* packed, const float* weight_scales, const int8_t* activations,
    const float* activation_scales, std::size_t tokens, std::size_t rows,
    std::size_t groups, float* output);
extern "C" int bitnet_group_scale_prepare_a8(
    const float* inputs, std::size_t tokens, std::size_t groups,
    int8_t* activations, float* activation_scales);

namespace {

struct graph_state {
    const uint8_t* packed;
    const float* weight_scales;
    const float* activation_scales;
    std::size_t tokens;
    std::size_t rows;
    std::size_t groups;
    int calls = 0;
    int status = 1;
    bool prepare_inputs = false;
    std::vector<int8_t> prepared_activations;
    std::vector<float> prepared_scales;
};

void grouped_dot(
    ggml_tensor* destination, const ggml_tensor* output_shape,
    const ggml_tensor* input, int thread, int thread_count, void* userdata) {
    if (thread != 0) {
        return;
    }
    auto* state = static_cast<graph_state*>(userdata);
    ++state->calls;
    if (thread_count != 1 || destination->type != GGML_TYPE_F32 ||
        output_shape->type != GGML_TYPE_F32 ||
        input->type != (state->prepare_inputs ? GGML_TYPE_F32 : GGML_TYPE_I8) ||
        destination->ne[0] != static_cast<int64_t>(state->rows) ||
        destination->ne[1] != static_cast<int64_t>(state->tokens) ||
        input->ne[0] != static_cast<int64_t>(state->groups * 128) ||
        input->ne[1] != static_cast<int64_t>(state->tokens) ||
        !ggml_is_contiguous(destination) || !ggml_is_contiguous(input) ||
        !destination->data || !input->data) {
        state->status = 4;
        return;
    }
    const int8_t* activations = static_cast<const int8_t*>(input->data);
    const float* activation_scales = state->activation_scales;
    if (state->prepare_inputs) {
        state->status = bitnet_group_scale_prepare_a8(
            static_cast<const float*>(input->data), state->tokens, state->groups,
            state->prepared_activations.data(), state->prepared_scales.data());
        if (state->status != 0) {
            return;
        }
        activations = state->prepared_activations.data();
        activation_scales = state->prepared_scales.data();
    }
    state->status = bitnet_group_scale_matmul_avx2(
        state->packed, state->weight_scales, activations,
        activation_scales, state->tokens, state->rows, state->groups,
        static_cast<float*>(destination->data));
}

struct bitnet_tensor_traits final : ggml::cpu::tensor_traits {
    std::size_t tokens;
    std::size_t rows;
    std::size_t groups;
    std::size_t calls = 0;
    std::size_t repacks = 0;
    int status = 4;
    std::vector<uint8_t> packed;
    std::vector<float> weight_scales;
    std::vector<int8_t> activations;
    std::vector<float> activation_scales;

    bitnet_tensor_traits(std::size_t token_count, std::size_t row_count, std::size_t group_count)
        : tokens(token_count), rows(row_count), groups(group_count),
          packed(rows * groups * 32), weight_scales(rows * groups),
          activations(tokens * groups * 128), activation_scales(tokens * groups) {}

    bool work_size(int n_threads, const ggml_tensor*, std::size_t& size) override {
        size = 0;
        return n_threads == 1;
    }

    int prepare_weight(const uint8_t* blocks) {
        std::fill(packed.begin(), packed.end(), 0);
        for (std::size_t block = 0; block < rows * groups; ++block) {
            const uint8_t* source = blocks + block * 34;
            const ggml_fp16_t scale_bits = static_cast<ggml_fp16_t>(source[0] | (source[1] << 8));
            const float scale = ggml_fp16_to_fp32(scale_bits);
            if (!std::isfinite(scale) || scale < 0.0f) {
                return 5;
            }
            weight_scales[block] = scale;
            for (std::size_t column = 0; column < 128; ++column) {
                const uint8_t code = (source[2 + column / 4] >> (2 * (column % 4))) & 3;
                if (code == 3) {
                    return 5;
                }
                packed[block * 32 + column % 32] |= code << (6 - 2 * (column / 32));
            }
        }
        ++repacks;
        return 0;
    }

    bool compute_forward(ggml_compute_params* params, ggml_tensor* op) override {
        if (params->ith != 0) {
            return true;
        }
        if (params->nth != 1 || !op->src[0]->data || !op->src[1]->data || !op->data) {
            status = 4;
            return true;
        }
        if (repacks == 0) {
            status = prepare_weight(static_cast<const uint8_t*>(op->src[0]->data));
            if (status != 0) {
                return true;
            }
        }
        status = bitnet_group_scale_prepare_a8(
            static_cast<const float*>(op->src[1]->data), tokens, groups,
            activations.data(), activation_scales.data());
        if (status == 0) {
            ++calls;
            status = bitnet_group_scale_matmul_avx2(
                packed.data(), weight_scales.data(), activations.data(),
                activation_scales.data(), tokens, rows, groups, static_cast<float*>(op->data));
        }
        return true;
    }
};

struct bitnet_buffer_type final : ggml::cpu::extra_buffer_type {
    bitnet_tensor_traits& traits;
    ggml_backend_buffer_type type;

    explicit bitnet_buffer_type(bitnet_tensor_traits& tensor_traits)
        : traits(tensor_traits), type(*ggml_backend_cpu_buffer_type()) {
        type.context = this;
        type.iface.get_name = [](ggml_backend_buffer_type_t) { return "JEV_BITNET_GROUP128"; };
        type.iface.is_host = [](ggml_backend_buffer_type_t) { return false; };
        type.iface.alloc_buffer = [](ggml_backend_buffer_type_t buft, std::size_t size) {
            auto* buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
            if (buffer) {
                buffer->buft = buft;
                buffer->iface.init_tensor = [](ggml_backend_buffer_t owner, ggml_tensor* tensor) {
                    auto* extra = static_cast<bitnet_buffer_type*>(owner->buft->context);
                    tensor->extra = &extra->traits;
                    return GGML_STATUS_SUCCESS;
                };
            }
            return buffer;
        };
        ggml_backend_cpu_get_extra_buffer_types().push_back(&type);
    }

    ~bitnet_buffer_type() override {
        auto& types = ggml_backend_cpu_get_extra_buffer_types();
        types.erase(std::find(types.begin(), types.end(), &type));
    }

    bool supports_op(ggml_backend_dev_t, const ggml_tensor* op) override {
        const ggml_tensor* weight = op->src[0];
        const ggml_tensor* input = op->src[1];
        return op->op == GGML_OP_MUL_MAT && weight && input &&
            weight->buffer && weight->buffer->buft == &type && weight->extra == &traits &&
            weight->type == GGML_TYPE_PQ2_0 && input->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
            weight->ne[0] == static_cast<int64_t>(traits.groups * 128) &&
            weight->ne[1] == static_cast<int64_t>(traits.rows) &&
            input->ne[0] == weight->ne[0] && input->ne[1] == static_cast<int64_t>(traits.tokens) &&
            op->ne[0] == weight->ne[1] && op->ne[1] == input->ne[1] &&
            weight->ne[2] == 1 && weight->ne[3] == 1 && input->ne[2] == 1 && input->ne[3] == 1 &&
            op->ne[2] == 1 && op->ne[3] == 1 &&
            ggml_is_contiguous(weight) && ggml_is_contiguous(input) && ggml_is_contiguous(op);
    }

    ggml::cpu::tensor_traits* get_tensor_traits(const ggml_tensor* op) override {
        return supports_op(nullptr, op) ? &traits : nullptr;
    }
};

std::mutex registration_mutex;

}

extern "C" const char* prism_group_scale_library_path(int index) {
    const void* address = nullptr;
    if (index == 0) {
        address = reinterpret_cast<const void*>(&ggml_backend_cpu_init);
    } else if (index == 1) {
        address = reinterpret_cast<const void*>(&ggml_map_custom2);
    } else {
        return nullptr;
    }
    Dl_info info {};
    return dladdr(address, &info) ? info.dli_fname : nullptr;
}

static int run_grouped_graph(
    const uint8_t* packed, const float* weight_scales, const void* activations,
    const float* activation_scales, std::size_t tokens, std::size_t rows,
    std::size_t groups, float* output, bool prepare_inputs, const float* signs = nullptr) {
    if (!packed || !weight_scales || !activations || (!prepare_inputs && !activation_scales) || !output ||
        tokens == 0 || tokens > 128 || rows == 0 || rows > 4096 || groups == 0 || groups > 96) {
        return 1;
    }
    if (signs) {
        if (!prepare_inputs) {
            return 1;
        }
        for (std::size_t index = 0; index < groups * 128; ++index) {
            if (signs[index] != 1.0f && signs[index] != -1.0f) {
                return 1;
            }
        }
    }
    graph_state state { packed, weight_scales, activation_scales, tokens, rows, groups };
    state.prepare_inputs = prepare_inputs;
    if (prepare_inputs) {
        state.prepared_activations.resize(tokens * groups * 128);
        state.prepared_scales.resize(tokens * groups);
    }
    ggml_init_params params = { 1024 * 1024, nullptr, true };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);
    if (!context) {
        return 2;
    }
    ggml_tensor* input = ggml_new_tensor_2d(
        context.get(), prepare_inputs ? GGML_TYPE_F32 : GGML_TYPE_I8, groups * 128, tokens);
    ggml_tensor* ordered_input = input;
    ggml_tensor* sign_tensor = nullptr;
    ggml_tensor* rotation = nullptr;
    if (signs) {
        sign_tensor = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, groups * 128);
        rotation = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 128, 128);
        ggml_tensor* signed_input = ggml_mul(context.get(), input, sign_tensor);
        ggml_tensor* grouped = ggml_reshape_2d(context.get(), signed_input, 128, groups * tokens);
        ggml_tensor* rotated = ggml_mul_mat(context.get(), rotation, grouped);
        ggml_mul_mat_set_hint(rotated, GGML_HINT_SRC0_IS_HADAMARD);
        ordered_input = ggml_reshape_2d(context.get(), rotated, groups * 128, tokens);
        ggml_set_input(sign_tensor);
        ggml_set_input(rotation);
    }
    ggml_tensor* shape = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, rows, tokens);
    ggml_tensor* result = ggml_map_custom2(context.get(), shape, ordered_input, grouped_dot, 1, &state);
    if (result->op != GGML_OP_MAP_CUSTOM2) {
        return 3;
    }
    ggml_set_input(input);
    ggml_set_input(shape);
    ggml_set_output(result);
    ggml_cgraph* graph = ggml_new_graph(context.get());
    ggml_build_forward_expand(graph, result);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
        ggml_backend_cpu_init(), ggml_backend_free);
    if (!backend) {
        return 2;
    }
    ggml_backend_cpu_set_n_threads(backend.get(), 1);
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
        ggml_gallocr_new(ggml_backend_cpu_buffer_type()), ggml_gallocr_free);
    if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
        return 2;
    }
    std::vector<float> empty(rows * tokens, 0.0f);
    ggml_backend_tensor_set(input, activations, 0, tokens * groups * 128 * (prepare_inputs ? sizeof(float) : sizeof(int8_t)));
    if (signs) {
        std::vector<float> unused_rotation(128 * 128, 0.0f);
        ggml_backend_tensor_set(sign_tensor, signs, 0, groups * 128 * sizeof(float));
        ggml_backend_tensor_set(rotation, unused_rotation.data(), 0, unused_rotation.size() * sizeof(float));
    }
    ggml_backend_tensor_set(shape, empty.data(), 0, empty.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        return 3;
    }
    if (state.calls != 1 || state.status != 0) {
        return state.status != 0 ? state.status : 4;
    }
    ggml_backend_tensor_get(result, output, 0, rows * tokens * sizeof(float));
    return 0;
}

extern "C" int prism_bitnet_group_scale_matmul_avx2(
    const uint8_t* packed, const float* weight_scales, const int8_t* activations,
    const float* activation_scales, std::size_t tokens, std::size_t rows,
    std::size_t groups, float* output) {
    return run_grouped_graph(packed, weight_scales, activations, activation_scales, tokens, rows, groups, output, false);
}

extern "C" int prism_bitnet_group_scale_matmul_f32(
    const uint8_t* packed, const float* weight_scales, const float* inputs,
    const float*, std::size_t tokens, std::size_t rows,
    std::size_t groups, float* output) {
    return run_grouped_graph(packed, weight_scales, inputs, nullptr, tokens, rows, groups, output, true);
}

extern "C" int prism_bitnet_group_scale_matmul_hadamard128(
    const uint8_t* packed, const float* weight_scales, const float* inputs,
    const float*, std::size_t tokens, std::size_t rows,
    std::size_t groups, float* output, const float* signs) {
    if (!signs) {
        return 1;
    }
    return run_grouped_graph(packed, weight_scales, inputs, nullptr, tokens, rows, groups, output, true, signs);
}

static int run_registered_tensor_matmul(
    const uint8_t* pq2_blocks, const float* inputs, std::size_t tokens,
    std::size_t rows, std::size_t groups, std::size_t evaluations, float* output,
    std::size_t* dispatch_calls, std::size_t* weight_repacks) {
    if (!dispatch_calls || !weight_repacks) {
        return 1;
    }
    *dispatch_calls = 0;
    *weight_repacks = 0;
    if (!pq2_blocks || !inputs || !output || tokens == 0 || tokens > 128 ||
        rows == 0 || rows > 4096 || groups == 0 || groups > 96 ||
        evaluations == 0 || evaluations > 2) {
        return 1;
    }
    const std::lock_guard<std::mutex> lock(registration_mutex);
    bitnet_tensor_traits traits(tokens, rows, groups);
    bitnet_buffer_type registered(traits);
    ggml_init_params params = { 1024 * 1024, nullptr, true };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);
    if (!context) {
        return 2;
    }
    ggml_tensor* weight = ggml_new_tensor_2d(context.get(), GGML_TYPE_PQ2_0, groups * 128, rows);
    ggml_tensor* input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, groups * 128, tokens);
    ggml_tensor* result = ggml_mul_mat(context.get(), weight, input);
    const std::size_t weight_bytes = rows * groups * 34;
    if (ggml_nbytes(weight) != weight_bytes || result->op != GGML_OP_MUL_MAT) {
        return 3;
    }
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> weight_buffer(
        ggml_backend_buft_alloc_buffer(&registered.type, weight_bytes), ggml_backend_buffer_free);
    if (!weight_buffer || ggml_backend_tensor_alloc(weight_buffer.get(), weight,
            ggml_backend_buffer_get_base(weight_buffer.get())) != GGML_STATUS_SUCCESS) {
        return 2;
    }
    ggml_backend_buffer_set_usage(weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_set_input(input);
    ggml_set_output(result);
    ggml_cgraph* graph = ggml_new_graph(context.get());
    ggml_build_forward_expand(graph, result);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
        ggml_backend_cpu_init(), ggml_backend_free);
    if (!backend) {
        return 2;
    }
    ggml_backend_cpu_set_n_threads(backend.get(), 1);
    if (!ggml_backend_supports_op(backend.get(), result)) {
        return 4;
    }
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
        ggml_gallocr_new(ggml_backend_cpu_buffer_type()), ggml_gallocr_free);
    if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
        return 2;
    }
    ggml_backend_tensor_set(weight, pq2_blocks, 0, weight_bytes);
    std::vector<float> results(evaluations * tokens * rows);
    for (std::size_t evaluation = 0; evaluation < evaluations; ++evaluation) {
        ggml_backend_tensor_set(input, inputs + evaluation * tokens * groups * 128,
            0, tokens * groups * 128 * sizeof(float));
        const ggml_status graph_status = ggml_backend_graph_compute(backend.get(), graph);
        *dispatch_calls = traits.calls;
        *weight_repacks = traits.repacks;
        if (graph_status != GGML_STATUS_SUCCESS) {
            return 3;
        }
        if (traits.status != 0 || traits.calls != evaluation + 1 || traits.repacks != 1) {
            return traits.status != 0 ? traits.status : 4;
        }
        ggml_backend_tensor_get(result, results.data() + evaluation * tokens * rows,
            0, tokens * rows * sizeof(float));
    }
    std::copy(results.begin(), results.end(), output);
    return 0;
}

extern "C" int prism_bitnet_registered_tensor_matmul(
    const uint8_t* pq2_blocks, const float* inputs, std::size_t tokens,
    std::size_t rows, std::size_t groups, float* output, std::size_t* dispatch_calls) {
    std::size_t repacks = 0;
    return run_registered_tensor_matmul(pq2_blocks, inputs, tokens, rows, groups,
        1, output, dispatch_calls, &repacks);
}

extern "C" int prism_bitnet_registered_tensor_matmul_repeated(
    const uint8_t* pq2_blocks, const float* inputs, std::size_t tokens,
    std::size_t rows, std::size_t groups, std::size_t evaluations, float* output,
    std::size_t* dispatch_calls, std::size_t* weight_repacks) {
    return run_registered_tensor_matmul(pq2_blocks, inputs, tokens, rows, groups,
        evaluations, output, dispatch_calls, weight_repacks);
}

extern "C" std::size_t prism_bitnet_registered_tensor_registry_size() {
    const std::lock_guard<std::mutex> lock(registration_mutex);
    return ggml_backend_cpu_get_extra_buffer_types().size();
}

extern "C" int prism_pq2_tensor_matmul(
    const uint8_t* pq2_blocks, const float* inputs, std::size_t tokens,
    std::size_t rows, std::size_t groups, float* output) {
    if (!pq2_blocks || !inputs || !output || tokens == 0 || tokens > 128 ||
        rows == 0 || rows > 4096 || groups == 0 || groups > 96) {
        return 1;
    }
    ggml_init_params params = { 1024 * 1024, nullptr, true };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);
    if (!context) {
        return 2;
    }
    ggml_tensor* weight = ggml_new_tensor_2d(context.get(), GGML_TYPE_PQ2_0, groups * 128, rows);
    ggml_tensor* input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, groups * 128, tokens);
    ggml_tensor* result = ggml_mul_mat(context.get(), weight, input);
    if (result->op != GGML_OP_MUL_MAT || ggml_nbytes(weight) != rows * groups * 34) {
        return 3;
    }
    ggml_set_input(weight);
    ggml_set_input(input);
    ggml_set_output(result);
    ggml_cgraph* graph = ggml_new_graph(context.get());
    ggml_build_forward_expand(graph, result);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    if (!backend) {
        return 2;
    }
    ggml_backend_cpu_set_n_threads(backend.get(), 1);
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
        ggml_gallocr_new(ggml_backend_cpu_buffer_type()), ggml_gallocr_free);
    if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
        return 2;
    }
    ggml_backend_tensor_set(weight, pq2_blocks, 0, rows * groups * 34);
    ggml_backend_tensor_set(input, inputs, 0, tokens * groups * 128 * sizeof(float));
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        return 3;
    }
    ggml_backend_tensor_get(result, output, 0, rows * tokens * sizeof(float));
    return 0;
}