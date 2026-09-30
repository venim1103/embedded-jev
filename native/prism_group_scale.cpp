#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <dlfcn.h>
#include <memory>
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