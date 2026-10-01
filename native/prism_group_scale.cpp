#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "traits.h"
#include "prism_bitnet_runtime.h"

#include <algorithm>
#include <limits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <mutex>
#include <new>
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

struct bitnet_registration {
    ggml_backend_buffer_type_t type;
    bool persistent = false;

    explicit bitnet_registration(ggml_backend_buffer_type_t buffer_type) : type(buffer_type) {
        ggml_backend_cpu_get_extra_buffer_types().push_back(type);
    }

    ~bitnet_registration() {
        if (!persistent) {
            auto& types = ggml_backend_cpu_get_extra_buffer_types();
            types.erase(std::find(types.begin(), types.end(), type));
        }
    }
};

struct bitnet_loader_tensor_v1 final : ggml::cpu::tensor_traits {
    ggml_tensor* weight;
    bitnet_tensor_traits kernel;
    std::mutex mutex;
    std::size_t uploaded = 0;
    int load_status = 8;
    bool sealed = false;

    explicit bitnet_loader_tensor_v1(ggml_tensor* tensor)
        : weight(tensor), kernel(1, tensor->ne[1], tensor->ne[0] / 128) {
        kernel.status = load_status;
    }

    bool work_size(int, const ggml_tensor*, std::size_t& size) override {
        size = 0;
        return true;
    }

    bool compute_forward(ggml_compute_params* params, ggml_tensor* op) override {
        if (params->ith != 0) {
            return true;
        }
        const std::lock_guard<std::mutex> lock(mutex);
        kernel.status = load_status;
        if (load_status == 0) {
            try {
                kernel.tokens = op->src[1]->ne[1];
                kernel.activations.resize(kernel.tokens * kernel.groups * 128);
                kernel.activation_scales.resize(kernel.tokens * kernel.groups);
                kernel.compute_forward(params, op);
            } catch (const std::bad_alloc&) {
                kernel.status = 2;
            }
        }
        if (kernel.status != 0 && op->data) {
            std::fill_n(static_cast<float*>(op->data), ggml_nelements(op),
                std::numeric_limits<float>::quiet_NaN());
        }
        return true;
    }
};

struct bitnet_loader_storage_v1 {
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> storage;
    std::vector<std::unique_ptr<bitnet_loader_tensor_v1>> tensors;

    explicit bitnet_loader_storage_v1(ggml_backend_buffer_t buffer)
        : storage(buffer, ggml_backend_buffer_free) {}

    bitnet_loader_tensor_v1* find(const ggml_tensor* tensor) {
        for (const auto& traits : tensors) {
            if (traits->weight == tensor) {
                return traits.get();
            }
        }
        return nullptr;
    }
};

struct bitnet_loader_buffer_v1 final : ggml::cpu::extra_buffer_type {
    ggml_backend_buffer_type type;

    static bool valid_weight(const ggml_tensor* tensor) {
        return tensor && std::strcmp(ggml_get_name(tensor), "blk.3.ffn_down.weight") == 0 &&
            tensor->type == GGML_TYPE_PQ2_0 && tensor->ne[0] > 0 && tensor->ne[0] <= 12288 &&
            tensor->ne[0] % 128 == 0 && tensor->ne[1] > 0 && tensor->ne[1] <= 4096 &&
            tensor->ne[2] == 1 && tensor->ne[3] == 1 && ggml_is_contiguous(tensor);
    }

    static bitnet_loader_storage_v1* state(ggml_backend_buffer_t buffer) {
        return static_cast<bitnet_loader_storage_v1*>(buffer->context);
    }

    bitnet_loader_buffer_v1() : type(*ggml_backend_cpu_buffer_type()) {
        type.context = this;
        type.iface.get_name = [](ggml_backend_buffer_type_t) { return "JEV_BITNET_LOADER_V1"; };
        type.iface.is_host = [](ggml_backend_buffer_type_t) { return false; };
        type.iface.alloc_buffer = [](ggml_backend_buffer_type_t buft, std::size_t size) {
            std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> storage(
                ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size), ggml_backend_buffer_free);
            if (!storage) {
                return static_cast<ggml_backend_buffer_t>(nullptr);
            }
            try {
                auto owner = std::make_unique<bitnet_loader_storage_v1>(storage.get());
                storage.release();
                auto iface = owner->storage->iface;
                iface.free_buffer = [](ggml_backend_buffer_t buffer) { delete state(buffer); };
                iface.get_base = [](ggml_backend_buffer_t buffer) {
                    return ggml_backend_buffer_get_base(state(buffer)->storage.get());
                };
                iface.init_tensor = [](ggml_backend_buffer_t buffer, ggml_tensor* tensor) {
                    if (!valid_weight(tensor) || state(buffer)->find(tensor)) {
                        return GGML_STATUS_FAILED;
                    }
                    try {
                        auto traits = std::make_unique<bitnet_loader_tensor_v1>(tensor);
                        auto* extra = traits.get();
                        state(buffer)->tensors.push_back(std::move(traits));
                        tensor->extra = extra;
                        return GGML_STATUS_SUCCESS;
                    } catch (const std::bad_alloc&) {
                        return GGML_STATUS_ALLOC_FAILED;
                    }
                };
                iface.set_tensor = [](ggml_backend_buffer_t buffer, ggml_tensor* tensor,
                                      const void* data, std::size_t offset, std::size_t count) {
                    auto* traits = state(buffer)->find(tensor);
                    if (!traits) {
                        return;
                    }
                    const std::lock_guard<std::mutex> lock(traits->mutex);
                    if (traits->sealed || !data || offset != traits->uploaded || count == 0 ||
                        offset > ggml_nbytes(tensor) || count > ggml_nbytes(tensor) - offset) {
                        traits->load_status = traits->kernel.status = 8;
                        traits->sealed = true;
                        return;
                    }
                    std::memcpy(static_cast<uint8_t*>(tensor->data) + offset, data, count);
                    traits->uploaded += count;
                    if (traits->uploaded == ggml_nbytes(tensor)) {
                        traits->sealed = true;
                        traits->load_status = traits->kernel.prepare_weight(static_cast<const uint8_t*>(tensor->data));
                    }
                    traits->kernel.status = traits->load_status;
                };
                iface.get_tensor = [](ggml_backend_buffer_t, const ggml_tensor* tensor,
                                      void* data, std::size_t offset, std::size_t count) {
                    std::memcpy(data, static_cast<const uint8_t*>(tensor->data) + offset, count);
                };
                iface.memset_tensor = [](ggml_backend_buffer_t buffer, ggml_tensor* tensor,
                                         uint8_t, std::size_t, std::size_t) {
                    if (auto* traits = state(buffer)->find(tensor)) {
                        const std::lock_guard<std::mutex> lock(traits->mutex);
                        traits->load_status = traits->kernel.status = 8;
                        traits->sealed = true;
                    }
                };
                iface.clear = [](ggml_backend_buffer_t buffer, uint8_t value) {
                    for (const auto& traits : state(buffer)->tensors) {
                        const std::lock_guard<std::mutex> lock(traits->mutex);
                        traits->load_status = traits->kernel.status = 8;
                        traits->sealed = traits->sealed || traits->uploaded != 0;
                    }
                    ggml_backend_buffer_clear(state(buffer)->storage.get(), value);
                };
                iface.cpy_tensor = nullptr;
                iface.set_tensor_2d = nullptr;
                iface.get_tensor_2d = nullptr;
                iface.reset = nullptr;
                auto* buffer = ggml_backend_buffer_init(buft, iface, owner.get(), size);
                if (buffer) {
                    owner.release();
                }
                return buffer;
            } catch (const std::bad_alloc&) {
                return static_cast<ggml_backend_buffer_t>(nullptr);
            }
        };
    }

    bool supports_op(ggml_backend_dev_t, const ggml_tensor* op) override {
        if (!op || op->op != GGML_OP_MUL_MAT || !valid_weight(op->src[0]) || !op->src[1]) {
            return false;
        }
        const auto* weight = op->src[0];
        const auto* input = op->src[1];
        return weight->buffer && weight->buffer->buft == &type &&
            weight->extra && state(weight->buffer)->find(weight) == weight->extra &&
            input->ne[1] > 0 && input->ne[1] <= 128 &&
            input->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && input->ne[0] == weight->ne[0] &&
            op->ne[0] == weight->ne[1] && op->ne[1] == input->ne[1] &&
            input->ne[2] == 1 && input->ne[3] == 1 && op->ne[2] == 1 && op->ne[3] == 1 &&
            ggml_is_contiguous(input) && ggml_is_contiguous(op);
    }

    ggml::cpu::tensor_traits* get_tensor_traits(const ggml_tensor* op) override {
        return supports_op(nullptr, op) ? state(op->src[0]->buffer)->find(op->src[0]) : nullptr;
    }
};

bitnet_loader_buffer_v1& loader_runtime_v1() {
    static bitnet_loader_buffer_v1 runtime;
    return runtime;
}

struct registered_projection {
    bitnet_tensor_traits traits;
    bitnet_buffer_type buffer_type;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context { nullptr, ggml_free };
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> weight_buffer { nullptr, ggml_backend_buffer_free };
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend { nullptr, ggml_backend_free };
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator { nullptr, ggml_gallocr_free };
    ggml_tensor* input = nullptr;
    ggml_tensor* result = nullptr;
    ggml_cgraph* graph = nullptr;

    registered_projection(std::size_t tokens, std::size_t rows, std::size_t groups)
        : traits(tokens, rows, groups), buffer_type(traits) {}

    int initialize(const uint8_t* pq2_blocks) {
        const bitnet_registration registration(&buffer_type.type);
        ggml_init_params params = { 1024 * 1024, nullptr, true };
        context.reset(ggml_init(params));
        if (!context) {
            return 2;
        }
        ggml_tensor* weight = ggml_new_tensor_2d(context.get(), GGML_TYPE_PQ2_0, traits.groups * 128, traits.rows);
        input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, traits.groups * 128, traits.tokens);
        result = ggml_mul_mat(context.get(), weight, input);
        const std::size_t weight_bytes = traits.rows * traits.groups * 34;
        if (ggml_nbytes(weight) != weight_bytes || result->op != GGML_OP_MUL_MAT) {
            return 3;
        }
        weight_buffer.reset(ggml_backend_buft_alloc_buffer(&buffer_type.type, weight_bytes));
        if (!weight_buffer || ggml_backend_tensor_alloc(weight_buffer.get(), weight,
                ggml_backend_buffer_get_base(weight_buffer.get())) != GGML_STATUS_SUCCESS) {
            return 2;
        }
        ggml_backend_buffer_set_usage(weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_set_input(input);
        ggml_set_output(result);
        graph = ggml_new_graph(context.get());
        ggml_build_forward_expand(graph, result);
        backend.reset(ggml_backend_cpu_init());
        if (!backend) {
            return 2;
        }
        ggml_backend_cpu_set_n_threads(backend.get(), 1);
        if (!ggml_backend_supports_op(backend.get(), result)) {
            return 4;
        }
        allocator.reset(ggml_gallocr_new(ggml_backend_cpu_buffer_type()));
        if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
            return 2;
        }
        ggml_backend_tensor_set(weight, pq2_blocks, 0, weight_bytes);
        return traits.prepare_weight(static_cast<const uint8_t*>(weight->data));
    }

    int compute(const float* inputs, float* output) {
        const bitnet_registration registration(&buffer_type.type);
        const std::size_t previous_calls = traits.calls;
        traits.status = 4;
        ggml_backend_tensor_set(input, inputs, 0, traits.tokens * traits.groups * 128 * sizeof(float));
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
            return 3;
        }
        if (traits.status != 0 || traits.calls != previous_calls + 1 || traits.repacks != 1) {
            return traits.status != 0 ? traits.status : 4;
        }
        ggml_backend_tensor_get(result, output, 0, traits.tokens * traits.rows * sizeof(float));
        return 0;
    }
};

std::mutex registration_mutex;

}

extern "C" int prism_bitnet_cpu_runtime_init_v1(
    const char* prism_revision, uint32_t abi_version,
    ggml_backend_buffer_type_t* buffer_type) {
    static_assert(GGML_BACKEND_API_VERSION == 2);
    if (!buffer_type) {
        return 1;
    }
    *buffer_type = nullptr;
    if (!prism_revision || std::strcmp(prism_revision, JEV_PRISM_SOURCE_REVISION) != 0 ||
        abi_version != JEV_BITNET_RUNTIME_ABI_V1 ||
        ggml_blck_size(GGML_TYPE_PQ2_0) != 128 || ggml_type_size(GGML_TYPE_PQ2_0) != 34) {
        return 1;
    }
    const std::lock_guard<std::mutex> lock(registration_mutex);
    auto& runtime = loader_runtime_v1();
    auto& types = ggml_backend_cpu_get_extra_buffer_types();
    if (std::find(types.begin(), types.end(), &runtime.type) != types.end()) {
        *buffer_type = &runtime.type;
        return 0;
    }
    try {
        const auto registry = ggml_backend_cpu_reg();
        const auto device = ggml_backend_reg_dev_get(registry, 0);
        const auto discovery = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
            ggml_backend_reg_get_proc_address(registry, "ggml_backend_dev_get_extra_bufts"));
        if (!discovery) {
            return 7;
        }
        bitnet_registration registration(&runtime.type);
        for (auto* candidate = discovery(device); candidate && *candidate; ++candidate) {
            if (*candidate == &runtime.type) {
                registration.persistent = true;
                *buffer_type = &runtime.type;
                return 0;
            }
        }
        return 7;
    } catch (const std::bad_alloc&) {
        return 2;
    }
}

extern "C" int prism_bitnet_cpu_tensor_status_v1(
    const ggml_tensor* weight, std::size_t* dispatch_calls, std::size_t* weight_repacks) {
    if (!dispatch_calls || !weight_repacks) {
        return 1;
    }
    *dispatch_calls = *weight_repacks = 0;
    if (!weight || !weight->buffer || weight->buffer->buft != &loader_runtime_v1().type) {
        return 1;
    }
    auto* traits = bitnet_loader_buffer_v1::state(weight->buffer)->find(weight);
    if (!traits || weight->extra != traits) {
        return 1;
    }
    const std::lock_guard<std::mutex> lock(traits->mutex);
    *dispatch_calls = traits->kernel.calls;
    *weight_repacks = traits->kernel.repacks;
    return traits->kernel.status;
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
    try {
        registered_projection projection(tokens, rows, groups);
        const int initialization_status = projection.initialize(pq2_blocks);
        if (initialization_status != 0) {
            return initialization_status;
        }
        std::vector<float> results(evaluations * tokens * rows);
        for (std::size_t evaluation = 0; evaluation < evaluations; ++evaluation) {
            const int status = projection.compute(inputs + evaluation * tokens * groups * 128,
                results.data() + evaluation * tokens * rows);
            *dispatch_calls = projection.traits.calls;
            *weight_repacks = projection.traits.repacks;
            if (status != 0) {
                return status;
            }
        }
        std::copy(results.begin(), results.end(), output);
        return 0;
    } catch (const std::bad_alloc&) {
        return 2;
    }
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

extern "C" int prism_bitnet_registered_projection_create(
    const uint8_t* pq2_blocks, std::size_t tokens, std::size_t rows,
    std::size_t groups, void** handle) {
    if (!handle) {
        return 1;
    }
    *handle = nullptr;
    if (!pq2_blocks || tokens == 0 || tokens > 128 || rows == 0 || rows > 4096 || groups == 0 || groups > 96) {
        return 1;
    }
    const std::lock_guard<std::mutex> lock(registration_mutex);
    try {
        auto projection = std::make_unique<registered_projection>(tokens, rows, groups);
        const int status = projection->initialize(pq2_blocks);
        if (status == 0) {
            *handle = projection.release();
        }
        return status;
    } catch (const std::bad_alloc&) {
        return 2;
    }
}

extern "C" int prism_bitnet_registered_projection_create_from_gguf(
    const char* path, std::size_t tokens, void** handle) {
    if (!handle) {
        return 1;
    }
    *handle = nullptr;
    if (!path || tokens == 0 || tokens > 128) {
        return 1;
    }
    std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(path, "rb"), std::fclose);
    if (!file || std::fseek(file.get(), 0, SEEK_END) != 0) {
        return 6;
    }
    const long length = std::ftell(file.get());
    if (length <= 0 || length > 14 * 1024 * 1024 || std::fseek(file.get(), 0, SEEK_SET) != 0) {
        return 6;
    }
    try {
        gguf_init_params params = { true, nullptr };
        std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
            gguf_init_from_file_ptr(file.get(), params), gguf_free);
        if (!metadata || gguf_get_version(metadata.get()) != GGUF_VERSION ||
            gguf_get_n_tensors(metadata.get()) != 1 ||
            gguf_get_tensor_type(metadata.get(), 0) != GGML_TYPE_PQ2_0 ||
            std::strcmp(gguf_get_tensor_name(metadata.get(), 0), "blk.3.ffn_down.weight") != 0) {
            return 6;
        }
        const int64_t policy = gguf_find_key(metadata.get(), "jev.bitnet.execution");
        if (policy < 0 || gguf_get_kv_type(metadata.get(), policy) != GGUF_TYPE_STRING ||
            std::strcmp(gguf_get_val_str(metadata.get(), policy), "group128-a8-fp32-nearest-even-identity-v1") != 0) {
            return 6;
        }
        for (int64_t key = 0; key < gguf_get_n_kv(metadata.get()); ++key) {
            if (std::strncmp(gguf_get_key(metadata.get(), key), "prism.hadamard.", sizeof("prism.hadamard.") - 1) == 0) {
                return 6;
            }
        }
        const int64_t* shape = gguf_get_tensor_ne(metadata.get(), 0);
        if (shape[0] <= 0 || shape[0] > 12288 || shape[0] % 128 != 0 ||
            shape[1] <= 0 || shape[1] > 4096 || shape[2] != 1 || shape[3] != 1) {
            return 6;
        }
        const std::size_t rows = static_cast<std::size_t>(shape[1]);
        const std::size_t groups = static_cast<std::size_t>(shape[0]) / 128;
        const std::size_t bytes = rows * groups * 34;
        const std::size_t file_bytes = static_cast<std::size_t>(length);
        const std::size_t data_offset = gguf_get_data_offset(metadata.get());
        const std::size_t tensor_offset = gguf_get_tensor_offset(metadata.get(), 0);
        if (gguf_get_tensor_size(metadata.get(), 0) != bytes || data_offset > file_bytes ||
            tensor_offset > file_bytes - data_offset || bytes > file_bytes - data_offset - tensor_offset) {
            return 6;
        }
        std::vector<uint8_t> blocks(bytes);
        if (std::fseek(file.get(), static_cast<long>(data_offset + tensor_offset), SEEK_SET) != 0 ||
            std::fread(blocks.data(), 1, bytes, file.get()) != bytes) {
            return 6;
        }
        return prism_bitnet_registered_projection_create(blocks.data(), tokens, rows, groups, handle);
    } catch (const std::bad_alloc&) {
        return 2;
    }
}

extern "C" int prism_bitnet_registered_projection_compute(
    void* handle, const float* inputs, float* output,
    std::size_t* dispatch_calls, std::size_t* weight_repacks) {
    if (!dispatch_calls || !weight_repacks) {
        return 1;
    }
    *dispatch_calls = 0;
    *weight_repacks = 0;
    if (!handle || !inputs || !output) {
        return 1;
    }
    const std::lock_guard<std::mutex> lock(registration_mutex);
    auto* projection = static_cast<registered_projection*>(handle);
    try {
        const int status = projection->compute(inputs, output);
        *dispatch_calls = projection->traits.calls;
        *weight_repacks = projection->traits.repacks;
        return status;
    } catch (const std::bad_alloc&) {
        return 2;
    }
}

extern "C" void prism_bitnet_registered_projection_free(void* handle) {
    const std::lock_guard<std::mutex> lock(registration_mutex);
    delete static_cast<registered_projection*>(handle);
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