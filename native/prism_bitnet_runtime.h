#pragma once

#include "ggml-backend.h"

#include <stdint.h>

#define JEV_BITNET_RUNTIME_ABI_V1 1
#define JEV_BITNET_PROJECTION_ABI_V2 2
#define JEV_BITNET_MODEL_ABI_V2 2
#define JEV_PRISM_SOURCE_REVISION "842b1880415d6f508f03b789e5ce70194def7bfd"

struct jev_bitnet_projection_spec_v2 {
    uint32_t abi_version;
    const char* tensor_name;
    size_t rows;
    size_t columns;
    const uint8_t* reference_pq2;
    size_t reference_bytes;
};

struct jev_bitnet_model_tensor_v2 {
    const char* name;
    uint32_t type;
    int64_t ne[4];
};

struct jev_bitnet_model_metadata_v2 {
    const char* name;
    uint32_t kind;
    size_t count;
    const void* values;
};

struct jev_bitnet_model_spec_v2 {
    uint32_t abi_version;
    const char* profile_id;
    const char* profile_sha256;
    const char* source_revision;
    const char* source_tensor_sha256;
    uint32_t layers;
    uint32_t hidden;
    uint32_t intermediate;
    uint32_t vocabulary;
    const struct jev_bitnet_model_tensor_v2* tensors;
    size_t tensor_count;
    const struct jev_bitnet_model_metadata_v2* metadata;
    size_t metadata_count;
    struct jev_bitnet_projection_spec_v2 projection;
};

#ifdef __cplusplus
extern "C" {
#endif

int prism_bitnet_registered_projection_create_v2(
    const struct jev_bitnet_projection_spec_v2* specification,
    const uint8_t* pq2_blocks, size_t payload_bytes, size_t tokens, void** handle);

int prism_bitnet_registered_projection_compute(
    void* handle, const float* inputs, float* output,
    size_t* dispatch_calls, size_t* weight_repacks);

void prism_bitnet_registered_projection_free(void* handle);

int prism_bitnet_cpu_runtime_init_v1(
    const char* prism_revision, uint32_t abi_version,
    ggml_backend_buffer_type_t* buffer_type);

int prism_bitnet_cpu_tensor_status_v1(
    const struct ggml_tensor* weight, size_t* dispatch_calls, size_t* weight_repacks);

int prism_bitnet_cpu_tensor_last_input_tokens_v1(
    const struct ggml_tensor* weight, size_t* input_tokens);

int prism_bitnet_cpu_tensor_storage_bytes_v1(
    const struct ggml_tensor* weight, size_t* resident_weight_bytes, size_t* auxiliary_weight_bytes);

struct llama_model_tensor_buft_override;

int prism_bitnet_cpu_loader_override_from_gguf_v1(
    const char* path, const char* prism_revision, uint32_t abi_version,
    const struct llama_model_tensor_buft_override** overrides);

int prism_bitnet_cpu_model_override_from_gguf_v1(
    const char* model_path, const char* projection_path,
    const char* prism_revision, uint32_t abi_version,
    const struct llama_model_tensor_buft_override** overrides);

int prism_bitnet_cpu_model_override_from_gguf_v2(
    const char* model_path, const struct jev_bitnet_model_spec_v2* specification,
    const char* prism_revision, uint32_t abi_version, void** policy,
    const struct llama_model_tensor_buft_override** overrides);

void prism_bitnet_cpu_model_policy_free_v2(void* policy);

int prism_bitnet_cpu_tensor_status_v2(
    void* policy, const struct ggml_tensor* weight, size_t* dispatch_calls, size_t* weight_repacks);

#ifdef __cplusplus
}
#endif