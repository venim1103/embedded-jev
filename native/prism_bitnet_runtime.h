#pragma once

#include "ggml-backend.h"

#include <stdint.h>

#define JEV_BITNET_RUNTIME_ABI_V1 1
#define JEV_PRISM_SOURCE_REVISION "842b1880415d6f508f03b789e5ce70194def7bfd"

#ifdef __cplusplus
extern "C" {
#endif

int prism_bitnet_cpu_runtime_init_v1(
    const char* prism_revision, uint32_t abi_version,
    ggml_backend_buffer_type_t* buffer_type);

int prism_bitnet_cpu_tensor_status_v1(
    const struct ggml_tensor* weight, size_t* dispatch_calls, size_t* weight_repacks);

#ifdef __cplusplus
}
#endif