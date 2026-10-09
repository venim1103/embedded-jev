"""Exact ternary-subset PQ2_0 byte conversion for the pinned Prism format."""

import ctypes
import hashlib
import os

import numpy as np

PRISM_SOURCE_REVISION = "842b1880415d6f508f03b789e5ce70194def7bfd"


def pack_ternary_pq2_0(codes, scales) -> np.ndarray:
    """Pack adjacent low-bit-first codes with one little-endian FP16 scale per 128 values."""
    values = np.asarray(codes)
    saved_scales = np.asarray(scales)
    if (
        values.ndim != 3 or values.dtype != np.int8 or values.shape[2] != 128
        or not 1 <= values.shape[0] <= 4096 or not 1 <= values.shape[1] <= 96
        or not np.isin(values, (-1, 0, 1)).all()
        or saved_scales.dtype != np.float16 or saved_scales.shape != values.shape[:2]
        or not np.isfinite(saved_scales).all() or np.any(saved_scales < 0)
        or np.any((saved_scales == 0) & np.any(values != 0, axis=-1))
    ):
        raise ValueError("invalid bounded ternary PQ2_0 codes or scales")
    rows, groups, _ = values.shape
    blocks = np.zeros((rows, groups, 34), dtype=np.uint8)
    blocks[:, :, :2] = np.ascontiguousarray(saved_scales, dtype="<f2").view(np.uint8).reshape(rows, groups, 2)
    lanes = (values + 1).astype(np.uint8).reshape(rows, groups, 32, 4)
    for lane in range(4):
        blocks[:, :, 2:] |= lanes[..., lane] << (2 * lane)
    return blocks


def unpack_ternary_pq2_0(blocks) -> tuple[np.ndarray, np.ndarray]:
    """Decode only {-1,0,+1}; native PQ2_0's additional +2 code is not ternary."""
    packed = np.asarray(blocks)
    if (
        packed.ndim != 3 or packed.dtype != np.uint8 or packed.shape[2] != 34
        or not 1 <= packed.shape[0] <= 4096 or not 1 <= packed.shape[1] <= 96
    ):
        raise ValueError("invalid bounded PQ2_0 block array")
    rows, groups, _ = packed.shape
    scales = np.ascontiguousarray(packed[:, :, :2]).view("<f2").reshape(rows, groups)
    if not np.isfinite(scales).all() or np.any(scales < 0):
        raise ValueError("invalid PQ2_0 FP16 scales")
    codes = np.empty((rows, groups, 32, 4), dtype=np.int8)
    for lane in range(4):
        values = (packed[:, :, 2:] >> (2 * lane)) & 3
        if np.any(values == 3):
            raise ValueError("PQ2_0 +2 code is outside the ternary subset")
        codes[..., lane] = values.astype(np.int8) - 1
    codes = codes.reshape(rows, groups, 128)
    if np.any((scales == 0) & np.any(codes != 0, axis=-1)):
        raise ValueError("zero PQ2_0 scale with nonzero ternary codes")
    return codes, scales


def profile_projection_specification(profile, inventory: dict, manifest: dict, pq2_blocks) -> dict:
    """Validate a declared identity-only projection against verified profile headers."""
    from embedded_jev.inventory import _profile_tensor_shapes
    from embedded_jev.model_profile import SHA256, profile_source
    from embedded_jev.ternary_artifact import HF_PROJECTION, PRISM_FOLDABLE

    fields = {
        "schema_version", "format", "profile_id", "profile_sha256", "tensor", "gguf_tensor",
        "shape", "group_size", "transform", "source_tensor_sha256", "payload_bytes", "payload_sha256",
    }
    if (
        not isinstance(manifest, dict) or set(manifest) != fields
        or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 2
        or manifest["format"] != "jev-profile-projection-v2"
        or manifest["profile_id"] != profile.profile_id or manifest["profile_sha256"] != profile.sha256
        or any(inventory.get("source", {}).get(key) != value for key, value in profile_source(profile).items())
        or type(manifest["group_size"]) is not int or manifest["group_size"] != 128
        or manifest["transform"] != "identity"
        or not isinstance(manifest["source_tensor_sha256"], str) or SHA256.fullmatch(manifest["source_tensor_sha256"]) is None
        or not isinstance(manifest["payload_sha256"], str) or SHA256.fullmatch(manifest["payload_sha256"]) is None
    ):
        raise ValueError("unverified profile projection manifest")
    match = PRISM_FOLDABLE.fullmatch(manifest["gguf_tensor"]) if isinstance(manifest["gguf_tensor"], str) else None
    if match is None:
        raise ValueError("unsupported profile projection tensor")
    name = f"{profile.packaging['text_prefix']}layers.{match[1]}.{HF_PROJECTION[match[2]]}.weight"
    expected_shape = _profile_tensor_shapes(profile).get(name)
    tensor = next((item for item in inventory["tensors"] if item["name"] == name), None)
    shape = manifest["shape"]
    if (
        tensor is None or manifest["tensor"] != name or not tensor["quantization_eligible"]
        or tensor["category"] != "language_projection"
        or tensor["dtype"] not in profile.packaging["dtype_policy"]["language_projection"]
        or HF_PROJECTION[match[2]] + ".weight" not in profile.quantization["eligible_suffixes"]
        or not isinstance(shape, list) or len(shape) != 2 or any(type(size) is not int or size < 1 for size in shape)
        or tuple(shape) != tuple(tensor["shape"]) or tuple(shape) != expected_shape
        or shape[0] > 12288 or shape[1] > 12288 or shape[1] % 128 or shape[0] * shape[1] > 4096 * 12288
    ):
        raise ValueError("profile projection name or geometry mismatch")
    blocks = np.asarray(pq2_blocks)
    if blocks.dtype != np.uint8 or blocks.shape != (shape[0], shape[1] // 128, 34) or not blocks.flags.c_contiguous:
        raise ValueError("profile projection payload shape or layout mismatch")
    if (
        type(manifest["payload_bytes"]) is not int or manifest["payload_bytes"] != blocks.nbytes
        or hashlib.sha256(memoryview(blocks)).hexdigest() != manifest["payload_sha256"]
    ):
        raise ValueError("profile projection payload identity mismatch")
    for first in range(0, shape[0], 4096):
        unpack_ternary_pq2_0(blocks[first:first + 4096])
    return {
        "abi_version": 2, "tensor_name": manifest["gguf_tensor"],
        "rows": shape[0], "columns": shape[1], "reference_bytes": blocks.nbytes,
        "profile_id": profile.profile_id, "profile_sha256": profile.sha256,
        "source_tensor_sha256": manifest["source_tensor_sha256"],
        "payload_sha256": manifest["payload_sha256"],
        "source_payload_verification": "declared_digest_not_rehashed_source_tensor",
    }


class NativeProjectionSpecification(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32), ("tensor_name", ctypes.c_char_p),
        ("rows", ctypes.c_size_t), ("columns", ctypes.c_size_t),
        ("reference_pq2", ctypes.POINTER(ctypes.c_uint8)), ("reference_bytes", ctypes.c_size_t),
    ]


def profile_model_specification(profile, inventory: dict, manifest: dict, pq2_blocks) -> dict:
    """Prepare a complete text-table policy, not model execution approval or source authentication."""
    from embedded_jev.inventory import _classify, _profile_tensor_shapes

    projection = profile_projection_specification(profile, inventory, manifest, pq2_blocks)
    runtime = profile.runtime
    geometry = profile.geometry
    if (
        runtime["architecture_adapter"] != "qwen35-safetensors-v1"
        or runtime["converter_revision"] != PRISM_SOURCE_REVISION
        or runtime["runtime_revision"] != PRISM_SOURCE_REVISION
        or tuple(runtime["converter_flags"]) != ("--no-nextn",)
        or geometry["num_hidden_layers"] > 32
        or geometry["hidden_size"] > 4096 or geometry["intermediate_size"] > 12288 or geometry["vocab_size"] > 248320
        or any(kind != ("full_attention" if (layer + 1) % geometry["full_attention_interval"] == 0 else "linear_attention")
               for layer, kind in enumerate(geometry["layer_types"]))
    ):
        raise ValueError("unsupported profile complete-model adapter or runtime pin")
    expected = _profile_tensor_shapes(profile)
    tensors = [item for item in inventory["tensors"] if item["category"] not in ("vision", "optional_mtp")]
    if (
        len(expected) != profile.packaging["text_tensor_count"] or len(tensors) != len(expected)
        or {item["name"] for item in tensors} != set(expected) or not 4 <= len(tensors) <= 427
    ):
        raise ValueError("profile complete-model tensor set mismatch")
    prefix = profile.packaging["text_prefix"]
    roots = {
        prefix + "embed_tokens.weight": "token_embd.weight",
        prefix + "norm.weight": "output_norm.weight", profile.packaging["output_head"]: "output.weight",
    }
    suffixes = {
        "input_layernorm.weight": "attn_norm.weight", "post_attention_layernorm.weight": "post_attention_norm.weight",
        "mlp.down_proj.weight": "ffn_down.weight", "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight", "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight", "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.o_proj.weight": "attn_output.weight", "self_attn.q_norm.weight": "attn_q_norm.weight",
        "self_attn.k_norm.weight": "attn_k_norm.weight", "linear_attn.in_proj_qkv.weight": "attn_qkv.weight",
        "linear_attn.in_proj_z.weight": "attn_gate.weight", "linear_attn.in_proj_a.weight": "ssm_alpha.weight",
        "linear_attn.in_proj_b.weight": "ssm_beta.weight", "linear_attn.out_proj.weight": "ssm_out.weight",
        "linear_attn.A_log": "ssm_a", "linear_attn.dt_bias": "ssm_dt.bias",
        "linear_attn.conv1d.weight": "ssm_conv1d.weight", "linear_attn.norm.weight": "ssm_norm.weight",
    }
    table = []
    weight_bytes = 0
    for tensor in sorted(tensors, key=lambda item: item["name"]):
        name = tensor["name"]
        shape = expected[name]
        category, _ = _classify(name, list(geometry["layer_types"]), profile)
        if (
            tuple(tensor["shape"]) != shape or tensor["category"] != category
            or tensor["dtype"] not in profile.packaging["dtype_policy"][category]
        ):
            raise ValueError("profile complete-model tensor geometry or dtype mismatch")
        gguf_name = roots.get(name)
        if gguf_name is None:
            layer, suffix = name[len(prefix + "layers."):].split(".", 1)
            gguf_name = f"blk.{layer}.{suffixes[suffix]}"
        if len(shape) == 3:
            shape = (shape[0], shape[2])
        dtype = "F32" if len(shape) == 1 or gguf_name.endswith(".ssm_conv1d.weight") else "BF16"
        if dtype == "BF16" and tensor["dtype"] != "BF16":
            raise ValueError("complete-model adapter refuses undeclared matrix dtype conversion")
        if gguf_name == projection["tensor_name"]:
            dtype = "PQ2_0"
            weight_bytes += projection["reference_bytes"]
        else:
            weight_bytes += int(np.prod(shape)) * (4 if dtype == "F32" else 2)
        table.append({"name": gguf_name, "type": dtype, "ne": [*reversed(shape), *([1] * (4 - len(shape)))]})
    if weight_bytes > 19 * 1024**3:
        raise ValueError("profile complete-model weight budget exceeded")
    metadata = [
        {"name": "qwen35." + name, "kind": "u32", "values": [value]} for name, value in {
            "attention.head_count": geometry["num_attention_heads"], "attention.head_count_kv": geometry["num_key_value_heads"],
            "attention.key_length": geometry["head_dim"], "attention.value_length": geometry["head_dim"],
            "rope.dimension_count": int(geometry["head_dim"] * geometry["partial_rotary_factor"]),
            "ssm.conv_kernel": geometry["linear_conv_kernel_dim"],
            "ssm.inner_size": geometry["linear_num_value_heads"] * geometry["linear_value_head_dim"],
            "ssm.state_size": geometry["linear_key_head_dim"], "ssm.time_step_rank": geometry["linear_num_value_heads"],
            "ssm.group_count": geometry["linear_num_key_heads"], "full_attention_interval": geometry["full_attention_interval"],
        }.items()
    ]
    metadata.extend([
        {"name": "qwen35.attention.layer_norm_rms_epsilon", "kind": "f32", "values": [geometry["rms_norm_eps"]]},
        {"name": "qwen35.rope.freq_base", "kind": "f32", "values": [geometry["rope_parameters"]["rope_theta"]]},
        {"name": "qwen35.rope.dimension_sections", "kind": "i32_array", "values": [*geometry["rope_parameters"]["mrope_section"], 0]},
    ])
    return {
        "abi_version": 2, "profile_id": profile.profile_id, "profile_sha256": profile.sha256,
        "source_revision": profile.revision, "source_tensor_sha256": projection["source_tensor_sha256"],
        "layers": geometry["num_hidden_layers"], "hidden": geometry["hidden_size"],
        "intermediate": geometry["intermediate_size"], "vocabulary": geometry["vocab_size"],
        "tensors": table, "metadata": metadata, "projection": projection, "maximum_file_bytes": weight_bytes + 1024**2,
        "source_payload_verification": projection["source_payload_verification"],
        "runtime_approval": "not_granted_by_policy_validation",
    }


class NativeModelTensor(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("type", ctypes.c_uint32), ("ne", ctypes.c_int64 * 4)]


class NativeModelMetadata(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("kind", ctypes.c_uint32), ("count", ctypes.c_size_t), ("values", ctypes.c_void_p)]


class NativeModelSpecification(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32), ("profile_id", ctypes.c_char_p), ("profile_sha256", ctypes.c_char_p),
        ("source_revision", ctypes.c_char_p), ("source_tensor_sha256", ctypes.c_char_p),
        ("layers", ctypes.c_uint32), ("hidden", ctypes.c_uint32), ("intermediate", ctypes.c_uint32),
        ("vocabulary", ctypes.c_uint32), ("tensors", ctypes.POINTER(NativeModelTensor)), ("tensor_count", ctypes.c_size_t),
        ("metadata", ctypes.POINTER(NativeModelMetadata)), ("metadata_count", ctypes.c_size_t),
        ("projection", NativeProjectionSpecification),
    ]


def create_profile_model_policy(native, model_path, profile, inventory: dict, manifest: dict, pq2_blocks):
    """Caller retains the library and frees the policy only after all model buffers and contexts."""
    specification = profile_model_specification(profile, inventory, manifest, pq2_blocks)
    blocks = np.asarray(pq2_blocks)
    projection = specification["projection"]
    table = (NativeModelTensor * len(specification["tensors"]))(*(
        NativeModelTensor(item["name"].encode("ascii"), {"F32": 0, "BF16": 30, "PQ2_0": 142}[item["type"]],
                          (ctypes.c_int64 * 4)(*item["ne"])) for item in specification["tensors"]
    ))
    values = [({"u32": ctypes.c_uint32, "f32": ctypes.c_float, "i32_array": ctypes.c_int32}[item["kind"]] * len(item["values"]))(*item["values"])
              for item in specification["metadata"]]
    metadata = (NativeModelMetadata * len(values))(*(
        NativeModelMetadata(item["name"].encode("ascii"), {"u32": 1, "f32": 2, "i32_array": 3}[item["kind"]],
                            len(value), ctypes.cast(value, ctypes.c_void_p))
        for item, value in zip(specification["metadata"], values, strict=True)
    ))
    record = NativeModelSpecification(
        2, profile.profile_id.encode("ascii"), profile.sha256.encode("ascii"), profile.revision.encode("ascii"),
        projection["source_tensor_sha256"].encode("ascii"), specification["layers"], specification["hidden"],
        specification["intermediate"], specification["vocabulary"], table, len(table), metadata, len(metadata),
        NativeProjectionSpecification(2, projection["tensor_name"].encode("ascii"), projection["rows"], projection["columns"],
                                      blocks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), blocks.nbytes),
    )
    function = native.prism_bitnet_cpu_model_override_from_gguf_v2
    function.argtypes = [ctypes.c_char_p, ctypes.POINTER(NativeModelSpecification), ctypes.c_char_p, ctypes.c_uint32,
                        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    function.restype = ctypes.c_int
    handle, overrides = ctypes.c_void_p(), ctypes.c_void_p()
    status = function(os.fsencode(model_path), ctypes.byref(record), PRISM_SOURCE_REVISION.encode("ascii"), 2,
                      ctypes.byref(handle), ctypes.byref(overrides))
    if status != 0 or not handle.value or not overrides.value:
        raise ValueError(f"native profile complete-model policy failed: {status}")
    return handle, overrides, specification


def create_profile_projection(native, profile, inventory: dict, manifest: dict, pq2_blocks, *, tokens: int):
    """Create only an owned projection; caller retains the library and frees its handle."""
    if type(tokens) is not int or not 1 <= tokens <= 128:
        raise ValueError("invalid profile projection token count")
    specification = profile_projection_specification(profile, inventory, manifest, pq2_blocks)
    blocks = np.asarray(pq2_blocks)
    pointer = blocks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    record = NativeProjectionSpecification(
        specification["abi_version"], specification["tensor_name"].encode("ascii"),
        specification["rows"], specification["columns"], pointer, specification["reference_bytes"],
    )
    function = native.prism_bitnet_registered_projection_create_v2
    function.argtypes = [ctypes.POINTER(NativeProjectionSpecification), ctypes.POINTER(ctypes.c_uint8),
                        ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
    function.restype = ctypes.c_int
    handle = ctypes.c_void_p()
    status = function(ctypes.byref(record), pointer, blocks.nbytes, tokens, ctypes.byref(handle))
    if status != 0 or not handle.value:
        raise ValueError(f"native profile projection creation failed: {status}")
    return handle, specification