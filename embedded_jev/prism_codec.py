"""Exact ternary-subset PQ2_0 byte conversion for the pinned Prism format."""

import ctypes
import hashlib
import os
import stat
import struct
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

PRISM_SOURCE_REVISION = "842b1880415d6f508f03b789e5ce70194def7bfd"
MODEL_METADATA_BYTES_V2 = 32 * 1024**2
MODEL_METADATA_KEYS_V2 = 256


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
        "tensors": table, "metadata": metadata, "projection": projection,
        "maximum_file_bytes": weight_bytes + MODEL_METADATA_BYTES_V2,
        "maximum_metadata_bytes": MODEL_METADATA_BYTES_V2, "maximum_metadata_keys": MODEL_METADATA_KEYS_V2,
        "source_payload_verification": projection["source_payload_verification"],
        "runtime_approval": "not_granted_by_policy_validation",
    }


def profile_model_conversion_plan(profile, inventory: dict, manifest: dict, pq2_blocks) -> dict:
    """Declare pinned dense-source transforms; this does not verify source or converted bytes."""
    specification = profile_model_specification(profile, inventory, manifest, pq2_blocks)
    sources = sorted((item for item in inventory["tensors"] if item["category"] not in ("vision", "optional_mtp")),
                     key=lambda item: item["name"])
    reordered = profile.geometry["linear_num_key_heads"] != profile.geometry["linear_num_value_heads"]
    table = []
    for source, destination in zip(sources, specification["tensors"], strict=True):
        name = source["name"]
        transforms = []
        if destination["type"] == "PQ2_0":
            transforms.append("frozen_projection_payload_not_dense_conversion")
        else:
            if reordered and ".linear_attn." in name:
                if name.endswith(("in_proj_qkv.weight", "conv1d.weight")):
                    transforms.append("v_rows_grouped_to_tiled")
                elif name.endswith("out_proj.weight"):
                    transforms.append("v_columns_grouped_to_tiled")
                elif name.endswith(("in_proj_z.weight", "in_proj_a.weight", "in_proj_b.weight", "A_log", "dt_bias")):
                    transforms.append("v_heads_grouped_to_tiled")
            if name.endswith("A_log"):
                transforms.append("negative_exp")
            elif name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
                transforms.append("add_one")
            if name.endswith("conv1d.weight"):
                transforms.append("squeeze")
            transforms.append("encode_" + destination["type"].lower())
        table.append({"source_name": name, "source_dtype": source["dtype"], "source_shape": list(source["shape"]),
                  "arithmetic_dtype": "not_applicable" if destination["type"] == "PQ2_0" else "F16" if source["dtype"] == "F16" else "F32",
                      **destination, "ne": destination["ne"].copy(), "transforms": transforms})
    return {"profile_id": profile.profile_id, "profile_sha256": profile.sha256,
            "source_revision": profile.revision, "converter_revision": PRISM_SOURCE_REVISION, "tensors": table,
            "projection_payload_sha256": specification["projection"]["payload_sha256"],
            "linear_head_layout": {key: profile.geometry[key] for key in (
                "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim")},
            "tensor_byte_verification": "not_checked", "source_payload_verification": specification["source_payload_verification"],
            "runtime_approval": "not_granted_by_conversion_plan"}


def profile_model_streaming_resource_plan_v2(
    profile, inventory: dict, manifest: dict, pq2_blocks, *,
    maximum_source_file_bytes: int, maximum_operand_bytes: int, maximum_storage_bytes: int,
) -> dict:
    """Bound declared serial storage, not observed memory, file identity or execution approval."""
    for value, ceiling in ((maximum_source_file_bytes, 6 * 1024**3),
                           (maximum_operand_bytes, 2 * 1024**3),
                           (maximum_storage_bytes, 48 * 1024**3)):
        if type(value) is not int or not 0 < value <= ceiling:
            raise ValueError("invalid v2 streaming resource budget")
    plan = profile_model_conversion_plan(profile, inventory, manifest, pq2_blocks)
    source_files = [{"name": name, "bytes": profile.files[name].bytes}
                    for name in profile.packaging["shards"]]
    metadata_inputs = [{"name": name, "bytes": pin.bytes} for name, pin in sorted(profile.files.items())
                       if name not in profile.packaging["shards"]]
    if any(item["bytes"] > maximum_source_file_bytes for item in (*source_files, *metadata_inputs)):
        raise ValueError("v2 streaming source file budget exceeded")
    tensors = []
    for entry in plan["tensors"]:
        elements = int(np.prod(entry["source_shape"]))
        source_bytes = elements * (4 if entry["source_dtype"] == "F32" else 2)
        converted_bytes = elements * 34 // 128 if entry["type"] == "PQ2_0" else elements * (4 if entry["type"] == "F32" else 2)
        if source_bytes > maximum_operand_bytes or converted_bytes > maximum_operand_bytes:
            raise ValueError("v2 streaming operand budget exceeded")
        tensors.append({"source_name": entry["source_name"], "name": entry["name"],
                        "source_bytes": source_bytes, "converted_bytes": converted_bytes,
                        "spool_bytes": (converted_bytes + 31) // 32 * 32,
                        "temporary_payload_bytes": 0 if entry["type"] == "PQ2_0" else converted_bytes})
    source_bytes = sum(item["bytes"] for item in source_files)
    metadata_input_bytes = sum(item["bytes"] for item in metadata_inputs)
    frozen_input_bytes = sum(item["converted_bytes"] for item in tensors if item["source_name"] == manifest["tensor"])
    spool_bytes = sum(item["spool_bytes"] for item in tensors)
    output_bytes = MODEL_METADATA_BYTES_V2 + spool_bytes
    temporary_bytes = max(item["temporary_payload_bytes"] for item in tensors)
    storage_bytes = source_bytes + metadata_input_bytes + frozen_input_bytes + spool_bytes + output_bytes + temporary_bytes
    if storage_bytes > maximum_storage_bytes:
        raise ValueError("v2 streaming aggregate storage budget exceeded")
    return {
        "resource_policy_version": 2, "profile_id": profile.profile_id, "profile_sha256": profile.sha256,
        "source_revision": profile.revision, "converter_revision": PRISM_SOURCE_REVISION,
        "projection_payload_sha256": plan["projection_payload_sha256"],
        "source_files": source_files, "metadata_inputs": metadata_inputs, "tensors": tensors,
        "source_storage_bytes": source_bytes, "metadata_input_storage_bytes": metadata_input_bytes,
        "frozen_payload_input_bytes": frozen_input_bytes,
        "spool_storage_bytes": spool_bytes, "output_file_ceiling_bytes": output_bytes,
        "temporary_payload_peak_bytes": temporary_bytes, "logical_storage_peak_bytes": storage_bytes,
        "maximum_source_file_bytes": maximum_source_file_bytes, "maximum_operand_bytes": maximum_operand_bytes,
        "maximum_storage_bytes": maximum_storage_bytes, "data_alignment_bytes": 32,
        "maximum_metadata_bytes": MODEL_METADATA_BYTES_V2, "ownership": "one_shard_one_operand_one_payload_serial",
        "source_file_verification": "not_checked", "tensor_byte_verification": "not_checked",
        "source_payload_verification": plan["source_payload_verification"],
        "memory_measurement": "not_performed", "runtime_approval": "not_granted_by_resource_plan",
    }


def prepare_profile_model_context_metadata(
    profile, *, metadata_files: dict, tokenizer_bytes: bytes, context_length: int,
):
    """Verify caller metadata bytes and bound context; do not infer tokenizer compatibility."""
    from embedded_jev.inventory import _json_object
    from embedded_jev.model_profile import profile_text_config, verify_profile_metadata

    if (not isinstance(metadata_files, dict) or any(type(data) is not bytes for data in metadata_files.values())
            or type(tokenizer_bytes) is not bytes
            or sum(len(data) for data in metadata_files.values()) + len(tokenizer_bytes) > MODEL_METADATA_BYTES_V2
            or type(context_length) is not int or not 0 < context_length <= 1024**2):
        raise ValueError("invalid source metadata preparation input or context budget")
    metadata = dict(metadata_files)
    verify_profile_metadata(profile, metadata)
    tokenizer_pin = profile.files["tokenizer.json"]
    if (tokenizer_pin.bytes > MODEL_METADATA_BYTES_V2 or len(tokenizer_bytes) != tokenizer_pin.bytes
            or hashlib.sha256(tokenizer_bytes).hexdigest() != tokenizer_pin.sha256):
        raise ValueError("tokenizer input identity differs from profile")
    _json_object(tokenizer_bytes, "tokenizer.json")
    config = _json_object(metadata["config.json"], "config.json")
    text = profile_text_config(profile, config)
    source_context = text.get("max_position_embeddings")
    if type(source_context) is not int or not context_length <= source_context <= 1024**2:
        raise ValueError("context length exceeds supported pinned source configuration")
    records = tuple(MappingProxyType({"name": name, "bytes": profile.files[name].bytes,
                                      "sha256": profile.files[name].sha256}) for name in sorted(metadata))
    return MappingProxyType({"profile_id": profile.profile_id, "profile_sha256": profile.sha256,
                            "source_revision": profile.revision, "context_length": context_length,
                            "source_context_length": source_context, "context_metadata_name": "qwen35.context_length",
                            "metadata_inputs": records, "tokenizer_bytes": tokenizer_pin.bytes,
                            "tokenizer_sha256": tokenizer_pin.sha256,
                            "source_metadata_verification": "profile_pinned_caller_bytes",
                            "tokenizer_semantic_verification": "not_checked",
                            "source_template_file": profile.tokenization["template_file"],
                            "template_verification": "profile_pinned_bytes_not_rendered",
                            "gguf_template_verification": "not_checked",
                            "source_weight_verification": "not_checked",
                            "runtime_approval": "not_granted_by_metadata_preparation"})


def write_profile_model_architecture_metadata(profile, inventory: dict, manifest: dict, pq2_blocks, *, writer):
    """Emit the checked architecture subset before headers; tokenizer/source files remain unverified."""
    specification = profile_model_specification(profile, inventory, manifest, pq2_blocks)
    records = [{"name": "qwen35." + key, "kind": "u32", "values": [value]} for key, value in {
        "block_count": specification["layers"], "embedding_length": specification["hidden"],
        "feed_forward_length": specification["intermediate"],
    }.items()] + specification["metadata"]
    mappings = getattr(writer, "kv_data", None)
    if (getattr(writer, "arch", None) != "qwen35" or not isinstance(mappings, list) or len(mappings) != 1
            or any(not isinstance(mapping, dict) for mapping in mappings)
            or any(not callable(getattr(writer, method, None)) for method in ("add_uint32", "add_float32", "add_array"))
            or any(record["name"] in mapping for record in records for mapping in mappings)):
        raise ValueError("invalid or conflicting architecture metadata writer")
    frozen_records = tuple(MappingProxyType({"name": record["name"], "kind": record["kind"],
                                            "values": tuple(record["values"])}) for record in records)
    for record in frozen_records:
        name, kind, values = record["name"], record["kind"], record["values"]
        if kind == "u32":
            writer.add_uint32(name, values[0])
        elif kind == "f32":
            writer.add_float32(name, values[0])
        else:
            writer.add_array(name, list(values))
    return MappingProxyType({"profile_id": profile.profile_id, "profile_sha256": profile.sha256,
                            "source_revision": profile.revision, "converter_revision": PRISM_SOURCE_REVISION,
                            "metadata_count": len(frozen_records), "records": frozen_records,
                            "metadata_verification": "declared_records_emitted_not_file_verified",
                            "tokenizer_verification": "not_checked", "source_file_verification": "not_checked",
                            "runtime_approval": "not_granted_by_metadata_emission"})


def verify_profile_converted_vector(
    profile, inventory: dict, manifest: dict, pq2_blocks, *, source_name: str,
    source_values: np.ndarray, converted_values: np.ndarray, expected_source_sha256: str,
    maximum_bytes: int = 64 * 1024,
):
    """Compare bounded operand arrays, not whole source files or a converted GGUF file."""
    from embedded_jev.model_profile import SHA256

    plan = profile_model_conversion_plan(profile, inventory, manifest, pq2_blocks)
    entry = next((item for item in plan["tensors"] if item["source_name"] == source_name), None)
    if (
        entry is None or entry["type"] != "F32" or len(entry["source_shape"]) != 1 or entry["arithmetic_dtype"] != "F32"
        or type(maximum_bytes) is not int or not 0 < maximum_bytes <= 64 * 1024
        or not isinstance(expected_source_sha256, str) or SHA256.fullmatch(expected_source_sha256) is None
        or not isinstance(source_values, np.ndarray) or not isinstance(converted_values, np.ndarray)
    ):
        raise ValueError("unsupported bounded converted vector or source identity")
    source_dtype = {"BF16": np.dtype("<u2"), "F32": np.dtype("<f4")}.get(entry["source_dtype"])
    if (
        source_dtype is None or source_values.dtype != source_dtype or converted_values.dtype != np.dtype("<f4")
        or source_values.shape != tuple(entry["source_shape"]) or converted_values.shape != source_values.shape
        or not source_values.flags.c_contiguous or not converted_values.flags.c_contiguous
        or source_values.nbytes + 2 * converted_values.nbytes > maximum_bytes
    ):
        raise ValueError("converted vector layout or byte budget mismatch")
    source_values = source_values.copy()
    converted_values = converted_values.copy()
    if hashlib.sha256(memoryview(source_values)).hexdigest() != expected_source_sha256:
        raise ValueError("source vector identity mismatch")
    values = (source_values.astype(np.uint32) << 16).view(np.float32) if entry["source_dtype"] == "BF16" else source_values
    if not np.isfinite(values).all() or not np.isfinite(converted_values).all():
        raise ValueError("converted vector operands must be finite")
    import torch

    with torch.inference_mode():
        reference = torch.from_numpy(values)
        for transform in entry["transforms"][:-1]:
            if transform == "v_heads_grouped_to_tiled":
                heads = plan["linear_head_layout"]
                groups = heads["linear_num_key_heads"]
                reference = reference.reshape(groups, heads["linear_num_value_heads"] // groups).transpose(0, 1).contiguous().reshape(-1)
            elif transform == "negative_exp":
                reference = -torch.exp(reference)
            elif transform == "add_one":
                reference = reference + 1
            else:
                raise ValueError("unsupported converted vector transform")
        expected = reference.numpy().astype("<f4", copy=False)
    if not np.isfinite(expected).all() or not np.array_equal(expected.view(np.uint8), converted_values.view(np.uint8)):
        raise ValueError("converted vector byte parity mismatch")
    return MappingProxyType({"profile_id": profile.profile_id, "profile_sha256": profile.sha256,
                            "source_name": source_name, "tensor_name": entry["name"],
                            "source_sha256": expected_source_sha256,
                            "converted_sha256": hashlib.sha256(memoryview(converted_values)).hexdigest(),
                            "converted_bytes": converted_values.nbytes,
                            "operand_byte_verification": "exact_f32_bytes_match_pinned_transform_reference",
                            "source_file_verification": "not_checked", "converted_file_verification": "not_checked",
                            "runtime_approval": "not_granted_by_operand_verification"})


def iter_profile_bf16_matrix_chunks(
    profile, inventory: dict, manifest: dict, pq2_blocks, *, source_name: str,
    source_values: np.ndarray, maximum_chunk_bytes: int = 64 * 1024,
):
    """Yield (first_row, owned read-only BF16 rows) from borrowed, unauthenticated source values."""
    plan = profile_model_conversion_plan(profile, inventory, manifest, pq2_blocks)
    entry = next((item for item in plan["tensors"] if item["source_name"] == source_name), None)
    if (
        entry is None or entry["type"] != "BF16" or entry["source_dtype"] != "BF16" or len(entry["source_shape"]) != 2
        or type(maximum_chunk_bytes) is not int or not 0 < maximum_chunk_bytes <= 1024**2
        or not isinstance(source_values, np.ndarray) or source_values.dtype != np.dtype("<u2")
        or source_values.shape != tuple(entry["source_shape"]) or not source_values.flags.c_contiguous
    ):
        raise ValueError("unsupported chunked BF16 matrix or output budget")
    source_values = np.asarray(source_values)
    transforms = entry["transforms"]
    if transforms not in (
        ["encode_bf16"], ["v_rows_grouped_to_tiled", "encode_bf16"],
        ["v_columns_grouped_to_tiled", "encode_bf16"], ["v_heads_grouped_to_tiled", "encode_bf16"],
    ):
        raise ValueError("unsupported chunked BF16 matrix transform")
    rows, columns = source_values.shape
    rows_per_chunk = maximum_chunk_bytes // (columns * 2)
    if rows_per_chunk == 0:
        raise ValueError("chunked BF16 matrix row exceeds output budget")
    heads = plan["linear_head_layout"]
    groups = heads["linear_num_key_heads"]
    per_group = heads["linear_num_value_heads"] // groups

    def source_indices(indices, head_dim, prefix=0):
        selected = indices >= prefix
        relative = indices[selected] - prefix
        head = relative // head_dim
        indices[selected] = prefix + ((head % groups) * per_group + head // groups) * head_dim + relative % head_dim
        return indices

    column_indices = None
    if transforms[0] == "v_columns_grouped_to_tiled":
        column_indices = source_indices(np.arange(columns), heads["linear_value_head_dim"])
    for first in range(0, rows, rows_per_chunk):
        last = min(first + rows_per_chunk, rows)
        if transforms[0] in ("v_rows_grouped_to_tiled", "v_heads_grouped_to_tiled"):
            prefix = 2 * groups * heads["linear_key_head_dim"] if transforms[0] == "v_rows_grouped_to_tiled" else 0
            width = heads["linear_value_head_dim"] if transforms[0] == "v_rows_grouped_to_tiled" or source_name.endswith("in_proj_z.weight") else 1
            indices = source_indices(np.arange(first, last), width, prefix)
            chunk = source_values[indices]
        elif column_indices is not None:
            chunk = np.ascontiguousarray(source_values[first:last, column_indices])
        else:
            chunk = source_values[first:last].copy()
        if np.any((chunk & 0x7F80) == 0x7F80):
            raise ValueError("chunked BF16 matrix operands must be finite")
        chunk.flags.writeable = False
        yield first, chunk


def temporary_profile_bf16_matrix_payload(
    profile, inventory: dict, manifest: dict, pq2_blocks, *, source_name: str,
    source_values: np.ndarray, expected_source_sha256: str, directory,
    maximum_bytes: int = 16 * 1024**2, maximum_chunk_bytes: int = 64 * 1024,
    maximum_seconds: float = 30,
):
    """Own a bounded raw payload; finish file users before exit and publish only after successful exit."""
    return _temporary_profile_bf16_matrix_payload(
        profile, inventory, manifest, pq2_blocks, source_name=source_name, source_values=source_values,
        expected_source_sha256=expected_source_sha256, directory=directory, maximum_bytes=maximum_bytes,
        maximum_chunk_bytes=maximum_chunk_bytes, maximum_seconds=maximum_seconds,
        payload_ceiling=16 * 1024**2, policy_version=1,
    )


def temporary_profile_bf16_matrix_payload_v2(
    profile, inventory: dict, manifest: dict, pq2_blocks, *, source_name: str,
    source_values: np.ndarray, expected_source_sha256: str, directory,
    maximum_bytes: int = 16 * 1024**2, maximum_chunk_bytes: int = 64 * 1024,
    maximum_seconds: float = 30,
):
    """Explicit v2 raw payload policy; larger extents require a caller budget."""
    return _temporary_profile_bf16_matrix_payload(
        profile, inventory, manifest, pq2_blocks, source_name=source_name, source_values=source_values,
        expected_source_sha256=expected_source_sha256, directory=directory, maximum_bytes=maximum_bytes,
        maximum_chunk_bytes=maximum_chunk_bytes, maximum_seconds=maximum_seconds,
        payload_ceiling=2 * 1024**3, policy_version=2,
    )


@contextmanager
def _temporary_profile_bf16_matrix_payload(
    profile, inventory: dict, manifest: dict, pq2_blocks, *, source_name, source_values,
    expected_source_sha256, directory, maximum_bytes, maximum_chunk_bytes, maximum_seconds,
    payload_ceiling, policy_version,
):
    from tempfile import TemporaryDirectory

    from embedded_jev.model_profile import SHA256

    if (
        type(maximum_bytes) is not int or not 0 < maximum_bytes <= payload_ceiling
        or type(maximum_seconds) not in (int, float) or not 0 < maximum_seconds <= 600
        or not isinstance(expected_source_sha256, str) or SHA256.fullmatch(expected_source_sha256) is None
        or not isinstance(source_values, np.ndarray) or source_values.dtype != np.dtype("<u2")
        or source_values.ndim != 2 or not source_values.flags.c_contiguous
        or not 0 < source_values.nbytes <= maximum_bytes
    ):
        raise ValueError("invalid temporary BF16 payload source or resource budget")
    source_values = np.asarray(source_values).view()
    source_bytes = memoryview(source_values).cast("B")
    started = time.monotonic()

    def check_deadline():
        if time.monotonic() - started > maximum_seconds:
            raise ValueError("temporary BF16 payload deadline exceeded")

    def check_source():
        digest = hashlib.sha256()
        for offset in range(0, len(source_bytes), 65536):
            check_deadline()
            digest.update(source_bytes[offset:offset + 65536])
        check_deadline()
        if digest.hexdigest() != expected_source_sha256:
            raise ValueError("temporary BF16 payload source identity mismatch")

    check_source()
    chunks = iter_profile_bf16_matrix_chunks(profile, inventory, manifest, pq2_blocks,
                 source_name=source_name, source_values=source_values, maximum_chunk_bytes=maximum_chunk_bytes)
    with TemporaryDirectory(prefix="jev-bf16-payload-", dir=directory) as temporary:
        path = Path(temporary) / "matrix.bf16"
        digest = hashlib.sha256()
        written = 0
        with path.open("xb") as output:
            for first, chunk in chunks:
                check_deadline()
                if first * source_values.shape[1] * 2 != written or written + chunk.nbytes > source_values.nbytes:
                    raise ValueError("temporary BF16 payload chunk extent mismatch")
                if output.write(memoryview(chunk)) != chunk.nbytes:
                    raise ValueError("temporary BF16 payload short write")
                digest.update(memoryview(chunk))
                written += chunk.nbytes
        if written != source_values.nbytes:
            raise ValueError("temporary BF16 payload size mismatch")
        converted_sha256 = digest.hexdigest()

        def identity(value):
            return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

        before = identity(path.lstat())

        def check_payload():
            check_deadline()
            try:
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(descriptor, "rb", buffering=0) as payload:
                    info = os.fstat(payload.fileno())
                    if not stat.S_ISREG(info.st_mode) or info.st_size != written or identity(info) != before:
                        raise ValueError("temporary BF16 payload file identity mismatch")
                    actual = hashlib.sha256()
                    remaining = written
                    while remaining:
                        check_deadline()
                        data = payload.read(min(65536, remaining))
                        if not data:
                            raise ValueError("temporary BF16 payload truncated")
                        actual.update(data)
                        remaining -= len(data)
                    check_deadline()
                    if identity(os.fstat(payload.fileno())) != before or identity(path.lstat()) != before or actual.hexdigest() != converted_sha256:
                        raise ValueError("temporary BF16 payload file identity mismatch")
            except OSError as error:
                raise ValueError("temporary BF16 payload file unavailable") from error

        check_source()
        check_payload()
        policy_receipt = {} if policy_version == 1 else {"payload_policy_version": policy_version,
                                   "payload_cap_bytes": payload_ceiling}
        receipt = MappingProxyType({**policy_receipt, "profile_id": profile.profile_id, "profile_sha256": profile.sha256,
                                   "source_name": source_name, "source_sha256": expected_source_sha256,
                                   "converted_sha256": converted_sha256, "converted_bytes": written,
                                   "source_array_verification": "entry_and_exit_checks_required",
                                   "source_file_verification": "not_checked",
                                   "payload_file_verification": "entry_and_exit_checks_required",
                                   "runtime_approval": "not_granted_by_temporary_payload"})
        yield path, receipt
        check_source()
        check_payload()


@contextmanager
def temporary_profile_safetensors_bf16_payload(
    profile, inventory: dict, manifest: dict, pq2_blocks, *, source_path, source_name: str,
    source_values: np.ndarray, expected_source_sha256: str, expected_file_bytes: int,
    expected_file_sha256: str, directory, maximum_bytes: int = 16 * 1024**2,
    maximum_file_bytes: int = 16 * 1024**2, maximum_chunk_bytes: int = 64 * 1024,
    maximum_seconds: float = 30,
):
    """Bind a disposable BF16 payload to unchanged profile-pinned source ranges at entry and exit."""
    if type(maximum_seconds) not in (int, float) or not 0 < maximum_seconds <= 600:
        raise ValueError("invalid source-bound BF16 payload deadline")
    deadline = time.monotonic() + maximum_seconds
    source_path = Path(source_path).absolute()

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise ValueError("source-bound BF16 payload deadline exceeded")
        return seconds

    def verify_source():
        try:
            return verify_profile_safetensors_operand(profile, inventory, manifest, pq2_blocks,
                       source_path=source_path, source_name=source_name, source_values=source_values,
                       expected_source_sha256=expected_source_sha256, expected_file_bytes=expected_file_bytes,
                       expected_file_sha256=expected_file_sha256, maximum_bytes=maximum_bytes,
                       maximum_file_bytes=maximum_file_bytes, maximum_seconds=remaining())
        except OSError as exc:
            raise ValueError("source-bound BF16 payload source file unavailable") from exc

    source_receipt = verify_source()
    with temporary_profile_bf16_matrix_payload(profile, inventory, manifest, pq2_blocks,
             source_name=source_name, source_values=source_values, expected_source_sha256=expected_source_sha256,
             directory=directory, maximum_bytes=maximum_bytes, maximum_chunk_bytes=maximum_chunk_bytes,
             maximum_seconds=remaining()) as (path, receipt):
        yield path, MappingProxyType({**receipt,
                                     "source_file_verification": "profile_pinned_shard_header_range_entry_and_exit_checks_required",
                                     "source_file": source_receipt,
                                     "runtime_approval": "not_granted_by_source_bound_payload"})
    if verify_source() != source_receipt:
        raise ValueError("source-bound BF16 payload source file changed")


@contextmanager
def temporary_profile_mapped_bf16_payload(
    profile, inventory: dict, manifest: dict, pq2_blocks, *, source_path, source_name: str,
    expected_source_sha256: str, expected_file_bytes: int, expected_file_sha256: str,
    directory, maximum_bytes: int = 16 * 1024**2, maximum_file_bytes: int = 16 * 1024**2,
    maximum_chunk_bytes: int = 64 * 1024, maximum_seconds: float = 30,
):
    """Compose verified source mapping and disposable payload ownership without caller source arrays."""
    if type(maximum_seconds) not in (int, float) or not 0 < maximum_seconds <= 600:
        raise ValueError("invalid mapped BF16 payload deadline")
    deadline = time.monotonic() + maximum_seconds

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise ValueError("mapped BF16 payload deadline exceeded")
        return seconds

    with mapped_profile_safetensors_bf16_matrix(profile, inventory, manifest, pq2_blocks,
             source_path=source_path, source_name=source_name, expected_source_sha256=expected_source_sha256,
             expected_file_bytes=expected_file_bytes, expected_file_sha256=expected_file_sha256,
             maximum_bytes=maximum_bytes, maximum_file_bytes=maximum_file_bytes,
             maximum_seconds=remaining()) as (values, source_receipt):
        with temporary_profile_bf16_matrix_payload(profile, inventory, manifest, pq2_blocks,
                 source_name=source_name, source_values=values, expected_source_sha256=expected_source_sha256,
                 directory=directory, maximum_bytes=maximum_bytes, maximum_chunk_bytes=maximum_chunk_bytes,
                 maximum_seconds=remaining()) as (path, receipt):
            yield path, MappingProxyType({**receipt, "source_file": source_receipt,
                                         "source_file_verification": source_receipt["source_file_verification"],
                                         "runtime_approval": "not_granted_by_mapped_source_payload"})


def verify_profile_converted_matrix(
    profile, inventory: dict, manifest: dict, pq2_blocks, *, source_name: str,
    source_values: np.ndarray, converted_values: np.ndarray, expected_source_sha256: str,
    maximum_bytes: int = 1024**2,
):
    """Compare bounded raw BF16 matrix operands, not whole source or GGUF files."""
    from embedded_jev.model_profile import SHA256

    plan = profile_model_conversion_plan(profile, inventory, manifest, pq2_blocks)
    entry = next((item for item in plan["tensors"] if item["source_name"] == source_name), None)
    if (
        entry is None or entry["type"] != "BF16" or entry["source_dtype"] != "BF16" or len(entry["source_shape"]) != 2
        or type(maximum_bytes) is not int or not 0 < maximum_bytes <= 16 * 1024**2
        or not isinstance(expected_source_sha256, str) or SHA256.fullmatch(expected_source_sha256) is None
        or not isinstance(source_values, np.ndarray) or not isinstance(converted_values, np.ndarray)
    ):
        raise ValueError("unsupported bounded converted matrix or source identity")
    if (
        source_values.dtype != np.dtype("<u2") or converted_values.dtype != np.dtype("<u2")
        or source_values.shape != tuple(entry["source_shape"]) or converted_values.shape != source_values.shape
        or not source_values.flags.c_contiguous or not converted_values.flags.c_contiguous
        or source_values.nbytes + 2 * converted_values.nbytes > maximum_bytes
    ):
        raise ValueError("converted matrix layout or byte budget mismatch")
    source_values = source_values.copy()
    converted_values = converted_values.copy()
    if hashlib.sha256(memoryview(source_values)).hexdigest() != expected_source_sha256:
        raise ValueError("source matrix identity mismatch")
    if np.any((source_values & 0x7F80) == 0x7F80) or np.any((converted_values & 0x7F80) == 0x7F80):
        raise ValueError("converted matrix operands must be finite")
    heads = plan["linear_head_layout"]
    groups = heads["linear_num_key_heads"]
    per_group = heads["linear_num_value_heads"] // groups

    def reorder(values, axis, head_dim):
        shape = list(values.shape)
        expanded = [*shape[:axis], groups, per_group, head_dim, *shape[axis + 1:]]
        return values.reshape(expanded).swapaxes(axis, axis + 1).reshape(shape)

    reference = source_values
    for transform in entry["transforms"][:-1]:
        if transform == "v_rows_grouped_to_tiled":
            prefix = 2 * groups * heads["linear_key_head_dim"]
            reference = np.concatenate((reference[:prefix], reorder(reference[prefix:], 0, heads["linear_value_head_dim"])))
        elif transform == "v_columns_grouped_to_tiled":
            reference = reorder(reference, 1, heads["linear_value_head_dim"])
        elif transform == "v_heads_grouped_to_tiled":
            reference = reorder(reference, 0, heads["linear_value_head_dim"] if source_name.endswith("in_proj_z.weight") else 1)
        else:
            raise ValueError("unsupported converted matrix transform")
    if not np.array_equal(reference, converted_values):
        raise ValueError("converted matrix byte parity mismatch")
    return MappingProxyType({"profile_id": profile.profile_id, "profile_sha256": profile.sha256,
                            "source_name": source_name, "tensor_name": entry["name"],
                            "source_sha256": expected_source_sha256,
                            "converted_sha256": hashlib.sha256(memoryview(converted_values)).hexdigest(),
                            "converted_bytes": converted_values.nbytes,
                            "operand_byte_verification": "exact_bf16_bits_match_declared_transform_reference",
                            "source_file_verification": "not_checked", "converted_file_verification": "not_checked",
                            "runtime_approval": "not_granted_by_operand_verification"})


def verify_profile_converted_convolution(
    profile, inventory: dict, manifest: dict, pq2_blocks, *, source_name: str,
    source_values: np.ndarray, converted_values: np.ndarray, expected_source_sha256: str,
    maximum_bytes: int = 512 * 1024,
):
    """Compare bounded convolution operand bits, not source or converted files."""
    from embedded_jev.model_profile import SHA256

    plan = profile_model_conversion_plan(profile, inventory, manifest, pq2_blocks)
    entry = next((item for item in plan["tensors"] if item["source_name"] == source_name), None)
    if (
        entry is None or entry["type"] != "F32" or not source_name.endswith("linear_attn.conv1d.weight")
        or len(entry["source_shape"]) != 3 or entry["source_shape"][1] != 1 or entry["source_shape"][2] <= 1
        or type(maximum_bytes) is not int or not 0 < maximum_bytes <= 16 * 1024**2
        or not isinstance(expected_source_sha256, str) or SHA256.fullmatch(expected_source_sha256) is None
        or not isinstance(source_values, np.ndarray) or not isinstance(converted_values, np.ndarray)
    ):
        raise ValueError("unsupported bounded converted convolution or source identity")
    source_dtype = {"BF16": np.dtype("<u2"), "F32": np.dtype("<f4")}.get(entry["source_dtype"])
    output_shape = (entry["source_shape"][0], entry["source_shape"][2])
    if (
        source_dtype is None or source_values.dtype != source_dtype or converted_values.dtype != np.dtype("<f4")
        or source_values.shape != tuple(entry["source_shape"]) or converted_values.shape != output_shape
        or not source_values.flags.c_contiguous or not converted_values.flags.c_contiguous
        or source_values.nbytes + 2 * converted_values.nbytes > maximum_bytes
    ):
        raise ValueError("converted convolution layout or byte budget mismatch")
    source_values = source_values.copy()
    converted_values = converted_values.copy()
    if hashlib.sha256(memoryview(source_values)).hexdigest() != expected_source_sha256:
        raise ValueError("source convolution identity mismatch")
    reference = (source_values.astype(np.uint32) << 16).view(np.float32) if entry["source_dtype"] == "BF16" else source_values
    if not np.isfinite(reference).all() or not np.isfinite(converted_values).all():
        raise ValueError("converted convolution operands must be finite")
    for transform in entry["transforms"][:-1]:
        if transform == "v_rows_grouped_to_tiled":
            heads = plan["linear_head_layout"]
            groups = heads["linear_num_key_heads"]
            prefix = 2 * groups * heads["linear_key_head_dim"]
            trailing = reference.shape[1:]
            reordered = reference[prefix:].reshape(groups, heads["linear_num_value_heads"] // groups,
                                                   heads["linear_value_head_dim"], *trailing).swapaxes(0, 1)
            reference = np.concatenate((reference[:prefix], reordered.reshape(reference[prefix:].shape)))
        elif transform == "squeeze":
            reference = reference.squeeze()
        else:
            raise ValueError("unsupported converted convolution transform")
    if reference.shape != output_shape or not np.array_equal(reference.view(np.uint8), converted_values.view(np.uint8)):
        raise ValueError("converted convolution byte parity mismatch")
    return MappingProxyType({"profile_id": profile.profile_id, "profile_sha256": profile.sha256,
                            "source_name": source_name, "tensor_name": entry["name"],
                            "source_sha256": expected_source_sha256,
                            "converted_sha256": hashlib.sha256(memoryview(converted_values)).hexdigest(),
                            "converted_bytes": converted_values.nbytes,
                            "operand_byte_verification": "exact_f32_bits_match_declared_convolution_reference",
                            "source_file_verification": "not_checked", "converted_file_verification": "not_checked",
                            "runtime_approval": "not_granted_by_operand_verification"})


def _verify_profile_gguf_operand(
    profile, inventory: dict, manifest: dict, pq2_blocks, *, reader, source_name: str,
    source_values: np.ndarray, expected_source_sha256: str, expected_file_bytes: int,
    expected_file_sha256: str, maximum_file_bytes: int = 64 * 1024, maximum_seconds: float = 30,
    kind: str = "vector", maximum_bytes: int = 64 * 1024,
    model_owner=None,
):
    """Bind a typed reader view to caller file identity, not whole-model parity."""
    from embedded_jev.model_profile import SHA256
    from embedded_jev.streamed_text import bounded_native_file_record, bounded_native_descriptor_record

    matrix = kind == "matrix"
    convolution = kind == "convolution"
    projection = kind == "projection"
    if (
        kind not in ("vector", "matrix", "convolution", "projection")
        or type(expected_file_bytes) is not int or type(maximum_file_bytes) is not int
        or not 0 < expected_file_bytes <= maximum_file_bytes <= 16 * 1024**2
        or not isinstance(expected_file_sha256, str) or SHA256.fullmatch(expected_file_sha256) is None
        or type(maximum_seconds) not in (int, float) or not 0 < maximum_seconds <= 600
        or type(maximum_bytes) is not int or not 0 < maximum_bytes <= (64 * 1024 if kind == "vector" else 64 * 1024**2 if projection else 16 * 1024**2)
    ):
        raise ValueError("invalid GGUF vector file identity or budget")
    plan = profile_model_conversion_plan(profile, inventory, manifest, pq2_blocks)
    entry = next((item for item in plan["tensors"] if item["source_name"] == source_name), None)
    storage = "PQ2_0" if projection else "BF16" if matrix else "F32"
    if entry is None or entry["type"] != storage or len(entry["source_shape"]) != (3 if convolution else 2 if matrix or projection else 1):
        raise ValueError("GGUF operand requires its declared storage type and rank")
    if convolution and (not source_name.endswith("linear_attn.conv1d.weight") or entry["source_shape"][1] != 1 or entry["source_shape"][2] <= 1):
        raise ValueError("GGUF convolution requires declared nonsingleton kernel geometry")
    output_shape = [entry["source_shape"][0], entry["source_shape"][2]] if convolution else entry["source_shape"]
    elements = int(np.prod(output_shape))
    size = elements // 128 * 34 if projection else elements * (2 if matrix else 4)
    source_dtype = np.dtype("uint8") if projection else {"BF16": np.dtype("<u2"), "F32": np.dtype("<f4")}.get(entry["source_dtype"])
    source_shape = np.asarray(pq2_blocks).shape if projection else tuple(entry["source_shape"])
    if (
        source_dtype is None or not isinstance(source_values, np.ndarray) or source_values.dtype != source_dtype
        or source_values.shape != source_shape or not source_values.flags.c_contiguous
        or not isinstance(expected_source_sha256, str) or SHA256.fullmatch(expected_source_sha256) is None
        or source_values.nbytes + 2 * size > maximum_bytes
    ):
        raise ValueError("GGUF source operand layout, identity or byte budget mismatch")
    import gguf

    if (
        type(reader) is not gguf.GGUFReader or reader.endianess != gguf.GGUFEndian.LITTLE
        or not isinstance(reader.data, np.memmap) or reader.data.mode != "r" or reader.data.offset != 0
        or reader.data.dtype != np.uint8 or reader.data.ndim != 1 or reader.data.size != expected_file_bytes
    ):
        raise ValueError("GGUF vector requires a read-only little-endian pinned reader")
    matches = [item for item in reader.tensors if item.name == entry["name"]]
    if len(matches) != 1:
        raise ValueError("GGUF vector name is missing or duplicated")
    tensor = matches[0]
    dimensions = list(reversed(output_shape))
    stored_shape = [*output_shape[:-1], output_shape[-1] // 128 * 34] if projection else [*output_shape[:-1], output_shape[-1] * 2] if matrix else output_shape
    offset = int(tensor.data_offset)
    alignment = reader.fields.get("general.alignment")
    alignment = 32 if alignment is None else alignment.contents()
    table_end = max(item.field.offset + sum(part.nbytes for part in item.field.parts) for item in reader.tensors)
    if (
        tensor.tensor_type != getattr(gguf.GGMLQuantizationType, storage) or list(tensor.shape) != dimensions
        or tensor.n_bytes != size or tensor.n_elements != elements
        or tensor.data.dtype != np.dtype("uint8" if matrix or projection else "<f4") or tensor.data.shape != tuple(stored_shape)
        or not tensor.data.flags.c_contiguous or tensor.data.flags.writeable
        or not np.shares_memory(reader.data, tensor.data)
        or offset < reader.data_offset or offset + size > expected_file_bytes
        or tensor.data.ctypes.data - reader.data.ctypes.data != offset
        or not 1 <= len(reader.tensors) <= 427 or not 3 <= len(reader.fields) <= 259
        or alignment != reader.alignment or alignment <= 0 or alignment & (alignment - 1)
        or (table_end + alignment - 1) // alignment * alignment != reader.data_offset
    ):
        raise ValueError("GGUF vector type, geometry or mapped range mismatch")
    descriptor = _verified_owner_descriptor(model_owner, profile, manifest, expected_file_bytes, expected_file_sha256)
    path = Path(reader.data.filename) if descriptor is None else Path(model_owner.model_path)
    deadline = time.monotonic() + maximum_seconds
    def file_record():
        return (bounded_native_file_record(path, maximum_bytes=expected_file_bytes, deadline=deadline) if descriptor is None
                else bounded_native_descriptor_record(descriptor, maximum_bytes=expected_file_bytes, deadline=deadline))

    before = file_record()
    if before["bytes"] != expected_file_bytes or before["sha256"] != expected_file_sha256:
        raise ValueError("GGUF vector file identity mismatch")
    copied = tensor.data.view("<u2").copy() if matrix else tensor.data.copy()
    name = tensor.name.encode("utf-8")
    information = (struct.pack("<Q", len(name)) + name + struct.pack("<I", len(dimensions))
                   + b"".join(struct.pack("<Q", dimension) for dimension in dimensions)
                   + struct.pack("<IQ", int(getattr(gguf.GGMLQuantizationType, storage)), offset - reader.data_offset))
    field_offset = int(tensor.field.offset)
    if not 24 <= field_offset < field_offset + len(information) <= reader.data_offset:
        raise ValueError("GGUF vector tensor table range mismatch")
    retained = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK) if descriptor is None else os.dup(descriptor)
    try:
        if os.pread(retained, 24, 0) != struct.pack("<4sIQQ", b"GGUF", 3, len(reader.tensors), len(reader.fields) - 3):
            raise ValueError("GGUF vector header differs from reader")
        if os.pread(retained, len(information), field_offset) != information:
            raise ValueError("GGUF vector tensor table differs from reader")
        if os.pread(retained, size, offset) != copied.tobytes():
            raise ValueError("GGUF vector mapping differs from file range")
    finally:
        os.close(retained)
    if projection:
        if expected_source_sha256 != manifest["payload_sha256"] or not np.array_equal(copied.reshape(source_shape), pq2_blocks):
            raise ValueError("GGUF frozen projection byte parity mismatch")
        receipt = {"profile_id": profile.profile_id, "profile_sha256": profile.sha256, "source_name": source_name,
                   "tensor_name": entry["name"], "source_file_verification": "not_checked",
                   "source_payload_verification": "declared_digest_not_rehashed_source_tensor",
                   "projection_payload_sha256": manifest["payload_sha256"],
                   "converted_sha256": hashlib.sha256(memoryview(copied)).hexdigest(), "converted_bytes": copied.nbytes,
                   "operand_byte_verification": "exact_frozen_pq2_bytes"}
    else:
        verifier = verify_profile_converted_convolution if convolution else verify_profile_converted_matrix if matrix else verify_profile_converted_vector
        receipt = verifier(
            profile, inventory, manifest, pq2_blocks, source_name=source_name, source_values=source_values,
            converted_values=copied, expected_source_sha256=expected_source_sha256, maximum_bytes=maximum_bytes,
        )
    if file_record() != before:
        raise ValueError("GGUF vector file changed during verification")
    return MappingProxyType({**receipt, "converted_file_verification": "caller_sha256_and_reader_range_match",
                            "gguf_reader_verification": "not_checked",
                            "converted_file_sha256": expected_file_sha256, "converted_file_bytes": expected_file_bytes,
                            "converted_tensor_offset": offset,
                            "runtime_approval": "not_granted_by_" + kind + "_file_verification"})


def verify_profile_gguf_vector(
    profile, inventory: dict, manifest: dict, pq2_blocks, *, reader, source_name: str,
    source_values: np.ndarray, expected_source_sha256: str, expected_file_bytes: int,
    expected_file_sha256: str, maximum_file_bytes: int = 64 * 1024, maximum_seconds: float = 30,
):
    """Bind a reader F32 vector view to caller file identity, not whole-model parity."""
    return _verify_profile_gguf_operand(
        profile, inventory, manifest, pq2_blocks, reader=reader, source_name=source_name, source_values=source_values,
        expected_source_sha256=expected_source_sha256, expected_file_bytes=expected_file_bytes,
        expected_file_sha256=expected_file_sha256, maximum_file_bytes=maximum_file_bytes, maximum_seconds=maximum_seconds,
    )


def verify_profile_gguf_matrix(
    profile, inventory: dict, manifest: dict, pq2_blocks, *, reader, source_name: str,
    source_values: np.ndarray, expected_source_sha256: str, expected_file_bytes: int,
    expected_file_sha256: str, maximum_file_bytes: int = 16 * 1024**2, maximum_seconds: float = 30,
    maximum_bytes: int = 1024**2,
):
    """Bind a reader BF16 matrix view to caller file identity, not whole-model parity."""
    return _verify_profile_gguf_operand(
        profile, inventory, manifest, pq2_blocks, reader=reader, source_name=source_name, source_values=source_values,
        expected_source_sha256=expected_source_sha256, expected_file_bytes=expected_file_bytes,
        expected_file_sha256=expected_file_sha256, maximum_file_bytes=maximum_file_bytes, maximum_seconds=maximum_seconds,
        kind="matrix", maximum_bytes=maximum_bytes,
    )


def verify_profile_gguf_convolution(
    profile, inventory: dict, manifest: dict, pq2_blocks, *, reader, source_name: str,
    source_values: np.ndarray, expected_source_sha256: str, expected_file_bytes: int,
    expected_file_sha256: str, maximum_file_bytes: int = 16 * 1024**2, maximum_seconds: float = 30,
    maximum_bytes: int = 512 * 1024,
):
    """Bind a reader F32 convolution view to caller file identity, not whole-model parity."""
    return _verify_profile_gguf_operand(
        profile, inventory, manifest, pq2_blocks, reader=reader, source_name=source_name, source_values=source_values,
        expected_source_sha256=expected_source_sha256, expected_file_bytes=expected_file_bytes,
        expected_file_sha256=expected_file_sha256, maximum_file_bytes=maximum_file_bytes, maximum_seconds=maximum_seconds,
        kind="convolution", maximum_bytes=maximum_bytes,
    )


def verify_profile_gguf_tensor_table(
    profile, inventory: dict, manifest: dict, pq2_blocks, *, reader, source_values: dict,
    expected_source_sha256: dict, expected_file_bytes: int, expected_file_sha256: str,
    maximum_bytes: int = 64 * 1024**2, maximum_file_bytes: int = 16 * 1024**2, maximum_seconds: float = 30,
    model_owner=None,
):
    """Verify all bounded tensor ranges, not source files, model metadata or runtime approval."""
    from embedded_jev.model_profile import SHA256

    if (
        type(maximum_bytes) is not int or not 0 < maximum_bytes <= 64 * 1024**2
        or type(maximum_file_bytes) is not int or not 0 < maximum_file_bytes <= 16 * 1024**2
        or type(expected_file_bytes) is not int or not 0 < expected_file_bytes <= maximum_file_bytes
        or not isinstance(expected_file_sha256, str) or SHA256.fullmatch(expected_file_sha256) is None
        or type(maximum_seconds) not in (int, float) or not 0 < maximum_seconds <= 600
        or not isinstance(source_values, dict) or not isinstance(expected_source_sha256, dict)
    ):
        raise ValueError("invalid complete tensor-table identity or budget")
    deadline = time.monotonic() + maximum_seconds
    source_values = source_values.copy()
    _verified_owner_descriptor(model_owner, profile, manifest, expected_file_bytes, expected_file_sha256)
    plan = profile_model_conversion_plan(profile, inventory, manifest, pq2_blocks)
    pins = expected_source_sha256.copy()
    total, kinds = _profile_gguf_table_layout(plan, pq2_blocks, source_values, pins, expected_file_bytes, maximum_bytes)
    sources = {name: values.copy() for name, values in source_values.items()}
    blocks = np.array(pq2_blocks, copy=True)
    blocks.setflags(write=False)
    for name, values in sources.items():
        if hashlib.sha256(memoryview(values)).hexdigest() != pins[name]:
            raise ValueError("complete tensor-table source identity mismatch")
        values.setflags(write=False)
    import gguf

    if type(reader) is not gguf.GGUFReader or len(reader.tensors) != len(plan["tensors"]) or {item.name for item in reader.tensors} != {item["name"] for item in plan["tensors"]}:
        raise ValueError("complete tensor-table reader set mismatch")
    receipts = []
    for item in plan["tensors"]:
        name = item["source_name"]
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("complete tensor-table deadline exceeded")
        frozen = item["type"] == "PQ2_0"
        kind = "projection" if frozen else kinds[name]
        receipts.append(_verify_profile_gguf_operand(
            profile, inventory, manifest, blocks, reader=reader, source_name=name,
            source_values=blocks if frozen else sources[name], expected_source_sha256=manifest["payload_sha256"] if frozen else pins[name],
            expected_file_bytes=expected_file_bytes, expected_file_sha256=expected_file_sha256,
            maximum_file_bytes=maximum_file_bytes, maximum_seconds=remaining,
            maximum_bytes=64 * 1024 if kind == "vector" else 64 * 1024**2 if frozen else 16 * 1024**2, kind=kind,
            model_owner=model_owner,
        ))
    if time.monotonic() > deadline:
        raise ValueError("complete tensor-table deadline exceeded")
    return MappingProxyType({"profile_id": profile.profile_id, "profile_sha256": profile.sha256,
                            "converted_file_sha256": expected_file_sha256, "converted_file_bytes": expected_file_bytes,
                            "tensor_count": len(receipts), "operand_budget_bytes": total, "tensors": tuple(receipts),
                            "model_file_access": "verified_owner_descriptor" if model_owner is not None else "reader_filename",
                            "tensor_table_verification": "all_declared_ranges_match_caller_operands_and_frozen_payload",
                            "source_file_verification": "not_checked", "model_metadata_verification": "not_checked",
                            "gguf_reader_verification": "not_checked", "runtime_approval": "not_granted_by_tensor_table_verification"})


def _profile_gguf_table_layout(plan, pq2_blocks, source_values, pins, expected_file_bytes, maximum_bytes):
    from embedded_jev.model_profile import SHA256

    dense = [item for item in plan["tensors"] if item["type"] != "PQ2_0"]
    names = {item["source_name"] for item in dense}
    if set(source_values) != names or set(pins) != names:
        raise ValueError("complete tensor-table source set mismatch")
    total = 3 * np.asarray(pq2_blocks).nbytes
    payload_bytes = np.asarray(pq2_blocks).nbytes
    kinds = {}
    for item in dense:
        name = item["source_name"]
        values = source_values[name]
        dtype = {"BF16": np.dtype("<u2"), "F32": np.dtype("<f4")}.get(item["source_dtype"])
        size = int(np.prod(item["source_shape"])) * (2 if item["type"] == "BF16" else 4)
        kind = "matrix" if item["type"] == "BF16" else "vector" if len(item["source_shape"]) == 1 else "convolution"
        if (
            dtype is None or not isinstance(values, np.ndarray) or values.dtype != dtype
            or values.shape != tuple(item["source_shape"]) or not values.flags.c_contiguous
            or not isinstance(pins[name], str) or SHA256.fullmatch(pins[name]) is None
            or values.nbytes + 2 * size > (64 * 1024 if kind == "vector" else 16 * 1024**2)
            or (kind == "convolution" and (len(item["source_shape"]) != 3 or item["source_shape"][1] != 1 or item["source_shape"][2] <= 1))
        ):
            raise ValueError("complete tensor-table operand layout, pin or budget mismatch")
        total += values.nbytes + 2 * size
        payload_bytes += size
        kinds[name] = kind
    if payload_bytes > expected_file_bytes:
        raise ValueError("complete tensor-table payloads exceed declared file extent")
    if total > maximum_bytes:
        raise ValueError("complete tensor-table operands exceed aggregate byte budget")
    return total, kinds


def mapped_profile_safetensors_bf16_shard(
    profile, inventory: dict, manifest: dict, pq2_blocks, *, source_path, source_shard: str,
    expected_file_bytes: int, expected_file_sha256: str, maximum_file_bytes: int = 16 * 1024**2,
    maximum_seconds: float = 30,
):
    """Own one verified shard and serial borrowed matrix contexts under a shared deadline."""
    from embedded_jev.streamed_text import bounded_native_descriptor_record

    return _mapped_profile_safetensors_bf16_shard(
        profile, inventory, manifest, pq2_blocks, source_path=source_path, source_shard=source_shard,
        expected_file_bytes=expected_file_bytes, expected_file_sha256=expected_file_sha256,
        maximum_file_bytes=maximum_file_bytes, maximum_seconds=maximum_seconds,
        file_ceiling=16 * 1024**2, operand_ceiling=16 * 1024**2,
        descriptor_record=bounded_native_descriptor_record, policy_version=1,
    )


def mapped_profile_safetensors_bf16_shard_v2(
    profile, inventory: dict, manifest: dict, pq2_blocks, *, source_path, source_shard: str,
    expected_file_bytes: int, expected_file_sha256: str, maximum_file_bytes: int = 16 * 1024**2,
    maximum_seconds: float = 30,
):
    """Explicit v2 shard policy; larger file and child budgets require caller opt-in."""
    from embedded_jev.streamed_text import bounded_source_descriptor_record_v2

    return _mapped_profile_safetensors_bf16_shard(
        profile, inventory, manifest, pq2_blocks, source_path=source_path, source_shard=source_shard,
        expected_file_bytes=expected_file_bytes, expected_file_sha256=expected_file_sha256,
        maximum_file_bytes=maximum_file_bytes, maximum_seconds=maximum_seconds,
        file_ceiling=6 * 1024**3, operand_ceiling=2 * 1024**3,
        descriptor_record=bounded_source_descriptor_record_v2, policy_version=2,
    )


def mapped_profile_safetensors_conversion_shard_v2(
    profile, inventory: dict, manifest: dict, pq2_blocks, *, source_path, source_shard: str,
    expected_file_bytes: int, expected_file_sha256: str, maximum_file_bytes: int = 16 * 1024**2,
    maximum_seconds: float = 30, include_frozen_source: bool = False,
):
    """Own serial conversion sources; frozen provenance requires explicit opt-in, never refitting."""
    from embedded_jev.streamed_text import bounded_source_descriptor_record_v2

    return _mapped_profile_safetensors_bf16_shard(
        profile, inventory, manifest, pq2_blocks, source_path=source_path, source_shard=source_shard,
        expected_file_bytes=expected_file_bytes, expected_file_sha256=expected_file_sha256,
        maximum_file_bytes=maximum_file_bytes, maximum_seconds=maximum_seconds,
        file_ceiling=6 * 1024**3, operand_ceiling=2 * 1024**3,
        descriptor_record=bounded_source_descriptor_record_v2, policy_version=2, allow_auxiliary=True,
        include_frozen_source=include_frozen_source,
    )


@contextmanager
def _mapped_profile_safetensors_bf16_shard(
    profile, inventory: dict, manifest: dict, pq2_blocks, *, source_path, source_shard,
    expected_file_bytes, expected_file_sha256, maximum_file_bytes, maximum_seconds,
    file_ceiling, operand_ceiling, descriptor_record, policy_version, allow_auxiliary=False, include_frozen_source=False,
):
    from embedded_jev.inventory import _json_object, parse_safetensors_header
    from embedded_jev.model_profile import SHA256

    if (
        type(include_frozen_source) is not bool
        or type(maximum_file_bytes) is not int or not 0 < maximum_file_bytes <= file_ceiling
        or type(expected_file_bytes) is not int or not 0 < expected_file_bytes <= maximum_file_bytes
        or not isinstance(expected_file_sha256, str) or SHA256.fullmatch(expected_file_sha256) is None
        or type(maximum_seconds) not in (int, float) or not 0 < maximum_seconds <= 600
    ):
        raise ValueError("invalid mapped source identity or budget")
    deadline = time.monotonic() + maximum_seconds
    plan = profile_model_conversion_plan(profile, inventory, manifest, pq2_blocks)
    frozen_source_name, frozen_source_pin = manifest["tensor"], manifest["source_tensor_sha256"]
    frozen_payload_pin = plan["projection_payload_sha256"]
    frozen_layout = (np.asarray(pq2_blocks).shape, np.asarray(pq2_blocks).dtype) if include_frozen_source else None
    if source_shard not in profile.packaging["shards"]:
        raise ValueError("mapped source shard is not declared")
    source_shards = {item["name"]: item["shard"] for item in inventory["tensors"]}
    pin = profile.files[source_shard]
    if (
        pin.bytes != expected_file_bytes or pin.sha256 != expected_file_sha256
        or pin.header_bytes is None or not 8 < pin.header_bytes <= (1 << 20) + 8 or pin.header_bytes >= expected_file_bytes
        or pin.header_sha256 is None
    ):
        raise ValueError("mapped source file pin or operand budget mismatch")
    path = Path(source_path).absolute()
    policy_receipt = {} if policy_version == 1 else {"source_policy_version": policy_version,
                                                   "source_file_cap_bytes": file_ceiling,
                                                   "source_operand_cap_bytes": operand_ceiling}
    if allow_auxiliary:
        policy_receipt.update(source_operand_policy="bf16_matrices_and_bf16_f32_auxiliaries", source_auxiliary_cap_bytes=1 << 20)
    if include_frozen_source:
        policy_receipt.update(frozen_source_policy="manifest_bound_source_only_no_refitting", frozen_payload_sha256=frozen_payload_pin)

    def identity(value):
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb", buffering=0) as source:
        descriptor = source.fileno()
        before = identity(os.fstat(descriptor))
        active, live, failed = [], True, False

        def check_identity():
            if not live:
                raise ValueError("mapped source shard owner is closed")
            if time.monotonic() > deadline:
                raise ValueError("mapped source deadline exceeded")
            if identity(os.fstat(descriptor)) != before or identity(path.lstat()) != before:
                raise ValueError("mapped source file identity changed")

        def verify_file():
            record = descriptor_record(descriptor, maximum_bytes=expected_file_bytes, deadline=deadline)
            if record["bytes"] != expected_file_bytes or record["sha256"] != expected_file_sha256:
                raise ValueError("mapped source file identity changed")
            check_identity()

        verify_file()
        prefix = bytearray()
        while len(prefix) < pin.header_bytes:
            if time.monotonic() > deadline:
                raise ValueError("mapped source deadline exceeded")
            chunk = os.pread(descriptor, min(65536, pin.header_bytes - len(prefix)), len(prefix))
            if not chunk:
                raise ValueError("mapped source header truncated")
            prefix.extend(chunk)
        if hashlib.sha256(prefix).hexdigest() != pin.header_sha256:
            raise ValueError("mapped source header identity mismatch")
        headers = {item.name: item for item in parse_safetensors_header(prefix, expected_file_bytes)}
        offsets = _json_object(prefix[8:], source_shard)
        check_identity()

        @contextmanager
        def matrix(source_name: str, expected_source_sha256: str, *, maximum_bytes: int = 16 * 1024**2):
            nonlocal failed
            check_identity()
            if active:
                raise ValueError("mapped source shard already has a live matrix")
            if (
                type(maximum_bytes) is not int or not 0 < maximum_bytes <= operand_ceiling
                or not isinstance(expected_source_sha256, str) or SHA256.fullmatch(expected_source_sha256) is None
            ):
                raise ValueError("invalid mapped source identity or budget")
            entry = next((item for item in plan["tensors"] if item["source_name"] == source_name), None)
            matrix_entry = entry is not None and entry["type"] == "BF16" and entry["source_dtype"] == "BF16" and len(entry["source_shape"]) == 2
            auxiliary_entry = (allow_auxiliary and entry is not None and entry["type"] == "F32"
                               and entry["arithmetic_dtype"] == "F32" and entry["source_dtype"] in ("BF16", "F32")
                               and (len(entry["source_shape"]) == 1 or (len(entry["source_shape"]) == 3 and source_name.endswith("conv1d.weight"))))
            frozen_entry = (include_frozen_source and entry is not None and entry["type"] == "PQ2_0"
                            and entry["source_dtype"] == "BF16" and len(entry["source_shape"]) == 2
                            and source_name == frozen_source_name)
            if not matrix_entry and not auxiliary_entry and not frozen_entry:
                raise ValueError("mapped source requires an unfrozen BF16 matrix")
            if frozen_entry and expected_source_sha256 != frozen_source_pin:
                raise ValueError("frozen source digest differs from manifest")
            if source_shards.get(source_name) != source_shard:
                raise ValueError("mapped source matrix belongs to another shard")
            shape = tuple(entry["source_shape"])
            dtype = np.dtype("<f4" if entry["source_dtype"] == "F32" else "<u2")
            storage_bytes = int(np.prod(shape)) * dtype.itemsize
            if storage_bytes > maximum_bytes or (auxiliary_entry and storage_bytes > 1 << 20):
                raise ValueError("mapped source file pin or operand budget mismatch")
            selected = headers.get(source_name)
            if selected is None or selected.dtype != entry["source_dtype"] or selected.shape != shape or selected.storage_bytes != storage_bytes:
                raise ValueError("mapped source header geometry or dtype mismatch")
            offset = len(prefix) + offsets[source_name]["data_offsets"][0]
            mapped = None
            try:
                digest = hashlib.sha256()
                position = 0
                while position < storage_bytes:
                    check_identity()
                    chunk = os.pread(descriptor, min(65536, storage_bytes - position), offset + position)
                    if not chunk:
                        raise ValueError("mapped source tensor range truncated")
                    digest.update(chunk)
                    position += len(chunk)
                if digest.hexdigest() != expected_source_sha256:
                    raise ValueError("mapped source tensor identity mismatch")
                mapped = np.memmap(source, mode="r", dtype=dtype, offset=offset, shape=shape)
                active.append(mapped)
                check_identity()
                layout_receipt = {} if not allow_auxiliary else {"source_dtype": entry["source_dtype"], "source_shape": shape,
                                                               "source_operand_kind": "frozen_source" if frozen_entry else "auxiliary" if auxiliary_entry else "matrix"}
                if frozen_entry:
                    layout_receipt.update(frozen_source_verification="manifest_source_digest_and_profile_pinned_range_entry_and_exit_checks_required",
                                          source_to_frozen_quantization_verification="not_checked")
                yield mapped, MappingProxyType({**policy_receipt, **layout_receipt, "profile_id": profile.profile_id, "profile_sha256": profile.sha256,
                                               "source_name": source_name, "source_sha256": expected_source_sha256,
                                               "source_file_sha256": expected_file_sha256, "source_file_bytes": expected_file_bytes,
                                               "source_file_identity": before, "source_header_sha256": pin.header_sha256,
                                               "source_range_offset": offset, "source_range_bytes": storage_bytes,
                                               "source_file_verification": "profile_pinned_mapped_range_entry_and_exit_checks_required",
                                               "pin_issuer_verification": "not_checked", "runtime_approval": "not_granted_by_source_mapping"})
                check_identity()
                if mapped._mmap.closed:
                    raise ValueError("mapped source matrix was closed by its consumer")
            except BaseException:
                failed = True
                raise
            finally:
                if mapped is not None:
                    mapped._mmap.close()
                active.clear()

        try:
            yield matrix, MappingProxyType({**policy_receipt, "profile_id": profile.profile_id, "profile_sha256": profile.sha256,
                                           "source_shard": source_shard, "source_file_sha256": expected_file_sha256,
                                           "source_file_bytes": expected_file_bytes, "source_file_identity": before,
                                           "source_file_verification": "profile_pinned_shard_entry_and_exit_checks_required",
                                           "runtime_approval": "not_granted_by_source_shard"})
            if failed or active:
                raise ValueError("mapped source shard has failed or unfinished matrix consumers")
            if include_frozen_source:
                frozen_payload = np.asarray(pq2_blocks)
                if (not frozen_payload.flags.c_contiguous or (frozen_payload.shape, frozen_payload.dtype) != frozen_layout
                        or hashlib.sha256(memoryview(frozen_payload)).hexdigest() != frozen_payload_pin):
                    raise ValueError("frozen payload changed during source ownership")
            verify_file()
        finally:
            live = False
            for mapped in active:
                mapped._mmap.close()
            active.clear()


@contextmanager
def mapped_profile_safetensors_bf16_matrix(
    profile, inventory: dict, manifest: dict, pq2_blocks, *, source_path, source_name: str,
    expected_source_sha256: str, expected_file_bytes: int, expected_file_sha256: str,
    maximum_bytes: int = 16 * 1024**2, maximum_file_bytes: int = 16 * 1024**2,
    maximum_seconds: float = 30,
):
    """Own a verified read-only source mapping; finish all borrowed views before context exit."""
    from embedded_jev.model_profile import SHA256

    if (
        type(maximum_bytes) is not int or not 0 < maximum_bytes <= 16 * 1024**2
        or type(maximum_file_bytes) is not int or not 0 < maximum_file_bytes <= 16 * 1024**2
        or type(expected_file_bytes) is not int or not 0 < expected_file_bytes <= maximum_file_bytes
        or any(not isinstance(value, str) or SHA256.fullmatch(value) is None for value in (expected_source_sha256, expected_file_sha256))
        or type(maximum_seconds) not in (int, float) or not 0 < maximum_seconds <= 600
    ):
        raise ValueError("invalid mapped source identity or budget")
    deadline = time.monotonic() + maximum_seconds
    plan = profile_model_conversion_plan(profile, inventory, manifest, pq2_blocks)
    entry = next((item for item in plan["tensors"] if item["source_name"] == source_name), None)
    if entry is None or entry["type"] != "BF16" or entry["source_dtype"] != "BF16" or len(entry["source_shape"]) != 2:
        raise ValueError("mapped source requires an unfrozen BF16 matrix")
    shape = tuple(entry["source_shape"])
    storage_bytes = shape[0] * shape[1] * 2
    tensor = next(item for item in inventory["tensors"] if item["name"] == source_name)
    pin = profile.files[tensor["shard"]]
    if (
        storage_bytes > maximum_bytes or pin.bytes != expected_file_bytes or pin.sha256 != expected_file_sha256
        or pin.header_bytes is None or not 8 < pin.header_bytes <= (1 << 20) + 8 or pin.header_bytes >= expected_file_bytes
        or pin.header_sha256 is None
    ):
        raise ValueError("mapped source file pin or operand budget mismatch")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ValueError("mapped source deadline exceeded")
    with mapped_profile_safetensors_bf16_shard(profile, inventory, manifest, pq2_blocks,
             source_path=source_path, source_shard=tensor["shard"], expected_file_bytes=expected_file_bytes,
             expected_file_sha256=expected_file_sha256, maximum_file_bytes=maximum_file_bytes,
             maximum_seconds=remaining) as (matrix, _):
        with matrix(source_name, expected_source_sha256, maximum_bytes=maximum_bytes) as result:
            yield result


def verify_profile_safetensors_operand(
    profile, inventory: dict, manifest: dict, pq2_blocks, *, source_path, source_name: str,
    source_values: np.ndarray, expected_source_sha256: str, expected_file_bytes: int, expected_file_sha256: str,
    maximum_bytes: int = 16 * 1024**2, maximum_file_bytes: int = 16 * 1024**2, maximum_seconds: float = 30,
):
    """Match an unfrozen operand to a profile-pinned source range, not issuer identity or conversion quality."""
    return _verify_profile_safetensors_operand(
        profile, inventory, manifest, pq2_blocks, source_path=source_path, source_name=source_name,
        source_values=source_values, expected_source_sha256=expected_source_sha256,
        expected_file_bytes=expected_file_bytes, expected_file_sha256=expected_file_sha256,
        maximum_bytes=maximum_bytes, maximum_file_bytes=maximum_file_bytes, maximum_seconds=maximum_seconds,
    )


def verify_profile_frozen_source_tensor(
    profile, inventory: dict, manifest: dict, pq2_blocks, *, source_path, source_values: np.ndarray,
    expected_file_bytes: int, expected_file_sha256: str, maximum_bytes: int = 16 * 1024**2,
    maximum_file_bytes: int = 16 * 1024**2, maximum_seconds: float = 30,
):
    """Verify frozen source identity only; never quantize, refit or compare dense weights to frozen codes."""
    if not isinstance(manifest, dict) or not {"tensor", "source_tensor_sha256", "payload_sha256"} <= manifest.keys():
        raise ValueError("source operand frozen manifest is incomplete or invalid")
    receipt = _verify_profile_safetensors_operand(
        profile, inventory, manifest, pq2_blocks, source_path=source_path, source_name=manifest["tensor"],
        source_values=source_values, expected_source_sha256=manifest["source_tensor_sha256"],
        expected_file_bytes=expected_file_bytes, expected_file_sha256=expected_file_sha256,
        maximum_bytes=maximum_bytes, maximum_file_bytes=maximum_file_bytes, maximum_seconds=maximum_seconds, frozen=True,
    )
    return MappingProxyType({**receipt, "frozen_payload_sha256": manifest["payload_sha256"],
                            "frozen_source_verification": "manifest_source_digest_matches_profile_pinned_source_range",
                            "source_to_frozen_quantization_verification": "not_checked",
                            "runtime_approval": "not_granted_by_frozen_source_verification"})


def _verify_profile_safetensors_operand(
    profile, inventory: dict, manifest: dict, pq2_blocks, *, source_path, source_name: str,
    source_values: np.ndarray, expected_source_sha256: str, expected_file_bytes: int, expected_file_sha256: str,
    maximum_bytes: int = 16 * 1024**2, maximum_file_bytes: int = 16 * 1024**2, maximum_seconds: float = 30, frozen=False,
):
    from embedded_jev.inventory import _json_object, parse_safetensors_header
    from embedded_jev.model_profile import SHA256
    from embedded_jev.streamed_text import bounded_native_descriptor_record

    if (
        type(maximum_bytes) is not int or not 0 < maximum_bytes <= 16 * 1024**2
        or type(maximum_file_bytes) is not int or not 0 < maximum_file_bytes <= 16 * 1024**2
        or type(expected_file_bytes) is not int or not 0 < expected_file_bytes <= maximum_file_bytes
        or any(not isinstance(value, str) or SHA256.fullmatch(value) is None for value in (expected_file_sha256, expected_source_sha256))
        or type(maximum_seconds) not in (int, float) or not 0 < maximum_seconds <= 600
    ):
        raise ValueError("invalid source operand identity or budget")
    deadline = time.monotonic() + maximum_seconds
    plan = profile_model_conversion_plan(profile, inventory, manifest, pq2_blocks)
    entry = next((item for item in plan["tensors"] if item["source_name"] == source_name), None)
    if entry is None or (entry["type"] == "PQ2_0") != frozen:
        raise ValueError("source operand gate requires the declared frozen or unfrozen tensor kind")
    dtype = {"BF16": np.dtype("<u2"), "F32": np.dtype("<f4")}.get(entry["source_dtype"])
    if (
        dtype is None or not isinstance(source_values, np.ndarray) or source_values.dtype != dtype
        or source_values.shape != tuple(entry["source_shape"]) or not source_values.flags.c_contiguous
        or 3 * source_values.nbytes > maximum_bytes
    ):
        raise ValueError("source operand layout or byte budget mismatch")
    tensor = next(item for item in inventory["tensors"] if item["name"] == source_name)
    pin = profile.files[tensor["shard"]]
    if (
        pin.bytes != expected_file_bytes or pin.sha256 != expected_file_sha256
        or pin.header_bytes is None or not 8 < pin.header_bytes <= (1 << 20) + 8 or pin.header_bytes >= expected_file_bytes
        or pin.header_sha256 is None
    ):
        raise ValueError("source operand file pin differs from profile")
    values = source_values.copy()
    values.setflags(write=False)
    if hashlib.sha256(memoryview(values)).hexdigest() != expected_source_sha256:
        raise ValueError("source operand array identity mismatch")
    wanted = values.tobytes()
    with os.fdopen(os.open(source_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb", buffering=0) as source:
        descriptor = source.fileno()
        before_stat = os.fstat(descriptor)
        def identity(value):
            return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

        before = bounded_native_descriptor_record(descriptor, maximum_bytes=expected_file_bytes, deadline=deadline)
        if before["bytes"] != expected_file_bytes or before["sha256"] != expected_file_sha256:
            raise ValueError("source operand file identity mismatch")
        prefix = bytearray()
        while len(prefix) < pin.header_bytes:
            if time.monotonic() > deadline:
                raise ValueError("source operand deadline exceeded")
            chunk = os.pread(descriptor, min(65536, pin.header_bytes - len(prefix)), len(prefix))
            if not chunk:
                raise ValueError("source operand header truncated")
            prefix.extend(chunk)
        prefix = bytes(prefix)
        if hashlib.sha256(prefix).hexdigest() != pin.header_sha256:
            raise ValueError("source operand header identity mismatch")
        headers = parse_safetensors_header(prefix, expected_file_bytes)
        selected = next((item for item in headers if item.name == source_name), None)
        if selected is None or selected.shape != values.shape or selected.dtype != entry["source_dtype"] or selected.storage_bytes != len(wanted):
            raise ValueError("source operand header geometry or dtype mismatch")
        offsets = _json_object(prefix[8:], tensor["shard"])[source_name]["data_offsets"]
        offset = len(prefix) + offsets[0]
        position = 0
        while position < len(wanted):
            if time.monotonic() > deadline:
                raise ValueError("source operand deadline exceeded")
            size = min(65536, len(wanted) - position)
            if os.pread(descriptor, size, offset + position) != wanted[position:position + size]:
                raise ValueError("source operand file range byte mismatch")
            position += size
        if (
            bounded_native_descriptor_record(descriptor, maximum_bytes=expected_file_bytes, deadline=deadline) != before
            or identity(before_stat) != identity(os.fstat(descriptor)) or identity(before_stat) != identity(Path(source_path).lstat())
        ):
            raise ValueError("source operand file changed during verification")
    return MappingProxyType({"profile_id": profile.profile_id, "profile_sha256": profile.sha256,
                            "source_name": source_name, "source_dtype": entry["source_dtype"], "source_shape": tuple(values.shape),
                            "source_sha256": expected_source_sha256, "source_file_sha256": expected_file_sha256,
                            "source_file_bytes": expected_file_bytes, "source_header_sha256": pin.header_sha256,
                            "source_file_identity": identity(before_stat),
                            "source_range_offset": offset, "source_range_bytes": len(wanted),
                            "source_file_verification": "profile_pinned_shard_and_header_range_match_caller_operand",
                            "pin_issuer_verification": "not_checked", "converted_file_verification": "not_checked",
                            "runtime_approval": "not_granted_by_source_operand_verification"})


def verify_profile_safetensors_tensor_table(
    profile, inventory: dict, manifest: dict, pq2_blocks: np.ndarray, *, source_paths: dict, source_values: dict,
    expected_source_sha256: dict, maximum_bytes: int = 64 * 1024**2,
    maximum_file_bytes: int = 16 * 1024**2, maximum_seconds: float = 30,
):
    """Verify every selected source identity, not conversion, quantization or an atomic source snapshot."""
    if type(maximum_seconds) not in (int, float) or not 0 < maximum_seconds <= 600:
        raise ValueError("invalid source table deadline")
    deadline = time.monotonic() + maximum_seconds
    plan, sources, pins, paths, blocks, total = _profile_source_table_snapshot(
        profile, inventory, manifest, pq2_blocks, source_paths=source_paths, source_values=source_values,
        expected_source_sha256=expected_source_sha256, maximum_bytes=maximum_bytes, maximum_file_bytes=maximum_file_bytes,
    )
    tensors = {entry["name"]: entry for entry in inventory["tensors"]}
    shards = set(paths)

    def remaining():
        duration = deadline - time.monotonic()
        if duration <= 0:
            raise ValueError("source table deadline exceeded")
        return duration

    receipts, identities = [], {}
    for entry in plan["tensors"]:
        name = entry["source_name"]
        shard = tensors[name]["shard"]
        pin = profile.files[shard]
        arguments = {"source_path": paths[shard], "source_values": sources[name], "expected_file_bytes": pin.bytes,
                     "expected_file_sha256": pin.sha256, "maximum_file_bytes": maximum_file_bytes, "maximum_seconds": remaining()}
        if entry["type"] == "PQ2_0":
            receipt = verify_profile_frozen_source_tensor(profile, inventory, manifest, blocks, **arguments)
        else:
            receipt = verify_profile_safetensors_operand(profile, inventory, manifest, blocks, source_name=name, expected_source_sha256=pins[name], **arguments)
        if identities.setdefault(shard, receipt["source_file_identity"]) != receipt["source_file_identity"]:
            raise ValueError("source table shard identity changed between operands")
        receipts.append(receipt)
    _verify_profile_source_files(profile, paths, identities, deadline=deadline)
    remaining()
    return MappingProxyType({"profile_id": profile.profile_id, "profile_sha256": profile.sha256,
                            "tensor_count": len(receipts), "source_shard_count": len(shards), "operand_budget_bytes": total,
                            "source_file_bytes": sum(profile.files[name].bytes for name in shards), "tensors": tuple(receipts),
                            "source_table_verification": "all_selected_source_ranges_match_profile_pins_and_caller_operands",
                            "source_to_frozen_quantization_verification": "not_checked", "converted_file_verification": "not_checked",
                            "pin_issuer_verification": "not_checked", "cross_file_atomic_snapshot": "not_guaranteed",
                            "runtime_approval": "not_granted_by_source_table_verification"})


def _verify_profile_source_files(profile, paths: dict, identities: dict, *, deadline: float):
    from embedded_jev.streamed_text import bounded_native_descriptor_record

    def identity(value):
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

    for shard in sorted(paths):
        if time.monotonic() > deadline:
            raise ValueError("source file recheck deadline exceeded")
        with os.fdopen(os.open(paths[shard], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb", buffering=0) as source:
            descriptor = source.fileno()
            pin = profile.files[shard]
            record = bounded_native_descriptor_record(descriptor, maximum_bytes=pin.bytes, deadline=deadline)
            if (
                record["bytes"] != pin.bytes or record["sha256"] != pin.sha256
                or identity(os.fstat(descriptor)) != identities[shard] or identity(Path(paths[shard]).lstat()) != identities[shard]
            ):
                raise ValueError("source table shard identity changed after operands")
    if time.monotonic() > deadline:
        raise ValueError("source file recheck deadline exceeded")


def _profile_source_table_snapshot(
    profile, inventory: dict, manifest: dict, pq2_blocks, *, source_paths: dict, source_values: dict,
    expected_source_sha256: dict, maximum_bytes: int, maximum_file_bytes: int,
):
    from embedded_jev.model_profile import SHA256

    if (
        type(maximum_bytes) is not int or not 0 < maximum_bytes <= 64 * 1024**2
        or type(maximum_file_bytes) is not int or not 0 < maximum_file_bytes <= 16 * 1024**2
        or not isinstance(pq2_blocks, np.ndarray) or pq2_blocks.dtype != np.dtype("u1") or not pq2_blocks.flags.c_contiguous
    ):
        raise ValueError("invalid source table payload layout or budget")
    plan = profile_model_conversion_plan(profile, inventory, manifest, pq2_blocks)
    names = {entry["source_name"] for entry in plan["tensors"]}
    tensors = {entry["name"]: entry for entry in inventory["tensors"]}
    shards = {tensors[name]["shard"] for name in names}
    if (
        not isinstance(source_values, dict) or set(source_values) != names
        or not isinstance(expected_source_sha256, dict) or set(expected_source_sha256) != names
        or not isinstance(source_paths, dict) or set(source_paths) != shards
    ):
        raise ValueError("source table requires exact selected tensor and shard maps")
    pins, paths = dict(expected_source_sha256), dict(source_paths)
    if (
        any(not isinstance(value, str) or SHA256.fullmatch(value) is None for value in pins.values())
        or pins[manifest["tensor"]] != manifest["source_tensor_sha256"]
        or sum(profile.files[name].bytes for name in shards) > maximum_file_bytes
        or any(profile.files[name].header_bytes is None or profile.files[name].header_sha256 is None for name in shards)
    ):
        raise ValueError("source table source pins or aggregate file budget mismatch")
    sources = dict(source_values)
    total = 3 * pq2_blocks.nbytes
    for entry in plan["tensors"]:
        values = sources[entry["source_name"]]
        if (
            not isinstance(values, np.ndarray) or values.dtype != {"BF16": np.dtype("<u2"), "F32": np.dtype("<f4")}.get(entry["source_dtype"])
            or values.shape != tuple(entry["source_shape"]) or not values.flags.c_contiguous
            or 3 * values.nbytes > 16 * 1024**2
        ):
            raise ValueError("source table operand layout or per-tensor budget mismatch")
        total += 3 * values.nbytes
    if total > maximum_bytes:
        raise ValueError("source table aggregate operand budget exceeded")
    blocks = np.array(pq2_blocks, copy=True)
    blocks.setflags(write=False)
    for name, values in sources.items():
        values = values.copy()
        values.setflags(write=False)
        if hashlib.sha256(memoryview(values)).hexdigest() != pins[name]:
            raise ValueError("source table operand digest mismatch")
        sources[name] = values

    return plan, sources, pins, paths, blocks, total


def verify_profile_gguf_source_tensor_table(
    profile, inventory: dict, manifest: dict, pq2_blocks: np.ndarray, *, reader, source_paths: dict,
    source_values: dict, expected_source_sha256: dict, expected_file_bytes: int, expected_file_sha256: str,
    maximum_bytes: int = 64 * 1024**2, maximum_file_bytes: int = 16 * 1024**2,
    maximum_seconds: float = 30, model_owner=None,
):
    """Bind complete selected source and converted evidence, keeping frozen identity separate from quantization."""
    from embedded_jev.model_profile import SHA256

    if (
        type(maximum_seconds) not in (int, float) or not 0 < maximum_seconds <= 600
        or type(maximum_file_bytes) is not int or not 0 < maximum_file_bytes <= 16 * 1024**2
        or type(expected_file_bytes) is not int or not 0 < expected_file_bytes <= maximum_file_bytes
        or not isinstance(expected_file_sha256, str) or SHA256.fullmatch(expected_file_sha256) is None
    ):
        raise ValueError("invalid source/converted table file identity or deadline")
    deadline = time.monotonic() + maximum_seconds
    _verified_owner_descriptor(model_owner, profile, manifest, expected_file_bytes, expected_file_sha256)
    plan, sources, pins, paths, blocks, total = _profile_source_table_snapshot(
        profile, inventory, manifest, pq2_blocks, source_paths=source_paths, source_values=source_values,
        expected_source_sha256=expected_source_sha256, maximum_bytes=maximum_bytes, maximum_file_bytes=maximum_file_bytes,
    )
    dense = {entry["source_name"]: sources[entry["source_name"]] for entry in plan["tensors"] if entry["type"] != "PQ2_0"}
    dense_pins = {name: pins[name] for name in dense}
    converted_total, _ = _profile_gguf_table_layout(plan, blocks, dense, dense_pins, expected_file_bytes, maximum_bytes)
    total = max(total, converted_total + 3 * sources[manifest["tensor"]].nbytes)
    if total > maximum_bytes:
        raise ValueError("source/converted table aggregate operand budget exceeded")
    import gguf

    if type(reader) is not gguf.GGUFReader or len(reader.tensors) != len(plan["tensors"]) or {item.name for item in reader.tensors} != {item["name"] for item in plan["tensors"]}:
        raise ValueError("complete tensor-table reader set mismatch")
    def remaining():
        duration = deadline - time.monotonic()
        if duration <= 0:
            raise ValueError("source/converted table deadline exceeded")
        return duration

    def source_record():
        return verify_profile_safetensors_tensor_table(
            profile, inventory, manifest, blocks, source_paths=paths, source_values=sources,
            expected_source_sha256=pins, maximum_bytes=maximum_bytes,
            maximum_file_bytes=maximum_file_bytes, maximum_seconds=remaining(),
        )

    source = source_record()
    converted = verify_profile_gguf_tensor_table(
        profile, inventory, manifest, blocks, reader=reader, source_values=dense,
        expected_source_sha256=dense_pins, expected_file_bytes=expected_file_bytes, expected_file_sha256=expected_file_sha256,
        maximum_bytes=maximum_bytes, maximum_file_bytes=maximum_file_bytes, maximum_seconds=remaining(), model_owner=model_owner,
    )
    if source_record() != source:
        raise ValueError("source/converted table source identity changed")
    remaining()
    return MappingProxyType({"profile_id": profile.profile_id, "profile_sha256": profile.sha256,
                            "tensor_count": len(plan["tensors"]), "operand_budget_bytes": total, "source": source, "converted": converted,
                            "source_conversion_table_verification": "selected_source_and_converted_ranges_match_shared_operands_or_frozen_payload",
                            "source_to_frozen_quantization_verification": "not_checked", "source_file_rechecked_after_conversion": True,
                            "cross_file_atomic_snapshot": "not_guaranteed", "pin_issuer_verification": "not_checked",
                            "model_metadata_verification": "not_checked", "runtime_approval": "not_granted_by_source_conversion_table_verification"})


def verify_profile_gguf_source_operand(
    profile, inventory: dict, manifest: dict, pq2_blocks, *, reader, source_path, source_name: str,
    source_values: np.ndarray, expected_source_sha256: str,
    expected_source_file_bytes: int, expected_source_file_sha256: str,
    expected_file_bytes: int, expected_file_sha256: str,
    maximum_bytes: int = 16 * 1024**2, maximum_file_bytes: int = 16 * 1024**2,
    maximum_seconds: float = 30, model_owner=None,
):
    """Bind one source and converted range through one copied operand, not an atomic whole-model snapshot."""
    from embedded_jev.model_profile import SHA256

    if (
        type(maximum_bytes) is not int or not 0 < maximum_bytes <= 16 * 1024**2
        or type(maximum_file_bytes) is not int or not 0 < maximum_file_bytes <= 16 * 1024**2
        or any(type(size) is not int or not 0 < size <= maximum_file_bytes for size in (expected_source_file_bytes, expected_file_bytes))
        or any(not isinstance(value, str) or SHA256.fullmatch(value) is None for value in (expected_source_sha256, expected_source_file_sha256, expected_file_sha256))
        or type(maximum_seconds) not in (int, float) or not 0 < maximum_seconds <= 600
        or not isinstance(source_values, np.ndarray) or not source_values.flags.c_contiguous or 3 * source_values.nbytes > maximum_bytes
    ):
        raise ValueError("invalid combined source operand identity, layout or budget")
    deadline = time.monotonic() + maximum_seconds
    _verified_owner_descriptor(model_owner, profile, manifest, expected_file_bytes, expected_file_sha256)
    plan = profile_model_conversion_plan(profile, inventory, manifest, pq2_blocks)
    entry = next((item for item in plan["tensors"] if item["source_name"] == source_name), None)
    if entry is None or entry["type"] == "PQ2_0":
        raise ValueError("combined source operand requires an unfrozen declared tensor")
    if (
        source_values.dtype != {"BF16": np.dtype("<u2"), "F32": np.dtype("<f4")}.get(entry["source_dtype"])
        or source_values.shape != tuple(entry["source_shape"])
    ):
        raise ValueError("combined source operand dtype or shape mismatch")
    values = source_values.copy()
    values.setflags(write=False)
    if hashlib.sha256(memoryview(values)).hexdigest() != expected_source_sha256:
        raise ValueError("combined source operand array identity mismatch")
    def remaining():
        duration = deadline - time.monotonic()
        if duration <= 0:
            raise ValueError("combined source operand deadline exceeded")
        return duration

    def source_record():
        return verify_profile_safetensors_operand(
            profile, inventory, manifest, pq2_blocks, source_path=source_path, source_name=source_name, source_values=values,
            expected_source_sha256=expected_source_sha256, expected_file_bytes=expected_source_file_bytes,
            expected_file_sha256=expected_source_file_sha256, maximum_bytes=maximum_bytes,
            maximum_file_bytes=maximum_file_bytes, maximum_seconds=remaining(),
        )

    source = source_record()
    kind = "matrix" if entry["type"] == "BF16" else "vector" if len(entry["source_shape"]) == 1 else "convolution"
    converted = _verify_profile_gguf_operand(
        profile, inventory, manifest, pq2_blocks, reader=reader, source_name=source_name, source_values=values,
        expected_source_sha256=expected_source_sha256, expected_file_bytes=expected_file_bytes,
        expected_file_sha256=expected_file_sha256, maximum_file_bytes=maximum_file_bytes,
        maximum_seconds=remaining(), kind=kind, maximum_bytes=min(maximum_bytes, 64 * 1024) if kind == "vector" else maximum_bytes,
        model_owner=model_owner,
    )
    if source_record() != source:
        raise ValueError("combined source operand source identity changed")
    remaining()
    return MappingProxyType({"profile_id": profile.profile_id, "profile_sha256": profile.sha256,
                            "source_sha256": expected_source_sha256, "source": source, "converted": converted,
                            "source_conversion_verification": "pinned_source_range_and_converted_range_match_same_operand",
                            "source_file_rechecked_after_conversion": True, "cross_file_atomic_snapshot": "not_guaranteed",
                            "pin_issuer_verification": "not_checked", "model_metadata_verification": "not_checked",
                            "runtime_approval": "not_granted_by_source_conversion_verification"})


def verify_profile_gguf_policy_metadata(
    profile, inventory: dict, manifest: dict, pq2_blocks, *, reader,
    expected_file_bytes: int, expected_file_sha256: str, maximum_file_bytes: int = 16 * 1024**2,
    maximum_seconds: float = 30, model_owner=None,
):
    """Bind selected policy declarations to file bytes, not tokenizer semantics or runtime approval."""
    import sys

    from embedded_jev.model_profile import SHA256
    from embedded_jev.streamed_text import bounded_native_descriptor_record, bounded_native_file_record

    if (
        type(maximum_file_bytes) is not int or not 0 < maximum_file_bytes <= 16 * 1024**2
        or type(expected_file_bytes) is not int or not 0 < expected_file_bytes <= maximum_file_bytes
        or not isinstance(expected_file_sha256, str) or SHA256.fullmatch(expected_file_sha256) is None
        or type(maximum_seconds) not in (int, float) or not 0 < maximum_seconds <= 600
    ):
        raise ValueError("invalid policy metadata file identity or budget")
    deadline = time.monotonic() + maximum_seconds
    specification = profile_model_specification(profile, inventory, manifest, pq2_blocks)
    descriptor = _verified_owner_descriptor(model_owner, profile, manifest, expected_file_bytes, expected_file_sha256)
    import gguf

    if (
        type(reader) is not gguf.GGUFReader or sys.byteorder != "little"
        or not isinstance(reader.data, np.memmap) or reader.data.dtype != np.dtype("u1") or reader.data.ndim != 1
        or reader.data.mode != "r" or reader.data.flags.writeable or reader.data.offset != 0
        or reader.data.nbytes != expected_file_bytes or len(reader.tensors) != len(specification["tensors"])
        or not 3 <= len(reader.fields) <= MODEL_METADATA_KEYS_V2 + 3
        or not 24 <= reader.data_offset <= min(expected_file_bytes, MODEL_METADATA_BYTES_V2)
    ):
        raise ValueError("policy metadata reader layout or budget mismatch")
    declarations = [
        {"name": name, "kind": "string", "values": [value]} for name, value in {
            "general.architecture": "qwen35", "jev.model.profile_id": profile.profile_id,
            "jev.model.profile_sha256": profile.sha256, "jev.model.source_revision": profile.revision,
            "jev.model.converter_revision": PRISM_SOURCE_REVISION, "jev.bitnet.source_tensor_sha256": specification["source_tensor_sha256"],
            "jev.bitnet.model": "profile-qwen35-single-projection-v2", "jev.bitnet.execution": "group128-a8-fp32-nearest-even-identity-v1",
        }.items()
    ]
    declarations.extend({"name": "qwen35." + name, "kind": "u32", "values": [value]} for name, value in {
        "block_count": specification["layers"], "embedding_length": specification["hidden"], "feed_forward_length": specification["intermediate"],
    }.items())
    declarations.extend(specification["metadata"])
    table_offset = min(int(tensor.field.offset) for tensor in reader.tensors)
    if not 24 <= table_offset <= reader.data_offset:
        raise ValueError("policy metadata tensor table boundary mismatch")
    records = []
    consumed = 0
    address = reader.data.__array_interface__["data"][0]
    kinds = {"string": (8,), "u32": (4,), "f32": (6,), "i32_array": (9, 5)}
    for item in declarations:
        if time.monotonic() > deadline:
            raise ValueError("policy metadata deadline exceeded")
        name, kind, values = item["name"], item["kind"], item["values"]
        encoded_name = name.encode("ascii")
        if kind == "string":
            value = values[0].encode("ascii")
            encoded_value = struct.pack("<Q", len(value)) + value
        elif kind == "i32_array":
            encoded_value = struct.pack("<IQ", 5, len(values)) + struct.pack("<" + "i" * len(values), *values)
        else:
            encoded_value = struct.pack("<I" if kind == "u32" else "<f", values[0])
        wanted = struct.pack("<Q", len(encoded_name)) + encoded_name + struct.pack("<I", kinds[kind][0]) + encoded_value
        consumed += len(wanted)
        if consumed > 8192:
            raise ValueError("policy metadata records exceed byte budget")
        field = reader.fields.get(name)
        if (
            field is None or field.name != name or tuple(field.types) != kinds[kind]
            or not isinstance(field.offset, (int, np.integer)) or isinstance(field.offset, (bool, np.bool_))
            or not 24 <= field.offset < field.offset + len(wanted) <= table_offset
            or not 1 <= len(field.parts) <= 32
        ):
            raise ValueError("policy metadata field type or range mismatch: " + name)
        position = int(field.offset)
        parts = []
        for part in field.parts:
            if (
                not isinstance(part, np.ndarray) or part.ndim != 1 or part.dtype.hasobject or not part.flags.c_contiguous or part.flags.writeable
                or not 0 < part.nbytes <= len(wanted) or position + part.nbytes > int(field.offset) + len(wanted)
                or not np.shares_memory(part, reader.data) or part.__array_interface__["data"][0] != address + position
            ):
                raise ValueError("policy metadata field mapping mismatch: " + name)
            parts.append(part.tobytes())
            position += part.nbytes
        if b"".join(parts) != wanted:
            raise ValueError("policy metadata declaration byte mismatch: " + name)
        records.append((name, int(field.offset), wanted))
    path = Path(reader.data.filename) if descriptor is None else Path(model_owner.model_path)
    def file_record():
        return (bounded_native_file_record(path, maximum_bytes=expected_file_bytes, deadline=deadline) if descriptor is None
                else bounded_native_descriptor_record(descriptor, maximum_bytes=expected_file_bytes, deadline=deadline))

    before = file_record()
    if before["bytes"] != expected_file_bytes or before["sha256"] != expected_file_sha256:
        raise ValueError("policy metadata file identity mismatch")
    retained = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK) if descriptor is None else os.dup(descriptor)
    try:
        if os.pread(retained, 24, 0) != struct.pack("<4sIQQ", b"GGUF", 3, len(reader.tensors), len(reader.fields) - 3):
            raise ValueError("policy metadata header differs from reader")
        cursor = 24
        fixed_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        positions = {}
        def read(size):
            nonlocal cursor
            if time.monotonic() > deadline:
                raise ValueError("policy metadata deadline exceeded")
            if not 0 <= size <= 65536 or cursor + size > table_offset:
                raise ValueError("policy metadata file record range mismatch")
            value = os.pread(retained, size, cursor)
            if len(value) != size:
                raise ValueError("policy metadata file record truncated")
            cursor += size
            return value

        def skip(size):
            nonlocal cursor
            if time.monotonic() > deadline:
                raise ValueError("policy metadata deadline exceeded")
            if not 0 <= size <= table_offset - cursor:
                raise ValueError("policy metadata file record range mismatch")
            cursor += size

        def skip_string():
            skip(struct.unpack("<Q", read(8))[0])

        for _ in range(len(reader.fields) - 3):
            offset = cursor
            name = read(struct.unpack("<Q", read(8))[0]).decode("utf-8")
            if not name or name in positions:
                raise ValueError("policy metadata file key is empty or duplicate")
            positions[name] = offset
            kind = struct.unpack("<I", read(4))[0]
            if kind == 8:
                skip_string()
            elif kind == 9:
                element_kind, count = struct.unpack("<IQ", read(12))
                if element_kind == 8:
                    if count > (table_offset - cursor) // 8:
                        raise ValueError("policy metadata file string array exceeds range")
                    for _ in range(count):
                        skip_string()
                elif element_kind in fixed_sizes:
                    skip(count * fixed_sizes[element_kind])
                else:
                    raise ValueError("policy metadata file array type unsupported")
            elif kind in fixed_sizes:
                skip(fixed_sizes[kind])
            else:
                raise ValueError("policy metadata file value type unsupported")
        if (
            cursor != table_offset or set(positions) != set(reader.fields) - {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}
            or any(positions.get(name) != offset for name, offset, _ in records)
        ):
            raise ValueError("policy metadata file key set or table boundary mismatch")
        for name, offset, wanted in records:
            if time.monotonic() > deadline:
                raise ValueError("policy metadata deadline exceeded")
            if os.pread(retained, len(wanted), offset) != wanted:
                raise ValueError("policy metadata file range differs from declaration: " + name)
    finally:
        os.close(retained)
    if file_record() != before or time.monotonic() > deadline:
        raise ValueError("policy metadata file changed or deadline exceeded")
    return MappingProxyType({"profile_id": profile.profile_id, "profile_sha256": profile.sha256,
                            "converted_file_sha256": expected_file_sha256, "converted_file_bytes": expected_file_bytes,
                            "metadata_names": tuple(name for name, _, _ in records), "metadata_record_bytes": consumed,
                            "policy_metadata_verification": "selected_typed_policy_records_match_declarations_and_file",
                            "model_file_access": "verified_owner_descriptor" if model_owner is not None else "reader_filename",
                            "model_metadata_verification": "selected_policy_records_only", "tokenizer_semantics_verification": "not_checked",
                            "gguf_reader_verification": "not_checked", "runtime_approval": "not_granted_by_policy_metadata_verification"})


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


def native_model_specification(specification: dict, pq2_blocks):
    """Marshal a checked specification, retaining backing arrays for the record's lifetime."""
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
        2, specification["profile_id"].encode("ascii"), specification["profile_sha256"].encode("ascii"), specification["source_revision"].encode("ascii"),
        projection["source_tensor_sha256"].encode("ascii"), specification["layers"], specification["hidden"],
        specification["intermediate"], specification["vocabulary"], table, len(table), metadata, len(metadata),
        NativeProjectionSpecification(2, projection["tensor_name"].encode("ascii"), projection["rows"], projection["columns"],
                                      blocks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), blocks.nbytes),
    )
    record._keepalive = (blocks, table, values, metadata)
    return record


def create_profile_model_policy(native, model_path, profile, inventory: dict, manifest: dict, pq2_blocks):
    """Caller retains the library and frees the policy only after all model buffers and contexts."""
    specification = profile_model_specification(profile, inventory, manifest, pq2_blocks)
    record = native_model_specification(specification, pq2_blocks)
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


@dataclass(frozen=True)
class VerifiedModelPolicy:
    _model_path: str
    handle: ctypes.c_void_p
    overrides: ctypes.c_void_p
    _record: NativeModelSpecification
    verification: MappingProxyType

    @property
    def model_path(self):
        if self.handle.value is None:
            raise ValueError("verified model policy is closed")
        return self._model_path

    @property
    def record(self):
        if self.handle.value is None:
            raise ValueError("verified model policy is closed")
        return self._record


def _verified_owner_descriptor(owner, profile, manifest, expected_bytes, expected_sha256):
    if owner is None:
        return None
    if type(owner) is not VerifiedModelPolicy:
        raise ValueError("tensor verification requires a verified model owner")
    path = owner.model_path
    if (
        not isinstance(owner.record, NativeModelSpecification) or owner.record.abi_version != 2
        or any(owner.verification.get(name) != value for name, value in {
            "profile_id": profile.profile_id, "profile_sha256": profile.sha256,
            "model_bytes": expected_bytes, "model_sha256": expected_sha256,
            "projection_payload_sha256": manifest["payload_sha256"],
        }.items())
    ):
        raise ValueError("tensor verification owner identity mismatch")
    prefix = "/proc/self/fd/"
    if not path.startswith(prefix) or not path[len(prefix):].isdecimal():
        raise ValueError("tensor verification owner has no descriptor path")
    return int(path[len(prefix):])


class _DynamicLibraryInfo(ctypes.Structure):
    _fields_ = [("filename", ctypes.c_char_p), ("base", ctypes.c_void_p),
                ("symbol", ctypes.c_char_p), ("address", ctypes.c_void_p)]


def _model_runtime_library_paths(native, scorer=None):
    locator = ctypes.CDLL(None).dladdr
    locator.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DynamicLibraryInfo)]
    locator.restype = ctypes.c_int
    paths = {}
    symbols = {
        "bridge": (native, "prism_bitnet_cpu_model_override_from_gguf_v2"),
        "ggml_cpu": (native, "ggml_backend_cpu_init"), "ggml_base": (native, "ggml_map_custom2"),
    }
    if scorer is not None:
        symbols.update({
            "scorer": (scorer, "prism_bitnet_test_prefill_policy_v2"),
            "llama": (scorer, "llama_model_load_from_file"),
            "ggml_registry": (scorer, "ggml_backend_init_by_type"),
        })
    for name, (library, symbol) in symbols.items():
        info = _DynamicLibraryInfo()
        address = ctypes.cast(getattr(library, symbol), ctypes.c_void_p)
        if not locator(address, ctypes.byref(info)) or not info.filename:
            raise ValueError("runtime symbol has no defining library")
        paths[name] = Path(os.fsdecode(info.filename)).resolve()
    return paths


def verify_model_runtime_files(
    native, expected_files: dict, *, maximum_bytes: int = 64 * 1024**2, maximum_seconds: float = 30, scorer=None,
) -> dict:
    """Match bridge/dependency files to caller pins, not loaded pages or execution approval."""
    names = ("bridge", "ggml_cpu", "ggml_base") + (() if scorer is None else ("scorer", "llama", "ggml_registry"))
    if (
        not isinstance(expected_files, dict) or set(expected_files) != set(names)
        or type(maximum_bytes) is not int or not 0 < maximum_bytes <= 64 * 1024**2
        or type(maximum_seconds) not in (int, float) or not 0 < maximum_seconds <= 600
    ):
        raise ValueError("invalid runtime file identity or budget")
    for expected in expected_files.values():
        if (
            not isinstance(expected, dict) or set(expected) != {"bytes", "sha256"}
            or type(expected["bytes"]) is not int or expected["bytes"] <= 0
            or not isinstance(expected["sha256"], str) or len(expected["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in expected["sha256"])
        ):
            raise ValueError("invalid runtime file identity or budget")
    if sum(expected["bytes"] for expected in expected_files.values()) > maximum_bytes:
        raise ValueError("runtime files exceed byte budget")
    expected_files = {name: item.copy() for name, item in expected_files.items()}
    deadline = time.monotonic() + maximum_seconds
    locator = native.prism_group_scale_library_path
    locator.argtypes = [ctypes.c_int]
    locator.restype = ctypes.c_char_p
    paths = _model_runtime_library_paths(native, scorer)
    for index, name in enumerate(("ggml_cpu", "ggml_base")):
        raw_path = locator(index)
        if not raw_path:
            raise ValueError("runtime did not identify its loaded dependency")
        if paths[name] != Path(os.fsdecode(raw_path)).resolve():
            raise ValueError("runtime dependency symbol origin mismatch")
    for name, path in paths.items():
        observed = path.stat()
        if not stat.S_ISREG(observed.st_mode) or observed.st_size != expected_files[name]["bytes"]:
            raise ValueError("runtime file size or type mismatch")
    from embedded_jev.streamed_text import bounded_native_file_record

    result = {name: bounded_native_file_record(paths[name], maximum_bytes=expected_files[name]["bytes"], deadline=deadline)
              for name in names}
    for name in names:
        if result[name] != {"path": str(paths[name]), **expected_files[name]}:
            raise ValueError("runtime file identity mismatch")
    return result


@contextmanager
def verified_profile_model_policy(
    native, model_path, profile, inventory: dict, manifest: dict, pq2_blocks, *,
    expected_bytes: int, expected_sha256: str, maximum_bytes: int = 16 * 1024**2,
    maximum_seconds: float = 30, expected_runtime_files: dict | None = None, runtime_scorer=None,
    tensor_reader=None, source_values: dict | None = None, expected_source_sha256: dict | None = None,
    maximum_tensor_bytes: int = 64 * 1024**2,
    verify_policy_metadata: bool = False,
    source_paths: dict | None = None, frozen_source_values: np.ndarray | None = None,
    maximum_source_file_bytes: int = 16 * 1024**2,
    verify_source_files_on_exit: bool = False,
):
    """Own a policy with explicit bounded evidence gates; destroy all native users before exit."""
    specification = profile_model_specification(profile, inventory, manifest, pq2_blocks)
    if (
        type(expected_bytes) is not int or type(maximum_bytes) is not int
        or not 0 < expected_bytes <= maximum_bytes <= specification["maximum_file_bytes"]
        or not isinstance(expected_sha256, str) or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
        or type(maximum_seconds) not in (int, float) or not 0 < maximum_seconds <= 600
    ):
        raise ValueError("invalid verified model identity or resource budget")
    if not hasattr(os, "O_NOFOLLOW") or not os.path.isdir("/proc/self/fd"):
        raise ValueError("verified model policy requires Linux descriptor paths")
    if runtime_scorer is not None and expected_runtime_files is None:
        raise ValueError("scorer runtime verification requires runtime file pins")
    requested_table = any(value is not None for value in (tensor_reader, source_values, expected_source_sha256))
    if type(verify_policy_metadata) is not bool or (verify_policy_metadata and not requested_table):
        raise ValueError("policy metadata verification requires a tensor-table reader and inputs")
    requested_source = source_paths is not None or frozen_source_values is not None
    if type(verify_source_files_on_exit) is not bool or (verify_source_files_on_exit and not requested_source):
        raise ValueError("source file exit verification requires explicit source-table inputs")
    if requested_source and (not requested_table or not isinstance(source_paths, dict) or frozen_source_values is None):
        raise ValueError("source verification requires tensor-table inputs, source paths and frozen source values")
    if requested_table:
        if (
            tensor_reader is None or not isinstance(source_values, dict) or not isinstance(expected_source_sha256, dict)
            or expected_bytes > 16 * 1024**2 or type(maximum_tensor_bytes) is not int or not 0 < maximum_tensor_bytes <= 64 * 1024**2
        ):
            raise ValueError("invalid verified model tensor-table inputs or budget")
        source_values, expected_source_sha256 = source_values.copy(), expected_source_sha256.copy()
    if requested_source:
        source_started = time.monotonic()
        if manifest["tensor"] in source_values or manifest["tensor"] in expected_source_sha256:
            raise ValueError("source-gated model requires unfrozen tensor maps plus separate frozen source values")
        _, source_values, expected_source_sha256, source_paths, _, _ = _profile_source_table_snapshot(
            profile, inventory, manifest, pq2_blocks, source_paths=source_paths,
            source_values={**source_values, manifest["tensor"]: frozen_source_values},
            expected_source_sha256={**expected_source_sha256, manifest["tensor"]: manifest["source_tensor_sha256"]},
            maximum_bytes=maximum_tensor_bytes, maximum_file_bytes=maximum_source_file_bytes,
        )
        if verify_source_files_on_exit:
            source_paths = {name: Path(path).absolute() for name, path in source_paths.items()}
    if expected_runtime_files is not None:
        if not isinstance(expected_runtime_files, dict) or any(not isinstance(item, dict) for item in expected_runtime_files.values()):
            raise ValueError("invalid runtime file identity or budget")
        expected_runtime_files = {name: item.copy() for name, item in expected_runtime_files.items()}
    blocks = np.array(pq2_blocks, copy=True)
    blocks.setflags(write=False)
    with os.fdopen(os.open(model_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb", buffering=0) as model:
        before = os.fstat(model.fileno())
        def identity(value):
            return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_bytes:
            raise ValueError("verified model file size or type mismatch")
        started = source_started if requested_source else time.monotonic()
        digest = hashlib.sha256()
        remaining = expected_bytes
        while remaining:
            if time.monotonic() - started > maximum_seconds:
                raise ValueError("verified model hashing deadline exceeded")
            chunk = model.read(min(65536, remaining))
            if not chunk:
                raise ValueError("verified model file truncated")
            digest.update(chunk)
            remaining -= len(chunk)
        if time.monotonic() - started > maximum_seconds:
            raise ValueError("verified model hashing deadline exceeded")
        if identity(os.fstat(model.fileno())) != identity(before) or digest.hexdigest() != expected_sha256:
            raise ValueError("verified model file identity mismatch")
        descriptor_path = f"/proc/self/fd/{model.fileno()}"
        runtime_files = None if expected_runtime_files is None else verify_model_runtime_files(native, expected_runtime_files, scorer=runtime_scorer)
        free = native.prism_bitnet_cpu_model_policy_free_v2
        free.argtypes = [ctypes.c_void_p]
        free.restype = None
        handle, overrides, specification = create_profile_model_policy(
            native, descriptor_path, profile, inventory, manifest, blocks,
        )
        source_identities = None
        try:
            record = native_model_specification(specification, blocks)
            verification = MappingProxyType({
                "profile_id": profile.profile_id, "profile_sha256": profile.sha256,
                "source_revision": profile.revision, "converter_revision": PRISM_SOURCE_REVISION,
                "model_bytes": expected_bytes, "model_sha256": expected_sha256,
                "projection_payload_sha256": manifest["payload_sha256"],
                "model_file_verification": "sha256_matches_caller_supplied_identity",
                "source_payload_verification": specification["source_payload_verification"],
                "runtime_approval": "not_granted_by_file_verification",
                "runtime_file_verification": "not_checked" if runtime_files is None else "sha256_matches_caller_supplied_runtime_files",
                "runtime_file_scope": "not_checked" if runtime_files is None else "policy_cpu_base" if runtime_scorer is None else "policy_cpu_base_scorer_llama_registry",
                **({name + "_sha256": item["sha256"] for name, item in runtime_files.items()} if runtime_files is not None else {}),
            })
            if requested_table:
                owner = VerifiedModelPolicy(descriptor_path, handle, overrides, record, verification)
                remaining = maximum_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    raise ValueError("verified model tensor-table deadline exceeded")
                if requested_source:
                    evidence = verify_profile_gguf_source_tensor_table(
                        profile, inventory, manifest, blocks, reader=tensor_reader, source_paths=source_paths,
                        source_values=source_values, expected_source_sha256=expected_source_sha256,
                        expected_file_bytes=expected_bytes, expected_file_sha256=expected_sha256,
                        maximum_bytes=maximum_tensor_bytes, maximum_seconds=remaining, model_owner=owner,
                    )
                    table = evidence["converted"]
                    verification = MappingProxyType({**verification, "source_conversion_table": evidence,
                                                     "source_table_verification": evidence["source"]["source_table_verification"],
                                                     "source_conversion_table_verification": evidence["source_conversion_table_verification"],
                                                     "source_file_lifetime_verification": "entry_and_exit_checks_required" if verify_source_files_on_exit else "pre_yield_only"})
                    if verify_source_files_on_exit:
                        tensor_shards = {item["name"]: item["shard"] for item in inventory["tensors"]}
                        source_identities = {tensor_shards[item["source_name"]]: item["source_file_identity"] for item in evidence["source"]["tensors"]}
                else:
                    table = verify_profile_gguf_tensor_table(
                        profile, inventory, manifest, blocks, reader=tensor_reader,
                        source_values=source_values, expected_source_sha256=expected_source_sha256,
                        expected_file_bytes=expected_bytes, expected_file_sha256=expected_sha256,
                        maximum_bytes=maximum_tensor_bytes, maximum_seconds=remaining, model_owner=owner,
                    )
                verification = MappingProxyType({**verification, "tensor_table_verification": table["tensor_table_verification"], "tensor_table": table})
                if verify_policy_metadata:
                    remaining = maximum_seconds - (time.monotonic() - started)
                    if remaining <= 0:
                        raise ValueError("verified model policy metadata deadline exceeded")
                    metadata = verify_profile_gguf_policy_metadata(
                        profile, inventory, manifest, blocks, reader=tensor_reader,
                        expected_file_bytes=expected_bytes, expected_file_sha256=expected_sha256,
                        maximum_seconds=remaining, model_owner=owner,
                    )
                    verification = MappingProxyType({**verification, "policy_metadata_verification": metadata["policy_metadata_verification"], "policy_metadata": metadata})
            if identity(os.fstat(model.fileno())) != identity(before):
                raise ValueError("verified model changed during policy creation")
            if expected_runtime_files is not None:
                verify_model_runtime_files(native, expected_runtime_files, scorer=runtime_scorer)
            if requested_table and time.monotonic() - started > maximum_seconds:
                raise ValueError("verified model tensor-table deadline exceeded")
            yield VerifiedModelPolicy(descriptor_path, handle, overrides, record, verification)
        finally:
            try:
                changed = identity(os.fstat(model.fileno())) != identity(before)
                if expected_runtime_files is not None:
                    verify_model_runtime_files(native, expected_runtime_files, scorer=runtime_scorer)
                if source_identities is not None:
                    _verify_profile_source_files(profile, source_paths, source_identities, deadline=time.monotonic() + maximum_seconds)
            finally:
                free(handle)
                handle.value = overrides.value = None
            if changed:
                raise ValueError("verified model changed while policy was active")


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