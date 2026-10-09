"""Exact ternary-subset PQ2_0 byte conversion for the pinned Prism format."""

import ctypes
import hashlib

import numpy as np


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