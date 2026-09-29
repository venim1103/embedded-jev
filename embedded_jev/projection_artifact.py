"""Hashed single-projection native fixture, not a GGUF or loadable MiMo model."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from embedded_jev.inventory import (
    MODEL_ID, MODEL_REVISION, _json_object, build_inventory, read_local_headers,
)
from embedded_jev.ternary import (
    pack_group128_codes, quantize_ternary_rtn, unpack_group128_codes,
)
from embedded_jev.weight_slice import DEFAULT_TENSOR, STREAM_ROWS


MAX_MANIFEST_BYTES = 1 << 20
MAX_PACKED_BYTES = 16 << 20
MAX_SCALE_BYTES = 2 << 20
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
PINNED_SHARD = "model-00002-of-00004.safetensors"
PINNED_SHARD_SHA256 = "7a0486565f06d25ac4628e9dba470dc3f604353471d240d5a0bf7128f64df396"


class ProjectionArtifactError(ValueError):
    """An unsupported or altered native projection fixture."""


def _file_record(path: Path) -> dict:
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": digest}


def save_projection_artifact(directory: Path, codes, scales, *, shard_sha256: str) -> dict:
    """Store one explicitly packed ternary candidate with immutable FP16 group scales."""
    ternary = np.asarray(codes)
    stored_scales = np.asarray(scales)
    if (
        ternary.ndim != 3 or ternary.dtype != np.int8 or 0 in ternary.shape
        or ternary.shape[2] != 128 or ternary.shape[0] > 4096 or ternary.shape[1] > 96
        or stored_scales.dtype != np.float16
        or stored_scales.shape != ternary.shape[:2]
        or not np.isfinite(stored_scales).all() or np.any(stored_scales < 0)
        or not isinstance(shard_sha256, str) or SHA256.fullmatch(shard_sha256) is None
    ):
        raise ProjectionArtifactError("invalid projection candidate bounds or scales")
    try:
        packed = pack_group128_codes(ternary)
    except ValueError as exc:
        raise ProjectionArtifactError("invalid projection ternary codes") from exc
    if packed.nbytes > MAX_PACKED_BYTES or stored_scales.nbytes > MAX_SCALE_BYTES:
        raise ProjectionArtifactError("projection candidate exceeds file budget")
    if directory.exists():
        raise ProjectionArtifactError("projection candidate already exists")
    directory.mkdir()
    np.save(directory / "packed.npy", packed, allow_pickle=False)
    np.save(directory / "scales.npy", stored_scales, allow_pickle=False)
    manifest = {
        "schema_version": 1,
        "format": "single_projection_bitnet_derived_native_fixture_not_gguf",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "tensor": DEFAULT_TENSOR,
        "algorithm": "rtn-searched-fp16-group128",
        "rotation": "identity",
        "shape": [int(ternary.shape[0]), int(ternary.shape[1] * 128)],
        "group_size": 128,
        "source_shard_sha256": shard_sha256,
        "arrays": {
            name: _file_record(directory / name) for name in ("packed.npy", "scales.npy")
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest


def quantize_pinned_projection(snapshot: Path, directory: Path) -> dict:
    """Save only one real searched-FP16 candidate from a verified pinned shard."""
    if directory.exists():
        raise ProjectionArtifactError("projection candidate already exists")
    metadata, headers = read_local_headers(snapshot)
    report = build_inventory(metadata, headers)
    tensor = next((entry for entry in report["tensors"] if entry["name"] == DEFAULT_TENSOR), None)
    if (
        tensor is None or tensor["shard"] != PINNED_SHARD or tensor["dtype"] != "BF16"
        or tensor["shape"] != (4096, 12288) or tensor["storage_bytes"] != 100663296
        or not tensor["quantization_eligible"]
    ):
        raise ProjectionArtifactError("pinned projection does not match expected BF16 layout")
    path = snapshot / PINNED_SHARD
    if _file_record(path)["sha256"] != PINNED_SHARD_SHA256:
        raise ProjectionArtifactError("pinned source shard SHA-256 mismatch")
    header, file_bytes = headers[PINNED_SHARD]
    offsets = _json_object(header[8:], PINNED_SHARD)[DEFAULT_TENSOR]["data_offsets"]
    if offsets[1] - offsets[0] != tensor["storage_bytes"] or len(header) + offsets[1] > file_bytes:
        raise ProjectionArtifactError("pinned projection byte span mismatch")
    codes = np.empty((4096, 96, 128), dtype=np.int8)
    scales = np.empty((4096, 96), dtype=np.float16)
    with path.open("rb") as source:
        source.seek(len(header) + offsets[0])
        for first_row in range(0, 4096, STREAM_ROWS):
            count = min(STREAM_ROWS, 4096 - first_row)
            data = source.read(count * 12288 * 2)
            if len(data) != count * 12288 * 2:
                raise ProjectionArtifactError("short pinned BF16 projection batch")
            weights = (np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16).view("<f4")
            weights = weights.reshape(count, 12288)
            if not np.isfinite(weights).all():
                raise ProjectionArtifactError("nonfinite pinned BF16 projection")
            batch_codes, batch_scales = quantize_ternary_rtn(weights, scale_search=True)
            codes[first_row:first_row + count] = batch_codes.reshape(count, 96, 128)
            scales[first_row:first_row + count] = batch_scales
    directory.parent.mkdir(parents=True, exist_ok=True)
    return save_projection_artifact(directory, codes, scales, shard_sha256=PINNED_SHARD_SHA256)


def load_projection_artifact(directory: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Refuse changed metadata or arrays before opening the non-pickle NumPy data."""
    try:
        manifest_path = directory / "manifest.json"
        if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ProjectionArtifactError("projection manifest exceeds byte budget")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            not isinstance(manifest, dict)
            or set(manifest) != {"schema_version", "format", "model", "revision", "tensor",
                                 "algorithm", "rotation", "shape", "group_size", "source_shard_sha256", "arrays"}
            or manifest["schema_version"] != 1
            or manifest["format"] != "single_projection_bitnet_derived_native_fixture_not_gguf"
            or manifest["model"] != MODEL_ID or manifest["revision"] != MODEL_REVISION
            or manifest["tensor"] != DEFAULT_TENSOR or manifest["algorithm"] != "rtn-searched-fp16-group128"
            or manifest["rotation"] != "identity" or manifest["group_size"] != 128
            or not isinstance(manifest["shape"], list) or len(manifest["shape"]) != 2
            or any(type(dimension) is not int or dimension < 1 for dimension in manifest["shape"])
            or manifest["shape"][0] > 4096 or manifest["shape"][1] > 12288
            or manifest["shape"][1] % 128
            or not isinstance(manifest["source_shard_sha256"], str)
            or SHA256.fullmatch(manifest["source_shard_sha256"]) is None
            or not isinstance(manifest["arrays"], dict)
            or set(manifest["arrays"]) != {"packed.npy", "scales.npy"}
            or set(directory.iterdir()) != {manifest_path, directory / "packed.npy", directory / "scales.npy"}
        ):
            raise ProjectionArtifactError("unsupported projection manifest")
        for name, limit in (("packed.npy", MAX_PACKED_BYTES), ("scales.npy", MAX_SCALE_BYTES)):
            record = manifest["arrays"][name]
            if (
                not isinstance(record, dict) or set(record) != {"bytes", "sha256"}
                or type(record["bytes"]) is not int or not 0 < record["bytes"] <= limit
                or not isinstance(record["sha256"], str) or SHA256.fullmatch(record["sha256"]) is None
                or _file_record(directory / name) != record
            ):
                raise ProjectionArtifactError("projection array size or hash mismatch")
        packed = np.load(directory / "packed.npy", allow_pickle=False, mmap_mode="r")
        scales = np.load(directory / "scales.npy", allow_pickle=False, mmap_mode="r")
        rows, width = manifest["shape"]
        groups = width // 128
        if (
            packed.dtype != np.uint8 or packed.shape != (rows, groups, 32)
            or scales.dtype != np.float16 or scales.shape != (rows, groups)
            or not np.isfinite(scales).all() or np.any(scales < 0)
        ):
            raise ProjectionArtifactError("invalid projection array shapes or values")
        try:
            unpack_group128_codes(packed)
        except ValueError as exc:
            raise ProjectionArtifactError("invalid packed projection trits") from exc
        return packed, scales, manifest
    except (OSError, json.JSONDecodeError, TypeError, KeyError, ValueError) as exc:
        if isinstance(exc, ProjectionArtifactError):
            raise
        raise ProjectionArtifactError("invalid projection artifact") from exc