"""Bounded, non-pickle container for toy ternary codes and FP16 scales."""

import hashlib
import io
import json
import zipfile

import numpy as np

from embedded_jev.inventory import _json_object
from embedded_jev.ternary import reconstruct_ternary


MAX_ARTIFACT_BYTES = 1 << 20
ALGORITHMS = {"rtn-maxabs-fp16-v1", "gptq-style-maxabs-fp16-v1"}


class TernaryArtifactError(ValueError):
    """Invalid or unsupported toy ternary artifact."""


def _encode_array(array):
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def save_toy_artifact(
    codes, scales, *, algorithm: str, signs=None, block_size: int | None = None
) -> bytes:
    try:
        reconstruct_ternary(codes, scales)
    except ValueError as exc:
        raise TernaryArtifactError("invalid codes or FP16 scales") from exc
    if algorithm not in ALGORITHMS:
        raise TernaryArtifactError("unsupported toy ternary algorithm")
    transform = {"kind": "identity"}
    if signs is not None:
        sign_vector = np.asarray(signs)
        if (
            type(block_size) is not int or block_size < 1
            or block_size & (block_size - 1)
            or codes.shape[1] % block_size
            or sign_vector.shape != (codes.shape[1],)
            or sign_vector.dtype.kind not in ("i", "u")
            or not np.isin(sign_vector, (-1, 1)).all()
        ):
            raise TernaryArtifactError("invalid signed transform")
        transform = {
            "kind": "signed_normalized_hadamard",
            "axis": "input-last-dimension",
            "block_size": block_size,
            "order": "signs_then_hadamard",
            "signs": sign_vector.astype(np.int8).tolist(),
        }
    elif block_size is not None:
        raise TernaryArtifactError("block size requires explicit signs")
    arrays = {"codes.npy": _encode_array(codes), "scales.npy": _encode_array(scales)}
    manifest = {
        "schema_version": 1,
        "format": "toy_non_packed_ternary",
        "origin": "toy_no_model_weights",
        "algorithm": algorithm,
        "group_size": codes.shape[1] // scales.shape[1],
        "transform": transform,
        "arrays": {
            name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            for name, data in sorted(arrays.items())
        },
    }
    entries = {"manifest.json": json.dumps(manifest, sort_keys=True).encode("utf-8"), **arrays}
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as container:
        for name, data in sorted(entries.items()):
            member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_STORED
            container.writestr(member, data)
    if archive.tell() > MAX_ARTIFACT_BYTES:
        raise TernaryArtifactError("toy ternary artifact exceeds size bound")
    return archive.getvalue()


def load_toy_artifact(payload: bytes) -> tuple[np.ndarray, np.ndarray, dict]:
    if not isinstance(payload, bytes) or len(payload) > MAX_ARTIFACT_BYTES:
        raise TernaryArtifactError("invalid or oversized toy ternary artifact")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as container:
            infos = container.infolist()
            if (
                {member.filename for member in infos}
                != {"manifest.json", "codes.npy", "scales.npy"}
                or len(infos) != 3
                or any(
                    member.compress_type != zipfile.ZIP_STORED
                    or member.file_size > MAX_ARTIFACT_BYTES
                    for member in infos
                )
            ):
                raise TernaryArtifactError("unexpected toy artifact entries")
            manifest = _json_object(container.read("manifest.json"), "manifest.json")
            if (
                set(manifest) != {
                    "schema_version", "format", "origin", "algorithm", "group_size",
                    "transform", "arrays",
                }
                or type(manifest["schema_version"]) is not int
                or manifest["schema_version"] != 1
                or manifest["format"] != "toy_non_packed_ternary"
                or manifest["origin"] != "toy_no_model_weights"
                or manifest["algorithm"] not in ALGORITHMS
                or set(manifest["arrays"]) != {"codes.npy", "scales.npy"}
            ):
                raise TernaryArtifactError("unsupported toy artifact manifest")
            arrays = {}
            for name in ("codes.npy", "scales.npy"):
                data = container.read(name)
                entry = manifest["arrays"][name]
                if (
                    set(entry) != {"sha256", "bytes"}
                    or entry["bytes"] != len(data)
                    or entry["sha256"] != hashlib.sha256(data).hexdigest()
                ):
                    raise TernaryArtifactError("toy artifact hash or size mismatch")
                arrays[name] = np.load(io.BytesIO(data), allow_pickle=False)
            codes, scales = arrays["codes.npy"], arrays["scales.npy"]
            reconstruct_ternary(codes, scales)
            transform = manifest["transform"]
            if not isinstance(transform, dict):
                raise TernaryArtifactError("unsupported toy transform")
            if transform != {"kind": "identity"}:
                signs = transform.get("signs")
                block_size = transform.get("block_size")
                if (
                    set(transform) != {"kind", "axis", "block_size", "order", "signs"}
                    or transform["kind"] != "signed_normalized_hadamard"
                    or transform["axis"] != "input-last-dimension"
                    or transform["order"] != "signs_then_hadamard"
                    or type(block_size) is not int or block_size < 1
                    or block_size & (block_size - 1)
                    or codes.shape[1] % block_size
                    or not isinstance(signs, list)
                    or len(signs) != codes.shape[1]
                    or any(type(sign) is not int or sign not in (-1, 1) for sign in signs)
                ):
                    raise TernaryArtifactError("unsupported toy transform")
            if type(manifest["group_size"]) is not int or (
                codes.shape[1] // scales.shape[1] != manifest["group_size"]
            ):
                raise TernaryArtifactError("toy artifact group size mismatch")
            return codes, scales, manifest
    except (OSError, zipfile.BadZipFile, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, TernaryArtifactError):
            raise
        raise TernaryArtifactError("invalid toy ternary artifact") from exc