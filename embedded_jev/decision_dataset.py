"""Bounded, split-aware text decisions with explicit provenance and leakage checks."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from embedded_jev.inventory import MODEL_ID, MODEL_REVISION, InventoryError, _json_object
from embedded_jev.label_probe import LabelProbeError, validate_decision_cases
from embedded_jev.weight_slice import DEFAULT_TENSOR


MAX_DATASET_BYTES = 1 << 20
DATASET_PURPOSES = {"synthetic_split_contract_smoke", "user_labeled_text_decisions"}
SPLITS = ("calibration", "validation", "held_out")
MAX_CAPTURE_TOKENS = 128
MAX_CAPTURE_BYTES = MAX_CAPTURE_TOKENS * 12288 * 4 + 4096


class DecisionDatasetError(ValueError):
    """Invalid dataset provenance, split assignment, or decision data."""


def _prompt_identity(case: dict) -> str:
    identity = {
        "state": " ".join(case["state"].split()),
        "question": " ".join(case["question"].split()),
        "options": sorted(" ".join(option["description"].split()) for option in case["options"]),
    }
    encoded = json.dumps(identity, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_decision_dataset(path: Path) -> tuple[dict, str]:
    """Validate all three splits before permitting calibration or scoring."""
    try:
        with path.open("rb") as source:
            data = source.read(MAX_DATASET_BYTES + 1)
        if len(data) > MAX_DATASET_BYTES:
            raise DecisionDatasetError("decision dataset exceeds 1 MiB")
        dataset = _json_object(data, "decision dataset")
        if (
            set(dataset) != {"schema_version", "purpose", "provenance", "splits"}
            or type(dataset["schema_version"]) is not int or dataset["schema_version"] != 1
            or not isinstance(dataset["purpose"], str) or dataset["purpose"] not in DATASET_PURPOSES
            or not isinstance(dataset["provenance"], dict)
            or set(dataset["provenance"]) != {"source", "license"}
            or any(not isinstance(value, str) or not value.strip() or len(value) > 1024
                   for value in dataset["provenance"].values())
            or not isinstance(dataset["splits"], dict) or set(dataset["splits"]) != set(SPLITS)
        ):
            raise DecisionDatasetError("unsupported decision dataset schema or provenance")
        seen_ids = set()
        groups = {}
        prompts = {}
        for split in SPLITS:
            cases = dataset["splits"][split]
            validate_decision_cases(cases)
            for case in cases:
                if case["id"] in seen_ids:
                    raise DecisionDatasetError("case IDs must be unique across dataset splits")
                seen_ids.add(case["id"])
                group = case["group"]
                if group in groups and groups[group] != split:
                    raise DecisionDatasetError("decision group crosses dataset splits")
                groups[group] = split
                identity = _prompt_identity(case)
                if identity in prompts and prompts[identity] != split:
                    raise DecisionDatasetError("equivalent decision prompt crosses dataset splits")
                prompts[identity] = split
        return dataset, hashlib.sha256(data).hexdigest()
    except (OSError, InventoryError, LabelProbeError) as exc:
        raise DecisionDatasetError(f"invalid decision dataset: {exc}") from exc


def select_dataset_case(dataset: dict, *, split: str, case_id: str) -> dict:
    """Select within an explicit, validated split without fallback to another split."""
    if split not in SPLITS:
        raise DecisionDatasetError("unknown decision dataset split")
    matches = [case for case in dataset["splits"][split] if case["id"] == case_id]
    if len(matches) != 1:
        raise DecisionDatasetError("decision case not found in requested split")
    return matches[0]


def save_calibration_capture(
    directory: Path, values, *, dataset_path: Path, dataset_sha256: str, case_id: str,
) -> dict:
    """Save bounded unrotated input features for a validated calibration case only."""
    dataset, digest = load_decision_dataset(dataset_path)
    if not isinstance(dataset_sha256, str) or digest != dataset_sha256:
        raise DecisionDatasetError("decision dataset changed since activation capture was planned")
    case = select_dataset_case(dataset, split="calibration", case_id=case_id)
    activations = np.asarray(values)
    if (
        activations.ndim != 2 or activations.dtype != np.float32
        or not 1 <= activations.shape[0] <= MAX_CAPTURE_TOKENS or activations.shape[1] != 12288
        or not np.isfinite(activations).all()
    ):
        raise DecisionDatasetError("invalid calibration activation shape, dtype, or values")
    if directory.exists():
        raise DecisionDatasetError("calibration capture already exists")
    directory.mkdir()
    path = directory / "activations.npy"
    np.save(path, activations, allow_pickle=False)
    with path.open("rb") as source:
        array_digest = hashlib.file_digest(source, "sha256").hexdigest()
    manifest = {
        "schema_version": 1,
        "format": "single_case_mimo_calibration_activations",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "tensor": DEFAULT_TENSOR,
        "transform": "identity",
        "shape": list(activations.shape),
        "dataset": {
            "sha256": digest, "purpose": dataset["purpose"], "provenance": dataset["provenance"],
            "split": "calibration", "case_id": case["id"], "group": case["group"],
        },
        "array": {"bytes": path.stat().st_size, "sha256": array_digest},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest


def load_calibration_capture(directory: Path) -> tuple[np.ndarray, dict]:
    """Check a calibration-only array and its provenance before memory-mapped reuse."""
    try:
        with (directory / "manifest.json").open("rb") as source:
            data = source.read((64 << 10) + 1)
        if len(data) > 64 << 10:
            raise DecisionDatasetError("calibration manifest exceeds byte budget")
        manifest = _json_object(data, "calibration capture manifest")
        if (
            set(manifest) != {"schema_version", "format", "model", "revision", "tensor", "transform", "shape", "dataset", "array"}
            or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
            or manifest["format"] != "single_case_mimo_calibration_activations"
            or manifest["model"] != MODEL_ID or manifest["revision"] != MODEL_REVISION
            or manifest["tensor"] != DEFAULT_TENSOR or manifest["transform"] != "identity"
            or not isinstance(manifest["shape"], list) or len(manifest["shape"]) != 2
            or any(type(size) is not int for size in manifest["shape"])
            or not 1 <= manifest["shape"][0] <= MAX_CAPTURE_TOKENS or manifest["shape"][1] != 12288
        ):
            raise DecisionDatasetError("unsupported calibration capture manifest")
        dataset = manifest["dataset"]
        record = manifest["array"]
        if (
            not isinstance(dataset, dict)
            or set(dataset) != {"sha256", "purpose", "provenance", "split", "case_id", "group"}
            or dataset["split"] != "calibration"
            or not isinstance(dataset["purpose"], str) or dataset["purpose"] not in DATASET_PURPOSES
            or any(not isinstance(dataset[key], str) or not dataset[key].strip() for key in ("case_id", "group"))
            or not isinstance(dataset["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", dataset["sha256"]) is None
            or not isinstance(dataset["provenance"], dict) or set(dataset["provenance"]) != {"source", "license"}
            or any(not isinstance(value, str) or not value.strip() or len(value) > 1024
                   for value in dataset["provenance"].values())
            or not isinstance(record, dict) or set(record) != {"bytes", "sha256"}
            or type(record["bytes"]) is not int or not 0 < record["bytes"] <= MAX_CAPTURE_BYTES
            or not isinstance(record["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
        ):
            raise DecisionDatasetError("invalid calibration provenance or array record")
        path = directory / "activations.npy"
        if set(directory.iterdir()) != {path, directory / "manifest.json"} or path.stat().st_size != record["bytes"]:
            raise DecisionDatasetError("calibration array size mismatch")
        with path.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != record["sha256"]:
                raise DecisionDatasetError("calibration array hash mismatch")
        values = np.load(path, allow_pickle=False, mmap_mode="r")
        if values.dtype != np.float32 or list(values.shape) != manifest["shape"] or not np.isfinite(values).all():
            raise DecisionDatasetError("invalid calibration activation array")
        return values, manifest
    except (OSError, InventoryError, TypeError, ValueError) as exc:
        if isinstance(exc, DecisionDatasetError):
            raise
        raise DecisionDatasetError(f"invalid calibration capture: {exc}") from exc