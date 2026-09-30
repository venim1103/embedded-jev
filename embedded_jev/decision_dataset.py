"""Bounded, split-aware text decisions with explicit provenance and leakage checks."""

import hashlib
import json
from pathlib import Path

from embedded_jev.inventory import InventoryError, _json_object
from embedded_jev.label_probe import LabelProbeError, validate_decision_cases


MAX_DATASET_BYTES = 1 << 20
DATASET_PURPOSES = {"synthetic_split_contract_smoke", "user_labeled_text_decisions"}
SPLITS = ("calibration", "validation", "held_out")


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