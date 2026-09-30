"""Bounded BF16-versus-frozen-native observations without quantizer fitting."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
from pathlib import Path

from embedded_jev.decision_dataset import (
    SPLITS, DecisionDatasetError, load_decision_dataset, select_dataset_case,
)
from embedded_jev.inventory import MODEL_ID, MODEL_REVISION
from embedded_jev.projection_artifact import PINNED_SHARD_SHA256, load_projection_artifact
from embedded_jev.streamed_text import native_backend_dependencies, run_streamed_text


MAX_COMPARISON_CASES = 4


def _candidate_identity(directory: Path) -> dict:
    _, _, manifest = load_projection_artifact(directory)
    if manifest["shape"] != [4096, 12288] or manifest["source_shard_sha256"] != PINNED_SHARD_SHA256:
        raise DecisionDatasetError("comparison requires the pinned full FFN-down candidate")
    with (directory / "manifest.json").open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"manifest_sha256": digest, "metadata": manifest}


def _source_identity() -> dict:
    digests = {}
    for path in sorted(Path(__file__).parent.glob("*.py")):
        with path.open("rb") as source:
            digests[path.name] = hashlib.file_digest(source, "sha256").hexdigest()
    return digests


def _runtime_versions() -> dict:
    versions = {"python": sys.version.split()[0]}
    for package in ("numpy", "torch", "transformers", "safetensors", "accelerate", "tokenizers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _case_observation(case: dict, dense: dict, native: dict) -> dict:
    if (
        dense.get("model") != MODEL_ID or native.get("model") != MODEL_ID
        or dense.get("revision") != MODEL_REVISION or native.get("revision") != MODEL_REVISION
        or dense.get("generated_tokens") != 0 or native.get("generated_tokens") != 0
        or dense.get("layers") != 32 or native.get("layers") != 32
        or dense.get("prompt_sha256") != native.get("prompt_sha256")
        or not isinstance(dense.get("prompt_sha256"), str) or len(dense["prompt_sha256"]) != 64
        or dense.get("tokens") != native.get("tokens")
        or dense.get("ffn_down_input_sha256") != native.get("ffn_down_input_sha256")
        or not isinstance(dense.get("ffn_down_input_sha256"), str) or len(dense["ffn_down_input_sha256"]) != 64
        or dense.get("dataset") != native.get("dataset")
        or dense.get("dataset", {}).get("case_id") != case["id"]
        or native.get("native_ffn_down", {}).get("candidate_origin") != "saved_hash_checked_native_fixture"
    ):
        raise DecisionDatasetError("paired model reports have incompatible provenance or execution")
    observations = []
    for report in (dense, native):
        options = report["decision"]["options"]
        if [option["id"] for option in options] != [option["id"] for option in case["options"]]:
            raise DecisionDatasetError("paired option IDs disagree with the selected case")
        probabilities = [option["conditional_probability"] for option in options]
        logits = [option["logit"] for option in options]
        if (
            any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities)
            or not math.isclose(sum(probabilities), 1.0, rel_tol=1e-7, abs_tol=1e-7)
            or any(not math.isfinite(logit) for logit in logits)
        ):
            raise DecisionDatasetError("invalid paired conditional scores")
        exponentials = [math.exp(logit - max(logits)) for logit in logits]
        total = sum(exponentials)
        if any(not math.isclose(probability, exponential / total, rel_tol=1e-6, abs_tol=1e-7)
               for probability, exponential in zip(probabilities, exponentials, strict=True)):
            raise DecisionDatasetError("paired logits disagree with conditional probabilities")
        chosen = max(options, key=lambda option: option["conditional_probability"])["id"]
        if chosen != report["decision"]["chosen_option_id"]:
            raise DecisionDatasetError("paired choice disagrees with conditional scores")
        observations.append({
            "chosen_option_id": chosen,
            "matches_expected": chosen == case["expected_option_id"],
            "options": options,
        })
    changes = []
    for dense_option, native_option in zip(observations[0]["options"], observations[1]["options"], strict=True):
        if any(dense_option[key] != native_option[key] for key in ("label", "token_id", "row_sha256")):
            raise DecisionDatasetError("paired label/head mapping changed")
        changes.append({
            "id": dense_option["id"],
            "conditional_probability_delta": native_option["conditional_probability"] - dense_option["conditional_probability"],
            "logit_delta": native_option["logit"] - dense_option["logit"],
        })
    return {
        "case_id": case["id"], "group": case["group"], "expected_option_id": case["expected_option_id"],
        "prompt_sha256": dense["prompt_sha256"], "tokens": dense["tokens"],
        "bf16": observations[0], "native": observations[1], "changes": changes,
        "native_execution": native["native_ffn_down"],
        "choice_changed": observations[0]["chosen_option_id"] != observations[1]["chosen_option_id"],
        "conditional_total_variation": 0.5 * sum(abs(change["conditional_probability_delta"]) for change in changes),
    }


def compare_dataset_cases(
    snapshot: Path, *, dataset_path: Path, split: str, case_ids: list[str],
    native_library: Path, candidate: Path,
    native_backend: str = "direct",
) -> dict:
    """Compare explicit cases using one saved candidate, refusing changed inputs."""
    if (
        split not in SPLITS or not isinstance(case_ids, list)
        or not 1 <= len(case_ids) <= MAX_COMPARISON_CASES
        or any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
        or len(set(case_ids)) != len(case_ids) or not native_library.is_file()
        or native_backend not in ("direct", "prism_ggml", "prism_ggml_f32")
    ):
        raise DecisionDatasetError("comparison requires 1-4 unique case IDs, an explicit split, and a native library")
    dataset, digest = load_decision_dataset(dataset_path)
    cases = [select_dataset_case(dataset, split=split, case_id=case_id) for case_id in case_ids]
    identity = _candidate_identity(candidate)
    source_identity = _source_identity()
    runtime_versions = _runtime_versions()
    dependencies = native_backend_dependencies(native_library, native_backend)
    with native_library.open("rb") as source:
        library_digest = hashlib.file_digest(source, "sha256").hexdigest()

    def verify_frozen_inputs():
        _, current = load_decision_dataset(dataset_path)
        with native_library.open("rb") as source:
            current_library = hashlib.file_digest(source, "sha256").hexdigest()
        if (
            current != digest or _candidate_identity(candidate) != identity
            or current_library != library_digest or _source_identity() != source_identity
            or _runtime_versions() != runtime_versions
            or native_backend_dependencies(native_library, native_backend) != dependencies
        ):
            raise DecisionDatasetError("dataset, candidate, native library, scoring source, or runtime changed during comparison")

    results = []
    for case in cases:
        verify_frozen_inputs()
        common = {
            "layers": 32, "prompt": "", "dataset_path": dataset_path, "split": split, "case_id": case["id"],
        }
        dense = run_streamed_text(snapshot, **common)
        verify_frozen_inputs()
        native = run_streamed_text(
            snapshot, **common, native_ffn_library=native_library, projection_artifact=candidate,
            native_ffn_backend=native_backend,
        )
        verify_frozen_inputs()
        execution = native.get("native_ffn_down", {})
        if execution.get("backend") != native_backend or execution.get("backend_dependencies") != dependencies:
            raise DecisionDatasetError("native execution backend/dependencies disagree with frozen comparison inputs")
        if any(report["dataset"]["sha256"] != digest or report["dataset"]["split"] != split for report in (dense, native)):
            raise DecisionDatasetError("model reports disagree with the frozen dataset split")
        results.append(_case_observation(case, dense, native))
    return {
        "schema_version": 1,
        "purpose": "fixed_candidate_pairwise_observations_not_calibrated_confidence",
        "model": MODEL_ID, "revision": MODEL_REVISION,
        "dataset": {"sha256": digest, "purpose": dataset["purpose"], "provenance": dataset["provenance"], "split": split},
        "candidate": identity, "native_library_sha256": library_digest,
        "native_backend": native_backend,
        "native_backend_dependencies": dependencies,
        "runtime_versions": runtime_versions, "scoring_source_sha256": source_identity,
        "generated_tokens": 0, "quantizer_fitting": False,
        "cases": results,
        "summary": {
            "cases": len(results),
            "bf16_expected_matches": sum(result["bf16"]["matches_expected"] for result in results),
            "native_expected_matches": sum(result["native"]["matches_expected"] for result in results),
            "changed_choices": sum(result["choice_changed"] for result in results),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare BF16 and one frozen native projection on explicit decision cases")
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--case-ids", nargs="+", required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--native-backend", choices=("direct", "prism_ggml", "prism_ggml_f32"), default="direct")
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare_dataset_cases(
        args.local_dir, dataset_path=args.dataset, split=args.split, case_ids=args.case_ids,
        native_library=args.native_library, candidate=args.candidate,
        native_backend=args.native_backend,
    ), sort_keys=True))


if __name__ == "__main__":
    main()