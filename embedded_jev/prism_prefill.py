"""Generated-only typed native prefill with verified-owner result publication."""

import ctypes
import math
import os
import time
from tempfile import TemporaryFile

import numpy as np

from embedded_jev.inventory import _json_object
from embedded_jev.model_profile import ModelProfile
from embedded_jev.prism_codec import (
    PRISM_SOURCE_REVISION,
    NativeModelSpecification,
    native_model_specification,
    profile_model_specification,
    verified_profile_model_policy,
)


class NativePrefillError(ValueError):
    """A refused native request or scoring operation; no scores are returned."""

    def __init__(self, stage: str, status: int):
        self.stage = stage
        self.status = status
        super().__init__(f"native prefill {stage} failed: {status}")


def _require_success(stage: str, status: int) -> None:
    if type(status) is not int:
        raise ValueError(f"native prefill {stage} returned an invalid status")
    if status:
        raise NativePrefillError(stage, status)


def _validate_result(raw: bytes, request: dict, model_sha256: str, mode: str) -> dict:
    report = _json_object(raw, "native prefill result", max_bytes=8191)
    tokens = request["input_token_ids"]
    calls = 2 if mode == "reset" else 1
    expected = {
        "format": "jev-prefill-result-v1", "id": request["id"],
        "profile_id": request["profile_id"], "profile_sha256": request["profile_sha256"],
        "model_sha256": model_sha256, "prompt_sha256": request["prompt_sha256"],
        "runtime_revision": PRISM_SOURCE_REVISION, "selected_metadata_version": 5,
        "selected_metadata_records": 38, "model_policy_abi": 2,
        "prefilled_tokens": len(tokens), "generated_answer_tokens": 0,
        "calibration_status": "not_calibrated", "backend": "bitnet", "mode": mode,
        "prefill_calls": calls, "bitnet_dispatch_calls": calls, "weight_repacks": 1,
        "bitnet_last_input_tokens": len(tokens),
        "rejected_prefill_calls": int(mode == "kernel-recovery"),
        "kernel_error_status": int(mode == "kernel-recovery"),
        "memory_position_min": len(tokens) - 1, "memory_position_max": len(tokens) - 1,
    }
    if (
        set(report) != set(expected) | {
            "input_tokens", "options", "chosen_option_id", "max_option_probability",
            "allowed_label_probability_mass", "vocabulary_argmax", "runtime_elapsed_seconds",
        }
        or any(type(report[key]) is not type(value) or report[key] != value for key, value in expected.items())
        or not isinstance(report["input_tokens"], list) or report["input_tokens"] != tokens
        or any(type(token) is not int for token in report["input_tokens"])
        or not isinstance(report["options"], list) or len(report["options"]) != len(request["options"])
        or type(report["vocabulary_argmax"]) is not int or not 0 <= report["vocabulary_argmax"] < 64
    ):
        raise ValueError("native prefill result identity, execution or mapping mismatch")
    logits, probabilities = [], []
    for option, mapping in zip(report["options"], request["options"], strict=True):
        if (
            not isinstance(option, dict) or set(option) != {"id", "label", "token_id", "logit", "conditional_probability"}
            or any(type(option[key]) is not type(value) or option[key] != value for key, value in mapping.items())
            or any(type(option[key]) is not float or not math.isfinite(option[key]) for key in ("logit", "conditional_probability"))
            or not 0 <= option["conditional_probability"] <= 1
        ):
            raise ValueError("native prefill result option or score mismatch")
        logits.append(option["logit"])
        probabilities.append(option["conditional_probability"])
    best = max(range(len(logits)), key=logits.__getitem__)
    exponentials = [math.exp(value - logits[best]) for value in logits]
    denominator = sum(exponentials)
    if (
        not math.isclose(sum(probabilities), 1.0, rel_tol=1e-7, abs_tol=1e-7)
        or any(not math.isclose(probability, value / denominator, rel_tol=1e-6, abs_tol=1e-7)
               for probability, value in zip(probabilities, exponentials, strict=True))
        or report["chosen_option_id"] != request["options"][best]["id"]
        or any(type(report[key]) is not float or not math.isfinite(report[key])
               for key in ("max_option_probability", "allowed_label_probability_mass", "runtime_elapsed_seconds"))
        or not math.isclose(report["max_option_probability"], probabilities[best], rel_tol=1e-6, abs_tol=1e-7)
        or not 0 <= report["allowed_label_probability_mass"] <= 1
        or report["runtime_elapsed_seconds"] <= 0
    ):
        raise ValueError("native prefill result scores are inconsistent")
    return report


def run_generated_prefill_request(
    native, scorer, model_path, profile: ModelProfile, inventory: dict, manifest: dict, pq2_blocks, *,
    metadata: bytes, request: bytes, expected_bytes: int, expected_sha256: str,
    expected_runtime_files: dict, mode: str = "full", maximum_seconds: float = 30,
) -> dict:
    """Run only the generated control, never learned weights or runtime authorization."""
    started = time.monotonic()
    if (
        not isinstance(profile, ModelProfile) or profile.profile_id != "synthetic-typed-prefill-v1"
        or any(profile.geometry[key] != value for key, value in {
            "num_hidden_layers": 4, "hidden_size": 32, "intermediate_size": 1024, "vocab_size": 64,
        }.items())
        or type(expected_bytes) is not int or not 0 < expected_bytes <= 2 * 1024**2
        or type(maximum_seconds) not in (int, float) or not 0 < maximum_seconds <= 60
        or mode not in ("full", "reset", "kernel-error", "kernel-recovery")
        or any(type(value) is not bytes or not 0 < len(value) <= 65536 for value in (metadata, request))
        or not isinstance(expected_runtime_files, dict)
        or set(expected_runtime_files) != {"bridge", "ggml_cpu", "ggml_base", "scorer", "llama", "ggml_registry"}
        or any(not isinstance(value, dict) for value in expected_runtime_files.values())
    ):
        raise ValueError("invalid generated prefill profile, inputs or budget")
    packet = _json_object(request, "generated prefill request", max_bytes=65536)
    if packet.get("profile_id") != profile.profile_id or packet.get("profile_sha256") != profile.sha256:
        raise ValueError("generated prefill request profile mismatch")
    specification = profile_model_specification(profile, inventory, manifest, pq2_blocks)
    runtime_files = {name: value.copy() for name, value in expected_runtime_files.items()}
    blocks = np.array(pq2_blocks, copy=True)
    record = native_model_specification(specification, blocks)
    preflight = scorer.prism_bitnet_test_prefill_request_preflight_v1
    preflight.argtypes = [ctypes.c_char_p, ctypes.POINTER(NativeModelSpecification), ctypes.c_char_p, ctypes.c_char_p]
    preflight.restype = ctypes.c_int
    score = scorer.prism_bitnet_test_prefill_request_policy_result_v1
    score.argtypes = [
        ctypes.c_char_p, ctypes.POINTER(NativeModelSpecification), ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t),
    ]
    score.restype = ctypes.c_int
    registry = native.prism_bitnet_registered_tensor_registry_size
    registry.argtypes = []
    registry.restype = ctypes.c_size_t
    initial_registry = registry()

    def remaining():
        value = maximum_seconds - (time.monotonic() - started)
        if value <= 0:
            raise ValueError("generated prefill deadline exceeded")
        return value

    output = ctypes.create_string_buffer(8192)
    output_bytes = ctypes.c_size_t()
    with TemporaryFile() as metadata_file, TemporaryFile() as request_file:
        metadata_file.write(metadata)
        request_file.write(request)
        metadata_file.flush()
        request_file.flush()
        metadata_path = os.fsencode(f"/proc/self/fd/{metadata_file.fileno()}")
        request_path = os.fsencode(f"/proc/self/fd/{request_file.fileno()}")

        def check_request(path):
            _require_success("preflight", preflight(os.fsencode(path), ctypes.byref(record), metadata_path, request_path))
            if registry() != initial_registry:
                raise ValueError("native prefill preflight changed backend registration")
            return 0

        with verified_profile_model_policy(
            native, model_path, profile, inventory, manifest, blocks,
            expected_bytes=expected_bytes, expected_sha256=expected_sha256, maximum_bytes=2 * 1024**2,
            maximum_seconds=remaining(), expected_runtime_files=runtime_files, runtime_scorer=scorer,
            request_preflight=check_request,
        ) as owner:
            remaining()
            status = score(
                os.fsencode(owner.model_path), ctypes.byref(owner.record), owner.handle, owner.overrides,
                metadata_path, request_path, mode.encode("ascii"), output, len(output), ctypes.byref(output_bytes),
            )
            _require_success("scoring", status)
            verification = dict(owner.verification)
        remaining()
        if not 0 < output_bytes.value < len(output) or any(output.raw[output_bytes.value:]):
            raise ValueError("native prefill result buffer is invalid")
        raw = output.raw[:output_bytes.value]
        report = _validate_result(raw, packet, expected_sha256, mode)
    remaining()
    report["verified_owner"] = {
        "model_sha256": verification["model_sha256"], "runtime_files": len(runtime_files),
        "native_preflight_before_owner": True, "native_preflight_anchored": True,
        "native_output_transport": "caller_buffer", "scores_released_after_exit": True,
        "runtime_approval": verification["runtime_approval"], "buffered_native_output_bytes": len(raw),
    }
    return report
