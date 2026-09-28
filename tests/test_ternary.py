"""Offline code/scale identity checks for the deterministic ternary baseline."""

import hashlib
import io
import json
import zipfile

import numpy as np
import pytest

from embedded_jev.activation import rotate_signed_hadamard
from embedded_jev.ternary_artifact import TernaryArtifactError, load_toy_artifact, save_toy_artifact
from embedded_jev.ternary import (
    quantize_ternary_compensated,
    quantize_ternary_rtn,
    reconstruct_ternary,
)


def test_fp16_group_scales_are_used_to_assign_and_reconstruct_codes():
    weights = np.zeros((2, 256), dtype=np.float32)
    weights[0, :3] = [1.0004, 0.5001, -0.75]
    weights[1, 0] = 2.25
    weights[1, 128] = -3.5

    codes, scales = quantize_ternary_rtn(weights)
    assert codes.dtype == np.int8 and scales.dtype == np.float16
    np.testing.assert_array_equal(scales, [[1.0, 0.0], [2.25, 3.5]])
    np.testing.assert_array_equal(codes[0, :3], [1, 1, -1])
    assert np.all(codes[0, 128:] == 0)
    assert codes[1, 0] == 1 and codes[1, 128] == -1
    restored = reconstruct_ternary(codes, scales)
    assert restored.dtype == np.float32
    np.testing.assert_array_equal(
        restored.reshape(2, 2, 128),
        codes.reshape(2, 2, 128).astype(np.float32) * scales.astype(np.float32)[..., None],
    )
    assert np.array_equal(codes, quantize_ternary_rtn(weights)[0])


@pytest.mark.parametrize(
    "weights",
    [np.zeros((2, 129)), np.full((1, 128), 70000.0),
     np.full((1, 128), 1e-9), np.full((1, 128), np.nan), np.zeros((0, 128))],
)
def test_quantizer_rejects_tails_overflow_underflow_and_nonfinite(weights):
    with pytest.raises(ValueError, match="invalid ternary|not representable"):
        quantize_ternary_rtn(weights)


@pytest.mark.parametrize(
    ("codes", "scales"),
    [
        (np.full((1, 128), 2, dtype=np.int8), np.ones((1, 1), dtype=np.float16)),
        (np.zeros((1, 128), dtype=np.int16), np.ones((1, 1), dtype=np.float16)),
        (np.zeros((1, 128), dtype=np.int8), np.ones((1, 1), dtype=np.float32)),
        (np.zeros((1, 129), dtype=np.int8), np.ones((1, 2), dtype=np.float16)),
        (np.zeros((1, 128), dtype=np.int8), np.array([[-1]], dtype=np.float16)),
        (np.zeros((1, 0), dtype=np.int8), np.ones((1, 1), dtype=np.float16)),
        (np.ones((1, 128), dtype=np.int8), np.zeros((1, 1), dtype=np.float16)),
    ],
)
def test_reconstruction_refuses_invalid_stored_artifacts(codes, scales):
    with pytest.raises(ValueError, match="invalid stored ternary"):
        reconstruct_ternary(codes, scales)


def test_compensation_reduces_one_correlated_toy_reconstruction_error():
    inputs = np.array(
        [[1, 1, 0, 0], [1, 0.8, 0, 0], [-1, -1, 0, 0],
         [-1, -0.8, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32,
    )
    weights = np.array([[0.49, 0.49, 1, 0]], dtype=np.float32)
    baseline_codes, baseline_scales = quantize_ternary_rtn(weights, group_size=4)
    codes, scales, details = quantize_ternary_compensated(weights, inputs, group_size=4)
    np.testing.assert_array_equal(baseline_codes, [[0, 0, 1, 0]])
    np.testing.assert_array_equal(codes, [[0, 1, 1, 0]])
    np.testing.assert_array_equal(scales, baseline_scales)
    assert details["damping"] > 0 and details["damping_attempts"] == 1
    baseline_loss = np.linalg.norm(inputs @ (weights - reconstruct_ternary(baseline_codes, baseline_scales)).T)
    compensated_loss = np.linalg.norm(inputs @ (weights - reconstruct_ternary(codes, scales)).T)
    assert compensated_loss < baseline_loss / 10
    assert np.array_equal(codes, quantize_ternary_compensated(weights, inputs, 4)[0])


def test_compensation_matches_rtn_on_diagonal_curvature_and_zero_groups():
    inputs = np.eye(4, dtype=np.float32)
    weights = np.array([[1.0004, 0.5001, 0, 0], [0, 0, -2.25, 0]], dtype=np.float32)
    baseline = quantize_ternary_rtn(weights, group_size=2)
    codes, scales, _ = quantize_ternary_compensated(weights, inputs, group_size=2)
    np.testing.assert_array_equal(codes, baseline[0])
    np.testing.assert_array_equal(scales, baseline[1])
    assert codes[0, 1] == 1 and scales[0, 1] == 0
    np.testing.assert_array_equal(reconstruct_ternary(codes, scales)[0, 2:], 0)


@pytest.mark.parametrize(
    ("weights", "inputs", "options"),
    [
        (np.ones((1, 4)), np.zeros((8, 4)), {}),
        (np.ones((1, 4)), np.eye(4), {"damping_ratio": 0}),
        (np.ones((1, 4)), np.eye(4), {"max_damping_attempts": 5}),
        (np.ones((1, 257)), np.ones((8, 257)), {}),
        (np.full((1, 4), 70000), np.eye(4), {}),
        (np.full((1, 4), 1e-9), np.eye(4), {}),
    ],
)
def test_compensated_quantizer_rejects_invalid_inputs_and_scales(weights, inputs, options):
    with pytest.raises(ValueError, match="invalid compensated|curvature|not representable"):
        quantize_ternary_compensated(weights, inputs, group_size=1, **options)


def test_compensated_quantizer_never_falls_back_to_pseudoinverse(monkeypatch):
    def fail_factorization(*args, **kwargs):
        raise np.linalg.LinAlgError("deliberate failure")

    monkeypatch.setattr(np.linalg, "cholesky", fail_factorization)
    with pytest.raises(ValueError, match="failed after bounded damping"):
        quantize_ternary_compensated(np.ones((1, 4)), np.eye(4), group_size=2)


@pytest.mark.parametrize(
    ("width", "group_size", "processing_block_size"),
    [(12, 4, 4), (12, 4, 8), (256, 128, 128)],
)
def test_processing_blocks_preserve_fixed_group_scales(width, group_size, processing_block_size):
    generator = np.random.default_rng(width + processing_block_size)
    weights = generator.normal(size=(3, width)).astype(np.float32)
    activations = generator.normal(size=(32, width)).astype(np.float32)
    activations[:, 1] = activations[:, 0] * 0.8 + activations[:, 1] * 0.2
    expected_codes, expected_scales, _ = quantize_ternary_compensated(
        weights, activations, group_size=group_size
    )
    codes, scales, _ = quantize_ternary_compensated(
        weights, activations, group_size=group_size,
        processing_block_size=processing_block_size,
    )
    np.testing.assert_array_equal(codes, expected_codes)
    np.testing.assert_array_equal(scales, expected_scales)
    np.testing.assert_array_equal(
        reconstruct_ternary(codes, scales), reconstruct_ternary(expected_codes, expected_scales)
    )


@pytest.mark.parametrize("processing_block_size", [0, 3, 6, 16])
def test_processing_block_size_cannot_split_groups_or_exceed_width(processing_block_size):
    with pytest.raises(ValueError, match="invalid compensated"):
        quantize_ternary_compensated(
            np.ones((1, 12)), np.eye(12), group_size=4,
            processing_block_size=processing_block_size,
        )


def artifact_entries(payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def replace_entries(entries, *, compression=zipfile.ZIP_STORED):
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", compression=compression) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return result.getvalue()


def test_toy_artifact_is_deterministic_and_retains_fp16_scales():
    weights = np.zeros((2, 256), dtype=np.float32)
    weights[0, 0] = 1.0004
    codes, scales = quantize_ternary_rtn(weights)
    payload = save_toy_artifact(codes, scales, algorithm="rtn-maxabs-fp16-v1")
    assert payload == save_toy_artifact(codes, scales, algorithm="rtn-maxabs-fp16-v1")
    loaded_codes, loaded_scales, manifest = load_toy_artifact(payload)
    np.testing.assert_array_equal(loaded_codes, codes)
    np.testing.assert_array_equal(loaded_scales, scales)
    assert manifest["group_size"] == 128
    assert manifest["transform"] == {"kind": "identity"}
    assert manifest["origin"] == "toy_no_model_weights"


def test_toy_artifact_rejects_tampering_pickle_and_unsupported_metadata():
    codes, scales = quantize_ternary_rtn(np.ones((1, 128), dtype=np.float32))
    payload = save_toy_artifact(codes, scales, algorithm="gptq-style-maxabs-fp16-v1")
    entries = artifact_entries(payload)
    with pytest.raises(TernaryArtifactError, match="unsupported"):
        save_toy_artifact(codes, scales, algorithm="invented")
    with pytest.raises(TernaryArtifactError, match="invalid or oversized"):
        load_toy_artifact(b"x" * ((1 << 20) + 1))

    manifest = json.loads(entries["manifest.json"])
    manifest["group_size"] = 64
    with pytest.raises(TernaryArtifactError, match="group size mismatch"):
        load_toy_artifact(replace_entries({**entries, "manifest.json": json.dumps(manifest).encode()}))
    manifest["group_size"] = 128
    manifest["transform"] = "unverified_hadamard"
    with pytest.raises(TernaryArtifactError, match="unsupported"):
        load_toy_artifact(replace_entries({**entries, "manifest.json": json.dumps(manifest).encode()}))

    changed_codes = bytearray(entries["codes.npy"])
    changed_codes[-1] ^= 1
    with pytest.raises(TernaryArtifactError, match="hash or size"):
        load_toy_artifact(replace_entries({**entries, "codes.npy": changed_codes}))
    with pytest.raises(TernaryArtifactError, match="unexpected"):
        load_toy_artifact(replace_entries(entries, compression=zipfile.ZIP_DEFLATED))
    with pytest.raises(TernaryArtifactError, match="unexpected"):
        load_toy_artifact(replace_entries({**entries, "extra.npy": b"x"}))

    object_array = io.BytesIO()
    np.save(object_array, np.array([object()], dtype=object), allow_pickle=True)
    manifest = json.loads(entries["manifest.json"])
    object_bytes = object_array.getvalue()
    manifest["arrays"]["codes.npy"] = {
        "bytes": len(object_bytes), "sha256": hashlib.sha256(object_bytes).hexdigest(),
    }
    with pytest.raises(TernaryArtifactError, match="invalid toy"):
        load_toy_artifact(replace_entries({
            **entries, "codes.npy": object_bytes, "manifest.json": json.dumps(manifest).encode(),
        }))


def test_signed_transform_round_trip_and_strict_schema():
    generator = np.random.default_rng(153)
    signs = generator.choice([-1, 1], size=256)
    weights = generator.normal(size=(2, 256)).astype(np.float32)
    rotated = rotate_signed_hadamard(weights, signs, 128)
    codes, scales = quantize_ternary_rtn(rotated)
    payload = save_toy_artifact(
        codes, scales, algorithm="rtn-maxabs-fp16-v1", signs=signs, block_size=128
    )
    loaded_codes, loaded_scales, manifest = load_toy_artifact(payload)
    np.testing.assert_array_equal(loaded_codes, codes)
    np.testing.assert_array_equal(loaded_scales, scales)
    assert manifest["transform"] == {
        "kind": "signed_normalized_hadamard", "axis": "input-last-dimension",
        "block_size": 128, "order": "signs_then_hadamard", "signs": signs.tolist(),
    }
    np.testing.assert_array_equal(
        rotate_signed_hadamard(weights, manifest["transform"]["signs"], 128), rotated
    )

    for invalid_signs, invalid_size in ((np.ones(255), 128), (np.ones(256), 3),
                                        (np.zeros(256), 128)):
        with pytest.raises(TernaryArtifactError, match="invalid signed transform"):
            save_toy_artifact(
                codes, scales, algorithm="rtn-maxabs-fp16-v1",
                signs=invalid_signs, block_size=invalid_size,
            )
    with pytest.raises(TernaryArtifactError, match="block size requires"):
        save_toy_artifact(codes, scales, algorithm="rtn-maxabs-fp16-v1", block_size=128)

    entries = artifact_entries(payload)
    manifest["transform"]["signs"][0] = 0
    with pytest.raises(TernaryArtifactError, match="unsupported toy transform"):
        load_toy_artifact(replace_entries({**entries, "manifest.json": json.dumps(manifest).encode()}))
    manifest["transform"]["signs"][0] = int(signs[0])
    manifest["transform"]["order"] = "hadamard_then_signs"
    with pytest.raises(TernaryArtifactError, match="unsupported toy transform"):
        load_toy_artifact(replace_entries({**entries, "manifest.json": json.dumps(manifest).encode()}))
    manifest["transform"]["order"] = "signs_then_hadamard"
    manifest["schema_version"] = True
    with pytest.raises(TernaryArtifactError, match="unsupported toy artifact manifest"):
        load_toy_artifact(replace_entries({**entries, "manifest.json": json.dumps(manifest).encode()}))


def test_scale_search_retains_fp16_codes_and_never_worsens_local_weight_mse():
    generator = np.random.default_rng(441)
    weights = generator.normal(size=(3, 256)).astype(np.float32)
    weights[0, :128] = 0.0
    baseline = quantize_ternary_rtn(weights)
    searched = quantize_ternary_rtn(weights, scale_search=True)
    assert searched[1].dtype == np.float16
    np.testing.assert_array_equal(searched[1][0, 0], 0)
    np.testing.assert_array_equal(searched[0][0, :128], 0)
    def local_mse(artifact):
        return np.mean((weights - reconstruct_ternary(*artifact)) ** 2)

    assert local_mse(searched) <= local_mse(baseline) + 1e-9
    saved_scales = np.where(searched[1] == 0, 1, searched[1]).astype(np.float32)
    expected_codes = np.clip(
        np.rint(weights.reshape(3, 2, 128) / saved_scales[..., None]), -1, 1
    ).astype(np.int8)
    np.testing.assert_array_equal(searched[0], expected_codes.reshape(3, 256))
    with pytest.raises(ValueError, match="invalid ternary"):
        quantize_ternary_rtn(weights, scale_search=1)


def test_compensated_scale_search_preserves_group_aligned_block_result():
    generator = np.random.default_rng(442)
    weights = generator.normal(size=(2, 256)).astype(np.float32)
    calibration = generator.normal(size=(48, 256)).astype(np.float32)
    full = quantize_ternary_compensated(weights, calibration, scale_search=True)
    blocked = quantize_ternary_compensated(
        weights, calibration, processing_block_size=128, scale_search=True
    )
    np.testing.assert_array_equal(blocked[0], full[0])
    np.testing.assert_array_equal(blocked[1], full[1])