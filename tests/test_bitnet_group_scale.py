"""Golden checks for the isolated BitNet-derived AVX2 group-scaled dot."""

import ctypes
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import hadamard

from embedded_jev.activation import quantize_a8_per_group, rotate_signed_hadamard
from embedded_jev.inventory import (
    MODEL_REVISION, _json_object, build_inventory, fetch_pinned_headers, read_local_headers,
)
from embedded_jev.ternary import (
    quantize_ternary_compensated,
    quantize_ternary_rtn,
    pack_group128_codes,
    reconstruct_ternary,
    unpack_group128_codes,
)
from embedded_jev.ternary_artifact import load_toy_artifact, save_toy_artifact
from embedded_jev.weight_slice import (
    DEFAULT_TENSOR, MAX_STREAM_TENSOR_BYTES, STREAM_ROWS, fetch_bf16_projection_slice,
)


SOURCE = Path(__file__).resolve().parents[1] / "native" / "bitnet_group_scale.cpp"
AVX2 = platform.machine() == "x86_64" and "avx2" in Path("/proc/cpuinfo").read_text()
pytestmark = pytest.mark.skipif(not AVX2, reason="requires x86-64 AVX2")


@pytest.fixture(scope="module")
def native_dot(tmp_path_factory):
    compiler = shutil.which("clang++-18")
    if compiler is None:
        pytest.skip("clang++-18 is not installed")
    library = tmp_path_factory.mktemp("bitnet-group-scale") / "kernel.so"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-mavx2", "-shared", "-fPIC", str(SOURCE),
         "-o", str(library)],
        check=True, capture_output=True, text=True,
    )
    binary = ctypes.CDLL(str(library))
    function = binary.bitnet_group_scale_matvec_avx2
    function.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_float),
    ]
    function.restype = ctypes.c_int
    batch = binary.bitnet_group_scale_matmul_avx2
    batch.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
    ]
    batch.restype = ctypes.c_int
    return function, batch, binary.bitnet_group_scale_prepare_a8, binary.bitnet_group_scale_matmul_fp16_avx2


def pack_codes(codes):
    return pack_group128_codes(codes)


def call_native(function, codes, activations, weight_scales, activation_scales):
    packed = pack_codes(codes)
    rows, groups, _ = codes.shape
    output = np.empty(rows, dtype=np.float32)
    status = function(
        packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        weight_scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        activations.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        activation_scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        rows, groups, output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    assert status == 0
    return packed, output


def reference(codes, activations, weight_scales, activation_scales):
    group_dots = np.einsum(
        "rgi,gi->rg", codes.astype(np.int32), activations.astype(np.int32)
    )
    return (group_dots * weight_scales * activation_scales).sum(axis=1)


def call_native_batch(function, codes, activations, weight_scales, activation_scales):
    tokens, groups, _ = activations.shape
    rows = codes.shape[0]
    packed = pack_codes(codes)
    output = np.empty((tokens, rows), dtype=np.float32)
    status = function(
        packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        weight_scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        activations.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        activation_scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        tokens, rows, groups, output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    assert status == 0
    return output


def test_bitnet_layout_zero_compensation_and_row_dependent_scales(native_dot):
    codes = np.zeros((3, 2, 128), dtype=np.int8)
    codes[1, 0, [0, 32, 64, 96]] = [-1, 0, 1, 0]
    codes[1, 1, :] = 1
    codes[2, 0, :] = -1
    codes[2, 1, ::2] = 1
    activations = np.tile(np.array([-128, 127, 7, -3], dtype=np.int8), 64).reshape(2, 128)
    weight_scales = np.array([[0.5, 1.0], [0.25, 3.0], [5.0, 0.125]], dtype=np.float32)
    activation_scales = np.array([0.25, 2.0], dtype=np.float32)
    packed, actual = call_native(native_dot[0], codes, activations, weight_scales, activation_scales)
    assert packed[1, 0, 0] == 0x19
    assert actual[0] == 0.0
    np.testing.assert_allclose(actual, reference(codes, activations, weight_scales, activation_scales))
    with pytest.raises(ValueError, match="group-128 ternary codes"):
        pack_group128_codes(np.full((1, 1, 128), 2, dtype=np.int8))
    np.testing.assert_array_equal(unpack_group128_codes(packed), codes)
    with pytest.raises(ValueError, match="invalid code"):
        unpack_group128_codes(np.full((1, 1, 32), 255, dtype=np.uint8))


def test_native_a8_preparation_matches_python_rounding_and_scale_guards(native_dot):
    prepare = native_dot[2]
    prepare.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
                        ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_float)]
    prepare.restype = ctypes.c_int
    values = np.random.default_rng(886).normal(size=(2, 3, 128)).astype(np.float32)
    values[0, 0] = 0
    values[0, 1, :8] = [127, -127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5]
    codes = np.empty_like(values, dtype=np.int8)
    scales = np.empty(values.shape[:2], dtype=np.float32)
    status = prepare(
        values.ctypes.data_as(prepare.argtypes[0]), 2, 3,
        codes.ctypes.data_as(prepare.argtypes[3]), scales.ctypes.data_as(prepare.argtypes[4]),
    )
    assert status == 0
    expected_codes, expected_scales = quantize_a8_per_group(values.reshape(2, -1))
    np.testing.assert_array_equal(codes.reshape(2, -1), expected_codes)
    np.testing.assert_array_equal(scales, expected_scales)
    for invalid in (np.nan, np.inf):
        values[1, 2, 0] = invalid
        assert prepare(values.ctypes.data_as(prepare.argtypes[0]), 2, 3,
                       codes.ctypes.data_as(prepare.argtypes[3]), scales.ctypes.data_as(prepare.argtypes[4])) == 2
    values.fill(np.nextafter(np.float32(0), np.float32(1)))
    assert prepare(values.ctypes.data_as(prepare.argtypes[0]), 2, 3,
                   codes.ctypes.data_as(prepare.argtypes[3]), scales.ctypes.data_as(prepare.argtypes[4])) == 2
    codes.fill(11)
    for tokens, groups in ((0, 3), (129, 3), (2, 97)):
        assert prepare(values.ctypes.data_as(prepare.argtypes[0]), tokens, groups,
                       codes.ctypes.data_as(prepare.argtypes[3]), scales.ctypes.data_as(prepare.argtypes[4])) == 1
        np.testing.assert_array_equal(codes, 11)


@pytest.mark.parametrize("groups", [32, 96])
def test_bitnet_dot_real_projection_widths_and_multiple_rows(native_dot, groups):
    generator = np.random.default_rng(76 + groups)
    codes = generator.integers(-1, 2, size=(5, groups, 128), dtype=np.int8)
    activations = generator.integers(-128, 128, size=(groups, 128), dtype=np.int8)
    weight_scales = generator.uniform(0.05, 1.5, size=(5, groups)).astype(np.float32)
    activation_scales = generator.uniform(0.01, 0.3, size=groups).astype(np.float32)
    _, actual = call_native(native_dot[0], codes, activations, weight_scales, activation_scales)
    expected = reference(codes, activations, weight_scales, activation_scales)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=0.005)


@pytest.mark.parametrize("groups", [2, 32, 96])
def test_multi_token_rows_and_group_scales_match_reference(native_dot, groups):
    generator = np.random.default_rng(221 + groups)
    tokens, rows = 3, 5
    codes = generator.integers(-1, 2, size=(rows, groups, 128), dtype=np.int8)
    activations = generator.integers(-128, 128, size=(tokens, groups, 128), dtype=np.int8)
    weight_scales = generator.uniform(0.05, 1.5, size=(rows, groups)).astype(np.float32)
    activation_scales = generator.uniform(0.01, 0.3, size=(tokens, groups)).astype(np.float32)
    activation_scales[1, 0] = 0.0
    output = call_native_batch(native_dot[1], codes, activations, weight_scales, activation_scales)
    group_dots = np.einsum(
        "rgi,tgi->trg", codes.astype(np.int32), activations.astype(np.int32)
    )
    expected = (group_dots * weight_scales[None, :, :] * activation_scales[:, None, :]).sum(axis=2)
    np.testing.assert_allclose(output, expected, rtol=2e-5, atol=0.005)


@pytest.mark.parametrize("block_size", [128, 1024])
def test_matching_rotation_then_a8_then_native_batch(native_dot, block_size):
    generator = np.random.default_rng(300 + block_size)
    width = block_size * 2
    groups = width // 128
    signs = generator.choice([-1, 1], size=width)
    inputs = generator.normal(size=(3, width)).astype(np.float32)
    inputs[0, 0] = 12.0
    codes = generator.integers(-1, 2, size=(4, groups, 128), dtype=np.int8)
    weight_scales = generator.uniform(0.05, 0.4, size=(4, groups)).astype(np.float32)
    rotated_weights = (codes * weight_scales[..., None]).reshape(4, width)
    rotation = np.zeros((width, width))
    for start in range(0, width, block_size):
        rotation[start : start + block_size, start : start + block_size] = (
            signs[start : start + block_size, None] * hadamard(block_size)
            / np.sqrt(block_size)
        )
    original_weights = rotated_weights @ rotation.T
    rotated_inputs = rotate_signed_hadamard(inputs, signs, block_size)
    np.testing.assert_allclose(
        inputs @ original_weights.T, rotated_inputs @ rotated_weights.T,
        rtol=1e-5, atol=0.001,
    )

    activations, activation_scales = quantize_a8_per_group(rotated_inputs)
    grouped_activations = activations.reshape(3, groups, 128)
    native = call_native_batch(
        native_dot[1], codes, grouped_activations,
        weight_scales, activation_scales,
    )
    group_dots = np.einsum(
        "rgi,tgi->trg", codes.astype(np.int32), grouped_activations.astype(np.int32)
    )
    quantized_reference = (
        group_dots * weight_scales[None, :, :] * activation_scales[:, None, :]
    ).sum(axis=-1)
    np.testing.assert_allclose(native, quantized_reference, rtol=2e-5, atol=0.005)
    error_bound = (
        np.abs(codes).sum(axis=-1)[None, :, :] * weight_scales[None, :, :]
        * activation_scales[:, None, :] / 2
    ).sum(axis=-1)
    assert np.all(np.abs(native - rotated_inputs @ rotated_weights.T) < error_bound + 0.005)


@pytest.mark.parametrize("groups", [2, 32])
def test_saved_fp16_group_scales_feed_native_batch(native_dot, groups):
    generator = np.random.default_rng(901 + groups)
    weights = generator.normal(size=(4, groups * 128)).astype(np.float32)
    weights[0, :128] = 0.0
    weights[1, 0] = 1.0004
    weights[1, 1] = 0.5001
    codes, stored_scales = quantize_ternary_rtn(weights)
    restored = reconstruct_ternary(codes, stored_scales)
    activations = generator.integers(-128, 128, size=(3, groups, 128), dtype=np.int8)
    activation_scales = generator.uniform(0.01, 0.3, size=(3, groups)).astype(np.float32)
    actual = call_native_batch(
        native_dot[1], codes.reshape(4, groups, 128), activations,
        stored_scales.astype(np.float32), activation_scales,
    )
    scaled_activations = (
        activations.astype(np.float32) * activation_scales[..., None]
    ).reshape(3, groups * 128)
    expected = np.einsum("ri,ti->tr", restored, scaled_activations)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=0.01)


@pytest.mark.parametrize("groups", [2, 32, 96])
def test_resident_fp16_blocks_match_expanded_scale_kernel_exactly(native_dot, groups):
    function = native_dot[3]
    function.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int8),
                        ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
                        ctypes.c_size_t, ctypes.POINTER(ctypes.c_float)]
    function.restype = ctypes.c_int
    generator = np.random.default_rng(1519 + groups)
    codes = generator.integers(-1, 2, size=(5, groups, 128), dtype=np.int8)
    activations = generator.integers(-128, 128, size=(3, groups, 128), dtype=np.int8)
    scales = generator.uniform(0, 1, size=(5, groups)).astype("<f2")
    scales.reshape(-1)[:5] = [0, -0.0, np.nextafter(np.float16(0), np.float16(1)), 1, 65504]
    activation_scales = generator.uniform(0.01, 0.2, size=(3, groups)).astype(np.float32)
    blocks = np.concatenate((scales.view(np.uint8).reshape(5, groups, 2), pack_codes(codes)), axis=-1)
    output = np.full((3, 5), -19, dtype=np.float32)
    def compute(payload):
        return function(payload.ctypes.data_as(function.argtypes[0]),
                        activations.ctypes.data_as(function.argtypes[1]),
                        activation_scales.ctypes.data_as(function.argtypes[2]),
                        3, 5, groups, output.ctypes.data_as(function.argtypes[-1]))

    assert compute(blocks) == 0
    expected = call_native_batch(native_dot[1], codes, activations, scales.astype(np.float32), activation_scales)
    np.testing.assert_array_equal(output, expected)
    for scale_bits in (0xbc00, 0x7c00, 0x7e00):
        invalid = blocks.copy()
        invalid[-1, -1, :2] = [scale_bits & 255, scale_bits >> 8]
        output.fill(-19)
        assert compute(invalid) == 5
        np.testing.assert_array_equal(output, -19)
    invalid = blocks.copy()
    invalid[-1, -1, -1] |= 3
    assert compute(invalid) == 5
    np.testing.assert_array_equal(output, -19)


def test_compensated_artifact_executes_with_saved_scales(native_dot):
    generator = np.random.default_rng(119)
    rows, groups, tokens = 3, 2, 4
    width = groups * 128
    weights = generator.normal(size=(rows, width)).astype(np.float32)
    calibration = generator.normal(size=(24, width)).astype(np.float32)
    calibration[:, 1] = calibration[:, 0] * 0.9 + calibration[:, 1] * 0.1
    codes, saved_scales, details = quantize_ternary_compensated(weights, calibration)
    assert details["damping"] > 0
    artifact = save_toy_artifact(codes, saved_scales, algorithm="gptq-style-maxabs-fp16-v1")
    codes, saved_scales, manifest = load_toy_artifact(artifact)
    assert manifest["group_size"] == 128

    activations = generator.integers(-128, 128, size=(tokens, groups, 128), dtype=np.int8)
    activation_scales = generator.uniform(0.01, 0.3, size=(tokens, groups)).astype(np.float32)
    actual = call_native_batch(
        native_dot[1], codes.reshape(rows, groups, 128), activations,
        saved_scales.astype(np.float32), activation_scales,
    )
    restored = reconstruct_ternary(codes, saved_scales)
    scaled_activations = (
        activations.astype(np.float32) * activation_scales[..., None]
    ).reshape(tokens, width)
    np.testing.assert_allclose(
        actual, np.einsum("ri,ti->tr", restored, scaled_activations),
        rtol=2e-5, atol=0.01,
    )


def test_loaded_signed_transform_and_scales_feed_native_batch(native_dot):
    generator = np.random.default_rng(481)
    rows, tokens, width, block_size = 4, 3, 256, 128
    groups = width // 128
    signs = generator.choice([-1, 1], size=width)
    weights = generator.normal(size=(rows, width)).astype(np.float32)
    rotated_weights = rotate_signed_hadamard(weights, signs, block_size)
    codes, scales = quantize_ternary_rtn(rotated_weights)
    payload = save_toy_artifact(
        codes, scales, algorithm="rtn-maxabs-fp16-v1",
        signs=signs, block_size=block_size,
    )
    codes, scales, manifest = load_toy_artifact(payload)
    transform = manifest["transform"]
    inputs = generator.normal(size=(tokens, width)).astype(np.float32)
    rotated_inputs = rotate_signed_hadamard(
        inputs, transform["signs"], transform["block_size"]
    )
    activations, activation_scales = quantize_a8_per_group(rotated_inputs)
    actual = call_native_batch(
        native_dot[1], codes.reshape(rows, groups, 128),
        activations.reshape(tokens, groups, 128),
        scales.astype(np.float32), activation_scales,
    )
    dequantized_inputs = (
        activations.reshape(tokens, groups, 128).astype(np.float32)
        * activation_scales[..., None]
    ).reshape(tokens, width)
    expected = np.einsum("ri,ti->tr", reconstruct_ternary(codes, scales), dequantized_inputs)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=0.01)


@pytest.mark.skipif(
    os.environ.get("MIMO_BF16_SLICE_TEST") != "1",
    reason="explicitly opt in to a pinned 2 KiB MiMo weight range read",
)
def test_bounded_mimo_slice_rotates_and_executes_saved_native_artifact(native_dot):
    metadata, headers = fetch_pinned_headers()
    weights, source = fetch_bf16_projection_slice(metadata, headers)
    assert source["revision"] == MODEL_REVISION and source["payload_bytes"] == 2048
    rows, width = weights.shape
    groups = width // 128
    generator = np.random.default_rng(773)
    signs = generator.choice([-1, 1], size=width)
    inputs = generator.normal(size=(3, width)).astype(np.float32)
    rotated_weights = rotate_signed_hadamard(weights, signs, 128)
    rotated_inputs = rotate_signed_hadamard(inputs, signs, 128)
    np.testing.assert_allclose(
        inputs @ weights.T, rotated_inputs @ rotated_weights.T, rtol=1e-5, atol=1e-4
    )
    codes, scales = quantize_ternary_rtn(rotated_weights, scale_search=True)
    payload = save_toy_artifact(
        codes, scales, algorithm="rtn-maxabs-fp16-v1", signs=signs, block_size=128
    )
    saved_codes, saved_scales, manifest = load_toy_artifact(payload)
    transformed_inputs = rotate_signed_hadamard(
        inputs, manifest["transform"]["signs"], manifest["transform"]["block_size"]
    )
    activations, activation_scales = quantize_a8_per_group(transformed_inputs)
    actual = call_native_batch(
        native_dot[1], saved_codes.reshape(rows, groups, 128),
        activations.reshape(3, groups, 128),
        saved_scales.astype(np.float32), activation_scales,
    )
    dequantized_inputs = (
        activations.reshape(3, groups, 128).astype(np.float32)
        * activation_scales[..., None]
    ).reshape(3, width)
    expected = np.einsum(
        "ri,ti->tr", reconstruct_ternary(saved_codes, saved_scales), dequantized_inputs
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=0.01)


@pytest.mark.skipif(
    os.environ.get("MIMO_FULL_PROJECTION_TEST") != "1",
    reason="explicitly opt in to streaming one complete MiMo projection through the native kernel",
)
def test_full_mimo_projection_matches_bitnet_derived_native_dot(native_dot):
    local_dir = os.environ.get("MIMO_LOCAL_DIR")
    if not local_dir:
        pytest.fail("MIMO_LOCAL_DIR must point to the verified pinned MiMo snapshot")
    directory = Path(local_dir)
    metadata, headers = read_local_headers(directory)
    report = build_inventory(metadata, headers)
    assert report["source"]["revision"] == MODEL_REVISION
    tensor = next(entry for entry in report["tensors"] if entry["name"] == DEFAULT_TENSOR)
    rows, columns = tensor["shape"]
    assert (rows, columns) == (4096, 12288)
    assert tensor["storage_bytes"] <= MAX_STREAM_TENSOR_BYTES
    groups = columns // 128
    header, file_bytes = headers[tensor["shard"]]
    start, end = _json_object(header[8:], tensor["shard"])[DEFAULT_TENSOR]["data_offsets"]
    assert end - start == tensor["storage_bytes"] and len(header) + end <= file_bytes

    inputs = np.random.default_rng(903).normal(size=(1, columns)).astype(np.float32)
    activations, activation_scales = quantize_a8_per_group(inputs)
    grouped_activations = activations.reshape(groups, 128)
    packed = np.empty((rows, groups, 32), dtype=np.uint8)
    weight_scales = np.empty((rows, groups), dtype=np.float32)
    expected = np.empty(rows, dtype=np.float32)
    dense = np.empty(rows, dtype=np.float32)
    with (directory / tensor["shard"]).open("rb") as source:
        source.seek(len(header) + start)
        for start_row in range(0, rows, STREAM_ROWS):
            batch_rows = min(STREAM_ROWS, rows - start_row)
            data = source.read(batch_rows * columns * 2)
            assert len(data) == batch_rows * columns * 2
            weights = (np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16).view("<f4")
            weights = weights.reshape(batch_rows, columns)
            assert np.isfinite(weights).all()
            codes, scales = quantize_ternary_rtn(weights, scale_search=True)
            grouped_codes = codes.reshape(batch_rows, groups, 128)
            stop_row = start_row + batch_rows
            packed[start_row:stop_row] = pack_codes(grouped_codes)
            weight_scales[start_row:stop_row] = scales.astype(np.float32)
            group_dots = np.einsum(
                "rgi,gi->rg", grouped_codes.astype(np.int32), grouped_activations.astype(np.int32)
            )
            expected[start_row:stop_row] = (
                group_dots * weight_scales[start_row:stop_row] * activation_scales[0]
            ).sum(axis=1)
            dense[start_row:stop_row] = weights @ inputs[0]

    assert packed.nbytes == rows * groups * 32 and np.isfinite(weight_scales).all()
    actual = np.empty(rows, dtype=np.float32)
    status = native_dot[0](
        packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        weight_scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        grouped_activations.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        activation_scales[0].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        rows, groups, actual.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    assert status == 0
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=0.005)
    relative_synthetic_error = float(np.linalg.norm(actual - dense) / np.linalg.norm(dense))
    assert np.isfinite(relative_synthetic_error)
    print(json.dumps({
        "purpose": "full_projection_native_parity_not_model_quality",
        "tensor": DEFAULT_TENSOR,
        "rows": rows,
        "groups": groups,
        "packed_bytes": packed.nbytes,
        "max_native_reference_error": float(np.max(np.abs(actual - expected))),
        "relative_synthetic_output_error": relative_synthetic_error,
    }, sort_keys=True))


@pytest.mark.skipif(
    os.environ.get("MIMO_CALIBRATED_SLICE_TEST") != "1",
    reason="requires pinned MiMo snapshot, a calibration capture, and isolated Torch for live validation",
)
@pytest.mark.parametrize("width", [256, 12288])
def test_calibration_fitted_slice_native_dot_on_live_validation(native_dot, width):
    from embedded_jev.decision_dataset import (
        load_balanced_calibration_captures, load_calibration_capture, load_decision_dataset,
    )
    from embedded_jev.weight_slice import read_local_bf16_projection_rows
    from embedded_jev.ternary import quantize_block_diagonal_compensated

    local_dir = os.environ.get("MIMO_LOCAL_DIR")
    interpreter = os.environ.get("MIMO_DENSE_PYTHON")
    dataset_path = os.environ.get("MIMO_PUBLIC_DATASET")
    capture_path = os.environ.get("MIMO_CALIBRATION_CAPTURE")
    if not all((local_dir, interpreter, dataset_path, capture_path)):
        pytest.fail("set MiMo interpreter/snapshot, MIMO_PUBLIC_DATASET, and MIMO_CALIBRATION_CAPTURE")
    calibration, manifest = load_calibration_capture(Path(capture_path))
    additional = os.environ.get("MIMO_ADDITIONAL_CALIBRATION_CAPTURES")
    if additional:
        calibration, selection = load_balanced_calibration_captures(
            [Path(capture_path)] + [Path(path) for path in additional.split(os.pathsep)],
        )
        assert selection["dataset"]["sha256"] == manifest["dataset"]["sha256"]
    dataset, digest = load_decision_dataset(Path(dataset_path))
    assert manifest["dataset"]["sha256"] == digest
    weights, provenance = read_local_bf16_projection_rows(Path(local_dir))
    assert provenance["tensor"] == manifest["tensor"]
    weights = weights[:, :width]
    if width == 256:
        codes, scales, details = quantize_ternary_compensated(
            weights, calibration[:, :width], processing_block_size=128, scale_search=True,
        )
        assert details["damping"] > 0
    else:
        codes, scales, details = quantize_block_diagonal_compensated(
            weights, calibration[:, :width], scale_search=True,
        )
        assert details["cross_block_curvature"] is False
        assert len(details["blocks"]) == 48 and all(block["damping"] > 0 for block in details["blocks"])
    assert scales.dtype == np.float16
    validation_case = dataset["splits"]["validation"][0]["id"]
    script = """
import json
import sys
from pathlib import Path
from embedded_jev.streamed_text import run_streamed_text
observed = []
report = run_streamed_text(
    Path(sys.argv[1]), layers=4, prompt="", dataset_path=Path(sys.argv[2]),
    split="validation", case_id=sys.argv[3], activation_observer=observed.append,
)
assert len(observed) == 1 and not observed[0].flags.writeable
print(json.dumps({"values": observed[0][:, :int(sys.argv[4])].tolist(), "dataset_sha256": report["dataset"]["sha256"]}))
"""
    result = subprocess.run(
        [interpreter, "-c", script, local_dir, dataset_path, validation_case, str(width)],
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
             "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4"},
        check=True, capture_output=True, text=True, timeout=180,
    )
    assert len(result.stdout) <= 24 << 20
    validation = json.loads(result.stdout)
    assert validation["dataset_sha256"] == digest
    values = np.asarray(validation["values"], dtype=np.float32)
    assert values.ndim == 2 and 1 <= values.shape[0] <= 128 and values.shape[1] == width
    groups = width // 128
    activations, activation_scales = quantize_a8_per_group(values)
    actual = call_native_batch(
        native_dot[1], codes.reshape(4, groups, 128), activations.reshape(values.shape[0], groups, 128),
        scales.astype(np.float32), activation_scales,
    )
    partial = np.einsum(
        "rgi,tgi->trg", codes.reshape(4, groups, 128).astype(np.int32),
        activations.reshape(values.shape[0], groups, 128).astype(np.int32),
    )
    expected = (partial * scales.astype(np.float32)[None] * activation_scales[:, None]).sum(axis=-1)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-5)
    assert np.isfinite(actual).all()


def test_multi_token_rejects_empty_and_overflowed_shapes(native_dot):
    packed = np.zeros(32, dtype=np.uint8)
    scale = np.ones(1, dtype=np.float32)
    activations = np.zeros(128, dtype=np.int8)
    output = np.full(1, 123.0, dtype=np.float32)
    pointers = (
        packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        scale.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        activations.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        scale.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    output_pointer = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    limit = ctypes.c_size_t(-1).value
    for rows, groups in ((0, 1), (1, 0), (1, limit), (limit, 1), (limit, 2)):
        assert native_dot[0](*pointers, rows, groups, output_pointer) == 1
    assert native_dot[1](*pointers, 0, 1, 1, output_pointer) == 1
    assert native_dot[1](*pointers, 1, 1, limit, output_pointer) == 1
    assert native_dot[1](*pointers, limit // 128, 33, 1, output_pointer) == 1
    assert output[0] == 123.0