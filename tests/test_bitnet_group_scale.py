"""Golden checks for the isolated BitNet-derived AVX2 group-scaled dot."""

import ctypes
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import hadamard

from embedded_jev.activation import quantize_a8_per_group, rotate_signed_hadamard
from embedded_jev.ternary import quantize_ternary_rtn, reconstruct_ternary


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
    return function, batch


def pack_codes(codes):
    rows, groups, group_width = codes.shape
    assert group_width == 128 and np.isin(codes, [-1, 0, 1]).all()
    lanes = (codes + 1).astype(np.uint8).reshape(rows, groups, 4, 32)
    packed = np.zeros((rows, groups, 32), dtype=np.uint8)
    for lane in range(4):
        packed |= lanes[:, :, lane, :] << (6 - 2 * lane)
    return packed


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
    assert native_dot[1](*pointers, 0, 1, 1, output_pointer) == 1
    assert native_dot[1](*pointers, 1, 1, ctypes.c_size_t(-1).value, output_pointer) == 1
    assert output[0] == 123.0