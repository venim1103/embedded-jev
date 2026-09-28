"""Golden checks for the isolated BitNet-derived AVX2 group-scaled dot."""

import ctypes
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


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
    function = ctypes.CDLL(str(library)).bitnet_group_scale_matvec_avx2
    function.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_float),
    ]
    function.restype = ctypes.c_int
    return function


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


def test_bitnet_layout_zero_compensation_and_row_dependent_scales(native_dot):
    codes = np.zeros((3, 2, 128), dtype=np.int8)
    codes[1, 0, [0, 32, 64, 96]] = [-1, 0, 1, 0]
    codes[1, 1, :] = 1
    codes[2, 0, :] = -1
    codes[2, 1, ::2] = 1
    activations = np.tile(np.array([-128, 127, 7, -3], dtype=np.int8), 64).reshape(2, 128)
    weight_scales = np.array([[0.5, 1.0], [0.25, 3.0], [5.0, 0.125]], dtype=np.float32)
    activation_scales = np.array([0.25, 2.0], dtype=np.float32)
    packed, actual = call_native(native_dot, codes, activations, weight_scales, activation_scales)
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
    _, actual = call_native(native_dot, codes, activations, weight_scales, activation_scales)
    expected = reference(codes, activations, weight_scales, activation_scales)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=0.005)