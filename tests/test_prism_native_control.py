"""Optional pinned Prism CPU transform graph control without model weights."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


PRISM_REVISION = "842b1880415d6f508f03b789e5ce70194def7bfd"
BITNET_REVISION = "0b341e582afbf9e1011f24744b554c96a3477eb5"


def test_pinned_prism_cpu_fwht_matches_dense_signed_reference(tmp_path):
    source_dir = os.environ.get("PRISM_SOURCE_DIR")
    library_name = os.environ.get("PRISM_GGML_CPU_LIBRARY")
    if not source_dir or not library_name:
        pytest.skip("requires pinned Prism source and built CPU library")
    source = Path(source_dir)
    library = Path(library_name)
    if not source.is_dir() or not library.is_file():
        pytest.fail("missing pinned Prism source or CPU library")
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == PRISM_REVISION
    compiler = shutil.which("clang++-18")
    if compiler is None:
        pytest.skip("clang++-18 is not installed")

    program = Path(__file__).resolve().parents[1] / "native" / "prism_hadamard_smoke.cpp"
    binary = tmp_path / "prism-hadamard-smoke"
    library_dir = library.parent
    subprocess.run(
        [
            compiler, "-std=c++17", "-O2", "-I", str(source / "ggml" / "include"),
            str(program), "-L", str(library_dir), f"-Wl,-rpath,{library_dir}",
            "-lggml-cpu", "-lggml-base", "-o", str(binary),
        ],
        check=True, capture_output=True, text=True,
    )
    result = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True, timeout=15
    )
    report = json.loads(result.stdout.strip())
    assert report["tokens"] == 2 and report["width"] == 128
    assert report["max_abs_error"] < 1e-4


def test_pinned_prism_fwht_feeds_bitnet_derived_group_scale_kernel(tmp_path):
    prism_source = os.environ.get("PRISM_SOURCE_DIR")
    prism_library = os.environ.get("PRISM_GGML_CPU_LIBRARY")
    bitnet_source = os.environ.get("BITNET_SOURCE_DIR")
    if not prism_source or not prism_library or not bitnet_source:
        pytest.skip("requires pinned Prism and BitNet source plus Prism CPU library")
    revisions = ((prism_source, PRISM_REVISION), (bitnet_source, BITNET_REVISION))
    for directory, expected in revisions:
        revision = subprocess.check_output(
            ["git", "-C", directory, "rev-parse", "HEAD"], text=True
        ).strip()
        assert revision == expected
    compiler = shutil.which("clang++-18")
    if compiler is None:
        pytest.skip("clang++-18 is not installed")
    library_dir = Path(prism_library).parent
    sources = Path(__file__).resolve().parents[1] / "native"
    binary = tmp_path / "prism-bitnet-bridge"
    subprocess.run(
        [
            compiler, "-std=c++17", "-O2", "-mavx2",
            "-I", str(Path(prism_source) / "ggml" / "include"),
            str(sources / "prism_bitnet_bridge.cpp"),
            str(sources / "bitnet_group_scale.cpp"),
            "-L", str(library_dir), f"-Wl,-rpath,{library_dir}",
            "-lggml-cpu", "-lggml-base", "-o", str(binary),
        ],
        check=True, capture_output=True, text=True,
    )
    result = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True, timeout=15
    )
    report = json.loads(result.stdout.strip())
    assert report["tokens"] == 2 and report["groups"] == 2
    assert report["gdn_v_grouped"] is False
    assert report["graph_op"] == "map_custom2"
    assert report["graph_evaluations"] == report["callback_calls"] == 2
    assert report["max_transform_error"] < 1e-4
    assert report["max_output_error"] < 0.005
    assert report["repeat_scale_error"] < 0.005
    grouped_result = subprocess.run(
        [str(binary), "--grouped-v"], check=True,
        capture_output=True, text=True, timeout=15,
    )
    grouped = json.loads(grouped_result.stdout.strip())
    assert grouped["gdn_v_grouped"] is True
    assert grouped["tokens"] == report["tokens"] == 2
    assert grouped["graph_evaluations"] == grouped["callback_calls"] == 2
    assert grouped["max_transform_error"] < 1e-4
    assert grouped["max_output_error"] < 0.005
    assert grouped["repeat_scale_error"] < 0.005


def test_pinned_prism_shared_group_scale_graph_matches_direct_kernel(tmp_path):
    import ctypes
    import platform

    import numpy as np

    from embedded_jev.ternary import pack_group128_codes

    source_dir = os.environ.get("PRISM_SOURCE_DIR")
    library_name = os.environ.get("PRISM_GGML_CPU_LIBRARY")
    compiler = shutil.which("clang++-18")
    if not source_dir or not library_name or compiler is None:
        pytest.skip("requires pinned Prism CPU source/library and Clang 18")
    if platform.machine() != "x86_64" or "avx2" not in Path("/proc/cpuinfo").read_text():
        pytest.skip("requires x86-64 AVX2")
    assert subprocess.check_output(["git", "-C", source_dir, "rev-parse", "HEAD"], text=True).strip() == PRISM_REVISION
    library_dir = Path(library_name).parent
    sources = Path(__file__).resolve().parents[1] / "native"
    binary = tmp_path / "prism_group_scale.so"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-mavx2", "-shared", "-fPIC",
         "-I", str(Path(source_dir) / "ggml" / "include"),
         "-I", str(Path(source_dir) / "ggml" / "src"),
         "-I", str(Path(source_dir) / "ggml" / "src" / "ggml-cpu"),
         str(sources / "prism_group_scale.cpp"), str(sources / "bitnet_group_scale.cpp"),
         "-L", str(library_dir), f"-Wl,-rpath,{library_dir}", "-lggml-cpu", "-lggml-base", "-o", str(binary)],
        check=True, capture_output=True, text=True,
    )
    native = ctypes.CDLL(str(binary))
    from embedded_jev.streamed_text import native_backend_dependencies

    dependencies = native_backend_dependencies(binary, "prism_ggml")
    assert set(dependencies) == {"ggml_cpu", "ggml_base"}
    assert Path(dependencies["ggml_cpu"]["path"]) == Path(library_name).resolve()
    assert all(len(record["sha256"]) == 64 for record in dependencies.values())
    arguments = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
                 ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_float),
                 ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_float)]
    graph = native.prism_bitnet_group_scale_matmul_avx2
    direct = native.bitnet_group_scale_matmul_avx2
    for function in (graph, direct):
        function.argtypes = arguments
        function.restype = ctypes.c_int
    generator = np.random.default_rng(884)
    codes = generator.integers(-1, 2, size=(5, 3, 128), dtype=np.int8)
    packed = pack_group128_codes(codes)
    scales = generator.uniform(0.01, 1.0, size=(5, 3)).astype(np.float16).astype(np.float32)
    activations = generator.integers(-127, 128, size=(2, 3, 128), dtype=np.int8)
    activation_scales = generator.uniform(0.01, 0.2, size=(2, 3)).astype(np.float32)
    pointers = [packed.ctypes.data_as(arguments[0]), scales.ctypes.data_as(arguments[1]),
                activations.ctypes.data_as(arguments[2]), activation_scales.ctypes.data_as(arguments[3])]
    expected = np.empty((2, 5), dtype=np.float32)
    actual = np.empty_like(expected)
    for multiplier in (1, -1, 2):
        activations[:] = generator.integers(-63, 64, size=activations.shape, dtype=np.int8) * multiplier
        assert direct(*pointers, 2, 5, 3, expected.ctypes.data_as(arguments[-1])) == 0
        assert graph(*pointers, 2, 5, 3, actual.ctypes.data_as(arguments[-1])) == 0
        np.testing.assert_array_equal(actual, expected)
    actual.fill(123)
    assert graph(*pointers, 129, 5, 3, actual.ctypes.data_as(arguments[-1])) == 1
    np.testing.assert_array_equal(actual, 123)
    from embedded_jev.activation import quantize_a8_per_group

    float_graph = native.prism_bitnet_group_scale_matmul_f32
    float_arguments = list(arguments)
    float_arguments[2] = ctypes.POINTER(ctypes.c_float)
    float_graph.argtypes = float_arguments
    float_graph.restype = ctypes.c_int
    inputs = generator.normal(size=(2, 384)).astype(np.float32)
    inputs[0, :128] = 0
    prepared, input_scales = quantize_a8_per_group(inputs)
    assert direct(
        pointers[0], pointers[1], prepared.ctypes.data_as(arguments[2]), input_scales.ctypes.data_as(arguments[3]),
        2, 5, 3, expected.ctypes.data_as(arguments[-1]),
    ) == 0
    assert float_graph(
        pointers[0], pointers[1], inputs.ctypes.data_as(float_graph.argtypes[2]), None,
        2, 5, 3, actual.ctypes.data_as(arguments[-1]),
    ) == 0
    np.testing.assert_array_equal(actual, expected)
    inputs[0, 0] = np.nan
    assert float_graph(
        pointers[0], pointers[1], inputs.ctypes.data_as(float_graph.argtypes[2]), None,
        2, 5, 3, actual.ctypes.data_as(arguments[-1]),
    ) == 2
    from embedded_jev.activation import rotate_signed_hadamard

    rotated_graph = native.prism_bitnet_group_scale_matmul_hadamard128
    rotated_graph.argtypes = float_arguments + [ctypes.POINTER(ctypes.c_float)]
    rotated_graph.restype = ctypes.c_int
    inputs = generator.normal(size=(2, 384)).astype(np.float32)
    signs = np.random.default_rng(773).choice([-1, 1], size=384).astype(np.float32)
    prepared, input_scales = quantize_a8_per_group(rotate_signed_hadamard(inputs, signs, 128))
    assert direct(
        pointers[0], pointers[1], prepared.ctypes.data_as(arguments[2]), input_scales.ctypes.data_as(arguments[3]),
        2, 5, 3, expected.ctypes.data_as(arguments[-1]),
    ) == 0
    assert rotated_graph(
        pointers[0], pointers[1], inputs.ctypes.data_as(float_arguments[2]), None,
        2, 5, 3, actual.ctypes.data_as(arguments[-1]), signs.ctypes.data_as(rotated_graph.argtypes[-1]),
    ) == 0
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-5)
    signs[0] = 0
    assert rotated_graph(
        pointers[0], pointers[1], inputs.ctypes.data_as(float_arguments[2]), None,
        2, 5, 3, actual.ctypes.data_as(arguments[-1]), signs.ctypes.data_as(rotated_graph.argtypes[-1]),
    ) == 1
    from embedded_jev.prism_codec import pack_ternary_pq2_0
    from embedded_jev.ternary import reconstruct_ternary

    registered = native.prism_bitnet_registered_tensor_matmul
    registered.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
                           ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                           ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_size_t)]
    registered.restype = ctypes.c_int
    repeated = native.prism_bitnet_registered_tensor_matmul_repeated
    repeated.argtypes = registered.argtypes[:5] + [ctypes.c_size_t] + registered.argtypes[5:] + [ctypes.POINTER(ctypes.c_size_t)]
    repeated.restype = ctypes.c_int
    registry_size = native.prism_bitnet_registered_tensor_registry_size
    registry_size.argtypes = []
    registry_size.restype = ctypes.c_size_t
    create_projection = native.prism_bitnet_registered_projection_create
    create_projection.argtypes = [registered.argtypes[0], ctypes.c_size_t, ctypes.c_size_t,
                                  ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
    create_projection.restype = ctypes.c_int
    compute_projection = native.prism_bitnet_registered_projection_compute
    compute_projection.argtypes = [ctypes.c_void_p, registered.argtypes[1], registered.argtypes[-2],
                                   registered.argtypes[-1], registered.argtypes[-1]]
    compute_projection.restype = ctypes.c_int
    free_projection = native.prism_bitnet_registered_projection_free
    free_projection.argtypes = [ctypes.c_void_p]
    free_projection.restype = None

    def check_projection_lifetime(blocks, values, reference):
        before = registry_size()
        tokens, rows = reference.shape[1:]
        groups = values.shape[-1] // 128
        copied_blocks = blocks.copy()
        handle = ctypes.c_void_p()
        assert create_projection(copied_blocks.ctypes.data_as(create_projection.argtypes[0]),
                                 tokens, rows, groups, ctypes.byref(handle)) == 0
        assert handle.value and registry_size() == before
        copied_blocks.fill(255)
        calls, repacks = ctypes.c_size_t(0), ctypes.c_size_t(0)
        output = np.empty((tokens, rows), dtype=np.float32)
        peer_handle = ctypes.c_void_p()
        try:
            output.fill(123)
            assert compute_projection(None, values[0].ctypes.data_as(compute_projection.argtypes[1]),
                                      output.ctypes.data_as(compute_projection.argtypes[2]),
                                      ctypes.byref(calls), ctypes.byref(repacks)) == 1
            assert calls.value == repacks.value == 0
            np.testing.assert_array_equal(output, 123)
            if rows == 5:
                peer_blocks = blocks.copy()
                peer_blocks.reshape(-1, 34)[:, :2] = 0
                assert create_projection(peer_blocks.ctypes.data_as(create_projection.argtypes[0]),
                                         tokens, rows, groups, ctypes.byref(peer_handle)) == 0
            for evaluation in range(2):
                assert compute_projection(handle, values[evaluation].ctypes.data_as(compute_projection.argtypes[1]),
                                          output.ctypes.data_as(compute_projection.argtypes[2]),
                                          ctypes.byref(calls), ctypes.byref(repacks)) == 0
                assert calls.value == evaluation + 1 and repacks.value == 1
                assert registry_size() == before
                np.testing.assert_array_equal(output, reference[evaluation])
                if evaluation == 0:
                    invalid = values[0].copy()
                    invalid[0, 0] = np.nan
                    output.fill(123)
                    assert compute_projection(handle, invalid.ctypes.data_as(compute_projection.argtypes[1]),
                                              output.ctypes.data_as(compute_projection.argtypes[2]),
                                              ctypes.byref(calls), ctypes.byref(repacks)) == 2
                    assert calls.value == 1 and repacks.value == 1
                    assert registry_size() == before
                    np.testing.assert_array_equal(output, 123)
                    if peer_handle.value:
                        assert compute_projection(peer_handle, values[0].ctypes.data_as(compute_projection.argtypes[1]),
                                                  output.ctypes.data_as(compute_projection.argtypes[2]),
                                                  ctypes.byref(calls), ctypes.byref(repacks)) == 0
                        assert calls.value == repacks.value == 1
                        np.testing.assert_array_equal(output, 0)
                        assert registry_size() == before
        finally:
            free_projection(peer_handle)
            free_projection(handle)
        assert registry_size() == before
        handle = ctypes.c_void_p(123)
        assert create_projection(copied_blocks.ctypes.data_as(create_projection.argtypes[0]),
                                 tokens, rows, groups, ctypes.byref(handle)) == 5
        assert handle.value is None and registry_size() == before

    for groups in (2, 3):
        codes = generator.integers(-1, 2, size=(5, groups, 128), dtype=np.int8)
        fp16_scales = generator.uniform(0.01, 1.0, size=(5, groups)).astype(np.float16)
        fp16_scales[0, 0] = 0
        codes[0, 0] = 0
        pq2_blocks = pack_ternary_pq2_0(codes, fp16_scales)
        packed = pack_group128_codes(codes)
        scales = fp16_scales.astype(np.float32)
        inputs = generator.normal(size=(2, groups * 128)).astype(np.float32)
        inputs[0, :128] = 0
        for tokens, multiplier in ((1, 1), (2, -2)):
            values = np.ascontiguousarray(inputs[:tokens] * multiplier)
            prepared, input_scales = quantize_a8_per_group(values)
            assert direct(packed.ctypes.data_as(arguments[0]), scales.ctypes.data_as(arguments[1]),
                          prepared.ctypes.data_as(arguments[2]), input_scales.ctypes.data_as(arguments[3]),
                          tokens, 5, groups, expected.ctypes.data_as(arguments[-1])) == 0
            calls = ctypes.c_size_t(0)
            assert registered(pq2_blocks.ctypes.data_as(registered.argtypes[0]),
                              values.ctypes.data_as(registered.argtypes[1]), tokens, 5, groups,
                              actual.ctypes.data_as(registered.argtypes[-2]), ctypes.byref(calls)) == 0
            assert calls.value == 1
            np.testing.assert_array_equal(actual[:tokens], expected[:tokens])
        values = np.stack((inputs, -2 * inputs)).astype(np.float32)
        expected_repeated = np.empty((2, 2, 5), dtype=np.float32)
        actual_repeated = np.empty_like(expected_repeated)
        for evaluation in range(2):
            prepared, input_scales = quantize_a8_per_group(values[evaluation])
            assert direct(packed.ctypes.data_as(arguments[0]), scales.ctypes.data_as(arguments[1]),
                          prepared.ctypes.data_as(arguments[2]), input_scales.ctypes.data_as(arguments[3]),
                          2, 5, groups, expected_repeated[evaluation].ctypes.data_as(arguments[-1])) == 0
        before = registry_size()
        calls, repacks = ctypes.c_size_t(0), ctypes.c_size_t(0)
        assert repeated(pq2_blocks.ctypes.data_as(repeated.argtypes[0]), values.ctypes.data_as(repeated.argtypes[1]),
                        2, 5, groups, 2, actual_repeated.ctypes.data_as(repeated.argtypes[-3]),
                        ctypes.byref(calls), ctypes.byref(repacks)) == 0
        assert calls.value == 2 and repacks.value == 1
        assert registry_size() == before
        np.testing.assert_array_equal(actual_repeated, expected_repeated)
        check_projection_lifetime(pq2_blocks, values, expected_repeated)
        values[1, 0, 0] = np.nan
        actual_repeated.fill(123)
        assert repeated(pq2_blocks.ctypes.data_as(repeated.argtypes[0]), values.ctypes.data_as(repeated.argtypes[1]),
                        2, 5, groups, 2, actual_repeated.ctypes.data_as(repeated.argtypes[-3]),
                        ctypes.byref(calls), ctypes.byref(repacks)) == 2
        assert calls.value == 1 and repacks.value == 1
        assert registry_size() == before
        np.testing.assert_array_equal(actual_repeated, 123)

    valid_blocks = pq2_blocks.copy()
    for invalid_kind in ("plus_two", "negative_scale", "nan_scale", "inf_scale", "nan_input", "token_limit"):
        invalid_blocks = valid_blocks.copy()
        invalid_inputs = inputs.copy()
        tokens = 2
        if invalid_kind == "plus_two":
            invalid_blocks.flat[2] = (int(invalid_blocks.flat[2]) & ~3) | 3
        elif invalid_kind in ("negative_scale", "nan_scale", "inf_scale"):
            scale = {"negative_scale": -1, "nan_scale": np.nan, "inf_scale": np.inf}[invalid_kind]
            invalid_blocks.reshape(-1, 34)[0, :2] = np.array([scale], dtype="<f2").view(np.uint8)
        elif invalid_kind == "nan_input":
            invalid_inputs[0, 0] = np.nan
        else:
            tokens = 129
        actual.fill(123)
        calls = ctypes.c_size_t(99)
        before = registry_size()
        status = registered(invalid_blocks.ctypes.data_as(registered.argtypes[0]),
                            invalid_inputs.ctypes.data_as(registered.argtypes[1]), tokens, 5, groups,
                            actual.ctypes.data_as(registered.argtypes[-2]), ctypes.byref(calls))
        assert status == (1 if invalid_kind == "token_limit" else 2 if invalid_kind == "nan_input" else 5)
        assert calls.value == 0
        assert registry_size() == before
        np.testing.assert_array_equal(actual, 123)

    pq2 = native.prism_pq2_tensor_matmul
    pq2.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
                    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_float)]
    pq2.restype = ctypes.c_int
    for groups in (2, 3):
        codes = generator.integers(-1, 2, size=(5, groups, 128), dtype=np.int8)
        fp16_scales = generator.uniform(0.01, 1.0, size=(5, groups)).astype(np.float16)
        pq2_blocks = pack_ternary_pq2_0(codes, fp16_scales)
        inputs = np.ones((2, groups * 128), dtype=np.float32)
        inputs[1] *= -2
        base = ctypes.CDLL(str(library_dir / "libggml-base.so"))
        suffix = "q8_K" if groups % 2 == 0 else "q8_0"
        quantizer = getattr(base, f"quantize_row_{suffix}_ref")
        decoder = getattr(base, f"dequantize_row_{suffix}")
        quantizer.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int64]
        quantizer.restype = None
        decoder.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
        decoder.restype = None
        block_width, block_bytes = (256, 292) if suffix == "q8_K" else (32, 34)
        restored_inputs = np.empty_like(inputs)
        for token in range(2):
            encoded = np.empty(inputs.shape[1] // block_width * block_bytes, dtype=np.uint8)
            quantizer(inputs[token].ctypes.data_as(quantizer.argtypes[0]), encoded.ctypes.data, inputs.shape[1])
            decoder(encoded.ctypes.data, restored_inputs[token].ctypes.data_as(decoder.argtypes[1]), inputs.shape[1])
        expected = restored_inputs @ reconstruct_ternary(codes.reshape(5, -1), fp16_scales).T
        actual = np.empty_like(expected)
        assert pq2(pq2_blocks.ctypes.data_as(pq2.argtypes[0]), inputs.ctypes.data_as(pq2.argtypes[1]),
                   2, 5, groups, actual.ctypes.data_as(pq2.argtypes[-1])) == 0
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-5)
    if os.environ.get("MIMO_PQ2_CODEC_TEST") == "1":
        from embedded_jev.projection_artifact import load_projection_artifact
        from embedded_jev.ternary import unpack_group128_codes

        artifact = os.environ.get("MIMO_PROJECTION_ARTIFACT")
        if not artifact:
            pytest.fail("set MIMO_PROJECTION_ARTIFACT for native full-size PQ2 tensor control")
        stored_packed, stored_scales, manifest = load_projection_artifact(Path(artifact))
        assert manifest["shape"] == [4096, 12288]
        stored_codes = unpack_group128_codes(stored_packed)
        pq2_blocks = pack_ternary_pq2_0(stored_codes, stored_scales)
        inputs = np.ones((2, 12288), dtype=np.float32)
        inputs[1] *= -2
        actual = np.empty((2, 4096), dtype=np.float32)
        assert pq2(pq2_blocks.ctypes.data_as(pq2.argtypes[0]), inputs.ctypes.data_as(pq2.argtypes[1]),
                   2, 4096, 96, actual.ctypes.data_as(pq2.argtypes[-1])) == 0
        expected = np.empty_like(actual)
        for first in range(0, 4096, 64):
            decoded_weights = reconstruct_ternary(stored_codes[first:first + 64].reshape(-1, 12288), stored_scales[first:first + 64])
            expected[:, first:first + 64] = inputs @ decoded_weights.T
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-5)
        inputs = generator.normal(size=(2, 12288)).astype(np.float32)
        inputs[0, :128] = 0
        prepared, input_scales = quantize_a8_per_group(inputs)
        scales = stored_scales.astype(np.float32)
        assert direct(stored_packed.ctypes.data_as(arguments[0]), scales.ctypes.data_as(arguments[1]),
                      prepared.ctypes.data_as(arguments[2]), input_scales.ctypes.data_as(arguments[3]),
                      2, 4096, 96, expected.ctypes.data_as(arguments[-1])) == 0
        calls = ctypes.c_size_t(0)
        assert registered(pq2_blocks.ctypes.data_as(registered.argtypes[0]),
                          inputs.ctypes.data_as(registered.argtypes[1]), 2, 4096, 96,
                          actual.ctypes.data_as(registered.argtypes[-2]), ctypes.byref(calls)) == 0
        assert calls.value == 1
        np.testing.assert_array_equal(actual, expected)
        values = np.stack((inputs, generator.normal(size=inputs.shape).astype(np.float32)))
        expected_repeated = np.empty((2, 2, 4096), dtype=np.float32)
        expected_repeated[0] = expected
        prepared, input_scales = quantize_a8_per_group(values[1])
        assert direct(stored_packed.ctypes.data_as(arguments[0]), scales.ctypes.data_as(arguments[1]),
                      prepared.ctypes.data_as(arguments[2]), input_scales.ctypes.data_as(arguments[3]),
                      2, 4096, 96, expected_repeated[1].ctypes.data_as(arguments[-1])) == 0
        actual_repeated = np.empty_like(expected_repeated)
        before = registry_size()
        calls, repacks = ctypes.c_size_t(0), ctypes.c_size_t(0)
        assert repeated(pq2_blocks.ctypes.data_as(repeated.argtypes[0]), values.ctypes.data_as(repeated.argtypes[1]),
                        2, 4096, 96, 2, actual_repeated.ctypes.data_as(repeated.argtypes[-3]),
                        ctypes.byref(calls), ctypes.byref(repacks)) == 0
        assert calls.value == 2 and repacks.value == 1
        assert registry_size() == before
        np.testing.assert_array_equal(actual_repeated, expected_repeated)
        check_projection_lifetime(pq2_blocks, values, expected_repeated)
    if os.environ.get("MIMO_PRISM_GRAPH_TEST") == "1":
        interpreter = os.environ.get("MIMO_DENSE_PYTHON")
        local_dir = os.environ.get("MIMO_LOCAL_DIR")
        artifact = os.environ.get("MIMO_PROJECTION_ARTIFACT")
        if not all((interpreter, local_dir, artifact)):
            pytest.fail("set MiMo interpreter/snapshot/candidate for the optional real projection graph test")
        environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4"}
        reports = []
        for backend in ("direct", "prism_ggml", "prism_ggml_f32"):
            command = [
                interpreter, "-m", "embedded_jev.streamed_text", "--local-dir", local_dir,
                "--layers", "32", "--native-ffn-library", str(binary), "--native-ffn-backend", backend,
                "--projection-artifact", artifact,
            ]
            if backend == "prism_ggml_f32":
                guard = """
import runpy
from unittest.mock import patch
import embedded_jev.activation as activation
original = activation.quantize_a8_per_group
def reference_only(values, *args, **kwargs):
    if values.shape != (1, 12288):
        raise AssertionError("Python prepared A8 for the production token batch")
    return original(values, *args, **kwargs)
with patch.object(activation, "quantize_a8_per_group", reference_only):
    runpy.run_module("embedded_jev.streamed_text", run_name="__main__")
"""
                command = [interpreter, "-c", guard] + command[3:]
            result = subprocess.run(
                command,
                env=environment, check=True, capture_output=True, text=True, timeout=180,
            )
            reports.append(json.loads(result.stdout))
        direct_report = reports[0]
        for graph_report in reports[1:]:
            assert graph_report["native_ffn_down"]["graph_op"] == "map_custom2"
            assert graph_report["native_ffn_down"]["backend_dependencies"] == dependencies
            assert graph_report["native_ffn_down"]["bf16_projection_materialized"] is False
            assert graph_report["native_ffn_down"]["max_native_reference_error"] < 1e-4
            assert graph_report["native_ffn_down"]["calls"] == 1
            assert graph_report["ffn_down_input_sha256"] == direct_report["ffn_down_input_sha256"]
            assert graph_report["last_token_sha256"] == direct_report["last_token_sha256"]
            assert graph_report["selected_head"]["options"] == direct_report["selected_head"]["options"]
            assert graph_report["generated_tokens"] == direct_report["generated_tokens"] == 0
        assert reports[2]["native_ffn_down"]["activation_preparation"] == "native_graph_callback"
        if os.environ.get("MIMO_ROTATED_GRAPH_TEST") == "1":
            result = subprocess.run(
                [interpreter, "-m", "embedded_jev.streamed_text", "--local-dir", local_dir,
                 "--layers", "32", "--native-ffn-library", str(binary),
                 "--native-ffn-backend", "prism_ggml_hadamard128"],
                env=environment, check=True, capture_output=True, text=True, timeout=180,
            )
            rotated = json.loads(result.stdout)
            execution = rotated["native_ffn_down"]
            assert execution["candidate_origin"] == "in_memory_signed_hadamard_rtn"
            assert execution["transform"]["kind"] == "signed_normalized_hadamard"
            assert execution["transform"]["sign_seed"] == 773
            assert execution["dense_rotation_max_abs_error"] < 1e-4
            assert execution["max_native_reference_error"] < 1e-4
            assert execution["activation_preparation"] == "native_graph_callback"
            assert rotated["ffn_down_input_sha256"] == direct_report["ffn_down_input_sha256"]
            assert rotated["generated_tokens"] == 0 and set(rotated["selected_head"]["options"]) == {"A", "B"}


def test_pinned_prism_pq2_ternary_subset_matches_actual_decoder():
    import ctypes

    import numpy as np

    from embedded_jev.prism_codec import pack_ternary_pq2_0
    from embedded_jev.ternary import reconstruct_ternary

    source_dir = os.environ.get("PRISM_SOURCE_DIR")
    library_name = os.environ.get("PRISM_GGML_CPU_LIBRARY")
    if not source_dir or not library_name:
        pytest.skip("requires pinned Prism source and native GGML base library")
    assert subprocess.check_output(["git", "-C", source_dir, "rev-parse", "HEAD"], text=True).strip() == PRISM_REVISION
    binary = ctypes.CDLL(str(Path(library_name).parent / "libggml-base.so"))

    class InitParams(ctypes.Structure):
        _fields_ = [("mem_size", ctypes.c_size_t), ("mem_buffer", ctypes.c_void_p), ("no_alloc", ctypes.c_bool)]

    binary.ggml_init.argtypes = [InitParams]
    binary.ggml_init.restype = ctypes.c_void_p
    binary.ggml_free.argtypes = [ctypes.c_void_p]
    decoder = binary.dequantize_row_pq2_0
    decoder.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
    decoder.restype = None
    context = binary.ggml_init(InitParams(1 << 20, None, True))
    assert context
    try:
        generator = np.random.default_rng(885)
        codes = generator.integers(-1, 2, size=(3, 3, 128), dtype=np.int8)
        scales = generator.uniform(0.01, 2.0, size=(3, 3)).astype(np.float16)
        scales[0, 0] = 0
        codes[0, 0] = 0
        blocks = pack_ternary_pq2_0(codes, scales)
        expected = reconstruct_ternary(codes.reshape(3, 384), scales)
        actual = np.empty_like(expected)
        for row in range(3):
            decoder(blocks[row].ctypes.data, actual[row].ctypes.data_as(decoder.argtypes[1]), 384)
        np.testing.assert_array_equal(actual, expected)
        if os.environ.get("MIMO_PQ2_CODEC_TEST") == "1":
            from embedded_jev.projection_artifact import load_projection_artifact
            from embedded_jev.ternary import unpack_group128_codes

            artifact = os.environ.get("MIMO_PROJECTION_ARTIFACT")
            if not artifact:
                pytest.fail("set MIMO_PROJECTION_ARTIFACT for actual frozen-projection PQ2 decoder parity")
            packed, stored_scales, manifest = load_projection_artifact(Path(artifact))
            assert manifest["shape"] == [4096, 12288]
            native_codes = unpack_group128_codes(packed)
            blocks = pack_ternary_pq2_0(native_codes, stored_scales)
            assert blocks.nbytes == 4096 * 96 * 34
            for first in range(0, 4096, 64):
                expected = reconstruct_ternary(native_codes[first:first + 64].reshape(-1, 12288), stored_scales[first:first + 64])
                actual = np.empty_like(expected)
                for offset in range(actual.shape[0]):
                    decoder(blocks[first + offset].ctypes.data, actual[offset].ctypes.data_as(decoder.argtypes[1]), 12288)
                np.testing.assert_array_equal(actual, expected)
    finally:
        binary.ggml_free(context)


def test_pinned_prism_qwen35_filter_and_ssm_bias_map_mimo_text_names():
    source_dir = os.environ.get("PRISM_SOURCE_DIR")
    interpreter = os.environ.get("PRISM_CONVERTER_PYTHON")
    if not source_dir or not interpreter:
        pytest.skip("requires pinned Prism source and isolated GGUF Python environment")
    source = Path(source_dir)
    if not source.is_dir() or not Path(interpreter).is_file():
        pytest.fail("missing pinned Prism source or GGUF Python environment")
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == PRISM_REVISION
    code = """
from __future__ import annotations
import ast
import json
import sys
from pathlib import Path
from gguf import MODEL_ARCH, get_tensor_name_map

source = Path(sys.argv[1]) / "conversion/base.py"
base_tree = ast.parse(source.read_text())
model = next(node for node in base_tree.body
             if isinstance(node, ast.ClassDef) and node.name == "ModelBase")
method = next(node for node in model.body
              if isinstance(node, ast.FunctionDef) and node.name == "filter_tensors")
filtered_class = ast.ClassDef(name="FilteredModel", bases=[], keywords=[],
                              body=[method], decorator_list=[])
namespace = {}
exec(compile(ast.fix_missing_locations(ast.Module(body=[filtered_class], type_ignores=[])),
             str(source), "exec"), namespace)
text_model = next(node for node in base_tree.body
                  if isinstance(node, ast.ClassDef) and node.name == "TextModel")
text_filter = next(node for node in text_model.body
                   if isinstance(node, ast.FunctionDef) and node.name == "filter_tensors")
filtered_text_class = ast.ClassDef(
    name="FilteredTextModel", bases=[ast.Name(id="FilteredModel", ctx=ast.Load())],
    keywords=[], body=[text_filter], decorator_list=[])
exec(compile(ast.fix_missing_locations(ast.Module(body=[filtered_text_class], type_ignores=[])),
             str(source), "exec"), namespace)
qwen_source = Path(sys.argv[1]) / "conversion/qwen.py"
qwen = next(node for node in ast.parse(qwen_source.read_text()).body
            if isinstance(node, ast.ClassDef) and node.name == "Qwen3NextModel")
modify = next(node for node in qwen.body
              if isinstance(node, ast.FunctionDef) and node.name == "modify_tensors")
bias_branch = next(node for node in ast.walk(modify)
                   if isinstance(node, ast.If) and ast.unparse(node.test) == "name.endswith('.dt_bias')")
rename = ast.parse("def rename_dt_bias(name):\\n    return name").body[0]
rename.body.insert(0, bias_branch.body[0])
exec(compile(ast.fix_missing_locations(ast.Module(body=[rename], type_ignores=[])),
             str(qwen_source), "exec"), namespace)
mapper = get_tensor_name_map(MODEL_ARCH.QWEN35, 32)
results = []
for name in sys.argv[2:]:
    filtered = namespace["FilteredTextModel"].filter_tensors((name, lambda: None))
    if filtered is None:
        results.append([None, None, None])
        continue
    mapped_name = (namespace["rename_dt_bias"](filtered[0])
                   if filtered[0].endswith(".dt_bias") else filtered[0])
    results.append([mapper.get_name(name, try_suffixes=(".weight", ".bias")),
                    filtered[0], mapper.get_name(mapped_name, try_suffixes=(".weight", ".bias"))])
print(json.dumps(results))
"""
    result = subprocess.run(
        [interpreter, "-c", code, str(source),
         "model.language_model.layers.3.mlp.down_proj.weight",
         "model.language_model.layers.3.self_attn.q_proj.weight",
         "model.language_model.layers.0.linear_attn.dt_bias",
         "model.visual.blocks.0.attn.qkv.weight"],
        env={**os.environ, "PYTHONPATH": str(source / "gguf-py")},
        check=True, capture_output=True, text=True, timeout=15,
    )
    assert json.loads(result.stdout) == [
        [None, "model.layers.3.mlp.down_proj.weight", "blk.3.ffn_down.weight"],
        [None, "model.layers.3.self_attn.q_proj.weight", "blk.3.attn_q.weight"],
        [None, "model.layers.0.linear_attn.dt_bias", "blk.0.ssm_dt.bias"],
        [None, None, None],
    ]

    from embedded_jev.inventory import _json_object, _open_bounded

    index_bytes, _ = _open_bounded("model.safetensors.index.json")
    names = sorted(_json_object(index_bytes, "model.safetensors.index.json")["weight_map"])
    assert len(names) == 760
    all_results = subprocess.run(
        [interpreter, "-c", code, str(source), *names],
        env={**os.environ, "PYTHONPATH": str(source / "gguf-py")},
        check=True, capture_output=True, text=True, timeout=15,
    )
    mapped = json.loads(all_results.stdout)
    assert len(mapped) == len(names)
    assert sum(result[1] is None for result in mapped) == 333
    assert sum(result[1] is not None for result in mapped) == 427
    assert all(result[2] is not None for result in mapped if result[1] is not None)
    renamed_biases = [
        result for result in mapped if result[1] is not None and result[1].endswith(".dt_bias")
    ]
    assert len(renamed_biases) == 24
    assert all(result[2].endswith(".ssm_dt.bias") for result in renamed_biases)


def test_pinned_prism_qwen35_value_head_reorder_on_toy_tensors():
    source_dir = os.environ.get("PRISM_SOURCE_DIR")
    interpreter = os.environ.get("PRISM_CONVERTER_PYTHON")
    if not source_dir or not interpreter:
        pytest.skip("requires pinned Prism converter and isolated CPU Torch environment")
    source = Path(source_dir)
    assert subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip() == PRISM_REVISION
    code = """
from __future__ import annotations
import ast
import json
import sys
from pathlib import Path
import numpy as np
import torch

module = ast.parse((Path(sys.argv[1]) / "conversion/qwen.py").read_text())
next_model = next(node for node in module.body
                  if isinstance(node, ast.ClassDef) and node.name == "Qwen3NextModel")
value_model = next(node for node in module.body
                   if isinstance(node, ast.ClassDef) and node.name == "_LinearAttentionVReorderBase")
next_modify = next(node for node in next_model.body
                   if isinstance(node, ast.FunctionDef) and node.name == "modify_tensors")
reorder = next(node for node in value_model.body
               if isinstance(node, ast.FunctionDef) and node.name == "_reorder_v_heads")
folded_check = next(node for node in value_model.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_hadamard_folds_tensor")
value_modify = next(node for node in value_model.body
                    if isinstance(node, ast.FunctionDef) and node.name == "modify_tensors")
base = ast.parse("class Base:\\n def modify_tensors(self, data_torch, name, bid):\\n  yield name, data_torch").body[0]
next_class = ast.ClassDef(name="Qwen3NextModel", bases=[ast.Name(id="Base", ctx=ast.Load())],
                          keywords=[], body=[next_modify], decorator_list=[])
value_class = ast.ClassDef(name="ValueModel", bases=[ast.Name(id="Qwen3NextModel", ctx=ast.Load())],
                           keywords=[], body=[reorder, folded_check, value_modify], decorator_list=[])
namespace = {"torch": torch, "np": np}
exec(compile(ast.fix_missing_locations(ast.Module(body=[base, next_class, value_class], type_ignores=[])),
             str(Path(sys.argv[1]) / "conversion/qwen.py"), "exec"), namespace)
model = namespace["ValueModel"]()
model.hparams = {"linear_num_key_heads": 2, "linear_num_value_heads": 4,
                 "linear_key_head_dim": 2, "linear_value_head_dim": 2, "hidden_size": 8}
model.hadamard_folded_names = lambda: []
model._hadamard_gdn_v_grouped = False
name = "model.layers.0.linear_attn."
qkv = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8)
result = list(model.modify_tensors(qkv, name + "in_proj_qkv.weight", 0))
assert len(result) == 1
expected_v = qkv[8:].numpy().reshape(2, 2, 2, 8).transpose(1, 0, 2, 3).reshape(8, 8)
assert np.array_equal(result[0][1].numpy(), np.concatenate([qkv[:8].numpy(), expected_v]))
gate = torch.arange(8 * 8, dtype=torch.float32).reshape(8, 8)
result_z = list(model.modify_tensors(gate, name + "in_proj_z.weight", 0))
assert np.array_equal(result_z[0][1].numpy(), gate.numpy().reshape(2, 2, 2, 8).transpose(1, 0, 2, 3).reshape(8, 8))
log_values = torch.arange(4, dtype=torch.float32)
result_log = list(model.modify_tensors(log_values, name + "A_log", 0))
expected_heads = np.array([0, 2, 1, 3], dtype=np.float32)
np.testing.assert_allclose(result_log[0][1].numpy(), -np.exp(expected_heads).astype(np.float32), rtol=1e-6, atol=1e-6)
result_bias = list(model.modify_tensors(log_values, name + "dt_bias", 0))
assert result_bias[0][0].endswith(".dt_proj.bias")
assert np.array_equal(result_bias[0][1].numpy(), expected_heads)
alpha = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)
result_alpha = list(model.modify_tensors(alpha, name + "in_proj_a.weight", 0))
expected_alpha = alpha.numpy().reshape(2, 2, 1, 8).transpose(1, 0, 2, 3).reshape(4, 8)
assert np.array_equal(result_alpha[0][1].numpy(), expected_alpha)
conv = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 1, 3)
result_conv = list(model.modify_tensors(conv, name + "conv1d.weight", 0))
conv_values = conv.numpy().reshape(16, 3)
expected_conv_v = conv_values[8:].reshape(2, 2, 2, 3).transpose(1, 0, 2, 3).reshape(8, 3)
assert np.array_equal(result_conv[0][1].numpy(), np.concatenate([conv_values[:8], expected_conv_v]))
out = torch.arange(3 * 8, dtype=torch.float32).reshape(3, 8)
out_name = name + "out_proj.weight"
unfolded = list(model.modify_tensors(out, out_name, 0))
expected_columns = out.numpy().reshape(3, 2, 2, 2).transpose(0, 2, 1, 3).reshape(3, 8)
assert np.array_equal(unfolded[0][1].numpy(), expected_columns)
assert model._hadamard_gdn_v_grouped is False
model.hadamard_folded_names = lambda: [out_name]
folded = list(model.modify_tensors(out, out_name, 0))
assert np.array_equal(folded[0][1].numpy(), out.numpy())
assert model._hadamard_gdn_v_grouped is True
print(json.dumps({"qkv_shape": list(result[0][1].shape), "v_head_order": [0, 2, 1, 3]}))
"""
    result = subprocess.run(
        [interpreter, "-c", code, str(source)], check=True,
        capture_output=True, text=True, timeout=15,
    )
    assert json.loads(result.stdout) == {"qkv_shape": [16, 8], "v_head_order": [0, 2, 1, 3]}


def test_pinned_prism_grouped_v_metadata_writer_for_folded_out_projection():
    source_dir = os.environ.get("PRISM_SOURCE_DIR")
    interpreter = os.environ.get("PRISM_CONVERTER_PYTHON")
    if not source_dir or not interpreter:
        pytest.skip("requires pinned Prism converter and isolated GGUF Python environment")
    source = Path(source_dir)
    assert subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip() == PRISM_REVISION
    code = """
import ast
import json
import logging
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import gguf

source = Path(sys.argv[1]) / "conversion/base.py"
tree = ast.parse(source.read_text())
base = next(node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ModelBase")
method = next(node for node in base.body
              if isinstance(node, ast.FunctionDef) and node.name == "add_hadamard_metadata")
model_class = ast.ClassDef(name="Model", bases=[], keywords=[], body=[method], decorator_list=[])
namespace = {"json": json, "re": re, "gguf": gguf, "logger": logging.getLogger("prism")}
exec(compile(ast.fix_missing_locations(ast.Module(body=[model_class], type_ignores=[])),
             str(source), "exec"), namespace)
manifest = {
    "schema_version": 1, "kind": "hadamard-weight-fold",
    "status": "requires-matching-runtime",
    "transform": {"block_size": 128, "name": "normalized-signed-sylvester-walsh-hadamard",
                  "sign_mode": "identity"},
    "tensors": [{"name": "model.layers.0.linear_attn.out_proj.weight", "axis": -1}],
}
results = []
with tempfile.TemporaryDirectory() as directory:
    Path(directory, "hadamard_packing.json").write_text(json.dumps(manifest))
    for grouped in (False, True):
        metadata = {}
        bool_keys = []
        def put(key, value):
            metadata[key] = value
        def put_bool(key, value):
            bool_keys.append(key)
            put(key, value)
        model = namespace["Model"]()
        model.dir_model = Path(directory)
        model.model_arch = gguf.MODEL_ARCH.QWEN35
        model.hparams = {"tie_word_embeddings": False}
        model._hadamard_gdn_v_grouped = grouped
        model.filter_tensors = lambda tensor: tensor
        model.map_tensor_name = {
            "model.layers.0.linear_attn.out_proj.weight": "blk.0.ssm_out.weight"
        }.__getitem__
        model.gguf_writer = SimpleNamespace(
            add_bool=put_bool, add_uint32=put, add_string=put, add_array=put,
        )
        model.add_hadamard_metadata()
        assert metadata["prism.hadamard.weight_names"] == ["blk.0.ssm_out.weight"]
        key = "prism.hadamard.gdn_v_grouped"
        assert (key in bool_keys) == grouped
        assert metadata.get(key) is (True if grouped else None)
        gguf_path = Path(directory, f"grouped-{grouped}.gguf")
        writer = gguf.GGUFWriter(gguf_path, "qwen35")
        model.gguf_writer = writer
        model.add_hadamard_metadata()
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        reader = gguf.GGUFReader(gguf_path)
        assert not reader.tensors
        assert reader.get_field("prism.hadamard.weight_names").contents() == [
            "blk.0.ssm_out.weight"
        ]
        field = reader.get_field(key)
        assert (field is not None) == grouped
        if grouped:
            assert field.types[0].name == "BOOL"
            assert field.contents() is True
        results.append(grouped)
print(json.dumps(results))
"""
    result = subprocess.run(
        [interpreter, "-c", code, str(source)],
        env={**os.environ, "PYTHONPATH": str(source / "gguf-py")},
        check=True, capture_output=True, text=True, timeout=15,
    )
    assert json.loads(result.stdout) == [False, True]