"""Optional real BitNet-fork I2_S CPU kernel control, without a model."""

import ctypes
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from embedded_jev.label_probe import load_decision_fixture


BITNET_REVISION = "0b341e582afbf9e1011f24744b554c96a3477eb5"
LLAMA_REVISION = "390c307752ab78fd8189f359d6954c9ba1be74af"
CONTROL_SHA256 = "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162"
CONTROL_BYTES = 1187801280


@pytest.fixture(scope="module")
def upstream_dot():
    source = os.environ.get("BITNET_SOURCE_DIR")
    library = os.environ.get("BITNET_GGML_CPU_LIBRARY")
    if not source or not library:
        pytest.skip("requires pinned BitNet source and compiled ggml-cpu library")
    source_path = Path(source)
    if not source_path.is_dir() or not Path(library).is_file():
        pytest.fail("missing pinned BitNet source or CPU library")
    for directory, revision in (
        (source_path, BITNET_REVISION),
        (source_path / "3rdparty" / "llama.cpp", LLAMA_REVISION),
    ):
        actual = subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
        ).strip()
        assert actual == revision
    function = ctypes.CDLL(library).ggml_vec_dot_i2_i8_s
    function.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
    ]
    function.restype = None
    return function


def packed_group(codes):
    lanes = (codes.astype(np.uint8) + 1).reshape(4, 32)
    packed = np.zeros(32, dtype=np.uint8)
    for lane in range(4):
        packed |= lanes[lane] << (6 - 2 * lane)
    return packed


def raw_dot(function, codes, activations):
    packed = packed_group(codes)
    result = ctypes.c_float()
    function(
        128, ctypes.byref(result), 1, packed.ctypes.data, 128,
        activations.ctypes.data, 0, 1,
    )
    return result.value


def test_compiled_upstream_i2_s_requires_per_group_activation_compensation(upstream_dot):
    codes = np.zeros(128, dtype=np.int8)
    activations = np.ones(128, dtype=np.int8)
    assert raw_dot(upstream_dot, codes, activations) == 128.0
    assert raw_dot(upstream_dot, codes, activations) - activations.sum(dtype=np.int32) == 0


def test_compiled_upstream_group_scaled_outputs_match_scalar_reference(upstream_dot):
    generator = np.random.default_rng(470)
    codes = generator.integers(-1, 2, size=(4, 2, 128), dtype=np.int8)
    activations = generator.integers(-128, 128, size=(3, 2, 128), dtype=np.int8)
    weight_scales = generator.uniform(0.1, 0.7, size=(4, 2)).astype(np.float32)
    activation_scales = generator.uniform(0.01, 0.2, size=(3, 2)).astype(np.float32)
    actual = np.zeros((3, 4), dtype=np.float32)
    for token in range(3):
        for row in range(4):
            for group in range(2):
                values = activations[token, group]
                signed_dot = raw_dot(upstream_dot, codes[row, group], values) - values.sum(dtype=np.int32)
                actual[token, row] += signed_dot * weight_scales[row, group] * activation_scales[token, group]
    group_dots = np.einsum(
        "rgi,tgi->trg", codes.astype(np.int32), activations.astype(np.int32)
    )
    expected = (group_dots * weight_scales[None, :, :] * activation_scales[:, None, :]).sum(axis=-1)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=0.005)


def test_pinned_fork_ggml_graph_executes_signed_i2_s_rows(upstream_dot, tmp_path):
    compiler = shutil.which("clang++-18")
    if compiler is None:
        pytest.skip("clang++-18 is not installed")
    source = Path(__file__).resolve().parents[1] / "native" / "bitnet_ggml_graph_smoke.cpp"
    fork = Path(os.environ["BITNET_SOURCE_DIR"]) / "3rdparty" / "llama.cpp"
    library_dir = Path(os.environ["BITNET_GGML_CPU_LIBRARY"]).parent
    binary = tmp_path / "bitnet-graph-smoke"
    subprocess.run(
        [
            compiler, "-std=c++17", "-O2", "-I", str(fork / "ggml" / "include"),
            str(source), "-L", str(library_dir), f"-Wl,-rpath,{library_dir}",
            "-lggml-cpu", "-lggml-base", "-o", str(binary),
        ],
        check=True, capture_output=True, text=True,
    )
    result = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True, timeout=10
    )
    assert result.stdout.splitlines() == [
        "0.000000", "128.000000", "-128.000000", "0.000000"
    ]
    grouped = subprocess.run(
        [str(binary), "--groups"], check=True, capture_output=True, text=True, timeout=10
    )
    assert grouped.stdout.splitlines() == [
        "0.000000", "128.000000", "-128.000000", "128.000000",
        "grouped 32.000000", "grouped -320.000000",
        "grouped -256.000000", "grouped 64.000000",
    ]
    batched = subprocess.run(
        [str(binary), "--batch"], check=True, capture_output=True, text=True, timeout=10
    )
    assert batched.stdout.splitlines() == [
        "0.000000", "128.000000", "-128.000000", "128.000000",
        "0.000000", "-256.000000", "256.000000", "-256.000000",
        "grouped 32.000000", "grouped -320.000000",
        "grouped -256.000000", "grouped 64.000000",
        "grouped -64.000000", "grouped 640.000000",
        "grouped 512.000000", "grouped -128.000000",
    ]


@pytest.fixture(scope="module")
def prefill_control(upstream_dot, tmp_path_factory):
    model_name = os.environ.get("BITNET_CONTROL_GGUF")
    if not model_name:
        pytest.skip("requires opt-in pinned BitNet control GGUF")
    model = Path(model_name)
    if not model.is_file() or model.stat().st_size != CONTROL_BYTES:
        pytest.fail("native BitNet control GGUF size mismatch")
    with model.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != CONTROL_SHA256:
        pytest.fail("native BitNet control GGUF SHA-256 mismatch")

    compiler = shutil.which("clang++-18")
    if compiler is None:
        pytest.skip("clang++-18 is not installed")
    fork = Path(os.environ["BITNET_SOURCE_DIR"]) / "3rdparty" / "llama.cpp"
    library_dir = Path(os.environ["BITNET_GGML_CPU_LIBRARY"]).parent
    library = library_dir / "libllama.so.0"
    if not library.is_file():
        pytest.fail("pinned native BitNet llama library was not built")
    source = Path(__file__).resolve().parents[1] / "native" / "bitnet_prefill_control.cpp"
    binary = tmp_path_factory.mktemp("bitnet-prefill") / "control"
    subprocess.run(
        [
            compiler, "-std=c++17", "-O2", "-I", str(fork / "include"),
            "-I", str(fork / "ggml" / "include"), str(source), str(library),
            "-L", str(library_dir), f"-Wl,-rpath,{library_dir}",
            "-lggml-cpu", "-lggml-base", "-o", str(binary),
        ],
        check=True, capture_output=True, text=True,
    )
    return binary, model


def test_pinned_bitnet_model_prefills_only_and_returns_finite_logits(prefill_control):
    binary, model = prefill_control
    result = subprocess.run(
        [str(binary), str(model)], check=True, capture_output=True, text=True, timeout=90
    )
    report = json.loads(result.stdout.strip())
    assert report["backend"] == "pinned_bitnet_control"
    assert report["prompt_tokens"] == 22
    assert report["finite_logits"] == report["vocab_size"] == 128256
    assert 0 <= report["argmax_token_id"] < report["vocab_size"]
    assert report["generated_tokens"] == 0
    assert report["labels"] == ["A", "B", "C"]
    assert report["label_token_ids"] == [32, 33, 34]
    probabilities = report["conditional_probabilities"]
    assert len(probabilities) == 3
    assert all(0 <= probability <= 1 for probability in probabilities)
    assert abs(sum(probabilities) - 1) < 1e-7
    assert report["selected_label"] == report["labels"][probabilities.index(max(probabilities))]
    assert abs(report["max_option_probability"] - max(probabilities)) < 1e-7
    assert 0 < report["allowed_label_mass"] < 1
    assert report["calibration_status"] == "uncalibrated"


def test_pinned_bitnet_model_prefill_dispatches_i2_s_matmul(prefill_control):
    debugger = shutil.which("gdb")
    if debugger is None:
        pytest.skip("gdb is not installed")
    binary, model = prefill_control
    result = subprocess.run(
        [
            debugger, "--batch", "--quiet", "-ex", "set pagination off",
            "-ex", "set breakpoint pending on", "-ex", "break llamafile_sgemm_i2s",
            "-ex", "run", "-ex", "bt 4", "--args", str(binary), str(model),
        ],
        check=True, capture_output=True, text=True, timeout=90,
    )
    assert "hit Breakpoint" in result.stdout
    assert "llamafile_sgemm_i2s" in result.stdout
    assert "ggml_compute_forward_mul_mat" in result.stdout


def test_pinned_bitnet_control_scores_fixture_prompts_without_sampling(prefill_control):
    binary, model = prefill_control
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "agent_tool_smoke.json"
    fixture, _ = load_decision_fixture(fixture_path)
    for case in fixture["cases"]:
        labels = "ABCDEFGHIJKLMNOP"[:len(case["options"])]
        choices = "\n".join(
            f"{label}. {option['description']}"
            for label, option in zip(labels, case["options"], strict=True)
        )
        prompt = (
            f"State: {case['state']}\nQuestion: {case['question']}\n"
            f"Options:\n{choices}\nAnswer:"
        )
        result = subprocess.run(
            [str(binary), str(model), prompt], check=True,
            capture_output=True, text=True, timeout=90,
        )
        report = json.loads(result.stdout.strip())
        assert 0 < report["prompt_tokens"] <= 128
        assert report["labels"] == list(labels)
        assert report["label_token_ids"] == [32, 33, 34]
        assert report["finite_logits"] == report["vocab_size"] == 128256
        assert abs(sum(report["conditional_probabilities"]) - 1) < 1e-7
        assert report["selected_label"] in labels
        assert 0 < report["allowed_label_mass"] <= 1
        assert report["calibration_status"] == "uncalibrated"
        assert report["generated_tokens"] == 0