"""Opt-in prefix execution against the pinned local MiMo BF16 snapshot."""

import json
import math
import os
import platform
import shutil
import subprocess
from pathlib import Path

import pytest

from embedded_jev.inventory import MODEL_REVISION


@pytest.mark.skipif(
    os.environ.get("MIMO_DENSE_PREFIX_TEST") != "1",
    reason="requires isolated CPU Torch/Transformers and the verified local MiMo snapshot",
)
def test_local_dense_text_prefix_captures_real_ffn_input(tmp_path):
    interpreter = os.environ.get("MIMO_DENSE_PYTHON")
    local_dir = os.environ.get("MIMO_LOCAL_DIR")
    if not interpreter or not local_dir or not Path(interpreter).is_file():
        pytest.fail("set MIMO_DENSE_PYTHON and MIMO_LOCAL_DIR for the opt-in prefix test")
    command = [
        interpreter, "-m", "embedded_jev.dense_probe", "--local-dir", local_dir,
        "--layers", "4", "--compare-ternary",
    ]
    if os.environ.get("MIMO_CHAT_TEMPLATE_TEST") == "1":
        command.append("--chat-template")
    if os.environ.get("MIMO_NATIVE_ACTIVATION_TEST") == "1":
        compiler = shutil.which("clang++-18")
        if (
            compiler is None or platform.machine() != "x86_64"
            or "avx2" not in Path("/proc/cpuinfo").read_text()
        ):
            pytest.skip("requires Clang 18 and x86-64 AVX2 for native comparison")
        library = tmp_path / "bitnet_group_scale.so"
        source = Path(__file__).resolve().parents[1] / "native" / "bitnet_group_scale.cpp"
        subprocess.run(
            [compiler, "-std=c++17", "-O2", "-mavx2", "-shared", "-fPIC",
             str(source), "-o", str(library)],
            check=True, capture_output=True, text=True,
        )
        command.extend(("--native-library", str(library)))
    result = subprocess.run(
        command,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
             "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4"},
        check=True, capture_output=True, text=True, timeout=120,
    )
    report = json.loads(result.stdout)
    assert report["revision"] == MODEL_REVISION
    assert report["layers"] == 4 and report["tokens"] <= 32
    assert report["weight_bytes"] == 3764136064
    assert report["output_shape"] == [1, report["tokens"], 4096]
    assert report["output_dtype"] == "torch.bfloat16"
    assert report["ffn_down_input_shape"] == [1, report["tokens"], 12288]
    assert len(report["ffn_down_input_sha256"]) == 64
    assert math.isfinite(report["ffn_down_input_rms"])
    comparison = report["ternary_comparison"]
    assert comparison["policy"] == "unrotated_searched_fp16_group128"
    assert comparison["activation_quantized"] is False
    assert math.isfinite(comparison["relative_output_rmse"])
    assert math.isfinite(comparison["weight_mse"])
    if os.environ.get("MIMO_CHAT_TEMPLATE_TEST") == "1":
        assert report["tokens"] == 22 and report["generated_tokens"] == 0
        assert len(report["prompt_sha256"]) == 64
        assert set(report["label_token_ids"]) == set("ABCDEFGHIJKLMNOP")
    if os.environ.get("MIMO_NATIVE_ACTIVATION_TEST") == "1":
        native = report["native_comparison"]
        assert native["groups"] == 96 and native["packed_bytes"] == 12582912
        assert native["activation_quantized"] is True
        assert native["max_native_reference_error"] < 1e-4
        assert math.isfinite(native["relative_dense_output_rmse"])


@pytest.mark.skipif(
    os.environ.get("MIMO_STREAMED_TEXT_TEST") != "1",
    reason="requires verified pinned MiMo shards and isolated CPU Torch/Transformers",
)
def test_streamed_full_text_scores_only_selected_labels():
    interpreter = os.environ.get("MIMO_DENSE_PYTHON")
    local_dir = os.environ.get("MIMO_LOCAL_DIR")
    if not interpreter or not local_dir or not Path(interpreter).is_file():
        pytest.fail("set MIMO_DENSE_PYTHON and MIMO_LOCAL_DIR for the opt-in streamed test")
    environment = {
        **os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4",
    }
    reference_result = subprocess.run(
        [interpreter, "-m", "embedded_jev.dense_probe", "--local-dir", local_dir,
         "--layers", "4", "--chat-template"],
        env=environment, check=True, capture_output=True, text=True, timeout=180,
    )
    reference = json.loads(reference_result.stdout)
    reports = []
    for layers in (4, 32):
        result = subprocess.run(
            [interpreter, "-m", "embedded_jev.streamed_text", "--local-dir", local_dir,
             "--layers", str(layers), "--label-count", "2"],
            env=environment, check=True, capture_output=True, text=True, timeout=180,
        )
        reports.append(json.loads(result.stdout))
    prefix, full = reports
    assert prefix["revision"] == full["revision"] == MODEL_REVISION
    assert prefix["last_token_sha256"] == reference["last_token_sha256"]
    assert prefix["ffn_down_input_sha256"] == reference["ffn_down_input_sha256"]
    assert prefix["prompt_sha256"] == reference["prompt_sha256"]
    assert prefix["ffn_down_input_sha256"] == full["ffn_down_input_sha256"]
    assert prefix["tokens"] == full["tokens"] == 22
    assert prefix["generated_tokens"] == full["generated_tokens"] == 0
    assert "selected_head" not in prefix
    assert full["layers"] == 32 and full["output_shape"] == [1, 22, 4096]
    assert full["max_layer_bytes"] <= 512 * 1024**2
    selected = full["selected_head"]
    assert selected["head_payload_bytes"] == 16384
    assert selected["full_vocabulary_mass"] == "not_computed"
    assert selected["max_fp32_to_bf16_logit_gap"] < 0.125
    assert set(selected["options"]) == {"A", "B"}
    assert math.isclose(
        sum(option["conditional_probability"] for option in selected["options"].values()),
        1.0, rel_tol=1e-7,
    )
    assert all(math.isfinite(option["logit"]) for option in selected["options"].values())


@pytest.mark.skipif(
    os.environ.get("MIMO_TYPED_FIXTURE_TEST") != "1",
    reason="requires pinned BF16 MiMo text layers and isolated CPU Torch/Transformers",
)
def test_streamed_text_maps_synthetic_options_to_typed_scores():
    interpreter = os.environ.get("MIMO_DENSE_PYTHON")
    local_dir = os.environ.get("MIMO_LOCAL_DIR")
    if not interpreter or not local_dir or not Path(interpreter).is_file():
        pytest.fail("set MIMO_DENSE_PYTHON and MIMO_LOCAL_DIR for the opt-in fixture test")
    fixture = Path(__file__).parent / "fixtures" / "agent_tool_smoke.json"
    environment = {
        **os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4",
    }
    for case_id, option_ids, expected_id in (
        ("edit-reordered-options", ("ask", "inspect", "edit"), "edit"),
        ("publish-without-authorization", ("edit", "inspect", "ask"), "ask"),
    ):
        result = subprocess.run(
            [interpreter, "-m", "embedded_jev.streamed_text", "--local-dir", local_dir,
             "--layers", "32", "--fixture", str(fixture), "--case-id", case_id],
            env=environment, check=True, capture_output=True, text=True, timeout=180,
        )
        report = json.loads(result.stdout)
        assert report["revision"] == MODEL_REVISION and report["generated_tokens"] == 0
        assert report["fixture"]["purpose"] == "synthetic_engineering_smoke_not_calibration_or_benchmark"
        assert report["fixture"]["prompt_token_count"] == report["tokens"] <= 128
        decision = report["decision"]
        assert decision["scope"] == "synthetic_fixture_observation_not_quality_or_calibration"
        assert decision["expected_option_id"] == decision["chosen_option_id"] == expected_id
        assert [option["id"] for option in decision["options"]] == list(option_ids)
        assert [option["label"] for option in decision["options"]] == ["A", "B", "C"]
        assert math.isclose(
            sum(option["conditional_probability"] for option in decision["options"]),
            1.0, rel_tol=1e-7,
        )
        assert report["selected_head"]["full_vocabulary_mass"] == "not_computed"


@pytest.mark.skipif(
    os.environ.get("MIMO_IN_MODEL_NATIVE_TEST") != "1",
    reason="requires pinned MiMo BF16 shards, isolated Torch, Clang 18, and x86-64 AVX2",
)
def test_streamed_text_substitutes_one_bitnet_derived_ffn(tmp_path):
    interpreter = os.environ.get("MIMO_DENSE_PYTHON")
    local_dir = os.environ.get("MIMO_LOCAL_DIR")
    compiler = shutil.which("clang++-18")
    if not interpreter or not local_dir or not Path(interpreter).is_file():
        pytest.fail("set MIMO_DENSE_PYTHON and MIMO_LOCAL_DIR for the opt-in native model test")
    if compiler is None or platform.machine() != "x86_64" or "avx2" not in Path("/proc/cpuinfo").read_text():
        pytest.skip("requires Clang 18 and x86-64 AVX2")
    library = tmp_path / "bitnet_group_scale.so"
    source = Path(__file__).resolve().parents[1] / "native" / "bitnet_group_scale.cpp"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-mavx2", "-shared", "-fPIC",
         str(source), "-o", str(library)],
        check=True, capture_output=True, text=True,
    )
    environment = {
        **os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4",
    }
    command = [
        interpreter, "-m", "embedded_jev.streamed_text", "--local-dir", local_dir,
        "--layers", "32", "--label-count", "2",
    ]
    if os.environ.get("MIMO_FULL_HEAD_TEST") == "1":
        command.append("--full-vocabulary-mass")
    reports = []
    for extra in ([], ["--native-ffn-library", str(library)]):
        result = subprocess.run(
            command + extra, env=environment, check=True,
            capture_output=True, text=True,
            timeout=360 if os.environ.get("MIMO_FULL_HEAD_TEST") == "1" else 180,
        )
        reports.append(json.loads(result.stdout))
    dense, native = reports
    assert dense["revision"] == native["revision"] == MODEL_REVISION
    assert dense["ffn_down_input_sha256"] == native["ffn_down_input_sha256"]
    assert dense["last_token_sha256"] != native["last_token_sha256"]
    assert dense["generated_tokens"] == native["generated_tokens"] == 0
    assert "native_ffn_down" not in dense
    assert native["native_ffn_down"]["calls"] == 1
    assert native["native_ffn_down"]["packed_bytes"] == 12582912
    assert native["native_ffn_down"]["bf16_projection_materialized"] is True
    assert native["native_ffn_down"]["max_native_reference_error"] < 1e-4
    for report in reports:
        options = report["selected_head"]["options"]
        assert set(options) == {"A", "B"}
        assert math.isclose(
            sum(option["conditional_probability"] for option in options.values()), 1.0,
            rel_tol=1e-7,
        )
        if os.environ.get("MIMO_FULL_HEAD_TEST") == "1":
            mass = report["vocabulary_mass"]
            assert mass["vocabulary_rows"] == 248320
            assert mass["payload_bytes"] == 2034237440
            assert mass["max_token_id"] == options["B"]["token_id"]
            assert 0 < mass["selected_label_mass"] < 1
            assert report["selected_head"]["full_vocabulary_mass"] == mass["selected_label_mass"]
    artifact = os.environ.get("MIMO_PROJECTION_ARTIFACT")
    if artifact:
        if not Path(artifact).is_dir():
            pytest.fail("MIMO_PROJECTION_ARTIFACT must point to the saved native fixture")
        saved_command = [argument for argument in command if argument != "--full-vocabulary-mass"]
        checked_loader = """
import runpy
from unittest.mock import patch
import safetensors

original = safetensors.safe_open
class CheckedReader:
    def __init__(self, *args, **kwargs):
        self.context = original(*args, **kwargs)
    def __enter__(self):
        self.reader = self.context.__enter__()
        return self
    def __exit__(self, *args):
        return self.context.__exit__(*args)
    def get_tensor(self, name):
        if name == "model.language_model.layers.3.mlp.down_proj.weight":
            raise AssertionError("saved candidate unnecessarily loaded its BF16 source projection")
        return self.reader.get_tensor(name)
    def __getattr__(self, name):
        return getattr(self.reader, name)

with patch("safetensors.safe_open", CheckedReader):
    runpy.run_module("embedded_jev.streamed_text", run_name="__main__")
"""
        saved_result = subprocess.run(
            [interpreter, "-c", checked_loader] + saved_command[3:]
            + ["--native-ffn-library", str(library), "--projection-artifact", artifact],
            env=environment, check=True, capture_output=True, text=True, timeout=180,
        )
        saved = json.loads(saved_result.stdout)
        assert saved["native_ffn_down"]["candidate_origin"] == "saved_hash_checked_native_fixture"
        assert saved["native_ffn_down"]["bf16_projection_materialized"] is False
        assert saved["native_ffn_down"]["max_native_reference_error"] < 1e-4
        assert saved["ffn_down_input_sha256"] == native["ffn_down_input_sha256"]
        assert saved["last_token_sha256"] == native["last_token_sha256"]
        assert saved["selected_head"]["options"] == native["selected_head"]["options"]