"""Optional pinned Prism CPU transform graph control without model weights."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


PRISM_REVISION = "842b1880415d6f508f03b789e5ce70194def7bfd"
BITNET_REVISION = "0b341e582afbf9e1011f24744b554c96a3477eb5"


def _profile_parity_fixture(profile, inventory, manifest, blocks):
    from dataclasses import asdict

    record = {"schema_version": 1, "profile_id": profile.profile_id,
              "source": {"model": profile.model, "revision": profile.revision, "license": profile.license,
                         "format": "safetensors", "support_tier": "A"},
              "files": {name: {key: value for key, value in asdict(item).items() if value is not None}
                        for name, item in profile.files.items()},
              **{name: getattr(profile, name) for name in ("packaging", "geometry", "tokenization", "runtime", "quantization", "evidence")}}
    fixture = json.dumps({"profile": record, "inventory": inventory, "manifest": manifest,
                          "payload_hex": blocks.tobytes().hex()}, default=dict)
    assert len(fixture) < 1024**2
    return fixture


@pytest.mark.parametrize("backend", ["dense", "bitnet", "bitnet-profile"])
@pytest.mark.parametrize("recurrent", [False, True])
@pytest.mark.parametrize("attention", [False, True])
def test_pinned_prism_tiny_qwen35_native_prefill_scores_without_generation(tmp_path, backend, recurrent, attention):
    import numpy as np

    source_dir = os.environ.get("PRISM_SOURCE_DIR")
    converter = os.environ.get("PRISM_CONVERTER_PYTHON")
    runtime_build = os.environ.get("MIMO_PRISM_RUNTIME_BUILD")
    if not source_dir or not converter or not runtime_build:
        pytest.skip("requires pinned source, GGUF environment, and versioned full runtime build")
    assert subprocess.check_output(["git", "-C", source_dir, "rev-parse", "HEAD"], text=True).strip() == PRISM_REVISION
    model_file = tmp_path / "tiny-qwen35-prefill.gguf"
    reference_file = tmp_path / "tiny-projection-reference.gguf"
    bitnet = backend.startswith("bitnet")
    profile_policy = backend == "bitnet-profile"
    writer_code = """
import json
import sys
import gguf
import numpy as np
writer = gguf.GGUFWriter(sys.argv[1], "qwen35")
if sys.argv[2].startswith("bitnet-profile"):
    for key, value in {
        "jev.model.profile_id": "synthetic-qwen35-prefill-v2",
        "jev.model.profile_sha256": "a" * 64,
        "jev.model.source_revision": "1" * 40,
        "jev.model.converter_revision": "842b1880415d6f508f03b789e5ce70194def7bfd",
        "jev.bitnet.source_tensor_sha256": "f" * 64,
        "jev.bitnet.model": "profile-qwen35-single-projection-v2",
        "jev.bitnet.execution": "group128-a8-fp32-nearest-even-identity-v1",
    }.items():
        writer.add_string(key, "other-profile" if key == "jev.model.profile_id" and sys.argv[2].endswith("_wrong_profile") else value)
for key, value in {
    "context_length": 256, "embedding_length": 32, "block_count": 4,
    "feed_forward_length": 256, "attention.head_count": 4, "attention.head_count_kv": 2,
    "attention.key_length": 8, "attention.value_length": 8, "rope.dimension_count": 8,
    "ssm.conv_kernel": 3, "ssm.inner_size": 32, "ssm.state_size": 8,
    "ssm.time_step_rank": 4, "ssm.group_count": 2, "full_attention_interval": 2,
}.items():
    writer.add_uint32("qwen35." + key, value)
writer.add_float32("qwen35.attention.layer_norm_rms_epsilon", 1e-5)
writer.add_float32("qwen35.rope.freq_base", 10000)
writer.add_array("qwen35.rope.dimension_sections", [2, 1, 1, 0])
writer.add_tokenizer_model("gpt2")
writer.add_tokenizer_pre("qwen2")
vocabulary = ["token_" + str(index) for index in range(64)]
vocabulary[2:5] = ["a", "b", "ab"]
vocabulary[11], vocabulary[17], vocabulary[23] = "A", "B", "C"
if sys.argv[2] == "dense_context_merge":
    vocabulary[24] = "aA"
writer.add_token_list(vocabulary)
writer.add_token_merges(["a b", "a A"] if sys.argv[2] == "dense_context_merge" else ["a b"])
writer.add_bos_token_id(0)
writer.add_eos_token_id(1)
writer.add_add_bos_token(False)
generator = np.random.default_rng(1407)
embedding = generator.normal(0, 0.2, (64, 32)).astype(np.float32)
head = generator.normal(0, 0.1, (64, 32)).astype(np.float32)
if sys.argv[2] == "dense_nan":
    head[11, 0] = np.nan
writer.add_tensor("token_embd.weight", embedding)
writer.add_tensor("output.weight", head)
writer.add_tensor("output_norm.weight", np.ones(32, dtype=np.float32))
codes = ((np.arange(32 * 256).reshape(32, 256) * 7) % 3 - 1).astype(np.int8)
scales = ((np.arange(64).reshape(32, 2) % 7 + 1) / 256).astype(np.float16)
gate = np.zeros((256, 32), dtype=np.float32)
gate[:, 0] = 0.25
up = np.zeros((256, 32), dtype=np.float32)
up[np.arange(256), (np.arange(256) * 7) % 32] = np.repeat([0.125, 0.375], 128)
recurrent_qkv = np.zeros((64, 32), dtype=np.float32)
recurrent_qkv[np.arange(64), np.arange(64) % 32] = 0.25
recurrent_gate = np.eye(32, dtype=np.float32) * np.float32(0.5)
recurrent_out = np.eye(32, dtype=np.float32) * np.float32(0.125)
convolution = np.tile(np.array([0.125, 0.25, 1], dtype=np.float32), (64, 1))
attention_query = np.zeros((64, 32), dtype=np.float32)
attention_query[np.arange(4) * 16, np.arange(4)] = 0.25
attention_key = np.zeros((16, 32), dtype=np.float32)
attention_key[np.arange(2) * 8, np.arange(2) + 4] = 0.25
attention_value = np.zeros((16, 32), dtype=np.float32)
attention_value[np.arange(16), np.arange(16) + 3] = 0.375
attention_out = np.eye(32, dtype=np.float32) * np.float32(0.125)
for layer in range(4):
    prefix = "blk." + str(layer) + "."
    writer.add_tensor(prefix + "attn_norm.weight", np.ones(32, dtype=np.float32))
    writer.add_tensor(prefix + "post_attention_norm.weight", np.ones(32, dtype=np.float32))
    if layer % 2:
        for name, shape in {"attn_q.weight": (64, 32), "attn_k.weight": (16, 32),
                            "attn_v.weight": (16, 32), "attn_output.weight": (32, 32)}.items():
            values = np.zeros(shape, dtype=np.float32)
            if layer == 1 and sys.argv[4] == "nonzero":
                values = {"attn_q.weight": attention_query, "attn_k.weight": attention_key,
                          "attn_v.weight": attention_value, "attn_output.weight": attention_out}[name]
            writer.add_tensor(prefix + name, values)
        writer.add_tensor(prefix + "attn_q_norm.weight", np.ones(8, dtype=np.float32))
        writer.add_tensor(prefix + "attn_k_norm.weight", np.ones(8, dtype=np.float32))
    else:
        for name, shape in {
            "attn_qkv.weight": (64, 32), "attn_gate.weight": (32, 32), "ssm_conv1d.weight": (64, 3),
            "ssm_beta.weight": (4, 32), "ssm_alpha.weight": (4, 32), "ssm_out.weight": (32, 32),
        }.items():
            values = np.zeros(shape, dtype=np.float32)
            if layer == 0 and sys.argv[3] == "nonzero":
                values = {"attn_qkv.weight": recurrent_qkv, "attn_gate.weight": recurrent_gate,
                          "ssm_conv1d.weight": convolution, "ssm_out.weight": recurrent_out}.get(name, values)
            writer.add_tensor(prefix + name, values)
        writer.add_tensor(prefix + "ssm_norm.weight", np.ones(8, dtype=np.float32))
        writer.add_tensor(prefix + "ssm_dt.bias", np.zeros(4, dtype=np.float32))
        writer.add_tensor(prefix + "ssm_a", -np.ones(4, dtype=np.float32))
    writer.add_tensor(prefix + "ffn_gate.weight", gate if layer == 3 else np.zeros_like(gate))
    writer.add_tensor(prefix + "ffn_up.weight", up if layer == 3 else np.zeros_like(up))
    if layer != 3:
        writer.add_tensor(prefix + "ffn_down.weight", np.zeros((32, 256), dtype=np.float32))
    elif sys.argv[2].startswith("bitnet"):
        fields = (codes.reshape(32, 2, 32, 4) + 1).astype(np.uint8)
        if sys.argv[2].endswith("_invalid_code"):
            fields[0, 0, 0, 0] = 3
        elif sys.argv[2].endswith("_negative_scale"):
            scales[0, 0] = -1
        elif sys.argv[2].endswith("_nonfinite_scale"):
            scales[0, 0] = np.nan
        elif sys.argv[2].endswith("_changed_scale"):
            scales[0, 0] = np.float16(0.25)
        payload = np.bitwise_or.reduce(fields << (2 * np.arange(4, dtype=np.uint8)), axis=-1)
        encoded = np.concatenate((scales.view(np.uint8).reshape(32, 2, 2), payload), axis=-1)
        writer.add_tensor(prefix + "ffn_down.weight", encoded.reshape(32, 68),
                          raw_dtype=gguf.GGMLQuantizationType.PQ2_0)
        if sys.argv[2] == "bitnet-profile":
            reference = gguf.GGUFWriter(sys.argv[5], "qwen35")
            reference.add_string("jev.bitnet.execution", "group128-a8-fp32-nearest-even-identity-v1")
            reference.add_tensor(prefix + "ffn_down.weight", encoded.reshape(32, 68),
                                 raw_dtype=gguf.GGMLQuantizationType.PQ2_0)
            reference.write_header_to_file()
            reference.write_kv_data_to_file()
            reference.write_tensors_to_file()
            reference.close()
    else:
        writer.add_tensor(prefix + "ffn_down.weight", codes.astype(np.float32) * np.repeat(scales, 128, axis=1))
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
if sys.argv[2] not in ("dense", "bitnet", "bitnet-profile"):
    sys.exit(0)
def reference_logits(token_ids):
    sequence = embedding[token_ids].copy()
    hidden = sequence[-1].copy()
    if sys.argv[3] == "nonzero":
        history = np.zeros((3, 64), dtype=np.float32)
        state = np.zeros((4, 8, 8), dtype=np.float32)
        for index, token in enumerate(token_ids):
            current = embedding[token]
            normalized = current / np.sqrt(np.mean(current * current) + np.float32(1e-5))
            history[:-1] = history[1:]
            history[-1] = recurrent_qkv @ normalized
            convolved = np.sum(history * convolution.T, axis=0)
            convolved = convolved / (np.float32(1) + np.exp(-convolved))
            query = convolved[:16].reshape(2, 8)
            key = convolved[16:32].reshape(2, 8)
            query /= np.maximum(np.sqrt(np.sum(query * query, axis=1, keepdims=True)), np.float32(1e-5))
            key /= np.maximum(np.sqrt(np.sum(key * key, axis=1, keepdims=True)), np.float32(1e-5))
            query = np.tile(query, (2, 1)) / np.sqrt(np.float32(8))
            key = np.tile(key, (2, 1))
            value = convolved[32:].reshape(4, 8)
            state *= np.float32(0.5)
            predicted = np.einsum("hij,hj->hi", state, key)
            delta = (value - predicted) * np.float32(0.5)
            state += delta[:, :, None] * key[:, None, :]
            output = np.einsum("hij,hj->hi", state, query)
            output /= np.sqrt(np.mean(output * output, axis=1, keepdims=True) + np.float32(1e-5))
            gating = recurrent_gate @ normalized
            output = output.reshape(32) * (gating / (np.float32(1) + np.exp(-gating)))
            hidden = current + recurrent_out @ output
            sequence[index] = hidden
        assert np.linalg.norm(hidden - embedding[token_ids[-1]]) > 0.01
    if sys.argv[4] == "nonzero":
        normalized = sequence / np.sqrt(np.mean(sequence * sequence, axis=1, keepdims=True) + np.float32(1e-5))
        query = (normalized @ attention_query.T)[:, np.arange(4) * 16]
        query /= np.sqrt(query * query / np.float32(8) + np.float32(1e-5))
        key = (normalized @ attention_key.T)[:, np.arange(2) * 8]
        key /= np.sqrt(key * key / np.float32(8) + np.float32(1e-5))
        positions = np.arange(len(token_ids), dtype=np.float32)
        key_cosine = (key * np.cos(positions[:, None])).astype(np.float16).astype(np.float32)
        key_sine = (key * np.sin(positions[:, None])).astype(np.float16).astype(np.float32)
        value = (normalized @ attention_value.T).astype(np.float16).astype(np.float32).reshape(len(token_ids), 2, 8)
        original = sequence.copy()
        for index in range(len(token_ids)):
            output = np.zeros((4, 8), dtype=np.float32)
            for head_index in range(4):
                key_head = head_index // 2
                cosine = query[index, head_index] * np.cos(positions[index])
                sine = query[index, head_index] * np.sin(positions[index])
                scores = (cosine * key_cosine[:index + 1, key_head] + sine * key_sine[:index + 1, key_head]) / np.sqrt(np.float32(8))
                scores = np.exp(scores - np.max(scores))
                scores /= np.sum(scores)
                output[head_index] = np.sum(value[:index + 1, key_head] * scores[:, None], axis=0)
            sequence[index] += attention_out @ (output.reshape(32) * np.float32(0.5))
        assert np.linalg.norm(sequence - original) > 0.01
        hidden = sequence[-1]
    normalized = hidden / np.sqrt(np.mean(hidden * hidden) + np.float32(1e-5))
    gated = gate @ normalized
    activation = (gated / (np.float32(1) + np.exp(-gated))) * (up @ normalized)
    if sys.argv[2].startswith("bitnet"):
        groups = activation.reshape(2, 128)
        activation_scales = np.max(np.abs(groups), axis=1) / np.float32(127)
        activation_scales[activation_scales == 0] = 1
        quantized = np.clip(np.rint(groups / activation_scales[:, None]), -127, 127).astype(np.int32)
        partials = np.sum(codes.reshape(32, 2, 128).astype(np.int32) * quantized[None], axis=-1)
        hidden += np.sum(partials.astype(np.float32) * scales.astype(np.float32) * activation_scales, axis=1)
    else:
        hidden += (codes.astype(np.float32) * np.repeat(scales, 128, axis=1)) @ activation
    hidden = hidden / np.sqrt(np.mean(hidden * hidden) + np.float32(1e-5))
    return (head[[11, 17, 23]] @ hidden).tolist()
print(json.dumps({"logits": reference_logits([3, 5, 7]),
                  "peer_logits": reference_logits([4, 9, 19, 3, 5, 7]),
                  "text_logits": reference_logits([4, 2]),
                  "maximum_logits": reference_logits([2] * 128)}))
"""
    try:
        writer = subprocess.run([converter, "-c", writer_code, str(model_file), backend,
                                 "nonzero" if recurrent else "zero", "nonzero" if attention else "zero", str(reference_file)], capture_output=True, text=True,
                                timeout=30, env={**os.environ, "PYTHONPATH": str(Path(source_dir) / "gguf-py"),
                                                "PYTHONDONTWRITEBYTECODE": "1"})
        assert writer.returncode == 0, writer.stderr
        expected = json.loads(writer.stdout)
        command = ("--prefill-profile-bitnet-control" if profile_policy else
               "--prefill-bitnet-control" if bitnet else "--prefill-control")
        model_arguments = [str(model_file), str(reference_file)] if profile_policy else [str(model_file)]
        modes = ("full", "chunked", "reset", "reordered", "parallel", "maximum")
        if bitnet:
            modes += ("kernel-recovery",)
        for mode in modes:
            control = subprocess.run([str(Path(runtime_build) / "bin" / "prism_bitnet_loader_control"),
                                      command, *model_arguments, mode], capture_output=True, text=True, timeout=30)
            assert control.returncode == 0, control.stderr
            report = json.loads(control.stdout)
            if mode == "full":
                untraced_report = report
            token_ids = [2] * 128 if mode == "maximum" else [3, 5, 7]
            assert report["prefilled_tokens"] == len(token_ids) and report["generated_answer_tokens"] == 0
            assert report["input_tokens"] == token_ids
            assert report["backend"] == ("bitnet" if bitnet else "dense") and report["mode"] == mode
            assert report.get("model_policy_abi") == (2 if profile_policy else None)
            calls = 3 if mode == "parallel" else 2 if mode in ("chunked", "reset") else 1
            assert report["prefill_calls"] == calls
            assert report["rejected_prefill_calls"] == (mode == "kernel-recovery")
            assert report["kernel_error_status"] == (1 if mode == "kernel-recovery" else 0)
            assert report["bitnet_dispatch_calls"] == (calls if bitnet else 0)
            assert report["weight_repacks"] == bitnet
            input_count = 2 if mode == "chunked" else len(token_ids)
            assert report["bitnet_last_input_tokens"] == (input_count if bitnet else 0)
            assert report["memory_position_min"] == report["memory_position_max"] == len(token_ids) - 1
            order = [2, 0, 1] if mode == "reordered" else [0, 1, 2]
            assert report["option_ids"] == [(["inspect", "edit", "ask"][index]) for index in order]
            assert report["option_token_ids"] == [([11, 17, 23][index]) for index in order]
            assert report["option_labels"] == [(["A", "B", "C"][index]) for index in order]
            logits = np.asarray(expected["maximum_logits" if mode == "maximum" else "logits"], dtype=np.float64)[order]
            np.testing.assert_allclose(report["logits"], logits, rtol=2e-5, atol=2e-5)
            scores = np.exp(logits - logits.max())
            scores /= scores.sum()
            np.testing.assert_allclose(report["conditional_scores"], scores, rtol=2e-5, atol=2e-5)
            if mode == "parallel":
                np.testing.assert_allclose(report["peer_logits"], expected["peer_logits"], rtol=2e-5, atol=2e-5)
                if recurrent or attention:
                    assert not np.allclose(expected["peer_logits"], expected["logits"], rtol=1e-4, atol=1e-4)
            else:
                assert report["peer_logits"] == []
        trace_path = tmp_path / "trace"
        trace_command = ("--prefill-profile-bitnet-trace-control" if profile_policy else
                 "--prefill-bitnet-trace-control" if bitnet else "--prefill-trace-control")
        traced = subprocess.run([str(Path(runtime_build) / "bin" / "prism_bitnet_loader_control"),
                     trace_command, *model_arguments, str(trace_path)],
                                capture_output=True, text=True, timeout=30)
        assert traced.returncode == 0, traced.stderr
        trace = json.loads((trace_path / "manifest.json").read_text())
        assert trace["format"] == "jev-prefill-f32-trace-v1" and trace["byte_order"] == "little"
        assert trace["dtype"] == "float32" and len(trace["tensors"]) == 8
        shapes = {tensor["name"]: tensor["shape"] for tensor in trace["tensors"]}
        assert shapes == {"model.input_embed": [3, 32], "ffn_input": [3, 256], "ffn_output": [3, 32],
                  "final_norm": [1, 32], **{f"l_out-{layer}": [3, 32] for layer in range(4)}}
        for tensor in trace["tensors"]:
            values = np.fromfile(trace_path / (tensor["name"] + ".f32"), dtype="<f4")
            assert values.size == np.prod(tensor["shape"]) and np.isfinite(values).all()
        np.testing.assert_allclose(json.loads(traced.stdout)["logits"], expected["logits"], rtol=2e-5, atol=2e-5)
        assert json.loads(traced.stdout) == untraced_report
        refused = subprocess.run([str(Path(runtime_build) / "bin" / "prism_bitnet_loader_control"),
                                  trace_command, *model_arguments, str(trace_path)],
                                 capture_output=True, text=True, timeout=30)
        assert refused.returncode == 35 and refused.stdout == ""
        if profile_policy:
            for missing_arguments in ([str(model_file)], [str(model_file), str(tmp_path / "missing.gguf")]):
                rejected = subprocess.run([str(Path(runtime_build) / "bin" / "prism_bitnet_loader_control"),
                                           command, *missing_arguments], capture_output=True, text=True, timeout=30)
                assert rejected.returncode == (2 if len(missing_arguments) == 1 else 27)
                assert rejected.stdout == ""
            rejected = subprocess.run([str(Path(runtime_build) / "bin" / "prism_bitnet_loader_control"),
                                       "--prefill-model-bitnet-control", str(model_file), str(reference_file), "aba"],
                                      capture_output=True, text=True, timeout=30)
            assert rejected.returncode == 27 and rejected.stdout == "", rejected.stderr
            reference_bytes = reference_file.read_bytes()
            with reference_file.open("r+b") as oversized:
                oversized.truncate(1024 * 1024 + 1)
            rejected = subprocess.run([str(Path(runtime_build) / "bin" / "prism_bitnet_loader_control"),
                                       command, *model_arguments], capture_output=True, text=True, timeout=30)
            assert rejected.returncode == 27 and rejected.stdout == ""
            reference_file.write_bytes(reference_bytes[:-1])
            rejected = subprocess.run([str(Path(runtime_build) / "bin" / "prism_bitnet_loader_control"),
                                       command, *model_arguments], capture_output=True, text=True, timeout=30)
            assert rejected.returncode == 27 and rejected.stdout == ""
            reference_file.write_bytes(reference_bytes)
            model_bytes = model_file.read_bytes()
            with model_file.open("r+b") as oversized:
                oversized.truncate(1024 * 1024 + 1)
            rejected = subprocess.run([str(Path(runtime_build) / "bin" / "prism_bitnet_loader_control"),
                                       command, *model_arguments], capture_output=True, text=True, timeout=30)
            assert rejected.returncode == 27 and rejected.stdout == ""
            model_file.write_bytes(model_bytes)
        if backend == "dense":
            for prompt, token_ids, expected_key in (("aba", [4, 2], "text_logits"),
                                                    ("a" * 128, [2] * 128, "maximum_logits")):
                text_control = subprocess.run([str(Path(runtime_build) / "bin" / "prism_bitnet_loader_control"),
                                               "--prefill-text-control", str(model_file), prompt],
                                              capture_output=True, text=True, timeout=30)
                assert text_control.returncode == 0, text_control.stderr
                report = json.loads(text_control.stdout)
                assert report["input_tokens"] == token_ids and report["prefilled_tokens"] == len(token_ids)
                assert report["generated_answer_tokens"] == report["bitnet_dispatch_calls"] == report["weight_repacks"] == 0
                assert report["option_ids"] == ["inspect", "edit", "ask"]
                assert report["option_labels"] == ["A", "B", "C"] and report["option_token_ids"] == [11, 17, 23]
                assert report["memory_position_min"] == report["memory_position_max"] == len(token_ids) - 1
                np.testing.assert_allclose(report["logits"], expected[expected_key], rtol=2e-5, atol=2e-5)
                logits = np.asarray(expected[expected_key], dtype=np.float64)
                scores = np.exp(logits - logits.max())
                scores /= scores.sum()
                np.testing.assert_allclose(report["conditional_scores"], scores, rtol=2e-5, atol=2e-5)
            for prompt in ("", "a" * 129, "a" * 4097):
                rejected = subprocess.run([str(Path(runtime_build) / "bin" / "prism_bitnet_loader_control"),
                                           "--prefill-text-control", str(model_file), prompt],
                                          capture_output=True, text=True, timeout=30)
                assert rejected.returncode == 34 and rejected.stdout == "", rejected.stderr
        if bitnet:
            rejected = subprocess.run([str(Path(runtime_build) / "bin" / "prism_bitnet_loader_control"),
                                       command, *model_arguments, "kernel-error"], capture_output=True, text=True, timeout=30)
            assert rejected.returncode == 29 and rejected.stdout == "", rejected.stderr
        rejected = subprocess.run([str(Path(runtime_build) / "bin" / "prism_bitnet_loader_control"),
                                   command, *model_arguments, "unknown"], capture_output=True, text=True, timeout=30)
        assert rejected.returncode == 32 and rejected.stdout == ""
        for arguments in ([command], [command, *model_arguments, "full", "extra"],
                          ["--unknown"], ["--vocab-only", str(model_file)]):
            rejected = subprocess.run([str(Path(runtime_build) / "bin" / "prism_bitnet_loader_control"),
                                       *arguments], capture_output=True, text=True, timeout=30)
            assert rejected.returncode == 2 and rejected.stdout == ""
        corruptions = (
            tuple(backend + suffix for suffix in ("_invalid_code", "_negative_scale", "_nonfinite_scale"))
            if bitnet else ("dense_nan",)
        )
        if profile_policy:
            corruptions += (backend + "_wrong_profile", backend + "_changed_scale")
        for corruption in corruptions:
            writer = subprocess.run([converter, "-c", writer_code, str(model_file), corruption,
                                     "nonzero" if recurrent else "zero", "nonzero" if attention else "zero"],
                                    capture_output=True, text=True, timeout=30,
                                    env={**os.environ, "PYTHONPATH": str(Path(source_dir) / "gguf-py"),
                                         "PYTHONDONTWRITEBYTECODE": "1"})
            assert writer.returncode == 0, writer.stderr
            rejected = subprocess.run([str(Path(runtime_build) / "bin" / "prism_bitnet_loader_control"),
                                       command, *model_arguments], capture_output=True, text=True, timeout=30)
            assert rejected.returncode in (27, 30), rejected.stderr
            assert rejected.stdout == ""
            if backend == "dense":
                writer = subprocess.run([converter, "-c", writer_code, str(model_file), "dense_context_merge",
                                         "nonzero" if recurrent else "zero", "nonzero" if attention else "zero"],
                                        capture_output=True, text=True, timeout=30,
                                        env={**os.environ, "PYTHONPATH": str(Path(source_dir) / "gguf-py"),
                                             "PYTHONDONTWRITEBYTECODE": "1"})
                assert writer.returncode == 0, writer.stderr
                rejected = subprocess.run([str(Path(runtime_build) / "bin" / "prism_bitnet_loader_control"),
                                           "--prefill-text-control", str(model_file), "aba"],
                                          capture_output=True, text=True, timeout=30)
                assert rejected.returncode == 34 and rejected.stdout == "", rejected.stderr
    finally:
        model_file.unlink(missing_ok=True)
        reference_file.unlink(missing_ok=True)


def test_pinned_prism_complete_model_policy_binds_exact_projection(tmp_path):
    import ctypes

    source_dir = os.environ.get("PRISM_SOURCE_DIR")
    converter = os.environ.get("PRISM_CONVERTER_PYTHON")
    runtime_build = os.environ.get("MIMO_PRISM_RUNTIME_BUILD")
    if not all((source_dir, converter, runtime_build)):
        pytest.skip("requires pinned GGUF environment and full runtime; no real model conversion")
    writer_code = """
import sys
from pathlib import Path
import gguf
import numpy as np
directory = Path(sys.argv[1])
payload = np.zeros((4096, 96 * 34), dtype=np.uint8)
payload.reshape(-1, 34)[:, 1] = 60
projection = gguf.GGUFWriter(directory / "projection.gguf", "jev-tensor-control")
projection.add_string("jev.bitnet.execution", "group128-a8-fp32-nearest-even-identity-v1")
projection.add_tensor("blk.3.ffn_down.weight", payload, raw_dtype=gguf.GGMLQuantizationType.PQ2_0)
projection.write_header_to_file()
projection.write_kv_data_to_file()
projection.write_tensors_to_file()
projection.close()
for kind in ("valid", "wrong_tag", "wrong_revision", "wrong_arch", "wrong_layers", "other_pq2",
             "wrong_vocabulary", "payload_mismatch", "transform", "extra_tensor", "truncated"):
    writer = gguf.GGUFWriter(directory / (kind + ".gguf"), "qwen35" if kind != "wrong_arch" else "qwen3")
    writer.add_string("jev.bitnet.execution", "group128-a8-fp32-nearest-even-identity-v1")
    writer.add_string("jev.bitnet.model", "wrong" if kind == "wrong_tag" else "mimo-qwen35-layer3-ffn-down-v1")
    writer.add_string("jev.model.source_revision", "wrong" if kind == "wrong_revision" else "2367e865d009c13ac81713a2878291d33ab28177")
    writer.add_uint32("qwen35.block_count", 4 if kind == "wrong_layers" else 32)
    writer.add_uint32("qwen35.embedding_length", 4096)
    writer.add_uint32("qwen35.feed_forward_length", 12288)
    if kind == "transform":
        writer.add_uint32("prism.hadamard.version", 1)
    for name in ("token_embd.weight", "output.weight"):
        shape = (248319 if kind == "wrong_vocabulary" else 248320, 4096)
        writer.add_tensor_info(name, shape, np.dtype("uint16"), int(np.prod(shape)) * 2,
                               raw_dtype=gguf.GGMLQuantizationType.BF16)
    writer.add_tensor("blk.3.ffn_down.weight", payload, raw_dtype=gguf.GGMLQuantizationType.PQ2_0)
    for index in range(424 + (kind == "extra_tensor")):
        if index == 0 and kind == "other_pq2":
            writer.add_tensor("blk.0.control.weight", np.zeros((1, 34), dtype=np.uint8),
                              raw_dtype=gguf.GGMLQuantizationType.PQ2_0)
        else:
            writer.add_tensor("blk.0.control_" + str(index) + ".weight", np.zeros(1, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    output = writer.fout[0]
    writer.write_padding(output, output.tell())
    data_offset = output.tell()
    tensor_infos = list(writer.tensors[0].values())
    total_bytes = sum(gguf.GGUFWriter.ggml_pad(info.nbytes, writer.data_alignment) for info in tensor_infos)
    output.truncate(data_offset + total_bytes)
    target_offset = sum(gguf.GGUFWriter.ggml_pad(info.nbytes, writer.data_alignment) for info in tensor_infos[:2])
    output.seek(data_offset + target_offset)
    encoded = payload.copy()
    if kind == "payload_mismatch":
        encoded.flat[2] = 1
    output.write(encoded.tobytes())
    if kind == "truncated":
        output.truncate(data_offset + target_offset + 1)
    writer.close()
"""
    writer = subprocess.run(
        [converter, "-c", writer_code, str(tmp_path)], capture_output=True, text=True, timeout=60,
        env={**os.environ, "PYTHONPATH": str(Path(source_dir) / "gguf-py"), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert writer.returncode == 0, writer.stderr
    native = ctypes.CDLL(str(Path(runtime_build) / "bin" / "libprism_group_scale.so"))
    model_policy = native.prism_bitnet_cpu_model_override_from_gguf_v1
    model_policy.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32,
                            ctypes.POINTER(ctypes.c_void_p)]
    model_policy.restype = ctypes.c_int
    projection = os.fsencode(tmp_path / "projection.gguf")
    for kind in ("wrong_tag", "wrong_revision", "wrong_arch", "wrong_layers", "other_pq2",
                 "wrong_vocabulary", "payload_mismatch", "transform", "extra_tensor", "truncated"):
        overrides = ctypes.c_void_p(123)
        assert model_policy(os.fsencode(tmp_path / (kind + ".gguf")), projection,
                            PRISM_REVISION.encode(), 1, ctypes.byref(overrides)) == 6
        assert overrides.value is None
    overrides = ctypes.c_void_p()
    assert model_policy(os.fsencode(tmp_path / "valid.gguf"), projection, PRISM_REVISION.encode(),
                        1, ctypes.byref(overrides)) == 0
    assert overrides.value is not None
    legacy = native.prism_bitnet_cpu_loader_override_from_gguf_v1
    legacy.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
    legacy.restype = ctypes.c_int
    assert legacy(os.fsencode(tmp_path / "valid.gguf"), PRISM_REVISION.encode(), 1, ctypes.byref(overrides)) == 6
    assert overrides.value is None
    control = str(Path(runtime_build) / "bin" / "prism_bitnet_loader_control")
    arguments = ["--prefill-model-bitnet-control", str(tmp_path / "wrong_tag.gguf"),
                 str(tmp_path / "projection.gguf"), "a"]
    for options, status in ((arguments, 27), (arguments[:-1], 2),
                            ([*arguments, "full", "extra"], 2),
                            ([*arguments, "unknown"], 32), ([*arguments, "chunked"], 34)):
        rejected = subprocess.run([control, *options], capture_output=True, text=True, timeout=15)
        assert rejected.returncode == status and rejected.stdout == "", rejected.stderr


def test_pinned_prism_profile_complete_model_policy_binds_exact_table_and_projection(tmp_path):
    import ctypes

    from embedded_jev.prism_codec import create_profile_model_policy, profile_model_specification
    from test_inventory import _model_policy_fixture

    source_dir = os.environ.get("PRISM_SOURCE_DIR")
    converter = os.environ.get("PRISM_CONVERTER_PYTHON")
    runtime_build = os.environ.get("MIMO_PRISM_RUNTIME_BUILD")
    if not all((source_dir, converter, runtime_build)):
        pytest.skip("requires cached pinned GGUF environment and runtime; synthetic policy only")
    assert subprocess.check_output(["git", "-C", source_dir, "rev-parse", "HEAD"], text=True).strip() == PRISM_REVISION
    profile, inventory, manifest, blocks = _model_policy_fixture()
    specification = profile_model_specification(profile, inventory, manifest, blocks)
    (tmp_path / "payload.bin").write_bytes(blocks.tobytes())
    writer_code = """
import json
import sys
from pathlib import Path
import gguf
import numpy as np
spec = json.loads(sys.argv[1])
directory = Path(sys.argv[2])
payload = (directory / "payload.bin").read_bytes()
for kind in ("valid", "metadata-large", "metadata-near-limit", "metadata-keys-limit", "metadata-keys-over",
             "metadata-over", "file-limit", "file-over", "header-tensors", "header-keys",
             "profile", "fingerprint", "revision", "converter", "source_digest", "tag", "execution",
             "layers", "hidden", "intermediate", "architecture", "name", "shape", "dtype", "other_pq2",
             "target_shape", "payload", "transform", "truncated", "extra", "missing", "heads", "rope", "epsilon", "recurrent"):
    writer = gguf.GGUFWriter(directory / (kind + ".gguf"), "qwen3" if kind == "architecture" else "qwen35")
    for key, value, corruption in (
        ("jev.model.profile_id", spec["profile_id"], "profile"),
        ("jev.model.profile_sha256", spec["profile_sha256"], "fingerprint"),
        ("jev.model.source_revision", spec["source_revision"], "revision"),
        ("jev.model.converter_revision", "842b1880415d6f508f03b789e5ce70194def7bfd", "converter"),
        ("jev.bitnet.source_tensor_sha256", spec["source_tensor_sha256"], "source_digest"),
        ("jev.bitnet.model", "profile-qwen35-single-projection-v2", "tag"),
        ("jev.bitnet.execution", "group128-a8-fp32-nearest-even-identity-v1", "execution"),
    ):
        writer.add_string(key, "wrong" if kind == corruption else value)
    for key, value, corruption in (("block_count", spec["layers"], "layers"),
                                    ("embedding_length", spec["hidden"], "hidden"),
                                    ("feed_forward_length", spec["intermediate"], "intermediate")):
        writer.add_uint32("qwen35." + key, value + (kind == corruption))
    for item in spec["metadata"]:
        name, values = item["name"], item["values"].copy()
        if (kind, name) in (("heads", "qwen35.attention.head_count"),
                           ("rope", "qwen35.rope.dimension_sections"),
                           ("epsilon", "qwen35.attention.layer_norm_rms_epsilon")):
            values[0] += 1
        if item["kind"] == "u32":
            writer.add_uint32(name, values[0])
        elif item["kind"] == "f32":
            writer.add_float32(name, values[0])
        else:
            writer.add_array(name, values)
    if kind == "recurrent":
        writer.add_array("qwen35.attention.recurrent_layers", [True])
    if kind == "transform":
        writer.add_uint32("prism.hadamard.version", 1)
    target = spec["projection"]["tensor_name"]
    if kind == "metadata-large":
        writer.add_string("jev.test.metadata", "x" * (1024**2 + 1))
    if kind in ("metadata-near-limit", "metadata-over"):
        size = spec["maximum_metadata_bytes"] + (1 if kind == "metadata-over" else -65536)
        writer.add_string("jev.test.metadata", "x" * size)
    if kind in ("metadata-keys-limit", "metadata-keys-over"):
        count = spec["maximum_metadata_keys"] + (kind == "metadata-keys-over")
        for index in range(count - len(writer.kv_data[0])):
            writer.add_uint32("jev.test.key." + str(index), index)
    for item in spec["tensors"]:
        if kind == "missing" and item["name"] == "output_norm.weight":
            continue
        name = "blk.0.control.weight" if kind == "name" and item["name"] == "output_norm.weight" else item["name"]
        shape = list(reversed(item["ne"][:1 if item["ne"][1] == 1 else 2]))
        dtype = item["type"]
        if kind == "shape" and name == "output.weight":
            shape[0] -= 1
        if kind == "target_shape" and name == target:
            shape[0] -= 1
        if kind == "dtype" and name == "output_norm.weight":
            dtype = "BF16"
        if kind == "other_pq2" and name == "blk.0.ffn_up.weight":
            dtype = "PQ2_0"
        count = int(np.prod(shape))
        size = count // 128 * 34 if dtype == "PQ2_0" else count * (4 if dtype == "F32" else 2)
        if dtype == "PQ2_0":
            shape[-1] = shape[-1] // 128 * 34
        writer.add_tensor_info(name, shape, np.dtype("uint8" if dtype == "PQ2_0" else "uint16"), size,
                               raw_dtype=getattr(gguf.GGMLQuantizationType, dtype))
    if kind == "extra":
        writer.add_tensor("blk.0.control.weight", np.zeros(1, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    output = writer.fout[0]
    writer.write_padding(output, output.tell())
    data_offset = output.tell()
    infos = list(writer.tensors[0].items())
    output.truncate(data_offset + sum(gguf.GGUFWriter.ggml_pad(info.nbytes, writer.data_alignment) for _, info in infos))
    offset = sum(gguf.GGUFWriter.ggml_pad(info.nbytes, writer.data_alignment) for name, info in infos[:next(index for index, (name, _) in enumerate(infos) if name == target)])
    output.seek(data_offset + offset)
    encoded = bytearray(payload)
    if kind == "payload":
        encoded[-1] ^= 1
    output.write(encoded[:next(info.nbytes for name, info in infos if name == target)])
    if kind == "truncated":
        output.truncate(data_offset + offset + 1)
    if kind in ("metadata-over", "file-limit", "file-over"):
        output.truncate(spec["maximum_file_bytes"] + (kind == "file-over"))
    if kind in ("header-tensors", "header-keys"):
        import struct
        output.seek(8 if kind == "header-tensors" else 16)
        output.write(struct.pack("<Q", 2**64 - 1))
    writer.close()
"""
    writer = subprocess.run([converter, "-c", writer_code, json.dumps(specification), str(tmp_path)],
                            capture_output=True, text=True, timeout=30,
                            env={**os.environ, "PYTHONPATH": str(Path(source_dir) / "gguf-py"), "PYTHONDONTWRITEBYTECODE": "1"})
    assert writer.returncode == 0, writer.stderr
    native = ctypes.CDLL(str(Path(runtime_build) / "bin" / "libprism_group_scale.so"))
    initialize = native.prism_bitnet_cpu_runtime_init_v1
    initialize.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
    buffer_type = ctypes.c_void_p()
    assert initialize(PRISM_REVISION.encode(), 1, ctypes.byref(buffer_type)) == 0
    registry = native.prism_bitnet_registered_tensor_registry_size
    registry.restype = ctypes.c_size_t
    baseline = registry()
    free = native.prism_bitnet_cpu_model_policy_free_v2
    free.argtypes = [ctypes.c_void_p]
    free.restype = None

    class Override(ctypes.Structure):
        _fields_ = [("pattern", ctypes.c_char_p), ("buffer_type", ctypes.c_void_p)]

    accepted = {"valid", "metadata-large", "metadata-near-limit", "metadata-keys-limit", "file-limit"}
    for kind in (*sorted(accepted), "valid"):
        handle, overrides, result = create_profile_model_policy(native, tmp_path / (kind + ".gguf"), profile, inventory, manifest, blocks)
        try:
            rules = ctypes.cast(overrides, ctypes.POINTER(Override))
            assert rules[0].pattern == b"^blk\\.0\\.ffn_down\\.weight$" and rules[0].buffer_type != buffer_type.value
            assert rules[1].pattern is None and rules[1].buffer_type is None
            assert registry() == baseline + 1
            assert result["source_payload_verification"] == "declared_digest_not_rehashed_source_tensor"
        finally:
            free(handle)
        assert registry() == baseline
    for path in sorted(tmp_path.glob("*.gguf")):
        if path.stem in accepted:
            continue
        with pytest.raises(ValueError, match="policy failed: 6"):
            create_profile_model_policy(native, path, profile, inventory, manifest, blocks)
        assert registry() == baseline
    function = native.prism_bitnet_cpu_model_override_from_gguf_v2
    from embedded_jev.prism_codec import native_model_specification

    record = native_model_specification(specification, blocks)
    for kind in ("metadata-keys-over", "metadata-over", "file-over", "header-tensors", "header-keys"):
        handle, overrides = ctypes.c_void_p(123), ctypes.c_void_p(123)
        assert function(os.fsencode(tmp_path / (kind + ".gguf")), ctypes.byref(record), PRISM_REVISION.encode(), 2,
                        ctypes.byref(handle), ctypes.byref(overrides)) == 6
        assert handle.value is overrides.value is None and registry() == baseline
    handle, overrides = ctypes.c_void_p(123), ctypes.c_void_p(123)
    assert function(os.fsencode(tmp_path / "valid.gguf"), None, PRISM_REVISION.encode(), 2,
                    ctypes.byref(handle), ctypes.byref(overrides)) == 1
    assert handle.value is overrides.value is None


@pytest.mark.parametrize("kind", ["vector-BF16", "vector-F32", "matrix-BF16", "convolution-BF16", "convolution-F32", "frozen-BF16", "complete-BF16", "complete-F32", "complete-owner-BF16", "complete-owner-F32", "norm-nonzero-BF16", "norm-nonzero-F32", "exp-nonzero-BF16", "exp-nonzero-F32", "matrix-rows-nonzero-BF16", "matrix-columns-nonzero-BF16", "convolution-nonzero-BF16", "convolution-nonzero-F32", "complete-owner-two-shards-BF16", "complete-owner-two-shards-F32"])
def test_pinned_prism_source_to_gguf_selected_range_parity(tmp_path, kind):
    from test_inventory import _prefill_model_policy_fixture

    source_dir = os.environ.get("PRISM_SOURCE_DIR")
    interpreter = os.environ.get("PRISM_PARITY_PYTHON")
    runtime_build = os.environ.get("MIMO_PRISM_RUNTIME_BUILD")
    if not source_dir or not interpreter:
        pytest.skip("requires cached pinned GGUF/Torch parity environment; creates synthetic files only")
    if kind.startswith("complete-owner") and not runtime_build:
        pytest.skip("requires existing pinned native policy library; synthetic files only")
    if kind.startswith("complete-owner"):
        assert (Path(runtime_build) / "bin" / "libprism_group_scale.so").is_file()
        assert (Path(runtime_build) / "bin" / "libprism_bitnet_prefill_test.so").is_file()
    assert subprocess.check_output(["git", "-C", source_dir, "rev-parse", "HEAD"], text=True).strip() == PRISM_REVISION
    profile, inventory, manifest, blocks = _prefill_model_policy_fixture(
        source_shards=2 if "two-shards" in kind else 1,
        pin_zero_source=True, vector_dtype="F32" if kind == "vector-F32" or (kind.startswith(("complete", "norm-nonzero", "exp-nonzero")) and kind.endswith("F32")) else "BF16", conv_dtype="F32" if kind == "convolution-F32" or (kind.startswith(("complete", "convolution-nonzero")) and kind.endswith("F32")) else "BF16",
    )
    code = """
import hashlib
import json
import sys
import tempfile
from pathlib import Path
import gguf
import numpy as np
from embedded_jev.model_profile import parse_model_profile, profile_source
from embedded_jev.prism_codec import profile_model_conversion_plan, profile_model_specification, verified_profile_model_policy, verify_profile_gguf_source_operand, verify_profile_frozen_source_tensor, verify_profile_gguf_source_tensor_table, _verify_profile_gguf_operand
fixture = json.loads(sys.stdin.read())
profile = parse_model_profile(json.dumps(fixture["profile"]).encode())
inventory, manifest = fixture["inventory"], fixture["manifest"]
blocks = np.frombuffer(bytes.fromhex(fixture["payload_hex"]), dtype=np.uint8).reshape(manifest["shape"][0], manifest["shape"][1] // 128, 34)
plan = profile_model_conversion_plan(profile, inventory, manifest, blocks)
frozen = sys.argv[1] == "frozen-BF16"
complete = sys.argv[1].startswith("complete")
owned = sys.argv[1].startswith("complete-owner")
nonzero = "nonzero" in sys.argv[1]
selected_transform = "v_columns_grouped_to_tiled" if sys.argv[1].startswith("matrix-columns") else "v_rows_grouped_to_tiled" if sys.argv[1].startswith(("matrix-rows", "convolution-nonzero")) else "negative_exp" if sys.argv[1].startswith("exp-nonzero") else "add_one"
rank = 2 if frozen or sys.argv[1].startswith("matrix") else 3 if sys.argv[1].startswith("convolution") else 1
entry = next(item for item in plan["tensors"] if (item["type"] == "PQ2_0") == frozen and len(item["source_shape"]) == rank and (not nonzero or selected_transform in item["transforms"]))
values = np.zeros(entry["source_shape"], dtype="<f4" if entry["source_dtype"] == "F32" else "<u2")
if nonzero:
    import torch
    if rank == 1:
        supplied = np.linspace(-0.5, 0.5, values.size, dtype="<f4").reshape(values.shape)
        values = (supplied.view("<u4") >> 16).astype("<u2") if entry["source_dtype"] == "BF16" else supplied
    elif entry["source_dtype"] == "BF16":
        values = (0x3F00 + np.arange(values.size).reshape(values.shape) % 97).astype("<u2")
    else:
        values = (np.float32(0.125) + (np.arange(values.size).reshape(values.shape) % 97).astype("<f4") / np.float32(64)).astype("<f4")
    decoded = (values.astype("<u4") << 16).view("<f4") if entry["source_dtype"] == "BF16" else values
    with torch.inference_mode():
        reference = torch.from_numpy(decoded)
        if selected_transform == "negative_exp":
            assert profile.geometry["linear_num_key_heads"] == 2 and profile.geometry["linear_num_value_heads"] == 4 and values.size == 4
            expected_operand = -reference[[0, 2, 1, 3]].exp().numpy()
        elif selected_transform == "add_one":
            expected_operand = (reference + 1).numpy()
        else:
            assert profile.geometry["linear_num_key_heads"] == 2 and profile.geometry["linear_num_value_heads"] == 4
            assert profile.geometry["linear_key_head_dim"] == profile.geometry["linear_value_head_dim"] == 32
            value_order = np.array([*range(32), *range(64, 96), *range(32, 64), *range(96, 128)])
            if selected_transform == "v_columns_grouped_to_tiled":
                assert values.shape[1] == 128
                expected_operand = values[:, value_order].copy()
            else:
                assert values.shape[0] == 256
                row_order = np.concatenate((np.arange(128), 128 + value_order))
                expected_operand = values[row_order].copy() if rank == 2 else decoded[row_order, 0, :].copy()
    wrong_operand = values if rank == 2 else decoded[:, 0, :] if rank == 3 else -reference.exp().numpy() if selected_transform == "negative_exp" else decoded
    assert not np.array_equal(expected_operand.view("u1"), wrong_operand.view("u1"))
if frozen:
    manifest["source_tensor_sha256"] = hashlib.sha256(memoryview(values)).hexdigest()
if complete:
    source_arrays = {item["source_name"]: np.zeros(item["source_shape"], dtype="<f4" if item["source_dtype"] == "F32" else "<u2") for item in plan["tensors"]}
    source_digests = {name: hashlib.sha256(memoryview(array)).hexdigest() for name, array in source_arrays.items()}
    manifest["source_tensor_sha256"] = source_digests[manifest["tensor"]]
entries = {}
shard_entries = {name: {} for name in profile.packaging["shards"]}
payload_bytes = dict.fromkeys(shard_entries, 0)
for item in sorted(inventory["tensors"], key=lambda item: item["name"]):
    shard = item["shard"]
    size = int(np.prod(item["shape"])) * (4 if item["dtype"] == "F32" else 2)
    entries[item["name"]] = {"dtype": item["dtype"], "shape": item["shape"], "data_offsets": [payload_bytes[shard], payload_bytes[shard] + size]}
    shard_entries[shard][item["name"]] = entries[item["name"]]
    payload_bytes[shard] += size
prefixes = {}
for shard, selected in shard_entries.items():
    header = json.dumps(selected, sort_keys=True).encode()
    prefixes[shard] = len(header).to_bytes(8, "little") + header
with tempfile.TemporaryDirectory(dir=sys.argv[2]) as directory:
    directory = Path(directory)
    source_paths = {}
    for shard in profile.packaging["shards"]:
        source_path = directory / shard
        source_path.write_bytes(prefixes[shard] + bytes(payload_bytes[shard]))
        source_paths[shard] = source_path
        assert source_path.stat().st_size == profile.files[shard].bytes
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == profile.files[shard].sha256
    assert sum(path.stat().st_size for path in source_paths.values()) <= 16 * 1024**2
    selected_shard = next(item["shard"] for item in inventory["tensors"] if item["name"] == entry["source_name"])
    source_path, prefix = source_paths[selected_shard], prefixes[selected_shard]
    source_pin = profile.files[source_path.name]
    assert source_path.stat().st_size == source_pin.bytes <= 16 * 1024**2
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_pin.sha256
    if nonzero:
        with source_path.open("r+b") as output:
            output.seek(len(prefix) + entries[entry["source_name"]]["data_offsets"][0])
            output.write(values.tobytes())
        fixture["profile"]["files"][source_path.name]["sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
        profile = parse_model_profile(json.dumps(fixture["profile"]).encode())
        inventory["source"].update(profile_source(profile))
        manifest["profile_sha256"] = profile.sha256
        source_pin = profile.files[source_path.name]
    destination = directory / "selected.gguf"
    writer = gguf.GGUFWriter(destination, "qwen35")
    if owned:
        spec = profile_model_specification(profile, inventory, manifest, blocks)
        for key, value in {
            "jev.model.profile_id": spec["profile_id"], "jev.model.profile_sha256": spec["profile_sha256"],
            "jev.model.source_revision": spec["source_revision"], "jev.bitnet.source_tensor_sha256": spec["source_tensor_sha256"],
            "jev.model.converter_revision": "842b1880415d6f508f03b789e5ce70194def7bfd",
            "jev.bitnet.model": "profile-qwen35-single-projection-v2", "jev.bitnet.execution": "group128-a8-fp32-nearest-even-identity-v1",
        }.items():
            writer.add_string(key, value)
        for key, value in {"block_count": spec["layers"], "embedding_length": spec["hidden"], "feed_forward_length": spec["intermediate"]}.items():
            writer.add_uint32("qwen35." + key, value)
        for item in spec["metadata"]:
            if item["kind"] == "u32":
                writer.add_uint32(item["name"], item["values"][0])
            elif item["kind"] == "f32":
                writer.add_float32(item["name"], item["values"][0])
            else:
                writer.add_array(item["name"], item["values"])
        writer.add_uint32("qwen35.context_length", 256)
        writer.add_tokenizer_model("gpt2")
        writer.add_tokenizer_pre("qwen2")
        vocabulary = ["token_" + str(index) for index in range(32)]
        vocabulary[2:5] = ["a", "b", "ab"]
        for label, token in profile.tokenization["label_token_ids"].items():
            vocabulary[token] = label
        writer.add_token_list(vocabulary)
        writer.add_token_merges(["a b"])
        writer.add_bos_token_id(0)
        writer.add_eos_token_id(1)
        writer.add_add_bos_token(False)
    for output_entry in plan["tensors"] if complete else [entry]:
        if nonzero:
            if output_entry["type"] == "BF16":
                writer.add_tensor(output_entry["name"], expected_operand.view("u1"), raw_dtype=gguf.GGMLQuantizationType.BF16)
            else:
                writer.add_tensor(output_entry["name"], expected_operand)
        elif output_entry["type"] == "PQ2_0":
            writer.add_tensor(output_entry["name"], blocks.reshape(manifest["shape"][0], manifest["shape"][1] // 128 * 34), raw_dtype=gguf.GGMLQuantizationType.PQ2_0)
        elif output_entry["type"] == "BF16":
            encoded = gguf.quants.quantize(np.zeros(output_entry["source_shape"], dtype=np.float32), gguf.GGMLQuantizationType.BF16)
            writer.add_tensor(output_entry["name"], encoded, raw_dtype=gguf.GGMLQuantizationType.BF16)
        else:
            dimensions = 1 if len(output_entry["source_shape"]) == 1 else 2
            shape = tuple(reversed(output_entry["ne"][:dimensions]))
            converted = np.ones(shape, dtype=np.float32) if "add_one" in output_entry["transforms"] else -np.ones(shape, dtype=np.float32) if "negative_exp" in output_entry["transforms"] else np.zeros(shape, dtype=np.float32)
            writer.add_tensor(output_entry["name"], converted)
    if not complete:
        for index in range(3):
            writer.add_tensor("synthetic_placeholder_" + str(index), np.zeros(1, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    reader = gguf.GGUFReader(destination)
    arguments = {
        "reader": reader, "source_path": source_path, "source_name": entry["source_name"], "source_values": values,
        "expected_source_sha256": hashlib.sha256(memoryview(values)).hexdigest(),
        "expected_source_file_bytes": source_pin.bytes, "expected_source_file_sha256": source_pin.sha256,
        "expected_file_bytes": destination.stat().st_size, "expected_file_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    owner_witness, parsed_scores = [None], [0]
    def verify(active_source=None, source_effect="change"):
        if owned:
            import ctypes
            import os
            from embedded_jev.prism_codec import NativeModelSpecification
            native = ctypes.CDLL(str(Path(sys.argv[3]) / "bin" / "libprism_group_scale.so"))
            control = ctypes.CDLL(str(Path(sys.argv[3]) / "bin" / "libprism_bitnet_prefill_test.so"))
            initialize = native.prism_bitnet_cpu_runtime_init_v1
            initialize.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
            buffer_type = ctypes.c_void_p()
            assert initialize(b"842b1880415d6f508f03b789e5ce70194def7bfd", 1, ctypes.byref(buffer_type)) == 0
            registry = native.prism_bitnet_registered_tensor_registry_size
            registry.restype = ctypes.c_size_t
            baseline = registry()
            prefill = control.prism_bitnet_test_prefill_policy_v2
            prefill.argtypes = [ctypes.c_char_p, ctypes.POINTER(NativeModelSpecification), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
            prefill.restype = ctypes.c_int
            dense_sources = {name: array for name, array in source_arrays.items() if name != manifest["tensor"]}
            with verified_profile_model_policy(
                native, destination, profile, inventory, manifest, blocks,
                expected_bytes=arguments["expected_file_bytes"], expected_sha256=arguments["expected_file_sha256"],
                tensor_reader=reader, source_values=dense_sources,
                expected_source_sha256={name: source_digests[name] for name in dense_sources},
                source_paths=source_paths, frozen_source_values=source_arrays[manifest["tensor"]],
                verify_policy_metadata=True, verify_source_files_on_exit=True,
            ) as owner:
                owner_witness[0] = owner, registry, baseline
                assert owner.handle.value and owner.overrides.value
                assert owner.verification["source_table_verification"] == "all_selected_source_ranges_match_profile_pins_and_caller_operands"
                assert owner.verification["source_file_lifetime_verification"] == "entry_and_exit_checks_required"
                assert owner.verification["policy_metadata"]["model_metadata_verification"] == "selected_policy_records_only"
                evidence = owner.verification["source_conversion_table"]
                assert registry() == baseline + 1
                with tempfile.TemporaryFile(dir=directory) as output:
                    standard_output = os.dup(1)
                    ctypes.CDLL(None).fflush(None)
                    try:
                        os.dup2(output.fileno(), 1)
                        status = prefill(owner.model_path.encode(), ctypes.byref(owner.record), owner.handle, owner.overrides, b"full")
                    finally:
                        ctypes.CDLL(None).fflush(None)
                        os.dup2(standard_output, 1)
                        os.close(standard_output)
                    assert status == 0
                    assert output.tell() <= 65536
                    output.seek(0)
                    prefill_bytes = output.read(65537)
                    assert 0 < len(prefill_bytes) <= 65536
                    assert registry() == baseline + 1
                if active_source is not None:
                    import os
                    observed = active_source.stat()
                    if source_effect == "replace":
                        import shutil
                        replacement = active_source.with_suffix(".replacement")
                        with active_source.open("rb") as source, replacement.open("wb") as replacement_file:
                            shutil.copyfileobj(source, replacement_file, length=65536)
                        os.replace(replacement, active_source)
                    else:
                        offset = profile.files[active_source.name].header_bytes
                        with active_source.open("r+b") as source:
                            source.seek(offset)
                            saved = source.read(1)
                            source.seek(offset)
                            source.write(bytes([saved[0] ^ 1]))
                            if source_effect == "restored":
                                source.seek(offset)
                                source.write(saved)
                        if source_effect == "restored":
                            os.utime(active_source, ns=(observed.st_atime_ns, observed.st_mtime_ns))
            score = json.loads(prefill_bytes)
            parsed_scores[0] += 1
            assert owner.handle.value is owner.overrides.value is None and registry() == baseline
            assert score["model_policy_abi"] == 2 and score["backend"] == "bitnet"
            assert score["generated_answer_tokens"] == 0 and score["prefilled_tokens"] > 0
            assert score["bitnet_dispatch_calls"] == score["prefill_calls"] > 0 and score["weight_repacks"] == 1
            assert score["option_ids"] == ["inspect", "edit", "ask"] and score["option_token_ids"] == [11, 17, 23]
            assert score["option_labels"] == ["A", "B", "C"]
            np.testing.assert_allclose(score["logits"], np.zeros(3), rtol=2e-5, atol=2e-5)
            np.testing.assert_allclose(score["conditional_scores"], np.full(3, 1 / 3), rtol=2e-5, atol=2e-5)
            try:
                owner.model_path
                raise AssertionError("closed owner descriptor remained public")
            except ValueError as error:
                assert "closed" in str(error), str(error)
            return evidence
        if complete:
            return verify_profile_gguf_source_tensor_table(
                profile, inventory, manifest, blocks, reader=reader, source_paths=source_paths,
                source_values=source_arrays, expected_source_sha256=source_digests,
                expected_file_bytes=arguments["expected_file_bytes"], expected_file_sha256=arguments["expected_file_sha256"],
            )
        if not frozen:
            return verify_profile_gguf_source_operand(profile, inventory, manifest, blocks, **arguments)
        source_receipt = verify_profile_frozen_source_tensor(
            profile, inventory, manifest, blocks, source_path=source_path, source_values=values,
            expected_file_bytes=source_pin.bytes, expected_file_sha256=source_pin.sha256,
        )
        converted_receipt = _verify_profile_gguf_operand(
            profile, inventory, manifest, blocks, reader=reader, source_name=entry["source_name"], source_values=blocks,
            expected_source_sha256=manifest["payload_sha256"], expected_file_bytes=arguments["expected_file_bytes"],
            expected_file_sha256=arguments["expected_file_sha256"], maximum_file_bytes=16 * 1024**2,
            kind="projection", maximum_bytes=16 * 1024**2,
        )
        assert source_receipt["source_sha256"] == manifest["source_tensor_sha256"]
        assert source_receipt["source_to_frozen_quantization_verification"] == "not_checked"
        assert source_receipt["frozen_payload_sha256"] == converted_receipt["projection_payload_sha256"]
        return source_receipt

    receipt = verify()
    if owned and len(source_paths) == 2:
        for active_source in source_paths.values():
            offset = profile.files[active_source.name].header_bytes
            with active_source.open("rb") as source:
                source.seek(offset)
                saved = source.read(1)
            for source_effect in ("change", "restored", "replace"):
                before_scores = parsed_scores[0]
                try:
                    verify(active_source=active_source, source_effect=source_effect)
                    raise AssertionError("changed active shard accepted after native prefill")
                except ValueError as error:
                    assert "source table shard identity changed after operands" in str(error), str(error)
                    owner, registry, baseline = owner_witness[0]
                    assert owner.handle.value is owner.overrides.value is None and registry() == baseline
                    assert parsed_scores[0] == before_scores
                    try:
                        owner.model_path
                        raise AssertionError("source-exit failure retained public model descriptor")
                    except ValueError as closed:
                        assert "closed" in str(closed), str(closed)
                finally:
                    with active_source.open("r+b") as source:
                        source.seek(offset)
                        source.write(saved)
    if not frozen:
        from embedded_jev import prism_codec as codec
        original_check = codec._verify_profile_gguf_operand
        def changed_source(*received, **keywords):
            result = original_check(*received, **keywords)
            with source_path.open("r+b") as output:
                output.seek(source_pin.bytes - 1)
                output.write(bytes([1]))
                output.seek(source_pin.bytes - 1)
                output.write(bytes(1))
            return result
        codec._verify_profile_gguf_operand = changed_source
        try:
            verify()
            raise AssertionError("restored source write between file stages accepted")
        except ValueError as error:
            assert "source identity changed" in str(error), str(error)
        finally:
            codec._verify_profile_gguf_operand = original_check
    if complete:
        assert receipt["tensor_count"] == receipt["source"]["tensor_count"] == receipt["converted"]["tensor_count"] == 28
        assert receipt["source"]["source_shard_count"] == len(source_paths)
        assert receipt["source"]["source_file_bytes"] == sum(path.stat().st_size for path in source_paths.values())
        assert receipt["source_to_frozen_quantization_verification"] == "not_checked"
        source_records = {item["source_name"]: item for item in receipt["source"]["tensors"]}
        converted_records = {item["source_name"]: item for item in receipt["converted"]["tensors"]}
        for item in plan["tensors"]:
            name = item["source_name"]
            if item["type"] == "PQ2_0":
                assert source_records[name]["source_sha256"] == manifest["source_tensor_sha256"]
                assert converted_records[name]["projection_payload_sha256"] == source_records[name]["frozen_payload_sha256"] == manifest["payload_sha256"]
            else:
                assert source_records[name]["source_sha256"] == converted_records[name]["source_sha256"] == source_digests[name]
        assert receipt["converted"]["source_file_verification"] == receipt["model_metadata_verification"] == "not_checked"
        assert receipt["source_file_rechecked_after_conversion"] is True and receipt["cross_file_atomic_snapshot"] == "not_guaranteed"
    elif not frozen:
        assert receipt["source_conversion_verification"] == "pinned_source_range_and_converted_range_match_same_operand"
        assert receipt["source"]["source_sha256"] == receipt["converted"]["source_sha256"] == arguments["expected_source_sha256"]
        assert receipt["source"]["source_file_verification"] == "profile_pinned_shard_and_header_range_match_caller_operand"
        assert receipt["converted"]["source_file_verification"] == "not_checked"
        assert receipt["model_metadata_verification"] == receipt["pin_issuer_verification"] == "not_checked"
        assert receipt["source_file_rechecked_after_conversion"] is True and receipt["cross_file_atomic_snapshot"] == "not_guaranteed"
    tensor = next(item for item in reader.tensors if item.name == entry["name"])
    if nonzero:
        original_digest = arguments["expected_file_sha256"]
        with destination.open("r+b") as output:
            output.seek(tensor.data_offset)
            output.write(wrong_operand.tobytes())
        arguments["expected_file_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
        try:
            verify()
            raise AssertionError("no-op or unpermuted operand accepted with fresh converted pin")
        except ValueError as error:
            assert "byte parity mismatch" in str(error), str(error)
        finally:
            with destination.open("r+b") as output:
                output.seek(tensor.data_offset)
                output.write(expected_operand.tobytes())
            assert hashlib.sha256(destination.read_bytes()).hexdigest() == original_digest
            arguments["expected_file_sha256"] = original_digest
    with destination.open("r+b") as output:
        output.seek(tensor.data_offset + (2 if frozen else 0))
        saved = output.read(1)
        output.seek(tensor.data_offset + (2 if frozen else 0))
        output.write(bytes([saved[0] ^ 1]))
    arguments["expected_file_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    try:
        verify()
        raise AssertionError("changed converted bytes accepted with fresh file pin")
    except ValueError as error:
        assert "byte parity mismatch" in str(error), str(error)
    finally:
        with destination.open("r+b") as output:
            output.seek(tensor.data_offset + (2 if frozen else 0))
            output.write(saved)
        del reader
print(json.dumps({"kind": sys.argv[1], "selected_ranges": 28 if complete else 1, "runtime_approval": "not_granted_by_source_conversion_verification"}))
"""
    result = subprocess.run(
        [interpreter, "-c", code, kind, str(tmp_path), runtime_build or ""], capture_output=True, text=True, timeout=30,
        input=_profile_parity_fixture(profile, inventory, manifest, blocks),
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1",
             "PYTHONPATH": os.pathsep.join((str(Path.cwd()), str(Path(source_dir) / "gguf-py")))},
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"kind": kind, "selected_ranges": 28 if kind.startswith("complete") else 1, "runtime_approval": "not_granted_by_source_conversion_verification"}
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("target", ["down", "up", "gate", "q", "k", "v", "o", "full-only", "large-state"])
def test_pinned_prism_profile_manifest_native_prefill_without_generation(tmp_path, target):
    import hashlib
    import sys
    import numpy as np

    from embedded_jev.prism_codec import profile_model_specification
    from test_inventory import _prefill_model_policy_fixture

    source_dir = os.environ.get("PRISM_SOURCE_DIR")
    converter = os.environ.get("PRISM_CONVERTER_PYTHON")
    runtime_build = os.environ.get("MIMO_PRISM_RUNTIME_BUILD")
    if not all((source_dir, converter, runtime_build)):
        pytest.skip("requires pinned GGUF environment and bounded prefill test library; synthetic inputs only")
    assert subprocess.check_output(["git", "-C", source_dir, "rev-parse", "HEAD"], text=True).strip() == PRISM_REVISION
    assert (Path(runtime_build) / "bin" / "libprism_bitnet_prefill_test.so").is_file()
    profile, inventory, manifest, blocks = _prefill_model_policy_fixture(
        target if target in ("down", "up", "gate", "q", "k", "v", "o") else "down", hybrid=target != "full-only", state_size=256 if target == "large-state" else 32,
    )
    specification = profile_model_specification(profile, inventory, manifest, blocks)
    payload_file = tmp_path / "projection.bin"
    model_file = tmp_path / "profile-prefill.gguf"
    operands_file = tmp_path / "source-operands.npz"
    parity_interpreter = os.environ.get("PRISM_PARITY_PYTHON")
    payload_file.write_bytes(blocks.tobytes())
    writer_code = """
import json
import sys
from pathlib import Path
import gguf
import numpy as np
from embedded_jev.prism_codec import unpack_ternary_pq2_0
spec = json.loads(sys.argv[1])
rows, columns = spec["projection"]["rows"], spec["projection"]["columns"]
hidden, intermediate = spec["hidden"], spec["intermediate"]
metadata = {item["name"]: item["values"] for item in spec["metadata"]}
heads, kv_heads = metadata["qwen35.attention.head_count"][0], metadata["qwen35.attention.head_count_kv"][0]
head_dim = metadata["qwen35.attention.key_length"][0]
payload = np.frombuffer(Path(sys.argv[3]).read_bytes(), dtype=np.uint8).copy().reshape(rows, columns // 128, 34)
codes, scales = unpack_ternary_pq2_0(payload)
writer = gguf.GGUFWriter(sys.argv[2], "qwen35")
for key, value in {
    "jev.model.profile_id": spec["profile_id"], "jev.model.profile_sha256": spec["profile_sha256"],
    "jev.model.source_revision": spec["source_revision"], "jev.bitnet.source_tensor_sha256": spec["source_tensor_sha256"],
    "jev.model.converter_revision": "842b1880415d6f508f03b789e5ce70194def7bfd",
    "jev.bitnet.model": "profile-qwen35-single-projection-v2",
    "jev.bitnet.execution": "group128-a8-fp32-nearest-even-identity-v1",
}.items():
    writer.add_string(key, value)
for key, value in {"block_count": spec["layers"], "embedding_length": spec["hidden"],
                   "feed_forward_length": spec["intermediate"], "context_length": 256}.items():
    writer.add_uint32("qwen35." + key, value)
for item in spec["metadata"]:
    if item["kind"] == "u32":
        writer.add_uint32(item["name"], item["values"][0])
    elif item["kind"] == "f32":
        writer.add_float32(item["name"], item["values"][0])
    else:
        writer.add_array(item["name"], item["values"])
writer.add_tokenizer_model("gpt2")
writer.add_tokenizer_pre("qwen2")
vocabulary = ["token_" + str(index) for index in range(32)]
vocabulary[2:5] = ["a", "b", "ab"]
for label, token in spec["label_token_ids"].items():
    vocabulary[token] = label
writer.add_token_list(vocabulary)
writer.add_token_merges(["a b"])
writer.add_bos_token_id(0)
writer.add_eos_token_id(1)
writer.add_add_bos_token(False)
generator = np.random.default_rng(1407)
arrays = {
    "token_embd.weight": generator.normal(0, 0.2, (32, hidden)).astype(np.float32),
    "output.weight": generator.normal(0, 0.1, (32, hidden)).astype(np.float32),
    "blk.0.ffn_gate.weight": np.zeros((intermediate, hidden), dtype=np.float32),
    "blk.0.ffn_up.weight": np.zeros((intermediate, hidden), dtype=np.float32),
    "blk.0.ffn_down.weight": np.eye(hidden, intermediate, dtype=np.float32) * np.float32(0.125),
}
if spec["projection"]["tensor_name"] == "blk.1.attn_output.weight":
    arrays["token_embd.weight"][:, 0] = 0.125
arrays["blk.0.ffn_gate.weight"][:, 0] = 0.25
arrays["blk.0.ffn_up.weight"][np.arange(intermediate), (np.arange(intermediate) * 7) % hidden] = np.repeat([0.125, 0.375], intermediate // 2)
attention = spec["projection"]["tensor_name"].startswith("blk.1.attn_")
if attention:
    query = np.zeros((heads * head_dim * 2, hidden), dtype=np.float32)
    query[np.arange(heads) * head_dim * 2, np.arange(heads)] = 0.25
    key = np.zeros((kv_heads * head_dim, hidden), dtype=np.float32)
    key[np.arange(kv_heads) * head_dim, np.arange(kv_heads) + 4] = 0.25
    value = np.zeros((kv_heads * head_dim, hidden), dtype=np.float32)
    value[np.arange(kv_heads * head_dim), (np.arange(kv_heads * head_dim) + 3) % hidden] = 0.375
    output = np.zeros((hidden, heads * head_dim), dtype=np.float32)
    diagonal = np.arange(min(hidden, heads * head_dim))
    output[diagonal, diagonal] = 0.125
    arrays.update({"blk.1.attn_q.weight": query, "blk.1.attn_k.weight": key,
                   "blk.1.attn_v.weight": value, "blk.1.attn_output.weight": output})
source_operands = {}
for item in spec["tensors"]:
    name = item["name"]
    shape = list(reversed(item["ne"][:1 if item["ne"][1] == 1 else 2]))
    if item["type"] == "PQ2_0":
        writer.add_tensor(name, payload.reshape(rows, columns // 128 * 34), raw_dtype=gguf.GGMLQuantizationType.PQ2_0)
        continue
    values = arrays.get(name)
    if values is None:
        values = np.ones(shape, dtype=np.float32) if name.endswith("norm.weight") else np.zeros(shape, dtype=np.float32)
        if name.endswith(".ssm_a"):
            values = -np.ones(shape, dtype=np.float32)
    if item["type"] == "BF16":
        encoded = gguf.quants.quantize(values, gguf.GGMLQuantizationType.BF16)
        arrays[name] = gguf.quants.dequantize(encoded, gguf.GGMLQuantizationType.BF16)
        writer.add_tensor(name, encoded, raw_dtype=gguf.GGMLQuantizationType.BF16)
        if sys.argv[4]:
            source_operands[name] = encoded.view("<u2").copy()
    else:
        writer.add_tensor(name, values)
        if sys.argv[4]:
            source_values = values if name.endswith(".ssm_norm.weight") else np.zeros_like(values)
            if name.endswith(".ssm_conv1d.weight"):
                source_values = source_values[:, None, :]
            source_operands[name] = (source_values.view(np.uint32) >> 16).astype("<u2")
source_fixture = {}
if len(sys.argv) > 5 and sys.argv[5]:
    import hashlib
    from embedded_jev.model_profile import parse_model_profile, profile_source
    from embedded_jev.prism_codec import profile_model_conversion_plan
    source_fixture = json.loads(sys.stdin.read())
    profile = parse_model_profile(json.dumps(source_fixture["profile"]).encode())
    manifest = source_fixture["manifest"]
    plan = profile_model_conversion_plan(profile, source_fixture["inventory"], manifest, payload)
    source_arrays = {item["source_name"]: source_operands[item["name"]] for item in plan["tensors"] if item["type"] != "PQ2_0"}
    source_arrays[manifest["tensor"]] = np.zeros(manifest["shape"], dtype="<u2")
    headers = {name: {} for name in profile.packaging["shards"]}
    chunks = {name: [] for name in headers}
    offsets = dict.fromkeys(headers, 0)
    for item in source_fixture["inventory"]["tensors"]:
        shard = item["shard"]
        values = source_arrays[item["name"]]
        assert values.shape == tuple(item["shape"]) and values.dtype == np.dtype("<u2")
        headers[shard][item["name"]] = {"dtype": "BF16", "shape": item["shape"], "data_offsets": [offsets[shard], offsets[shard] + values.nbytes]}
        chunks[shard].append(values.tobytes())
        offsets[shard] += values.nbytes
    total_source_bytes = 0
    for shard, header in headers.items():
        encoded_header = json.dumps(header, sort_keys=True).encode()
        prefix = len(encoded_header).to_bytes(8, "little") + encoded_header
        source_path = Path(sys.argv[2]).parent / shard
        source_bytes = prefix + b"".join(chunks[shard])
        if sys.argv[5] == "pinned-source-range-mismatch":
            embedding_name = next(item["source_name"] for item in plan["tensors"] if item["name"] == "token_embd.weight")
            if embedding_name in header:
                changed_offset = len(prefix) + header[embedding_name]["data_offsets"][0]
                source_bytes = bytearray(source_bytes)
                source_bytes[changed_offset] ^= 1
                source_bytes = bytes(source_bytes)
        total_source_bytes += len(source_bytes)
        assert total_source_bytes <= 16 * 1024**2
        source_path.write_bytes(source_bytes)
        source_fixture["profile"]["files"][shard].update(bytes=len(source_bytes), sha256=hashlib.sha256(source_bytes).hexdigest(),
            header_bytes=len(prefix), header_sha256=hashlib.sha256(prefix).hexdigest())
    profile = parse_model_profile(json.dumps(source_fixture["profile"]).encode())
    source_fixture["inventory"]["source"].update(profile_source(profile))
    manifest["profile_sha256"] = profile.sha256
    manifest["source_tensor_sha256"] = hashlib.sha256(memoryview(source_arrays[manifest["tensor"]])).hexdigest()
    writer.add_string("jev.model.profile_sha256", profile.sha256)
    writer.add_string("jev.bitnet.source_tensor_sha256", manifest["source_tensor_sha256"])
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
def bitnet_projection(inputs):
    groups = inputs.reshape(columns // 128, 128)
    activation_scales = np.max(np.abs(groups), axis=1) / np.float32(127)
    activation_scales[activation_scales == 0] = 1
    quantized = np.clip(np.rint(groups / activation_scales[:, None]), -127, 127).astype(np.int32)
    partials = np.sum(codes.astype(np.int32) * quantized[None], axis=-1)
    return np.sum(partials.astype(np.float32) * scales.astype(np.float32) * activation_scales, axis=1)
def dense_projection(name, inputs):
    encoded = gguf.quants.quantize(inputs, gguf.GGMLQuantizationType.BF16)
    rounded = gguf.quants.dequantize(encoded, gguf.GGMLQuantizationType.BF16)
    return arrays[name] @ rounded
def pre_attention(token):
    hidden = arrays["token_embd.weight"][token].copy()
    normalized = hidden / np.sqrt(np.mean(hidden * hidden) + np.float32(1e-6))
    gate = bitnet_projection(normalized) if spec["projection"]["tensor_name"] == "blk.0.ffn_gate.weight" else dense_projection("blk.0.ffn_gate.weight", normalized)
    up = bitnet_projection(normalized) if spec["projection"]["tensor_name"] == "blk.0.ffn_up.weight" else dense_projection("blk.0.ffn_up.weight", normalized)
    activation = (gate / (np.float32(1) + np.exp(-gate))) * up
    projected = bitnet_projection(activation) if spec["projection"]["tensor_name"] == "blk.0.ffn_down.weight" else dense_projection("blk.0.ffn_down.weight", activation)
    assert np.linalg.norm(projected) > 0.01
    hidden += projected
    return hidden
def reference(tokens):
    sequence = np.stack([pre_attention(token) for token in tokens])
    if attention:
        normalized = sequence / np.sqrt(np.mean(sequence * sequence, axis=1, keepdims=True) + np.float32(1e-6))
        def project(name):
            return np.stack([bitnet_projection(current) if name == spec["projection"]["tensor_name"] else dense_projection(name, current)
                             for current in normalized])
        query = project("blk.1.attn_q.weight")[:, np.arange(heads) * head_dim * 2]
        query /= np.sqrt(query * query / np.float32(head_dim) + np.float32(1e-6))
        key = project("blk.1.attn_k.weight")[:, np.arange(kv_heads) * head_dim]
        key /= np.sqrt(key * key / np.float32(head_dim) + np.float32(1e-6))
        positions = np.arange(len(tokens), dtype=np.float32)
        key_cosine = (key * np.cos(positions[:, None])).astype(np.float16).astype(np.float32)
        key_sine = (key * np.sin(positions[:, None])).astype(np.float16).astype(np.float32)
        value = project("blk.1.attn_v.weight").astype(np.float16).astype(np.float32).reshape(len(tokens), kv_heads, head_dim)
        original = sequence.copy()
        for index in range(len(tokens)):
            output = np.zeros((heads, head_dim), dtype=np.float32)
            for head in range(heads):
                key_head = head // (heads // kv_heads)
                cosine = np.float32(np.float16(query[index, head] * np.cos(positions[index])))
                sine = np.float32(np.float16(query[index, head] * np.sin(positions[index])))
                scores = (cosine * key_cosine[:index + 1, key_head] + sine * key_sine[:index + 1, key_head]) / np.sqrt(np.float32(head_dim))
                scores = np.exp(scores - np.max(scores))
                scores /= np.sum(scores)
                scores = scores.astype(np.float16).astype(np.float32)
                output[head] = np.sum(value[:index + 1, key_head] * scores[:, None], axis=0)
            gated = output.reshape(heads * head_dim) * np.float32(0.5)
            sequence[index] += bitnet_projection(gated) if spec["projection"]["tensor_name"] == "blk.1.attn_output.weight" else dense_projection("blk.1.attn_output.weight", gated)
        assert np.linalg.norm(sequence - original) > 0.01
    hidden = sequence[-1]
    hidden /= np.sqrt(np.mean(hidden * hidden) + np.float32(1e-6))
    return dense_projection("output.weight", hidden)[[11, 17, 23]].tolist()
import hashlib
model_bytes = Path(sys.argv[2]).read_bytes()
archive_identity = {}
if sys.argv[4]:
    assert sum(value.nbytes for value in source_operands.values()) <= 16 * 1024**2
    np.savez(sys.argv[4], **source_operands)
    archive_bytes = Path(sys.argv[4]).read_bytes()
    assert len(archive_bytes) <= 16 * 1024**2
    archive_identity = {"source_archive_bytes": len(archive_bytes), "source_archive_sha256": hashlib.sha256(archive_bytes).hexdigest()}
print(json.dumps({"logits": reference([3, 5, 7]), "peer_logits": reference([4, 9, 19, 3, 5, 7]), "maximum_logits": reference([2] * 128),
                  "model_bytes": len(model_bytes), "model_sha256": hashlib.sha256(model_bytes).hexdigest(), **archive_identity, "source_fixture": source_fixture}))
"""
    driver_code = """
import ctypes
import gc
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from embedded_jev.prism_codec import create_profile_model_policy, native_model_specification, NativeModelSpecification, verified_profile_model_policy, profile_model_conversion_plan
if len(sys.argv) > 8 and sys.argv[8]:
    import numpy as np
    from embedded_jev.model_profile import parse_model_profile
    fixture = json.loads(sys.stdin.read())
    profile = parse_model_profile(json.dumps(fixture["profile"]).encode())
    inventory, manifest = fixture["inventory"], fixture["manifest"]
    blocks = np.frombuffer(bytes.fromhex(fixture["payload_hex"]), dtype=np.uint8).reshape(manifest["shape"][0], manifest["shape"][1] // 128, 34)
else:
    from test_inventory import _prefill_model_policy_fixture
    profile, inventory, manifest, blocks = _prefill_model_policy_fixture(
        sys.argv[2] if sys.argv[2] in ("down", "up", "gate", "q", "k", "v", "o") else "down", hybrid=sys.argv[2] != "full-only", state_size=256 if sys.argv[2] == "large-state" else 32)
directory = Path(sys.argv[3]) / "bin"
native = ctypes.CDLL(str(directory / "libprism_group_scale.so"))
control = ctypes.CDLL(str(directory / "libprism_bitnet_prefill_test.so"))
initialize = native.prism_bitnet_cpu_runtime_init_v1
initialize.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
buffer_type = ctypes.c_void_p()
assert initialize(b"842b1880415d6f508f03b789e5ce70194def7bfd", 1, ctypes.byref(buffer_type)) == 0
registry = native.prism_bitnet_registered_tensor_registry_size
registry.restype = ctypes.c_size_t
baseline = registry()
free = native.prism_bitnet_cpu_model_policy_free_v2
free.argtypes = [ctypes.c_void_p]
free.restype = None
function = control.prism_bitnet_test_prefill_policy_v2
function.argtypes = [ctypes.c_char_p, ctypes.POINTER(NativeModelSpecification), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
function.restype = ctypes.c_int
status = 27
owners = ExitStack()
buffers = ExitStack()
native_output = None
body_succeeded = False
source_exit_refusal = False
try:
    mode = sys.argv[4]
    model_path = sys.argv[1]
    if mode.startswith("verified-"):
        tensor_arguments = {}
        caller_refusal = mode in ("verified-table-source-key", "verified-table-source-parity", "verified-table-budget", "verified-policy-metadata-type", "verified-policy-metadata-copy", "verified-source-file", "verified-source-range")
        if len(sys.argv) > 8 and sys.argv[8]:
            import hashlib
            import time
            import zipfile
            import gguf
            import numpy as np
            from embedded_jev.streamed_text import bounded_native_file_record
            archive_record = bounded_native_file_record(Path(sys.argv[8]), maximum_bytes=int(sys.argv[9]), deadline=time.monotonic() + 30)
            assert archive_record["sha256"] == sys.argv[10] and archive_record["bytes"] == int(sys.argv[9]) <= 16 * 1024**2
            with zipfile.ZipFile(sys.argv[8]) as packed:
                entries = packed.infolist()
                assert all(item.compress_type == zipfile.ZIP_STORED for item in entries)
                assert sum(item.file_size for item in entries) <= 16 * 1024**2
                assert len(entries) == len({item.filename for item in entries})
            plan = profile_model_conversion_plan(profile, inventory, manifest, blocks)
            with np.load(sys.argv[8], allow_pickle=False) as archive:
                assert set(archive.files) == {item["name"] for item in plan["tensors"] if item["type"] != "PQ2_0"}
                sources = {item["source_name"]: archive[item["name"]] for item in plan["tensors"] if item["type"] != "PQ2_0"}
            pins = {name: hashlib.sha256(memoryview(value)).hexdigest() for name, value in sources.items()}
            if mode == "verified-table-source-key":
                sources.pop(next(iter(sources)))
            elif mode == "verified-table-source-parity":
                embedding = next(item for item in plan["tensors"] if item["name"] == "token_embd.weight")["source_name"]
                sources[embedding].flat[0] ^= 1
                pins[embedding] = hashlib.sha256(memoryview(sources[embedding])).hexdigest()
            elif mode == "verified-table-budget":
                tensor_arguments["maximum_tensor_bytes"] = 1
            tensor_arguments.update(tensor_reader=gguf.GGUFReader(model_path), source_values=sources, expected_source_sha256=pins)
            tensor_arguments["verify_policy_metadata"] = True
            if mode.startswith("verified-source-"):
                tensor_arguments.update(source_paths={name: Path(model_path).parent / name for name in profile.packaging["shards"]},
                    frozen_source_values=np.zeros(manifest["shape"], dtype="<u2"), verify_source_files_on_exit=True)
            if mode in ("verified-policy-metadata-type", "verified-policy-metadata-copy"):
                selected_name = "qwen35.attention.head_count"
                original_field = tensor_arguments["tensor_reader"].fields[selected_name]
                if mode == "verified-policy-metadata-type":
                    changed_field = original_field._replace(types=[gguf.GGUFValueType.FLOAT32])
                else:
                    parts = original_field.parts.copy()
                    parts[-1] = parts[-1].copy()
                    parts[-1].setflags(write=False)
                    changed_field = original_field._replace(parts=parts)
                tensor_arguments["tensor_reader"].fields[selected_name] = changed_field
        assert not caller_refusal or tensor_arguments
        try:
            owner = owners.enter_context(verified_profile_model_policy(
                native, model_path, profile, inventory, manifest, blocks,
                expected_bytes=int(sys.argv[5]), expected_sha256=sys.argv[6],
                expected_runtime_files=json.loads(sys.argv[7]),
                runtime_scorer=control, **tensor_arguments,
            ))
        except ValueError as error:
            if not caller_refusal:
                raise
            expected_error = {
                "verified-table-source-key": "complete tensor-table source set mismatch",
                "verified-table-source-parity": "converted matrix byte parity mismatch",
                "verified-table-budget": "complete tensor-table operands exceed aggregate byte budget",
                "verified-policy-metadata-type": "policy metadata field type or range mismatch",
                "verified-policy-metadata-copy": "policy metadata field mapping mismatch",
                "verified-source-file": "source operand file identity mismatch",
                "verified-source-range": "source operand file range byte mismatch",
            }[mode]
            assert expected_error in str(error), str(error)
            assert registry() == baseline
            sys.exit(33)
        assert not caller_refusal
        handle, overrides, record = owner.handle, owner.overrides, owner.record
        model_path = owner.model_path
        specification = None
        assert owner.verification["model_sha256"] == sys.argv[6]
        assert owner.verification["runtime_approval"] == "not_granted_by_file_verification"
        assert owner.verification["runtime_file_verification"] == "sha256_matches_caller_supplied_runtime_files"
        assert owner.verification["runtime_file_scope"] == "policy_cpu_base_scorer_llama_registry"
        if tensor_arguments:
            table = owner.verification["tensor_table"]
            assert table["tensor_count"] == 28 and table["model_file_access"] == "verified_owner_descriptor"
            assert table["converted_file_sha256"] == owner.verification["model_sha256"]
            assert owner.verification["tensor_table_verification"] == "all_declared_ranges_match_caller_operands_and_frozen_payload"
            assert table["source_file_verification"] == table["model_metadata_verification"] == table["gguf_reader_verification"] == "not_checked"
            assert table["runtime_approval"] == "not_granted_by_tensor_table_verification"
            metadata = owner.verification["policy_metadata"]
            assert metadata["model_file_access"] == "verified_owner_descriptor" and len(metadata["metadata_names"]) == 25
            assert metadata["converted_file_sha256"] == owner.verification["model_sha256"]
            assert owner.verification["policy_metadata_verification"] == "selected_typed_policy_records_match_declarations_and_file"
            assert metadata["model_metadata_verification"] == "selected_policy_records_only"
            assert metadata["tokenizer_semantics_verification"] == metadata["gguf_reader_verification"] == "not_checked"
            if mode.startswith("verified-source-"):
                source_table = owner.verification["source_conversion_table"]
                assert source_table["tensor_count"] == 28 and source_table["converted"]["model_file_access"] == "verified_owner_descriptor"
                assert source_table["source"]["source_shard_count"] == len(profile.packaging["shards"]) == 2
                assert source_table["source"]["source_file_bytes"] == sum(profile.files[name].bytes for name in profile.packaging["shards"]) <= 16 * 1024**2
                assert source_table["source_file_rechecked_after_conversion"] is True
                assert source_table["cross_file_atomic_snapshot"] == "not_guaranteed"
                assert owner.verification["source_file_lifetime_verification"] == "entry_and_exit_checks_required"
                import tempfile
                native_output = buffers.enter_context(tempfile.TemporaryFile(dir=Path(sys.argv[1]).parent))
        for name in ("bridge", "scorer", "llama", "ggml_registry"):
            assert owner.verification[name + "_sha256"] == json.loads(sys.argv[7])[name]["sha256"]
        assert owner.verification["ggml_cpu_sha256"] == "adaaacaf406df3700fb5f05cb5749990e9b17d410d91e13d0c58263ae971a150"
        assert owner.verification["ggml_base_sha256"] == "57a9b6060d3bc902c30fd97fb271ae7f027054d4a9f8b50e27dd6fe91951847d"
        mode = mode.removeprefix("verified-")
        mode = mode.removeprefix("source-")
    else:
        handle, overrides, specification = create_profile_model_policy(native, model_path, profile, inventory, manifest, blocks)
        owners.callback(free, handle)
        record = native_model_specification(specification, blocks)
    other_handle = None
    assert registry() == baseline + 1
    if mode == "null-policy":
        handle_for_call = None
        mode = "full"
    else:
        handle_for_call = handle
    overrides_for_call = overrides
    record_for_call = ctypes.byref(record)
    if mode == "wrong-policy":
        other_handle, overrides_for_call, _ = create_profile_model_policy(native, sys.argv[1], profile, inventory, manifest, blocks)
        owners.callback(free, other_handle)
        mode = "full"
    if mode == "null-overrides":
        overrides_for_call = None
        mode = "full"
    if mode == "null-specification":
        record_for_call = None
        mode = "full"
    if mode == "shape-mismatch":
        record.projection.columns = 512
        mode = "full"
    if mode == "hidden-mismatch":
        record.hidden = 512
        mode = "full"
    if mode == "vocabulary-mismatch":
        record.vocabulary = 24
        mode = "full"
    if mode == "intermediate-mismatch":
        record.intermediate = 512
        mode = "full"
    if mode == "oversized-file":
        with open(sys.argv[1], "r+b") as oversized:
            oversized.truncate(16 * 1024 * 1024 + 1)
        mode = "full"
    if mode == "oversized-geometry":
        record.hidden = 1025
        mode = "full"
    source_exit_refusal = mode in ("active-change", "active-restored", "active-replace", "active-last-change", "active-last-restored", "active-last-replace")
    if source_exit_refusal:
        source_effect = mode.rsplit("-", 1)[-1]
        source_path = list(tensor_arguments["source_paths"].values())[-1 if mode.startswith("active-last-") else 0]
        source_offset = profile.files[source_path.name].header_bytes
        source_stat = source_path.stat()
        mode = "full"
    del profile, inventory, manifest, blocks, specification
    gc.collect()
    if native_output is None:
        status = function(model_path.encode(), record_for_call, handle_for_call, overrides_for_call, mode.encode())
    else:
        import os
        ctypes.CDLL(None).fflush(None)
        saved_stdout = os.dup(1)
        try:
            os.dup2(native_output.fileno(), 1)
            status = function(model_path.encode(), record_for_call, handle_for_call, overrides_for_call, mode.encode())
        finally:
            ctypes.CDLL(None).fflush(None)
            os.dup2(saved_stdout, 1)
            os.close(saved_stdout)
        assert status == 0 and 0 < native_output.tell() <= 65536
    if source_exit_refusal:
        if source_effect == "replace":
            import shutil
            replacement = source_path.with_suffix(".replacement")
            with source_path.open("rb") as source, replacement.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=65536)
            os.replace(replacement, source_path)
        else:
            with source_path.open("r+b") as source:
                source.seek(source_offset)
                original_byte = source.read(1)
                source.seek(source_offset)
                source.write(bytes([original_byte[0] ^ 1]))
                if source_effect == "restored":
                    source.seek(source_offset)
                    source.write(original_byte)
            if source_effect == "restored":
                os.utime(source_path, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns))
    assert registry() == baseline + 1 + (other_handle is not None)
    body_succeeded = True
finally:
    if "tensor_arguments" in locals():
        tensor_arguments.clear()
    try:
        try:
            owners.close()
        except ValueError as error:
            if not source_exit_refusal:
                raise
            assert "source table shard identity changed after operands" in str(error), str(error)
            assert registry() == baseline and owner.handle.value is owner.overrides.value is None
            sys.exit(33)
        assert registry() == baseline
        if native_output is not None and body_succeeded:
            assert not source_exit_refusal
            assert owner.handle.value is owner.overrides.value is None
            native_output.seek(0)
            output_bytes = native_output.read(65537)
            assert 0 < len(output_bytes) <= 65536
            sys.stdout.buffer.write(output_bytes)
    finally:
        buffers.close()
ctypes.CDLL(None).fflush(None)
sys.exit(status)
"""
    try:
        writer = subprocess.run([converter, "-c", writer_code, json.dumps({**specification, "label_token_ids": dict(profile.tokenization["label_token_ids"])}), str(model_file), str(payload_file), str(operands_file) if parity_interpreter else ""],
                                capture_output=True, text=True, timeout=30,
                                env={**os.environ, "PYTHONPATH": os.pathsep.join((str(Path(source_dir) / "gguf-py"), str(Path.cwd()))),
                                     "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1"})
        assert writer.returncode == 0, writer.stderr
        expected = json.loads(writer.stdout)
        environment = {**os.environ, "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), str(Path.cwd()))),
                       "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1"}
        if target in ("full-only", "large-state"):
            control = subprocess.run([sys.executable, "-c", driver_code, str(model_file), target, runtime_build, "full"],
                                     capture_output=True, text=True, timeout=30, env=environment)
            assert control.returncode == 27 and control.stdout == "", control.stderr
            return
        vector_interpreter = os.environ.get("PRISM_VECTOR_PYTHON")
        if parity_interpreter or vector_interpreter:
            parity_code = """
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path
import gguf
import numpy as np
from embedded_jev.model_profile import parse_model_profile
from embedded_jev.prism_codec import profile_model_conversion_plan, verify_profile_gguf_vector, verify_profile_gguf_matrix, verify_profile_gguf_convolution
from embedded_jev.streamed_text import bounded_native_file_record

fixture = json.loads(sys.stdin.read())
profile = parse_model_profile(json.dumps(fixture["profile"]).encode())
payload = np.frombuffer(bytes.fromhex(fixture["payload_hex"]), dtype=np.uint8).reshape(
    fixture["manifest"]["shape"][0], fixture["manifest"]["shape"][1] // 128, 34)
plan = profile_model_conversion_plan(profile, fixture["inventory"], fixture["manifest"], payload)
reader = gguf.GGUFReader(sys.argv[1])
complete = bool(sys.argv[4])
archive = None
if complete:
    assert 0 < int(sys.argv[5]) <= 16 * 1024**2
    record = bounded_native_file_record(Path(sys.argv[4]), maximum_bytes=int(sys.argv[5]), deadline=time.monotonic() + 30)
    assert record["bytes"] == int(sys.argv[5]) <= 16 * 1024**2 and record["sha256"] == sys.argv[6]
    with zipfile.ZipFile(sys.argv[4]) as packed:
        assert all(item.compress_type == zipfile.ZIP_STORED for item in packed.infolist())
        assert sum(item.file_size for item in packed.infolist()) <= 16 * 1024**2
    archive = np.load(sys.argv[4], allow_pickle=False)
    assert set(archive.files) == {item["name"] for item in plan["tensors"] if item["type"] != "PQ2_0"}
    assert {item.name for item in reader.tensors} == {item["name"] for item in plan["tensors"]}
    assert len(reader.tensors) == len(plan["tensors"]) == 28
    from embedded_jev.prism_codec import verify_profile_gguf_tensor_table, verify_profile_gguf_policy_metadata
    sources = {item["source_name"]: archive[item["name"]] for item in plan["tensors"] if item["type"] != "PQ2_0"}
    pins = {name: hashlib.sha256(memoryview(value)).hexdigest() for name, value in sources.items()}
    table_receipt = verify_profile_gguf_tensor_table(
        profile, fixture["inventory"], fixture["manifest"], payload, reader=reader,
        source_values=sources, expected_source_sha256=pins,
        expected_file_bytes=int(sys.argv[2]), expected_file_sha256=sys.argv[3])
    assert table_receipt["tensor_count"] == 28 and table_receipt["operand_budget_bytes"] <= 64 * 1024**2
    assert table_receipt["source_file_verification"] == table_receipt["model_metadata_verification"] == table_receipt["gguf_reader_verification"] == "not_checked"
    assert table_receipt["runtime_approval"] == "not_granted_by_tensor_table_verification"
    metadata_arguments = {
        "reader": reader, "expected_file_bytes": int(sys.argv[2]), "expected_file_sha256": sys.argv[3],
    }
    metadata_receipt = verify_profile_gguf_policy_metadata(
        profile, fixture["inventory"], fixture["manifest"], payload, **metadata_arguments,
    )
    assert len(metadata_receipt["metadata_names"]) == 25 and metadata_receipt["metadata_record_bytes"] <= 8192
    assert metadata_receipt["policy_metadata_verification"] == "selected_typed_policy_records_match_declarations_and_file"
    assert metadata_receipt["model_metadata_verification"] == "selected_policy_records_only"
    assert metadata_receipt["tokenizer_semantics_verification"] == metadata_receipt["gguf_reader_verification"] == "not_checked"
    assert metadata_receipt["runtime_approval"] == "not_granted_by_policy_metadata_verification"
    try:
        metadata_receipt["metadata_names"] = ()
        raise AssertionError("mutable metadata receipt")
    except TypeError:
        pass
    from embedded_jev import prism_codec as codec
    selected_name = "qwen35.attention.head_count"
    original_field = reader.fields[selected_name]
    value_offset = original_field.parts[-1].__array_interface__["data"][0] - reader.data.__array_interface__["data"][0]
    for defect, expected_error in (
        ("missing", "field type or range mismatch"), ("type", "field type or range mismatch"),
        ("copy", "field mapping mismatch"), ("offset", "field mapping mismatch"),
        ("value", "declaration byte mismatch"), ("header", "header differs from reader"),
        ("pin", "file identity mismatch"), ("deadline", "deadline exceeded"),
        ("mid-read", "file range differs from declaration"),
    ):
        changed = metadata_arguments.copy()
        original_fields = reader.fields
        original_clock = time.monotonic
        original_pread = codec.os.pread
        saved = None
        changed_offset = 4 if defect == "header" else value_offset
        reader.fields = original_fields.copy()
        if defect == "missing":
            reader.fields.pop(selected_name)
        elif defect == "type":
            reader.fields[selected_name] = original_field._replace(types=[gguf.GGUFValueType.FLOAT32])
        elif defect == "copy":
            parts = original_field.parts.copy()
            parts[-1] = parts[-1].copy()
            parts[-1].setflags(write=False)
            reader.fields[selected_name] = original_field._replace(parts=parts)
        elif defect == "offset":
            reader.fields[selected_name] = original_field._replace(offset=original_field.offset + 1)
        elif defect == "pin":
            changed["expected_file_sha256"] = "0" * 64
        elif defect == "deadline":
            time.monotonic = iter((0, 31)).__next__
        if defect in ("value", "header", "mid-read"):
            with open(sys.argv[1], "rb") as output:
                output.seek(changed_offset)
                saved = output.read(1)
            def mutate():
                with open(sys.argv[1], "r+b") as output:
                    output.seek(changed_offset)
                    output.write(bytes([saved[0] ^ 1]))
            if defect == "mid-read":
                calls = []
                def read(descriptor, size, position):
                    result = original_pread(descriptor, size, position)
                    calls.append(1)
                    if len(calls) == 1:
                        mutate()
                    return result
                codec.os.pread = read
            else:
                mutate()
                changed["expected_file_sha256"] = bounded_native_file_record(
                    Path(sys.argv[1]), maximum_bytes=int(sys.argv[2]), deadline=time.monotonic() + 30,
                )["sha256"]
        try:
            verify_profile_gguf_policy_metadata(profile, fixture["inventory"], fixture["manifest"], payload, **changed)
            raise AssertionError("invalid policy metadata accepted: " + defect)
        except ValueError as error:
            assert expected_error in str(error), str(error)
        finally:
            reader.fields = original_fields
            time.monotonic = original_clock
            codec.os.pread = original_pread
            if saved is not None:
                with open(sys.argv[1], "r+b") as output:
                    output.seek(changed_offset)
                    output.write(saved)
    assert bounded_native_file_record(Path(sys.argv[1]), maximum_bytes=int(sys.argv[2]), deadline=time.monotonic() + 30)["sha256"] == sys.argv[3]
    unselected = reader.fields["qwen35.context_length"]
    type_offset = unselected.parts[2].__array_interface__["data"][0] - reader.data.__array_interface__["data"][0]
    with open(sys.argv[1], "r+b") as output:
        output.seek(type_offset)
        saved_type = output.read(4)
        output.seek(type_offset)
        output.write((11).to_bytes(4, "little"))
    try:
        changed_file = bounded_native_file_record(Path(sys.argv[1]), maximum_bytes=int(sys.argv[2]), deadline=time.monotonic() + 30)
        verify_profile_gguf_policy_metadata(
            profile, fixture["inventory"], fixture["manifest"], payload, reader=reader,
            expected_file_bytes=int(sys.argv[2]), expected_file_sha256=changed_file["sha256"],
        )
        raise AssertionError("changed unselected metadata type accepted by stale reader")
    except ValueError as error:
        assert "policy metadata file" in str(error), str(error)
    finally:
        with open(sys.argv[1], "r+b") as output:
            output.seek(type_offset)
            output.write(saved_type)
    table_entries = {item["tensor_name"]: item for item in table_receipt["tensors"]}
    assert len(table_entries) == 28
    try:
        table_receipt["tensor_count"] = 0
        raise AssertionError("mutable table receipt")
    except TypeError:
        pass
    for defect in ("source-key", "aggregate-budget", "duplicate", "deadline"):
        changed_sources = sources.copy()
        changed = {}
        original_tensors = reader.tensors
        original_clock = time.monotonic
        if defect == "source-key":
            changed_sources.pop(next(iter(changed_sources)))
        elif defect == "aggregate-budget":
            changed["maximum_bytes"] = 1
        elif defect == "duplicate":
            reader.tensors = [*reader.tensors, reader.tensors[0]]
        elif defect == "deadline":
            time.monotonic = iter((0, 31)).__next__
        try:
            verify_profile_gguf_tensor_table(
                profile, fixture["inventory"], fixture["manifest"], payload, reader=reader,
                source_values=changed_sources, expected_source_sha256=pins,
                expected_file_bytes=int(sys.argv[2]), expected_file_sha256=sys.argv[3], **changed)
            raise AssertionError("invalid complete-table request accepted: " + defect)
        except ValueError:
            pass
        finally:
            reader.tensors = original_tensors
            time.monotonic = original_clock
    projection = next(item for item in plan["tensors"] if item["type"] == "PQ2_0")
    tensor = next(item for item in reader.tensors if item.name == projection["name"])
    with open(sys.argv[1], "r+b") as output:
        output.seek(tensor.data_offset + 2)
        saved = output.read(1)
        output.seek(tensor.data_offset + 2)
        output.write(bytes([saved[0] ^ 1]))
    try:
        changed_file = bounded_native_file_record(Path(sys.argv[1]), maximum_bytes=int(sys.argv[2]), deadline=time.monotonic() + 30)
        verify_profile_gguf_tensor_table(
            profile, fixture["inventory"], fixture["manifest"], payload, reader=reader,
            source_values=sources, expected_source_sha256=pins,
            expected_file_bytes=int(sys.argv[2]), expected_file_sha256=changed_file["sha256"])
        raise AssertionError("changed frozen projection accepted by table gate")
    except ValueError as error:
        assert "frozen projection byte parity" in str(error)
    finally:
        with open(sys.argv[1], "r+b") as output:
            output.seek(tensor.data_offset + 2)
            output.write(saved)
count = 0
matrices = convolutions = frozen = 0
for item in plan["tensors"]:
    if item["type"] == "PQ2_0":
        if complete:
            tensor = next(value for value in reader.tensors if value.name == item["name"])
            assert tensor.tensor_type == gguf.GGMLQuantizationType.PQ2_0
            assert list(tensor.shape) == list(reversed(item["source_shape"]))
            assert tensor.n_bytes == payload.nbytes and tensor.n_elements == int(np.prod(item["source_shape"]))
            assert np.array_equal(tensor.data.reshape(payload.shape), payload)
            frozen += 1
        continue
    vector = item["type"] == "F32" and len(item["source_shape"]) == 1
    if not complete and not vector:
        continue
    if complete:
        source = archive[item["name"]]
    else:
        values = np.ones(item["source_shape"], dtype=np.float32) if item["source_name"].endswith("linear_attn.norm.weight") else np.zeros(item["source_shape"], dtype=np.float32)
        source = (values.view(np.uint32) >> 16).astype("<u2")
    verifier = verify_profile_gguf_vector if vector else verify_profile_gguf_matrix if item["type"] == "BF16" else verify_profile_gguf_convolution
    if complete:
        receipt = table_entries[item["name"]]
    else:
        receipt = verifier(
            profile, fixture["inventory"], fixture["manifest"], payload, reader=reader,
            source_name=item["source_name"], source_values=source,
            expected_source_sha256=hashlib.sha256(memoryview(source)).hexdigest(),
            expected_file_bytes=int(sys.argv[2]), expected_file_sha256=sys.argv[3], maximum_file_bytes=16 * 1024**2,
            **({} if vector else {"maximum_bytes": 16 * 1024**2}))
    assert receipt["converted_file_verification"] == "caller_sha256_and_reader_range_match"
    assert receipt["converted_file_sha256"] == sys.argv[3]
    assert receipt["source_file_verification"] == receipt["gguf_reader_verification"] == "not_checked"
    count += vector
    matrices += item["type"] == "BF16"
    convolutions += not vector and item["type"] == "F32"
assert count == 10
report = {"vectors": count, "file_sha256": sys.argv[3]}
if complete:
    assert (matrices, convolutions, frozen) == (16, 1, 1)
    embedding = next(item for item in plan["tensors"] if item["name"] == "token_embd.weight")
    tensor = next(item for item in reader.tensors if item.name == embedding["name"])
    source = archive[embedding["name"]]
    with open(sys.argv[1], "r+b") as output:
        output.seek(tensor.data_offset)
        saved = output.read(1)
        output.seek(tensor.data_offset)
        output.write(bytes([saved[0] ^ 1]))
    try:
        changed_file = bounded_native_file_record(Path(sys.argv[1]), maximum_bytes=int(sys.argv[2]), deadline=time.monotonic() + 30)
        verify_profile_gguf_matrix(
            profile, fixture["inventory"], fixture["manifest"], payload, reader=reader,
            source_name=embedding["source_name"], source_values=source,
            expected_source_sha256=hashlib.sha256(memoryview(source)).hexdigest(),
            expected_file_bytes=int(sys.argv[2]), expected_file_sha256=changed_file["sha256"], maximum_bytes=16 * 1024**2)
        raise AssertionError("changed complete-model matrix payload accepted")
    except ValueError as error:
        assert "byte parity" in str(error)
    finally:
        with open(sys.argv[1], "r+b") as output:
            output.seek(tensor.data_offset)
            output.write(saved)
    restored = bounded_native_file_record(Path(sys.argv[1]), maximum_bytes=int(sys.argv[2]), deadline=time.monotonic() + 30)
    assert restored["sha256"] == sys.argv[3]
    import ctypes
    import os
    from embedded_jev.prism_codec import verified_profile_model_policy
    native = ctypes.CDLL(str(Path(sys.argv[7]) / "bin" / "libprism_group_scale.so"))
    initialize = native.prism_bitnet_cpu_runtime_init_v1
    initialize.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
    buffer_type = ctypes.c_void_p()
    assert initialize(b"842b1880415d6f508f03b789e5ce70194def7bfd", 1, ctypes.byref(buffer_type)) == 0
    registry = native.prism_bitnet_registered_tensor_registry_size
    registry.restype = ctypes.c_size_t
    baseline = registry()
    original_path = Path(sys.argv[1])
    displaced_path = original_path.with_suffix(".displaced")
    try:
        with verified_profile_model_policy(
            native, original_path, profile, fixture["inventory"], fixture["manifest"], payload,
            expected_bytes=int(sys.argv[2]), expected_sha256=sys.argv[3],
        ) as owner:
            owner_reader = gguf.GGUFReader(owner.model_path)
            descriptor = int(owner.model_path.rsplit("/", 1)[1])
            position = os.lseek(descriptor, 0, os.SEEK_CUR)
            original_path.rename(displaced_path)
            original_path.write_bytes(b"replacement must never be opened")
            try:
                anchored = verify_profile_gguf_tensor_table(
                    profile, fixture["inventory"], fixture["manifest"], payload, reader=owner_reader,
                    source_values=sources, expected_source_sha256=pins,
                    expected_file_bytes=int(sys.argv[2]), expected_file_sha256=sys.argv[3], model_owner=owner,
                )
                assert anchored["model_file_access"] == "verified_owner_descriptor" and anchored["tensor_count"] == 28
                anchored_metadata = verify_profile_gguf_policy_metadata(
                    profile, fixture["inventory"], fixture["manifest"], payload, reader=owner_reader,
                    expected_file_bytes=int(sys.argv[2]), expected_file_sha256=sys.argv[3], model_owner=owner,
                )
                assert anchored_metadata["model_file_access"] == "verified_owner_descriptor" and len(anchored_metadata["metadata_names"]) == 25
                assert os.lseek(descriptor, 0, os.SEEK_CUR) == position
                for defect in ("pin", "profile", "payload"):
                    changed = {"expected_file_sha256": "0" * 64} if defect == "pin" else {}
                    changed_manifest = {**fixture["manifest"], "payload_sha256": "0" * 64} if defect == "payload" else fixture["manifest"]
                    changed_profile = parse_model_profile(json.dumps({**fixture["profile"], "profile_id": "wrong-owner"}).encode()) if defect == "profile" else profile
                    try:
                        verify_profile_gguf_tensor_table(
                            changed_profile, fixture["inventory"], changed_manifest, payload, reader=owner_reader,
                            source_values=sources, expected_source_sha256=pins,
                            expected_file_bytes=int(sys.argv[2]), expected_file_sha256=changed.pop("expected_file_sha256", sys.argv[3]),
                            model_owner=owner, **changed,
                        )
                        raise AssertionError("mismatched owner accepted: " + defect)
                    except ValueError:
                        pass
            finally:
                del owner_reader
                displaced_path.replace(original_path)
        raise AssertionError("renamed owner file accepted at owner exit")
    except ValueError as error:
        assert "changed while policy was active" in str(error)
    finally:
        if displaced_path.exists():
            displaced_path.replace(original_path)
    assert registry() == baseline and owner.handle.value is None
    with verified_profile_model_policy(
        native, original_path, profile, fixture["inventory"], fixture["manifest"], payload,
        expected_bytes=int(sys.argv[2]), expected_sha256=sys.argv[3],
        tensor_reader=reader, source_values=sources, expected_source_sha256=pins, verify_policy_metadata=True,
    ) as owner:
        assert owner.verification["policy_metadata"]["model_file_access"] == "verified_owner_descriptor"
        owner_reader = gguf.GGUFReader(owner.model_path)
        try:
            successful = verify_profile_gguf_tensor_table(
                profile, fixture["inventory"], fixture["manifest"], payload, reader=owner_reader,
                source_values=sources, expected_source_sha256=pins,
                expected_file_bytes=int(sys.argv[2]), expected_file_sha256=sys.argv[3], model_owner=owner,
            )
        finally:
            del owner_reader
    assert successful["model_file_access"] == "verified_owner_descriptor" and successful["tensor_count"] == 28
    assert registry() == baseline and owner.handle.value is None
    try:
        verify_profile_gguf_tensor_table(
            profile, fixture["inventory"], fixture["manifest"], payload, reader=reader,
            source_values=sources, expected_source_sha256=pins,
            expected_file_bytes=int(sys.argv[2]), expected_file_sha256=sys.argv[3], model_owner=owner,
        )
        raise AssertionError("closed owner accepted")
    except ValueError as error:
        assert "closed" in str(error)
    assert bounded_native_file_record(original_path, maximum_bytes=int(sys.argv[2]), deadline=time.monotonic() + 30)["sha256"] == sys.argv[3]
    archive.close()
    report.update(matrices=matrices, convolutions=convolutions, frozen=frozen, source_archive_sha256=sys.argv[6])
print(json.dumps(report))
"""
            parity = subprocess.run(
                [parity_interpreter or vector_interpreter, "-c", parity_code, str(model_file), str(expected["model_bytes"]), expected["model_sha256"],
                 str(operands_file) if parity_interpreter else "", str(expected.get("source_archive_bytes", 0)), expected.get("source_archive_sha256", ""), runtime_build],
                capture_output=True, text=True, input=_profile_parity_fixture(profile, inventory, manifest, blocks), timeout=30,
                env={**environment, "PYTHONPATH": os.pathsep.join((str(Path.cwd()), str(Path(source_dir) / "gguf-py")))},
            )
            assert parity.returncode == 0, parity.stderr
            expected_parity = {"vectors": 10, "file_sha256": expected["model_sha256"]}
            if parity_interpreter:
                expected_parity.update(matrices=16, convolutions=1, frozen=1, source_archive_sha256=expected["source_archive_sha256"])
            assert json.loads(parity.stdout) == expected_parity
        from embedded_jev.streamed_text import native_backend_dependencies

        bridge = Path(runtime_build) / "bin" / "libprism_group_scale.so"
        dependencies = native_backend_dependencies(bridge, "prism_ggml_registered")
        expected_runtime = {name: {"bytes": item["bytes"], "sha256": item["sha256"]} for name, item in dependencies.items()}
        expected_runtime["ggml_cpu"]["sha256"] = "adaaacaf406df3700fb5f05cb5749990e9b17d410d91e13d0c58263ae971a150"
        expected_runtime["ggml_base"]["sha256"] = "57a9b6060d3bc902c30fd97fb271ae7f027054d4a9f8b50e27dd6fe91951847d"
        with bridge.open("rb") as source:
            bridge_digest = hashlib.file_digest(source, "sha256").hexdigest()
        expected_runtime["bridge"] = {"bytes": bridge.stat().st_size, "sha256": bridge_digest}
        for name, filename in (("scorer", "libprism_bitnet_prefill_test.so"), ("llama", "libllama.so"), ("ggml_registry", "libggml.so")):
            library = Path(runtime_build) / "bin" / filename
            with library.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            expected_runtime[name] = {"bytes": library.stat().st_size, "sha256": digest}
        for requested_mode in ("full", "chunked", "reset", "reordered", "parallel", "maximum", "kernel-recovery",
                               "verified-full", "verified-parallel", "verified-kernel-recovery"):
            gated = parity_interpreter and requested_mode.startswith("verified-")
            control = subprocess.run([parity_interpreter if gated else sys.executable, "-c", driver_code, str(model_file), target, runtime_build, requested_mode,
                                      str(expected["model_bytes"]), expected["model_sha256"], json.dumps(expected_runtime),
                                      str(operands_file) if gated else "", str(expected.get("source_archive_bytes", 0)), expected.get("source_archive_sha256", "")],
                                     capture_output=True, text=True, timeout=30,
                                     input=_profile_parity_fixture(profile, inventory, manifest, blocks) if gated else None,
                                     env={**environment, "PYTHONPATH": os.pathsep.join((environment["PYTHONPATH"], str(Path(source_dir) / "gguf-py")))})
            assert control.returncode == 0, control.stderr
            report = json.loads(control.stdout)
            mode = requested_mode.removeprefix("verified-")
            tokens = [2] * 128 if mode == "maximum" else [3, 5, 7]
            assert report["model_policy_abi"] == 2 and report["generated_answer_tokens"] == 0
            assert report["input_tokens"] == tokens and report["prefilled_tokens"] == len(tokens)
            calls = 3 if mode == "parallel" else 2 if mode in ("chunked", "reset") else 1
            assert report["bitnet_dispatch_calls"] == report["prefill_calls"] == calls
            assert report["weight_repacks"] == 1
            assert report["bitnet_last_input_tokens"] == (2 if mode == "chunked" else len(tokens))
            assert report["rejected_prefill_calls"] == (mode == "kernel-recovery")
            assert report["kernel_error_status"] == (mode == "kernel-recovery")
            order = [2, 0, 1] if mode == "reordered" else [0, 1, 2]
            assert report["option_ids"] == [["inspect", "edit", "ask"][index] for index in order]
            assert report["option_labels"] == [["A", "B", "C"][index] for index in order]
            assert report["option_token_ids"] == [profile.tokenization["label_token_ids"]["ABC"[index]] for index in order]
            logits = np.asarray(expected["maximum_logits" if mode == "maximum" else "logits"])[order]
            np.testing.assert_allclose(report["logits"], logits, rtol=2e-5, atol=2e-5)
            scores = np.exp(logits - logits.max())
            scores /= scores.sum()
            np.testing.assert_allclose(report["conditional_scores"], scores, rtol=2e-5, atol=2e-5)
            if mode == "parallel":
                np.testing.assert_allclose(report["peer_logits"], expected["peer_logits"], rtol=2e-5, atol=2e-5)
        if parity_interpreter:
            for requested_mode in ("verified-table-source-key", "verified-table-source-parity", "verified-table-budget",
                                   "verified-policy-metadata-type", "verified-policy-metadata-copy"):
                control = subprocess.run(
                    [parity_interpreter, "-c", driver_code, str(model_file), target, runtime_build, requested_mode,
                     str(expected["model_bytes"]), expected["model_sha256"], json.dumps(expected_runtime),
                     str(operands_file), str(expected["source_archive_bytes"]), expected["source_archive_sha256"]],
                    capture_output=True, text=True, timeout=30,
                    input=_profile_parity_fixture(profile, inventory, manifest, blocks),
                    env={**environment, "PYTHONPATH": os.pathsep.join((environment["PYTHONPATH"], str(Path(source_dir) / "gguf-py")))},
                )
                assert control.returncode == 33 and control.stdout == "", control.stderr
        for mode, status in (("unknown", 32), ("kernel-error", 29), ("null-policy", 27), ("oversized-geometry", 27),
                     ("null-overrides", 27), ("null-specification", 27), ("wrong-policy", 30),
                             ("shape-mismatch", 30), ("hidden-mismatch", 27), ("vocabulary-mismatch", 27),
                             ("intermediate-mismatch", 27), ("oversized-file", 27)):
            control = subprocess.run([sys.executable, "-c", driver_code, str(model_file), target, runtime_build, mode],
                                     capture_output=True, text=True, timeout=30, env=environment)
            assert control.returncode == status and control.stdout == "", control.stderr
        if parity_interpreter:
            import tempfile
            with tempfile.TemporaryDirectory(dir=tmp_path) as source_directory:
                source_model = Path(source_directory) / "nonzero-source.gguf"
                source_archive = Path(source_directory) / "operands.npz"
                source_profile, source_inventory, source_manifest, source_blocks = _prefill_model_policy_fixture(target, source_shards=2)
                np.testing.assert_array_equal(source_blocks, blocks)
                assert source_manifest["payload_sha256"] == manifest["payload_sha256"]
                source_fixture_input = _profile_parity_fixture(source_profile, source_inventory, source_manifest, source_blocks)
                source_writer = subprocess.run(
                    [converter, "-c", writer_code, json.dumps({**specification, "label_token_ids": dict(profile.tokenization["label_token_ids"])}),
                     str(source_model), str(payload_file), str(source_archive), "pinned-source"],
                    capture_output=True, text=True, timeout=30, input=source_fixture_input,
                    env={**environment, "PYTHONPATH": os.pathsep.join((str(Path.cwd()), str(Path(source_dir) / "gguf-py")))},
                )
                assert source_writer.returncode == 0, source_writer.stderr
                source_expected = json.loads(source_writer.stdout)
                assert len(json.dumps(source_expected["source_fixture"])) < 1024**2
                logits = np.asarray(expected["logits"])
                assert np.ptp(logits) > 0.01
                np.testing.assert_allclose(source_expected["logits"], logits, rtol=0, atol=0)
                np.testing.assert_allclose(source_expected["peer_logits"], expected["peer_logits"], rtol=0, atol=0)
                np.testing.assert_allclose(source_expected["maximum_logits"], expected["maximum_logits"], rtol=0, atol=0)
                for source_mode in ("full", "chunked", "reset", "reordered", "parallel", "maximum", "kernel-recovery"):
                    control = subprocess.run(
                        [parity_interpreter, "-c", driver_code, str(source_model), target, runtime_build, "verified-source-" + source_mode,
                         str(source_expected["model_bytes"]), source_expected["model_sha256"], json.dumps(expected_runtime),
                         str(source_archive), str(source_expected["source_archive_bytes"]), source_expected["source_archive_sha256"]],
                        capture_output=True, text=True, timeout=30, input=json.dumps(source_expected["source_fixture"]),
                        env={**environment, "PYTHONPATH": os.pathsep.join((environment["PYTHONPATH"], str(Path(source_dir) / "gguf-py")))},
                    )
                    assert control.returncode == 0, control.stderr
                    report = json.loads(control.stdout)
                    tokens = [2] * 128 if source_mode == "maximum" else [3, 5, 7]
                    order = [2, 0, 1] if source_mode == "reordered" else [0, 1, 2]
                    calls = 3 if source_mode == "parallel" else 2 if source_mode in ("chunked", "reset") else 1
                    assert report["model_policy_abi"] == 2 and report["generated_answer_tokens"] == 0
                    assert report["input_tokens"] == tokens and report["prefilled_tokens"] == len(tokens)
                    assert report["option_ids"] == [["inspect", "edit", "ask"][index] for index in order]
                    assert report["option_labels"] == [["A", "B", "C"][index] for index in order]
                    assert report["option_token_ids"] == [[11, 17, 23][index] for index in order]
                    assert report["bitnet_dispatch_calls"] == report["prefill_calls"] == calls
                    assert report["weight_repacks"] == 1 and report["bitnet_last_input_tokens"] == (2 if source_mode == "chunked" else len(tokens))
                    assert report["rejected_prefill_calls"] == report["kernel_error_status"] == (source_mode == "kernel-recovery")
                    source_logits = np.asarray(expected["maximum_logits" if source_mode == "maximum" else "logits"])[order]
                    scores = np.exp(source_logits - source_logits.max())
                    scores /= scores.sum()
                    np.testing.assert_allclose(report["logits"], source_logits, rtol=2e-5, atol=2e-5)
                    np.testing.assert_allclose(report["conditional_scores"], scores, rtol=2e-5, atol=2e-5)
                    if source_mode == "parallel":
                        np.testing.assert_allclose(report["peer_logits"], expected["peer_logits"], rtol=2e-5, atol=2e-5)
                source_paths = [Path(source_directory) / name for name in source_expected["source_fixture"]["profile"]["packaging"]["shards"]]
                source_path = source_paths[0]
                for source_exit_mode in ("change", "restored", "replace", "last-change", "last-restored", "last-replace"):
                    active_source_path = source_paths[-1 if source_exit_mode.startswith("last-") else 0]
                    source_offset = source_expected["source_fixture"]["profile"]["files"][active_source_path.name]["header_bytes"]
                    with active_source_path.open("rb") as source:
                        source.seek(source_offset)
                        original_byte = source.read(1)
                    try:
                        control = subprocess.run(
                            [parity_interpreter, "-c", driver_code, str(source_model), target, runtime_build, "verified-source-active-" + source_exit_mode,
                             str(source_expected["model_bytes"]), source_expected["model_sha256"], json.dumps(expected_runtime),
                             str(source_archive), str(source_expected["source_archive_bytes"]), source_expected["source_archive_sha256"]],
                            capture_output=True, text=True, timeout=30, input=json.dumps(source_expected["source_fixture"]),
                            env={**environment, "PYTHONPATH": os.pathsep.join((environment["PYTHONPATH"], str(Path(source_dir) / "gguf-py")))},
                        )
                        assert control.returncode == 33 and control.stdout == "", control.stderr
                    finally:
                        with active_source_path.open("r+b") as source:
                            source.seek(source_offset)
                            source.write(original_byte)
                with source_path.open("r+b") as changed_source:
                    offset = source_expected["source_fixture"]["profile"]["files"][source_path.name]["header_bytes"]
                    changed_source.seek(offset)
                    original_byte = changed_source.read(1)
                    changed_source.seek(offset)
                    changed_source.write(bytes([original_byte[0] ^ 1]))
                for source_refusal in ("file", "range"):
                    if source_refusal == "range":
                        refusal_writer = subprocess.run(
                            [*source_writer.args[:-1], "pinned-source-range-mismatch"],
                            capture_output=True, text=True, timeout=30, input=source_fixture_input,
                            env={**environment, "PYTHONPATH": os.pathsep.join((str(Path.cwd()), str(Path(source_dir) / "gguf-py")))},
                        )
                        assert refusal_writer.returncode == 0, refusal_writer.stderr
                        source_expected = json.loads(refusal_writer.stdout)
                        assert len(json.dumps(source_expected["source_fixture"])) < 1024**2
                    control = subprocess.run(
                        [parity_interpreter, "-c", driver_code, str(source_model), target, runtime_build, "verified-source-" + source_refusal,
                         str(source_expected["model_bytes"]), source_expected["model_sha256"], json.dumps(expected_runtime),
                         str(source_archive), str(source_expected["source_archive_bytes"]), source_expected["source_archive_sha256"]],
                        capture_output=True, text=True, timeout=30, input=json.dumps(source_expected["source_fixture"]),
                        env={**environment, "PYTHONPATH": os.pathsep.join((environment["PYTHONPATH"], str(Path(source_dir) / "gguf-py")))},
                    )
                    assert control.returncode == 33 and control.stdout == "", control.stderr
    finally:
        model_file.unlink(missing_ok=True)
        payload_file.unlink(missing_ok=True)
        operands_file.unlink(missing_ok=True)


@pytest.mark.parametrize("use_bitnet,mode", [(False, "full"), (True, "full"),
                                 (True, "kernel-error"), (True, "kernel-recovery"), (True, "trace")],
                    ids=["dense-bf16", "one-bitnet", "one-bitnet-kernel-error", "one-bitnet-kernel-recovery",
                        "one-bitnet-trace"])
def test_pinned_prism_mimo_bf16_native_prompt_prefill_without_generation(use_bitnet, mode, tmp_path):
    import numpy as np

    model_file = os.environ.get("MIMO_NATIVE_BITNET_GGUF" if use_bitnet else "MIMO_NATIVE_REFERENCE_GGUF")
    if not model_file:
        pytest.skip("requires a separately approved temporary text-only model; never converts weights")
    projection_file = os.environ.get("MIMO_NATIVE_BITNET_PROJECTION_GGUF", "")
    artifact = os.environ.get("MIMO_PROJECTION_ARTIFACT", "")
    if use_bitnet and not all((projection_file, artifact)):
        pytest.fail("one-BitNet reference requires the tagged projection GGUF and frozen artifact")
    source_dir = os.environ.get("PRISM_SOURCE_DIR")
    dense_python = os.environ.get("MIMO_DENSE_PYTHON")
    local_model = os.environ.get("MIMO_LOCAL_DIR")
    runtime_build = os.environ.get("MIMO_PRISM_RUNTIME_BUILD")
    if not all((source_dir, dense_python, local_model, runtime_build)):
        pytest.fail("real native reference requires pinned source, dense environment, source snapshot, and runtime")
    assert subprocess.check_output(["git", "-C", source_dir, "rev-parse", "HEAD"], text=True).strip() == PRISM_REVISION
    source_floor = 17_820_312_576 if use_bitnet else 17_907_606_528
    assert Path(model_file).is_file() and source_floor < Path(model_file).stat().st_size < 19 * 1024**3
    fixture = Path(__file__).parent / "fixtures" / "agent_tool_smoke.json"
    reference_code = """
import hashlib
import json
import sys
from pathlib import Path
import gguf
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from embedded_jev.label_probe import decision_case_messages, load_decision_fixture, probe_label_boundary
model_file, local_model, fixture_path, projection_file, artifact, use_bitnet = sys.argv[1:]
snapshot = Path(local_model)
reader = gguf.GGUFReader(model_file)
tensors = {tensor.name: tensor for tensor in reader.tensors}
assert len(tensors) == 427
assert reader.fields["general.architecture"].contents() == "qwen35"
assert reader.fields["qwen35.block_count"].contents() == 32
assert not any("vision" in name or "mmproj" in name or "nextn" in name or "mtp" in name for name in tensors)
assert tensors["blk.3.ffn_down.weight"].shape.tolist() == [12288, 4096]
if use_bitnet == "1":
    import numpy as np
    from embedded_jev.prism_codec import pack_ternary_pq2_0
    from embedded_jev.projection_artifact import load_projection_artifact
    from embedded_jev.ternary import unpack_group128_codes
    packed, scales, manifest = load_projection_artifact(Path(artifact))
    expected = pack_ternary_pq2_0(unpack_group128_codes(packed), scales).reshape(4096, -1)
    projection = gguf.GGUFReader(projection_file)
    assert len(projection.tensors) == 1
    for target in (tensors["blk.3.ffn_down.weight"], projection.tensors[0]):
        assert target.name == "blk.3.ffn_down.weight" and target.tensor_type == gguf.GGMLQuantizationType.PQ2_0
        np.testing.assert_array_equal(target.data, expected)
    assert reader.fields["jev.model.source_revision"].contents() == manifest["revision"]
    assert reader.fields["jev.bitnet.model"].contents() == "mimo-qwen35-layer3-ffn-down-v1"
    assert all(tensor.tensor_type in (gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.F32)
               for name, tensor in tensors.items() if name != "blk.3.ffn_down.weight")
else:
    assert set(tensor.tensor_type for tensor in tensors.values()) == {gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.F32}
    assert tensors["blk.3.ffn_down.weight"].tensor_type == gguf.GGMLQuantizationType.BF16
fixture, digest = load_decision_fixture(Path(fixture_path))
case = next(case for case in fixture["cases"] if case["id"] == "inspect-before-answer")
messages = decision_case_messages(case)
tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
boundary = probe_label_boundary(tokenizer, messages, ("A", "B", "C"))
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
tokens = tokenizer.encode(prompt, add_special_tokens=False)
labels = list(boundary["label_token_ids"].values())
assert 0 < len(tokens) <= 128
weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
for converted_name, source_name, selected_rows in (
    ("token_embd.weight", "model.language_model.embed_tokens.weight", sorted(set(tokens))),
    ("output.weight", "lm_head.weight", labels),
):
    converted = tensors[converted_name]
    assert converted.shape.tolist() == [4096, 248320]
    assert converted.tensor_type == gguf.GGMLQuantizationType.BF16
    with safe_open(snapshot / weight_map[source_name], framework="pt", device="cpu") as source:
        source_rows = source.get_slice(source_name)
        for token_id in selected_rows:
            row = source_rows[token_id]
            assert row.dtype == torch.bfloat16
            original = row.contiguous().view(torch.int16).numpy().tobytes()
            assert hashlib.sha256(converted.data[token_id].tobytes()).digest() == hashlib.sha256(original).digest()
print(json.dumps({"prompt": prompt, "tokens": tokens, "labels": labels,
                  "option_ids": [option["id"] for option in case["options"]],
                  "prompt_sha256": boundary["prompt_sha256"], "fixture_sha256": digest}))
"""
    reference = subprocess.run(
        [dense_python, "-c", reference_code, model_file, local_model, str(fixture),
         projection_file, artifact, "1" if use_bitnet else "0"],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTHONPATH": str(Path(source_dir) / "gguf-py"),
             "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert reference.returncode == 0, reference.stderr
    expected = json.loads(reference.stdout)
    native_environment = dict(os.environ)
    native_environment.pop("OMP_NUM_THREADS", None)
    arguments = (["--prefill-model-bitnet-control", model_file, projection_file] if use_bitnet
                 else ["--prefill-text-control", model_file])
    arguments += [expected["prompt"], *([mode] if use_bitnet else [])]
    if mode == "trace":
        arguments = ["--prefill-model-bitnet-trace-control", model_file, projection_file,
                     expected["prompt"], str(tmp_path / "trace")]
    control_timeout = 1200 if mode == "trace" else 600
    try:
        control = subprocess.run(
            [str(Path(runtime_build) / "bin" / "prism_bitnet_loader_control"), *arguments],
            capture_output=True, text=True, timeout=control_timeout, env=native_environment,
        )
    except subprocess.TimeoutExpired as failure:
        diagnostics = failure.stderr or ""
        if isinstance(diagnostics, bytes):
            diagnostics = diagnostics.decode("utf-8", errors="replace")
        pytest.fail(f"native prefill exceeded {control_timeout} seconds; stderr tail: {diagnostics[-6000:]}")
    if mode == "kernel-error":
        assert control.returncode == 29 and control.stdout == "", control.stderr[-6000:]
        return
    assert control.returncode == 0, control.stderr[-6000:]
    report = json.loads(control.stdout)
    assert report["input_tokens"] == expected["tokens"]
    assert report["prefilled_tokens"] == len(expected["tokens"])
    assert report["option_ids"] == expected["option_ids"]
    assert report["option_labels"] == ["A", "B", "C"] and report["option_token_ids"] == expected["labels"]
    assert report["generated_answer_tokens"] == 0
    assert report["rejected_prefill_calls"] == int(mode == "kernel-recovery")
    assert report["kernel_error_status"] == int(mode == "kernel-recovery")
    assert report["bitnet_dispatch_calls"] == report["weight_repacks"] == int(use_bitnet)
    assert report["bitnet_last_input_tokens"] == (len(expected["tokens"]) if use_bitnet else 0)
    assert report["backend"] == ("bitnet" if use_bitnet else "dense")
    assert report["mode"] == ("full" if mode == "trace" else mode) and report["prefill_calls"] == 1
    assert report["memory_position_min"] == report["memory_position_max"] == len(expected["tokens"]) - 1
    logits = np.asarray(report["logits"], dtype=np.float64)
    scores = np.asarray(report["conditional_scores"], dtype=np.float64)
    assert logits.shape == scores.shape == (3,) and np.isfinite(logits).all() and np.isfinite(scores).all()
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    np.testing.assert_allclose(scores, probabilities, rtol=1e-12, atol=1e-12)
    if mode == "trace":
        trace = json.loads((tmp_path / "trace" / "manifest.json").read_text())
        assert trace["format"] == "jev-prefill-f32-trace-v1" and trace["byte_order"] == "little"
        assert trace["dtype"] == "float32" and len(trace["tensors"]) == 36
        shapes = {tensor["name"]: tensor["shape"] for tensor in trace["tensors"]}
        assert shapes == {"model.input_embed": [len(expected["tokens"]), 4096],
                          "ffn_input": [len(expected["tokens"]), 12288],
                          "ffn_output": [len(expected["tokens"]), 4096], "final_norm": [1, 4096],
                          **{f"l_out-{layer}": [len(expected["tokens"]), 4096] for layer in range(32)}}
        for tensor in trace["tensors"]:
            values = np.fromfile(tmp_path / "trace" / (tensor["name"] + ".f32"), dtype="<f4")
            assert values.size == np.prod(tensor["shape"]) and np.isfinite(values).all()


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

    import numpy as np

    result = subprocess.run(
        [str(binary), "--qk-norm-control"], check=True, capture_output=True, text=True, timeout=15,
    )
    report = json.loads(result.stdout)
    assert report["rows"] == 5 and report["width"] == 128
    assert report["epsilon"] == pytest.approx(1e-6)
    pattern = (np.arange(128) % 7 - 3).astype(np.float32)
    inputs = pattern[None, :] * np.array([0, 1e-8, 1e-4, 0.01, 1], dtype=np.float32)[:, None]
    squared = inputs * inputs
    norm = np.sqrt(squared.sum(axis=1, keepdims=True, dtype=np.float64)).astype(np.float32)
    native_expected = inputs * (np.float32(1) / np.maximum(norm, np.float32(1e-6)))
    additive_expected = inputs / np.sqrt(squared.sum(axis=1, keepdims=True) + np.float32(1e-6))
    actual = np.asarray(report["outputs"], dtype=np.float32).reshape(5, 128)
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[0], np.zeros(128, dtype=np.float32))
    np.testing.assert_allclose(actual, native_expected, rtol=2e-7, atol=1e-8)
    assert np.max(np.abs(actual[1] - additive_expected[1])) > 0.02
    assert np.max(np.abs(actual[2] - additive_expected[2])) > 0.005

    def recurrence(tokens, initialized, raw_gates=False, interleaved=False):
        token_index = np.arange(tokens)[:, None, None]
        key_head = np.arange(2)[None, :, None]
        column = np.arange(128)[None, None, :]
        query = ((column * 7 + token_index * 3 + key_head * 11) % 29 - 14).astype(np.float32) / 64
        key = ((column * 3 + token_index * 7 + key_head * 5) % 31 - 15).astype(np.float32) / 64
        for features in (query, key):
            norm = np.sqrt((features * features).sum(axis=-1, keepdims=True, dtype=np.float64).astype(np.float32))
            features *= np.float32(1) / np.maximum(norm, np.float32(1e-6))
        value_head = np.arange(4)[None, :, None]
        value = ((column * 5 + token_index * 2 + value_head * 13) % 23 - 11).astype(np.float32) / 32
        decay = -0.125 - ((token_index[:, :, 0] + 2 * np.arange(4)[None, :]) % 5) / 16
        beta = 0.25 + ((token_index[:, :, 0] + np.arange(4)[None, :]) % 7) / 16
        if raw_gates:
            alpha = np.array([-25, -3, 0, 20, 25])[(token_index[:, :, 0] + 2 * np.arange(4)[None, :]) % 5]
            biased = alpha + (np.arange(4)[None, :] - 2) / 8
            decay = -(np.arange(4)[None, :] + 1) / 4 * np.logaddexp(0, biased)
            raw_beta = ((token_index[:, :, 0] + np.arange(4)[None, :]) % 7 - 3) / 4
            beta = 1 / (1 + np.exp(-raw_beta))
        state = np.zeros((4, 128, 128), dtype=np.float64)
        if initialized:
            value_index = np.arange(128)[None, :, None]
            key_index = np.arange(128)[None, None, :]
            state = ((key_index * 3 + value_index * 5 + np.arange(4)[:, None, None] * 7) % 19 - 9) / 4096
        outputs = np.empty((tokens, 4, 128), dtype=np.float64)
        mapping = np.arange(4) // 2 if interleaved else np.arange(4) % 2
        for token in range(tokens):
            selected_key = key[token, mapping].astype(np.float64)
            selected_query = query[token, mapping].astype(np.float64)
            state *= np.exp(decay[token])[:, None, None]
            delta = (value[token] - np.einsum("hvk,hk->hv", state, selected_key)) * beta[token, :, None]
            state += delta[:, :, None] * selected_key[:, None, :]
            outputs[token] = np.einsum("hvk,hk->hv", state, selected_query) / np.sqrt(128)
        return outputs, state

    for tokens, initialized, raw_gates in (
        (tokens, initialized, raw_gates)
        for tokens in (1, 7, 64, 80) for initialized in (False, True) for raw_gates in (False, True)
    ):
        result = subprocess.run(
            [str(binary), "--gdn-control", str(tokens), "nonzero" if initialized else "zero", *(["raw"] if raw_gates else [])],
            check=True, capture_output=True, text=True, timeout=15,
        )
        report = json.loads(result.stdout)
        assert report["tokens"] == tokens and report["initialized"] is initialized
        assert report["raw_gates"] is raw_gates
        assert report["width"] == 128 and report["key_heads"] == 2 and report["value_heads"] == 4
        assert report["broadcast"] == "tiled" and report["state_layout"] == "value_by_key"
        outputs = np.asarray(report["outputs"], dtype=np.float32).reshape(tokens, 4, 128)
        state = np.asarray(report["state"], dtype=np.float32).reshape(4, 128, 128)
        expected_output, expected_state = recurrence(tokens, initialized, raw_gates)
        assert np.isfinite(outputs).all() and np.isfinite(state).all()
        np.testing.assert_allclose(outputs, expected_output, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(state, expected_state, rtol=2e-5, atol=2e-6)
        wrong_output, wrong_state = recurrence(tokens, initialized, raw_gates, interleaved=True)
        assert np.max(np.abs(outputs - wrong_output)) > 2e-5
        assert np.max(np.abs(state - wrong_state)) > 1e-4

    for arguments in (["--unknown"], ["--qk-norm-control", "extra"], ["--gdn-control"],
                      ["--gdn-control", "0", "zero"], ["--gdn-control", "129", "zero"],
                      ["--gdn-control", "7", "unknown"], ["--gdn-control", "7", "zero", "extra"]):
        rejected = subprocess.run([str(binary), *arguments], capture_output=True, text=True, timeout=15)
        assert rejected.returncode == 5 and rejected.stdout == ""


def test_streamed_native_qk_diagnostic_matches_pinned_torch_recurrence():
    dense_python = os.environ.get("MIMO_DENSE_PYTHON")
    if not dense_python:
        pytest.skip("requires the existing pinned CPU Torch/Transformers environment")
    code = """
import json
import numpy as np
import torch
import transformers
from embedded_jev.inventory import InventoryError
from embedded_jev.streamed_text import _ggml_qk_delta_rule
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    l2norm, torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule,
)
assert torch.__version__.startswith("2.10.0") and transformers.__version__ == "5.12.1"
generator = np.random.default_rng(1407)
query = generator.normal(size=(1, 80, 2, 128)).astype(np.float32)
key = generator.normal(size=query.shape).astype(np.float32)
amplitudes = np.resize(np.array([1e-8, 1e-4, 0.01, 1], dtype=np.float32), 80)
query *= amplitudes[None, :, None, None]
key *= amplitudes[None, :, None, None]
def normalize(features):
    norm = np.sqrt((features * features).sum(axis=-1, keepdims=True, dtype=np.float64).astype(np.float32))
    return torch.from_numpy(features / np.maximum(norm, np.float32(1e-6)))
native_query, native_key = normalize(query), normalize(key)
query, key = torch.from_numpy(query), torch.from_numpy(key)
np.testing.assert_allclose(l2norm(query).numpy(),
    query.numpy() / np.sqrt((query.numpy() ** 2).sum(axis=-1, keepdims=True) + np.float32(1e-6)),
    rtol=2e-7, atol=1e-8)
assert torch.max(torch.abs(native_query[:, 1] - l2norm(query)[:, 1])).item() > 0.05
value = torch.from_numpy(generator.normal(0, 0.25, (1, 80, 2, 8)).astype(np.float32))
g = torch.from_numpy(-generator.uniform(0.1, 0.5, (1, 80, 2)).astype(np.float32))
beta = torch.from_numpy(generator.uniform(0.1, 0.9, (1, 80, 2)).astype(np.float32))
initial = torch.from_numpy(generator.normal(0, 0.05, (1, 2, 128, 8)).astype(np.float32))
max_output_error = max_state_error = 0.0
cases = 0
for tokens in (1, 7, 64, 80):
    for state in (None, initial):
        output, final = _ggml_qk_delta_rule(torch_chunk_gated_delta_rule,
            query[:, :tokens], key[:, :tokens], value[:, :tokens], g[:, :tokens], beta[:, :tokens],
            initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True)
        expected_output, expected_final = torch_recurrent_gated_delta_rule(
            native_query[:, :tokens], native_key[:, :tokens], value[:, :tokens], g[:, :tokens], beta[:, :tokens],
            initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=False)
        torch.testing.assert_close(output, expected_output, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(final, expected_final, rtol=2e-5, atol=2e-6)
        max_output_error = max(max_output_error, (output - expected_output).abs().max().item())
        max_state_error = max(max_state_error, (final - expected_final).abs().max().item())
        cases += 1
output, final = _ggml_qk_delta_rule(torch_chunk_gated_delta_rule,
    query[:, :7], key[:, :7], value[:, :7], g[:, :7], beta[:, :7],
    initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
assert final is None and torch.isfinite(output).all()
for dtype, flag in ((torch.float32, False), (torch.bfloat16, True)):
    try:
        _ggml_qk_delta_rule(torch_chunk_gated_delta_rule, query.to(dtype), key.to(dtype), value, g, beta,
                           use_qk_l2norm_in_kernel=flag)
    except InventoryError:
        pass
    else:
        raise AssertionError("unsafe Q/K diagnostic input was accepted")
print(json.dumps({"cases": cases, "max_output_error": max_output_error, "max_state_error": max_state_error}))
"""
    result = subprocess.run(
        [dense_python, "-c", code], capture_output=True, text=True, timeout=60,
        env={**os.environ, "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1",
             "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["cases"] == 8
    assert report["max_output_error"] < 2e-6 and report["max_state_error"] < 2e-6


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
         "-I", str(Path(source_dir) / "include"),
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
    loader_control = tmp_path / "prism-bitnet-loader-control"
    loader_build = subprocess.run(
        [compiler, "-std=c++17", "-O2", "-I", str(Path(source_dir) / "ggml" / "include"),
         "-I", str(Path(source_dir) / "include"),
         str(sources / "prism_bitnet_loader_control.cpp"), str(binary),
         "-L", str(library_dir), f"-Wl,-rpath,{library_dir}", "-pthread", "-lggml-cpu", "-lggml-base",
         "-o", str(loader_control)],
        capture_output=True, text=True,
    )
    assert loader_build.returncode == 0, loader_build.stderr
    for options in ([], ["--late"]):
        loader_run = subprocess.run([str(loader_control), *options], capture_output=True, text=True, timeout=15)
        assert loader_run.returncode == 0, loader_run.stderr
        loader_report = json.loads(loader_run.stdout)
        assert loader_report == ({"late_init_refused": True} if options else
                                 {"discovery_matches": 1, "idempotent_init": True,
                                  "kernel_calls": 11, "weight_repacks": 1, "concurrent_graphs": True})
    if os.environ.get("MIMO_NATIVE_VOCAB_TEST") == "1":
        dense_python = os.environ.get("MIMO_DENSE_PYTHON")
        local_model = os.environ.get("MIMO_LOCAL_DIR")
        runtime_build = os.environ.get("MIMO_PRISM_RUNTIME_BUILD")
        if not dense_python or not local_model or not runtime_build:
            pytest.fail("native vocabulary control requires dense Python, local model, and versioned runtime build")
        runtime_dependencies = native_backend_dependencies(
            Path(runtime_build) / "bin" / "libprism_group_scale.so", "prism_ggml",
        )
        assert runtime_dependencies == dependencies
        vocab_file = tmp_path / "mimo-vocab-only.gguf"
        vocab_control = """
import json
import runpy
import sys
from pathlib import Path
import safetensors
import gguf
from transformers import AutoTokenizer
original_open = safetensors.safe_open
class HeaderOnly:
    def __init__(self, *args, **kwargs):
        self.source = original_open(*args, **kwargs)
    def __enter__(self):
        self.source.__enter__()
        return self
    def __exit__(self, *args):
        return self.source.__exit__(*args)
    def keys(self):
        return self.source.keys()
    def get_tensor(self, *args):
        raise AssertionError("vocabulary preflight must not load source weights")
    def get_slice(self, *args):
        raise AssertionError("vocabulary preflight must not access source weight slices")
safetensors.safe_open = HeaderOnly
source, model, destination = sys.argv[1:]
sys.path.insert(0, source)
sys.argv = [str(Path(source) / "convert_hf_to_gguf.py"), model, "--vocab-only",
            "--outtype", "bf16", "--no-nextn", "--outfile", destination]
runpy.run_path(sys.argv[0], run_name="__main__")
reader = gguf.GGUFReader(destination)
assert len(reader.tensors) == 0
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Choose A or B. A: inspect the file. B: edit the file."}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
assert isinstance(prompt, str)
labels = [tokenizer.encode(label, add_special_tokens=False) for label in ("A", "B", "C")]
assert all(len(tokens) == 1 for tokens in labels)
print(json.dumps({"prompt": prompt, "tokens": tokenizer.encode(prompt, add_special_tokens=False),
                  "labels": [tokens[0] for tokens in labels]}))
"""
        try:
            vocab_write = subprocess.run(
                [dense_python, "-c", vocab_control, source_dir, local_model, str(vocab_file)],
                capture_output=True, text=True, timeout=60,
                env={**os.environ, "PYTHONPATH": str(Path(source_dir) / "gguf-py"),
                     "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
            )
            assert vocab_write.returncode == 0, vocab_write.stderr
            vocab_reference = json.loads(vocab_write.stdout)
            native_vocab = subprocess.run(
                [str(Path(runtime_build) / "bin" / "prism_bitnet_loader_control"), "--vocab-only",
                 str(vocab_file), vocab_reference["prompt"]], capture_output=True, text=True, timeout=30,
            )
            assert native_vocab.returncode == 0, native_vocab.stderr
            assert json.loads(native_vocab.stdout) == {"vocab_only": True, "generated_answer_tokens": 0,
                                                     "tokens": vocab_reference["tokens"],
                                                     "labels": vocab_reference["labels"]}
        finally:
            vocab_file.unlink(missing_ok=True)
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
    from embedded_jev.ternary import reconstruct_ternary, unpack_group128_codes

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
    projection_storage = native.prism_bitnet_registered_projection_storage_bytes_v1
    projection_storage.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    projection_storage.restype = ctypes.c_int

    from embedded_jev.prism_codec import NativeProjectionSpecification as ProjectionSpecification

    create_v2 = native.prism_bitnet_registered_projection_create_v2
    create_v2.argtypes = [ctypes.POINTER(ProjectionSpecification), ctypes.POINTER(ctypes.c_uint8),
                         ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
    create_v2.restype = ctypes.c_int
    for rows, name in ((8192, "blk.3.attn_q.weight"), (12288, "blk.1.ffn_gate.weight"), (12288, "blk.2.ffn_up.weight")):
        large_groups = 32
        codes = generator.integers(-1, 2, size=(rows, large_groups, 128), dtype=np.int8)
        fp16_scales = generator.uniform(0.01, 0.2, size=(rows, large_groups)).astype(np.float16)
        blocks = np.concatenate([pack_ternary_pq2_0(codes[first:first + 4096], fp16_scales[first:first + 4096])
                                 for first in range(0, rows, 4096)])
        reference_blocks = blocks.copy()
        specification = ProjectionSpecification(2, name.encode(), rows, large_groups * 128,
            reference_blocks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), reference_blocks.nbytes)
        handle = ctypes.c_void_p()
        assert create_projection(blocks.ctypes.data_as(create_projection.argtypes[0]), 2, rows, large_groups, ctypes.byref(handle)) == 1
        assert handle.value is None
        assert create_v2(ctypes.byref(specification), blocks.ctypes.data_as(create_v2.argtypes[1]),
                         blocks.nbytes, 2, ctypes.byref(handle)) == 0
        try:
            values = generator.normal(size=(2, large_groups * 128)).astype(np.float32)
            prepared, input_scales = quantize_a8_per_group(values)
            packed = pack_group128_codes(codes)
            scales = fp16_scales.astype(np.float32)
            reference_output = np.empty((2, rows), dtype=np.float32)
            assert direct(packed.ctypes.data_as(arguments[0]), scales.ctypes.data_as(arguments[1]),
                          prepared.ctypes.data_as(arguments[2]), input_scales.ctypes.data_as(arguments[3]),
                          2, rows, large_groups, reference_output.ctypes.data_as(arguments[-1])) == 0
            output = np.empty_like(reference_output)
            calls, repacks = ctypes.c_size_t(), ctypes.c_size_t()
            assert compute_projection(handle, values.ctypes.data_as(compute_projection.argtypes[1]),
                                      output.ctypes.data_as(compute_projection.argtypes[2]),
                                      ctypes.byref(calls), ctypes.byref(repacks)) == 0
            np.testing.assert_array_equal(output, reference_output)
            assert calls.value == repacks.value == 1
            resident_bytes, auxiliary_bytes = ctypes.c_size_t(), ctypes.c_size_t()
            assert projection_storage(handle, ctypes.byref(resident_bytes), ctypes.byref(auxiliary_bytes)) == 0
            assert resident_bytes.value == blocks.nbytes and auxiliary_bytes.value == 0
        finally:
            free_projection(handle)
        changed = blocks.copy()
        changed.flat[-1] ^= 1
        handle = ctypes.c_void_p(123)
        assert create_v2(ctypes.byref(specification), changed.ctypes.data_as(create_v2.argtypes[1]),
                         changed.nbytes, 2, ctypes.byref(handle)) == 6
        assert handle.value is None
        specification.tensor_name = b"blk.3.ssm_out.weight"
        assert create_v2(ctypes.byref(specification), blocks.ctypes.data_as(create_v2.argtypes[1]),
                         blocks.nbytes, 2, ctypes.byref(handle)) == 6
        assert handle.value is None
        specification.tensor_name = name.encode()
        for field, value in (("abi_version", 1), ("rows", 12289), ("columns", 12288),
                             ("reference_bytes", blocks.nbytes + 1)):
            invalid = ProjectionSpecification.from_buffer_copy(specification)
            setattr(invalid, field, value)
            handle = ctypes.c_void_p(123)
            assert create_v2(ctypes.byref(invalid), blocks.ctypes.data_as(create_v2.argtypes[1]),
                             blocks.nbytes, 2, ctypes.byref(handle)) == 1
            assert handle.value is None
        for tokens in (0, 129):
            handle = ctypes.c_void_p(123)
            assert create_v2(ctypes.byref(specification), blocks.ctypes.data_as(create_v2.argtypes[1]),
                             blocks.nbytes, tokens, ctypes.byref(handle)) == 1
            assert handle.value is None

    import hashlib

    from embedded_jev.inventory import _json_object, build_inventory, read_local_headers
    from embedded_jev.model_profile import mimo_profile
    from embedded_jev.prism_codec import create_profile_projection

    snapshot = os.environ.get("MIMO_LOCAL_DIR")
    artifact = os.environ.get("MIMO_PROJECTION_ARTIFACT")
    if snapshot and artifact:
        from embedded_jev.projection_artifact import load_projection_artifact

        profile = mimo_profile()
        metadata, headers = read_local_headers(Path(snapshot), profile=profile)
        inventory = build_inventory(metadata, headers, profile=profile)
        stored_packed, stored_scales, stored_manifest = load_projection_artifact(Path(artifact))
        source_tensor = next(tensor for tensor in inventory["tensors"] if tensor["name"] == stored_manifest["tensor"])
        header, _ = headers[source_tensor["shard"]]
        offsets = _json_object(header[8:], source_tensor["shard"])[source_tensor["name"]]["data_offsets"]
        source_digest = hashlib.sha256()
        remaining = source_tensor["storage_bytes"]
        with (Path(snapshot) / source_tensor["shard"]).open("rb") as source:
            source.seek(len(header) + offsets[0])
            while remaining:
                data = source.read(min(1 << 20, remaining))
                assert data, "short existing source tensor payload"
                source_digest.update(data)
                remaining -= len(data)
        blocks = pack_ternary_pq2_0(unpack_group128_codes(stored_packed), stored_scales)
        manifest = {
            "schema_version": 2, "format": "jev-profile-projection-v2", "profile_id": profile.profile_id,
            "profile_sha256": profile.sha256, "tensor": stored_manifest["tensor"],
            "gguf_tensor": "blk.3.ffn_down.weight", "shape": stored_manifest["shape"], "group_size": 128,
            "transform": "identity", "source_tensor_sha256": source_digest.hexdigest(),
            "payload_bytes": blocks.nbytes, "payload_sha256": hashlib.sha256(memoryview(blocks)).hexdigest(),
        }
        handle, specification = create_profile_projection(native, profile, inventory, manifest, blocks, tokens=2)
        try:
            values = generator.normal(size=(2, 12288)).astype(np.float32)
            prepared, input_scales = quantize_a8_per_group(values)
            reference_output = np.empty((2, 4096), dtype=np.float32)
            scales = stored_scales.astype(np.float32)
            assert direct(stored_packed.ctypes.data_as(arguments[0]), scales.ctypes.data_as(arguments[1]),
                          prepared.ctypes.data_as(arguments[2]), input_scales.ctypes.data_as(arguments[3]),
                          2, 4096, 96, reference_output.ctypes.data_as(arguments[-1])) == 0
            output = np.empty_like(reference_output)
            calls, repacks = ctypes.c_size_t(), ctypes.c_size_t()
            assert compute_projection(handle, values.ctypes.data_as(compute_projection.argtypes[1]),
                                      output.ctypes.data_as(compute_projection.argtypes[2]),
                                      ctypes.byref(calls), ctypes.byref(repacks)) == 0
            np.testing.assert_array_equal(output, reference_output)
            assert specification["profile_id"] == profile.profile_id and calls.value == repacks.value == 1
        finally:
            free_projection(handle)

    def check_projection_lifetime(blocks, values, reference):
        before = registry_size()
        tokens, rows = reference.shape[1:]
        groups = values.shape[-1] // 128
        copied_blocks = blocks.copy()
        handle = ctypes.c_void_p()
        assert create_projection(copied_blocks.ctypes.data_as(create_projection.argtypes[0]),
                                 tokens, rows, groups, ctypes.byref(handle)) == 0
        assert handle.value and registry_size() == before
        resident_bytes, auxiliary_bytes = ctypes.c_size_t(99), ctypes.c_size_t(99)
        assert projection_storage(handle, ctypes.byref(resident_bytes), ctypes.byref(auxiliary_bytes)) == 0
        assert resident_bytes.value == rows * groups * 34 and auxiliary_bytes.value == 0
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

    converter = os.environ.get("PRISM_CONVERTER_PYTHON")
    if converter:
        import_projection = native.prism_bitnet_registered_projection_create_from_gguf
        import_projection.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
        import_projection.restype = ctypes.c_int
        writer_control = """
import json
import sys
from pathlib import Path
import gguf
import numpy as np
from embedded_jev.prism_codec import pack_ternary_pq2_0
directory = Path(sys.argv[1])
generator = np.random.default_rng(291)
codes = generator.integers(-1, 2, size=(5, 3, 128), dtype=np.int8)
scales = generator.uniform(0.01, 1, size=(5, 3)).astype(np.float16)
codes[0, 0] = 0
scales[0, 0] = 0
blocks = pack_ternary_pq2_0(codes, scales)
for kind in ("valid", "missing_contract", "wrong_contract", "contract_type", "wrong_name",
             "wrong_type", "extra_tensor", "row_limit", "group_limit", "batch_dimension",
             "bad_code", "bad_scale", "truncated", "file_limit", "transform_metadata"):
    path = directory / (kind + ".gguf")
    writer = gguf.GGUFWriter(str(path), "jev-tensor-control")
    if kind == "contract_type":
        writer.add_uint32("jev.bitnet.execution", 128)
    elif kind != "missing_contract":
        writer.add_string("jev.bitnet.execution", "q8_K" if kind == "wrong_contract"
                          else "group128-a8-fp32-nearest-even-identity-v1")
    if kind == "transform_metadata":
        writer.add_uint32("prism.hadamard.version", 1)
    payload = blocks.reshape(5, -1).copy()
    if kind == "bad_code":
        payload.flat[2] = (int(payload.flat[2]) & ~3) | 3
    if kind == "bad_scale":
        payload.reshape(-1, 34)[0, :2] = np.array([np.nan], dtype="<f2").view(np.uint8)
    if kind == "row_limit":
        payload = np.zeros((4097, 34), dtype=np.uint8)
    if kind == "group_limit":
        payload = np.zeros((1, 97 * 34), dtype=np.uint8)
    if kind == "batch_dimension":
        payload = np.stack((payload, payload))
    name = "other.weight" if kind == "wrong_name" else "blk.3.ffn_down.weight"
    if kind == "wrong_type":
        writer.add_tensor(name, np.zeros((5, 384), dtype=np.float32))
    else:
        writer.add_tensor(name, payload, raw_dtype=gguf.GGMLQuantizationType.PQ2_0)
    if kind == "extra_tensor":
        writer.add_tensor("extra.weight", np.zeros((1, 128), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    if kind in ("truncated", "file_limit"):
        with path.open("r+b") as source:
            source.truncate(gguf.GGUFReader(str(path)).data_offset + 1 if kind == "truncated" else 14 * 1024 * 1024 + 1)
print(json.dumps({"codes": codes.tolist(), "scales": scales.tolist()}))
"""
        writer_result = subprocess.run([converter, "-c", writer_control, str(tmp_path)],
                                       capture_output=True, text=True, timeout=30,
                                       env={**os.environ, "PYTHONPATH": str(Path(source_dir) / "gguf-py"),
                                            "PYTHONDONTWRITEBYTECODE": "1"})
        assert writer_result.returncode == 0, writer_result.stderr
        file_reference = json.loads(writer_result.stdout)
        file_packed = pack_group128_codes(np.asarray(file_reference["codes"], dtype=np.int8))
        file_scales = np.asarray(file_reference["scales"], dtype=np.float32)
        valid_path = tmp_path / "valid.gguf"
        real_loader_control = tmp_path / "prism-bitnet-real-loader-control"
        loader_sources = ["llama-model-loader.cpp", "llama-mmap.cpp", "llama-arch.cpp",
                          "llama-hparams.cpp", "llama-impl.cpp", "llama.cpp"]
        real_loader_build = subprocess.run(
            [compiler, "-std=c++17", "-O1", "-ffunction-sections", "-fdata-sections", "-DJEV_TEST_REAL_LOADER=1",
             "-DGGML_USE_CPU=1", "-I", str(Path(source_dir) / "ggml" / "src"),
             f'-DLLAMA_VERSION="jev-loader-control-{PRISM_REVISION[:7]}"',
             "-I", str(Path(source_dir) / "ggml" / "include"), "-I", str(Path(source_dir) / "include"),
             "-I", str(Path(source_dir) / "src"), str(sources / "prism_bitnet_loader_control.cpp"),
             *(str(Path(source_dir) / "src" / filename) for filename in loader_sources),
             str(Path(source_dir) / "ggml" / "src" / "ggml-backend-reg.cpp"), str(binary),
             "-L", str(library_dir), f"-Wl,-rpath,{library_dir}", "-Wl,--gc-sections", "-pthread",
             "-lggml-cpu", "-lggml-base", "-o", str(real_loader_control)],
            capture_output=True, text=True, timeout=180,
        )
        assert real_loader_build.returncode == 0, real_loader_build.stderr
        real_loader_run = subprocess.run(
            [str(real_loader_control), "--real-loader", str(valid_path), str(tmp_path / "missing_contract.gguf")],
            capture_output=True, text=True, timeout=15,
        )
        assert real_loader_run.returncode == 0, real_loader_run.stderr
        assert json.loads(real_loader_run.stdout) == {"real_loader_selected": True, "untagged_pq2_unselected": True,
                                                     "kernel_calls": 2, "weight_repacks": 1}
        for kind in ("valid", "missing_contract", "wrong_contract", "contract_type", "wrong_name", "wrong_type",
                     "extra_tensor", "row_limit", "group_limit", "batch_dimension", "bad_code",
                     "bad_scale", "truncated", "file_limit", "missing_file", "transform_metadata"):
            override_run = subprocess.run([str(loader_control), "--override", str(tmp_path / (kind + ".gguf"))],
                                          capture_output=True, text=True, timeout=15)
            assert override_run.returncode == 0, override_run.stderr
            expected_status = 0 if kind == "valid" else 5 if kind in ("bad_code", "bad_scale") else 6
            assert json.loads(override_run.stdout) == {"status": expected_status, "override_ready": kind == "valid"}
        before = registry_size()
        handle = ctypes.c_void_p()
        assert import_projection(os.fsencode(valid_path), 2, ctypes.byref(handle)) == 0
        valid_path.unlink()
        try:
            for values in (inputs, -2 * inputs):
                prepared, input_scales = quantize_a8_per_group(values)
                assert direct(file_packed.ctypes.data_as(arguments[0]), file_scales.ctypes.data_as(arguments[1]),
                              prepared.ctypes.data_as(arguments[2]), input_scales.ctypes.data_as(arguments[3]),
                              2, 5, 3, expected.ctypes.data_as(arguments[-1])) == 0
                assert compute_projection(handle, values.ctypes.data_as(compute_projection.argtypes[1]),
                                          actual.ctypes.data_as(compute_projection.argtypes[2]),
                                          ctypes.byref(calls), ctypes.byref(repacks)) == 0
                np.testing.assert_array_equal(actual, expected)
                assert repacks.value == 1 and registry_size() == before
            assert calls.value == 2
        finally:
            free_projection(handle)
        for kind in ("missing_contract", "wrong_contract", "contract_type", "wrong_name", "wrong_type",
                     "extra_tensor", "row_limit", "group_limit", "batch_dimension", "bad_code",
                     "bad_scale", "truncated", "file_limit", "missing_file", "transform_metadata"):
            handle = ctypes.c_void_p(123)
            status = import_projection(os.fsencode(tmp_path / (kind + ".gguf")), 2, ctypes.byref(handle))
            assert status == (5 if kind in ("bad_code", "bad_scale") else 6)
            assert handle.value is None and registry_size() == before
        assert import_projection(None, 2, ctypes.byref(handle)) == 1
        assert handle.value is None

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
        if converter:
            frozen_file = tmp_path / "frozen-loader-parity.gguf"
            frozen_writer = """
import sys
from pathlib import Path
import gguf
from embedded_jev.prism_codec import pack_ternary_pq2_0
from embedded_jev.projection_artifact import load_projection_artifact
from embedded_jev.ternary import unpack_group128_codes
packed, scales, manifest = load_projection_artifact(Path(sys.argv[1]))
assert manifest["shape"] == [4096, 12288]
blocks = pack_ternary_pq2_0(unpack_group128_codes(packed), scales).reshape(4096, -1)
writer = gguf.GGUFWriter(sys.argv[2], "jev-tensor-control")
writer.add_string("jev.bitnet.execution", "group128-a8-fp32-nearest-even-identity-v1")
writer.add_tensor("blk.3.ffn_down.weight", blocks, raw_dtype=gguf.GGMLQuantizationType.PQ2_0)
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
"""
            try:
                frozen_write = subprocess.run([converter, "-c", frozen_writer, artifact, str(frozen_file)],
                                              capture_output=True, text=True, timeout=30,
                                              env={**os.environ, "PYTHONPATH": str(Path(source_dir) / "gguf-py"),
                                                   "PYTHONDONTWRITEBYTECODE": "1"})
                assert frozen_write.returncode == 0, frozen_write.stderr
                frozen_load = subprocess.run(
                    [str(real_loader_control), "--real-loader", str(frozen_file), str(tmp_path / "missing_contract.gguf")],
                    capture_output=True, text=True, timeout=30,
                )
                assert frozen_load.returncode == 0, frozen_load.stderr
                assert json.loads(frozen_load.stdout) == {"real_loader_selected": True, "untagged_pq2_unselected": True,
                                                         "kernel_calls": 2, "weight_repacks": 1}
            finally:
                frozen_file.unlink(missing_ok=True)
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
        if os.environ.get("MIMO_REGISTERED_MODULE_TEST") == "1":
            interpreter = os.environ.get("MIMO_DENSE_PYTHON")
            if not interpreter:
                pytest.fail("set MIMO_DENSE_PYTHON for the registered projection module control")
            module_control = """
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import embedded_jev.activation as activation
from embedded_jev.inventory import InventoryError
from embedded_jev.streamed_text import make_native_ffn_down
library, artifact = map(Path, sys.argv[1:])
direct, _ = make_native_ffn_down(None, library, artifact, backend="direct")
values = torch.from_numpy(np.random.default_rng(178).normal(size=(1, 2, 12288)).astype(np.float32)).to(torch.bfloat16)
references = [direct(values), direct(-2 * values)]
original = activation.quantize_a8_per_group
def reference_only(values, *args, **kwargs):
    if values.shape != (1, 12288):
        raise AssertionError("Python prepared A8 for the production token batch")
    return original(values, *args, **kwargs)
with patch.object(activation, "quantize_a8_per_group", reference_only):
    module, report = make_native_ffn_down(None, library, artifact, backend="prism_ggml_registered")
    try:
        for index, features in enumerate((values, -2 * values)):
            torch.testing.assert_close(module(features), references[index], rtol=0, atol=0)
            assert report["native_dispatch_calls"] == index + 1
            assert report["weight_uploads"] == report["weight_repacks"] == 1
        assert report["graph_op"] == "mul_mat"
        assert report["activation_preparation"] == "native_tensor_trait"
        assert report["bf16_projection_materialized"] is False
        try:
            module(values[:, :1])
        except InventoryError:
            pass
        else:
            raise AssertionError("registered module silently reused a different token shape")
    finally:
        module.close()
    module.close()
    assert report["native_handle_released"] is True
    try:
        module(values)
    except InventoryError:
        pass
    else:
        raise AssertionError("closed native projection was reused")
"""
            module_result = subprocess.run([interpreter, "-c", module_control, str(binary), artifact],
                                           capture_output=True, text=True, timeout=60,
                                           env={**os.environ, "OMP_NUM_THREADS": "4", "PYTHONDONTWRITEBYTECODE": "1"})
            assert module_result.returncode == 0, module_result.stderr
    if os.environ.get("MIMO_PRISM_GRAPH_TEST") == "1":
        interpreter = os.environ.get("MIMO_DENSE_PYTHON")
        local_dir = os.environ.get("MIMO_LOCAL_DIR")
        artifact = os.environ.get("MIMO_PROJECTION_ARTIFACT")
        if not all((interpreter, local_dir, artifact)):
            pytest.fail("set MiMo interpreter/snapshot/candidate for the optional real projection graph test")
        environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4"}
        reports = []
        for backend in ("direct", "prism_ggml", "prism_ggml_f32", "prism_ggml_registered"):
            command = [
                interpreter, "-m", "embedded_jev.streamed_text", "--local-dir", local_dir,
                "--layers", "32", "--native-ffn-library", str(binary), "--native-ffn-backend", backend,
                "--projection-artifact", artifact,
            ]
            if backend in ("prism_ggml_f32", "prism_ggml_registered"):
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
            registered_backend = graph_report["native_ffn_down"]["backend"] == "prism_ggml_registered"
            assert graph_report["native_ffn_down"]["graph_op"] == ("mul_mat" if registered_backend else "map_custom2")
            assert graph_report["native_ffn_down"]["backend_dependencies"] == dependencies
            assert graph_report["native_ffn_down"]["bf16_projection_materialized"] is False
            assert graph_report["native_ffn_down"]["max_native_reference_error"] < 1e-4
            assert graph_report["native_ffn_down"]["calls"] == 1
            assert graph_report["ffn_down_input_sha256"] == direct_report["ffn_down_input_sha256"]
            assert graph_report["last_token_sha256"] == direct_report["last_token_sha256"]
            assert graph_report["selected_head"]["options"] == direct_report["selected_head"]["options"]
            assert graph_report["generated_tokens"] == direct_report["generated_tokens"] == 0
        assert reports[2]["native_ffn_down"]["activation_preparation"] == "native_graph_callback"
        registered_report = reports[3]["native_ffn_down"]
        assert registered_report["activation_preparation"] == "native_tensor_trait"
        assert registered_report["native_dispatch_calls"] == 1
        assert registered_report["weight_uploads"] == registered_report["weight_repacks"] == 1
        assert registered_report["weight_tensor_bytes"] == 4096 * 96 * 34
        assert registered_report["native_handle_released"] is True
        if os.environ.get("MIMO_REGISTERED_TYPED_TEST") == "1":
            fixture = Path(__file__).parent / "fixtures" / "agent_tool_smoke.json"
            typed_reports = []
            for backend in ("direct", "prism_ggml_registered"):
                typed_result = subprocess.run(
                    [interpreter, "-m", "embedded_jev.streamed_text", "--local-dir", local_dir,
                     "--layers", "32", "--native-ffn-library", str(binary), "--native-ffn-backend", backend,
                     "--projection-artifact", artifact, "--fixture", str(fixture),
                     "--case-id", "edit-reordered-options"],
                    env=environment, capture_output=True, text=True, timeout=180,
                )
                assert typed_result.returncode == 0, typed_result.stderr
                typed_reports.append(json.loads(typed_result.stdout))
            assert typed_reports[0]["last_token_sha256"] == typed_reports[1]["last_token_sha256"]
            assert typed_reports[0]["decision"] == typed_reports[1]["decision"]
            decision = typed_reports[1]["decision"]
            assert decision["scope"] == "synthetic_fixture_observation_not_quality_or_calibration"
            assert [option["id"] for option in decision["options"]] == ["ask", "inspect", "edit"]
            assert [option["label"] for option in decision["options"]] == ["A", "B", "C"]
            assert decision["expected_option_id"] == "edit"
            assert typed_reports[0]["generated_tokens"] == typed_reports[1]["generated_tokens"] == 0
            assert typed_reports[1]["native_ffn_down"]["native_handle_released"] is True
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

    from embedded_jev.inventory import _json_object, _open_bounded, read_local_headers

    local_dir = os.environ.get("MIMO_LOCAL_DIR")
    if local_dir:
        metadata, _ = read_local_headers(Path(local_dir))
        index_bytes = metadata["model.safetensors.index.json"]
    else:
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


@pytest.mark.parametrize("vector_dtype", ["BF16", "F32"])
def test_pinned_prism_qwen35_value_head_reorder_on_toy_tensors(vector_dtype):
    from embedded_jev.prism_codec import profile_model_conversion_plan
    from test_inventory import _prefill_model_policy_fixture

    source_dir = os.environ.get("PRISM_SOURCE_DIR")
    interpreter = os.environ.get("PRISM_CONVERTER_PYTHON")
    if not source_dir or not interpreter:
        pytest.skip("requires pinned Prism converter and isolated CPU Torch environment")
    source = Path(source_dir)
    assert subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip() == PRISM_REVISION
    profile, inventory, manifest, blocks = _prefill_model_policy_fixture("down", vector_dtype=vector_dtype, conv_dtype=vector_dtype)
    plan = profile_model_conversion_plan(profile, inventory, manifest, blocks)
    fixture = _profile_parity_fixture(profile, inventory, manifest, blocks)
    code = """
from __future__ import annotations
import ast
import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import torch
import gguf
from embedded_jev.model_profile import parse_model_profile
from embedded_jev.prism_codec import verify_profile_converted_vector, verify_profile_gguf_vector, verify_profile_converted_matrix, verify_profile_gguf_matrix, verify_profile_converted_convolution, verify_profile_gguf_convolution
import embedded_jev.prism_codec as prism_codec

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
plan = json.loads(sys.argv[2])
transforms = {item["name"]: item["transforms"] for item in plan["tensors"]}
assert plan["linear_head_layout"]["linear_num_key_heads"] == model.hparams["linear_num_key_heads"]
assert plan["linear_head_layout"]["linear_num_value_heads"] == model.hparams["linear_num_value_heads"]
assert transforms["blk.0.attn_qkv.weight"] == ["v_rows_grouped_to_tiled", "encode_bf16"]
assert transforms["blk.0.attn_gate.weight"] == ["v_heads_grouped_to_tiled", "encode_bf16"]
assert transforms["blk.0.ssm_a"] == ["v_heads_grouped_to_tiled", "negative_exp", "encode_f32"]
assert transforms["blk.0.ssm_dt.bias"] == ["v_heads_grouped_to_tiled", "encode_f32"]
assert transforms["blk.0.ssm_alpha.weight"] == ["v_heads_grouped_to_tiled", "encode_bf16"]
assert transforms["blk.0.ssm_beta.weight"] == ["v_heads_grouped_to_tiled", "encode_bf16"]
assert transforms["blk.0.ssm_conv1d.weight"] == ["v_rows_grouped_to_tiled", "squeeze", "encode_f32"]
assert transforms["blk.0.ssm_out.weight"] == ["v_columns_grouped_to_tiled", "encode_bf16"]
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
result_beta = list(model.modify_tensors(alpha, name + "in_proj_b.weight", 0))
assert np.array_equal(result_beta[0][1].numpy(), expected_alpha)
for norm_name, gguf_name in (("model.norm.weight", "output_norm.weight"),
                             ("model.layers.0.input_layernorm.weight", "blk.0.attn_norm.weight"),
                             ("model.layers.1.self_attn.q_norm.weight", "blk.1.attn_q_norm.weight"),
                             (name + "norm.weight", "blk.0.ssm_norm.weight")):
    values = torch.tensor([0, 2, -1], dtype=torch.float32)
    add_one = not norm_name.endswith("linear_attn.norm.weight")
    assert transforms[gguf_name] == (["add_one"] if add_one else []) + ["encode_f32"]
    result_norm = list(model.modify_tensors(values, norm_name, 0))
    assert np.array_equal(result_norm[0][1].numpy(), values.numpy() + int(add_one))
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
model.hadamard_folded_names = lambda: []
fixture = json.loads(sys.stdin.read())
profile = parse_model_profile(json.dumps(fixture["profile"]).encode())
assert profile.sha256 == plan["profile_sha256"]
payload = np.frombuffer(bytes.fromhex(fixture["payload_hex"]), dtype=np.uint8).reshape(
    fixture["manifest"]["shape"][0], fixture["manifest"]["shape"][1] // 128, 34)
model.hparams.update(plan["linear_head_layout"])
model.hparams["hidden_size"] = profile.geometry["hidden_size"]
convolution = next(item for item in plan["tensors"] if item["source_name"].endswith("linear_attn.conv1d.weight"))
source_conv = (torch.arange(int(np.prod(convolution["source_shape"])), dtype=torch.float32).reshape(convolution["source_shape"]) % 17 - 8) / 16
raw_conv = source_conv.to(torch.bfloat16).view(torch.uint16).numpy().copy() if convolution["source_dtype"] == "BF16" else source_conv.numpy().copy()
converted_conv = list(model.modify_tensors(source_conv, convolution["source_name"], 0))[0][1].numpy().copy()
conv_keywords = {"source_name": convolution["source_name"], "source_values": raw_conv, "converted_values": converted_conv,
                 "expected_source_sha256": hashlib.sha256(memoryview(raw_conv)).hexdigest()}
conv_receipt = verify_profile_converted_convolution(profile, fixture["inventory"], fixture["manifest"], payload, **conv_keywords)
assert conv_receipt["converted_sha256"] == hashlib.sha256(memoryview(converted_conv)).hexdigest()
assert conv_receipt["source_file_verification"] == conv_receipt["converted_file_verification"] == "not_checked"
changed_conv = converted_conv.copy()
changed_conv.view(np.uint32).flat[0] ^= 0x80000000
try:
    verify_profile_converted_convolution(profile, fixture["inventory"], fixture["manifest"], payload,
                                        **{**conv_keywords, "converted_values": changed_conv})
    raise AssertionError("changed convolution sign bit accepted")
except ValueError as error:
    assert "byte parity" in str(error)
with TemporaryDirectory(prefix="jev-convolution-gguf-") as directory:
    path = Path(directory) / "convolution.gguf"
    writer = gguf.GGUFWriter(path, "qwen35")
    writer.add_tensor(convolution["name"], converted_conv)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    assert path.stat().st_size < 64 * 1024
    reader = gguf.GGUFReader(path)
    tensor = reader.tensors[0]
    file_keywords = {"reader": reader, "source_name": convolution["source_name"], "source_values": raw_conv,
                     "expected_source_sha256": hashlib.sha256(memoryview(raw_conv)).hexdigest(),
                     "expected_file_bytes": path.stat().st_size, "expected_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    receipt = verify_profile_gguf_convolution(profile, fixture["inventory"], fixture["manifest"], payload, **file_keywords)
    assert receipt["converted_sha256"] == conv_receipt["converted_sha256"]
    assert receipt["converted_file_verification"] == "caller_sha256_and_reader_range_match"
    assert receipt["source_file_verification"] == receipt["gguf_reader_verification"] == "not_checked"
    assert receipt["runtime_approval"] == "not_granted_by_convolution_file_verification"
    for defect in ("file-pin", "source-pin", "operand-budget", "copied-view", "shape", "reader-mode", "deadline"):
        changed = file_keywords.copy()
        original_tensors = reader.tensors
        original_clock = prism_codec.time.monotonic
        if defect == "file-pin":
            changed["expected_file_sha256"] = "0" * 64
        elif defect == "source-pin":
            changed["expected_source_sha256"] = "0" * 64
        elif defect == "operand-budget":
            changed["maximum_bytes"] = 1
        elif defect == "reader-mode":
            changed["reader"] = gguf.GGUFReader(path, mode="r+")
        elif defect == "deadline":
            prism_codec.time.monotonic = iter((0, 31)).__next__
        else:
            modified = tensor._replace(data=tensor.data.copy()) if defect == "copied-view" else tensor._replace(shape=tensor.shape[::-1])
            reader.tensors = [modified]
        try:
            verify_profile_gguf_convolution(profile, fixture["inventory"], fixture["manifest"], payload, **changed)
            raise AssertionError("invalid convolution file binding accepted: " + defect)
        except ValueError:
            pass
        finally:
            reader.tensors = original_tensors
            prism_codec.time.monotonic = original_clock
    with path.open("r+b") as output:
        output.seek(tensor.data_offset)
        saved = output.read(1)
        output.seek(tensor.data_offset)
        output.write(bytes([saved[0] ^ 1]))
    try:
        verify_profile_gguf_convolution(profile, fixture["inventory"], fixture["manifest"], payload,
                                       **{**file_keywords, "expected_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        raise AssertionError("changed convolution file payload accepted")
    except ValueError as error:
        assert "byte parity" in str(error)
    del tensor, reader
vector_operands = {}
for source_name in ("model.norm.weight", "model.layers.0.input_layernorm.weight",
                    "model.layers.1.self_attn.q_norm.weight", "model.layers.1.self_attn.k_norm.weight",
                    name + "norm.weight", name + "A_log", name + "dt_bias"):
    entry = next(item for item in plan["tensors"] if item["source_name"] == source_name)
    source = torch.tensor(np.arange(entry["source_shape"][0]) % 5 - 2, dtype=torch.float32)
    raw_source = source.to(torch.bfloat16).view(torch.uint16).numpy().copy() if entry["source_dtype"] == "BF16" else source.numpy().copy()
    converted = list(model.modify_tensors(source, source_name, 0))[0][1].numpy().copy()
    vector_operands[entry["name"]] = (source_name, raw_source, converted)
    keywords = {"source_name": source_name, "source_values": raw_source, "converted_values": converted,
                "expected_source_sha256": hashlib.sha256(memoryview(raw_source)).hexdigest()}
    receipt = verify_profile_converted_vector(profile, fixture["inventory"], fixture["manifest"], payload, **keywords)
    assert receipt["converted_sha256"] == hashlib.sha256(memoryview(converted)).hexdigest()
    assert receipt["source_file_verification"] == receipt["converted_file_verification"] == "not_checked"
    assert receipt["runtime_approval"] == "not_granted_by_operand_verification"
    try:
        receipt["source_sha256"] = "0" * 64
        raise AssertionError("mutable vector receipt")
    except TypeError:
        pass
    changed = converted.copy()
    zero = np.flatnonzero(changed == 0)
    if zero.size:
        changed[zero[0]] = np.float32(-0.0)
    else:
        changed.view(np.uint32)[0] ^= 1
    keywords["converted_values"] = changed
    try:
        verify_profile_converted_vector(profile, fixture["inventory"], fixture["manifest"], payload, **keywords)
        raise AssertionError("changed vector bytes were accepted")
    except ValueError as error:
        assert "byte parity" in str(error)
with TemporaryDirectory(prefix="jev-vector-gguf-") as directory:
    path = Path(directory) / "vectors.gguf"
    writer = gguf.GGUFWriter(path, "qwen35")
    writer.add_uint32("general.alignment", 32)
    for tensor_name, (_, _, converted) in vector_operands.items():
        writer.add_tensor(tensor_name, converted)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    assert path.stat().st_size < 64 * 1024
    reader = gguf.GGUFReader(path)
    file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(reader.tensors) == len(vector_operands) == 7
    for tensor in reader.tensors:
        source_name, raw_source, converted = vector_operands[tensor.name]
        assert tensor.tensor_type == gguf.GGMLQuantizationType.F32
        assert tensor.data.shape == converted.shape
        receipt = verify_profile_converted_vector(
            profile, fixture["inventory"], fixture["manifest"], payload,
            source_name=source_name, source_values=raw_source, converted_values=tensor.data,
            expected_source_sha256=hashlib.sha256(memoryview(raw_source)).hexdigest())
        assert receipt["converted_sha256"] == hashlib.sha256(memoryview(converted)).hexdigest()
        file_receipt = verify_profile_gguf_vector(
            profile, fixture["inventory"], fixture["manifest"], payload,
            reader=reader, source_name=source_name, source_values=raw_source,
            expected_source_sha256=hashlib.sha256(memoryview(raw_source)).hexdigest(),
            expected_file_bytes=path.stat().st_size, expected_file_sha256=file_sha256)
        assert file_receipt["converted_file_verification"] == "caller_sha256_and_reader_range_match"
        assert file_receipt["gguf_reader_verification"] == "not_checked"
        assert file_receipt["converted_tensor_offset"] == tensor.data_offset
        assert file_receipt["source_file_verification"] == "not_checked"
        file_keywords = {"reader": reader, "source_name": source_name, "source_values": raw_source,
                         "expected_source_sha256": hashlib.sha256(memoryview(raw_source)).hexdigest(),
                         "expected_file_bytes": path.stat().st_size, "expected_file_sha256": file_sha256}
        for defect in ("digest", "file-budget", "reader-mode", "missing", "duplicate", "type", "offset", "copied-view", "deadline"):
            changed_keywords = file_keywords.copy()
            original_tensors = reader.tensors
            original_clock = prism_codec.time.monotonic
            if defect == "digest":
                changed_keywords["expected_file_sha256"] = "0" * 64
            elif defect == "file-budget":
                changed_keywords["maximum_file_bytes"] = file_keywords["expected_file_bytes"] - 1
            elif defect == "reader-mode":
                changed_keywords["reader"] = gguf.GGUFReader(path, mode="r+")
            elif defect == "missing":
                reader.tensors = [item for item in reader.tensors if item.name != tensor.name]
            elif defect == "duplicate":
                reader.tensors = [*reader.tensors, tensor]
            elif defect == "deadline":
                prism_codec.time.monotonic = iter((0, 31)).__next__
            else:
                changed_tensor = tensor._replace(**{
                    "type": {"tensor_type": gguf.GGMLQuantizationType.BF16},
                    "offset": {"data_offset": tensor.data_offset + 4},
                    "copied-view": {"data": tensor.data.copy()},
                }[defect])
                reader.tensors = [changed_tensor if item.name == tensor.name else item for item in reader.tensors]
            try:
                verify_profile_gguf_vector(profile, fixture["inventory"], fixture["manifest"], payload, **changed_keywords)
                raise AssertionError("invalid GGUF vector binding was accepted: " + defect)
            except ValueError:
                pass
            finally:
                reader.tensors = original_tensors
                prism_codec.time.monotonic = original_clock
        for defect in ("stale-name", "stale-alignment"):
            changed_offset = tensor.field.offset + 8 if defect == "stale-name" else reader.fields["general.alignment"].parts[-1].ctypes.data - reader.data.ctypes.data
            with path.open("r+b") as output:
                output.seek(changed_offset)
                saved = output.read(1)
                output.seek(changed_offset)
                output.write(b"x" if defect == "stale-name" else b"@")
            changed_keywords = {**file_keywords, "expected_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            try:
                verify_profile_gguf_vector(profile, fixture["inventory"], fixture["manifest"], payload, **changed_keywords)
                raise AssertionError("stale GGUF metadata was accepted: " + defect)
            except ValueError:
                pass
            finally:
                with path.open("r+b") as output:
                    output.seek(changed_offset)
                    output.write(saved)
        original_verify = prism_codec.verify_profile_converted_vector
        def changed_during_parity(*arguments, **keywords):
            result = original_verify(*arguments, **keywords)
            with path.open("r+b") as output:
                output.seek(tensor.data_offset)
                value = output.read(1)
                output.seek(tensor.data_offset)
                output.write(bytes([value[0] ^ 1]))
            return result
        prism_codec.verify_profile_converted_vector = changed_during_parity
        try:
            verify_profile_gguf_vector(profile, fixture["inventory"], fixture["manifest"], payload, **file_keywords)
            raise AssertionError("mid-verification file mutation was accepted")
        except ValueError as error:
            assert "changed" in str(error)
        finally:
            prism_codec.verify_profile_converted_vector = original_verify
            with path.open("r+b") as output:
                output.seek(tensor.data_offset)
                output.write(converted.tobytes()[:1])
        with path.open("r+b") as output:
            output.seek(tensor.data_offset)
            original = output.read(1)
            output.seek(tensor.data_offset)
            output.write(bytes([original[0] ^ 1]))
        try:
            verify_profile_converted_vector(
                profile, fixture["inventory"], fixture["manifest"], payload,
                source_name=source_name, source_values=raw_source, converted_values=tensor.data,
                expected_source_sha256=hashlib.sha256(memoryview(raw_source)).hexdigest())
            raise AssertionError("changed GGUF vector bytes were accepted")
        except ValueError as error:
            assert "byte parity" in str(error)
        try:
            verify_profile_gguf_vector(profile, fixture["inventory"], fixture["manifest"], payload, **file_keywords)
            raise AssertionError("changed file identity was accepted")
        except ValueError as error:
            assert "file identity" in str(error)
        finally:
            with path.open("r+b") as output:
                output.seek(tensor.data_offset)
                output.write(original)
    del tensor, reader
matrix_count = 0
matrix_operands = {}
if next(item["source_dtype"] for item in plan["tensors"] if item["name"] == "output_norm.weight") == "BF16":
    model.hparams.update(plan["linear_head_layout"])
    model.hparams["hidden_size"] = profile.geometry["hidden_size"]
    for item in plan["tensors"]:
        if item["type"] != "BF16":
            continue
        source = (torch.arange(int(np.prod(item["source_shape"])), dtype=torch.float32).reshape(item["source_shape"]) % 17 - 8) / 16
        raw_source = source.to(torch.bfloat16).view(torch.uint16).numpy().copy()
        converted = list(model.modify_tensors(source, item["source_name"], 0))[0][1]
        raw_converted = converted.to(torch.bfloat16).view(torch.uint16).numpy().copy()
        keywords = {"source_name": item["source_name"], "source_values": raw_source, "converted_values": raw_converted,
                    "expected_source_sha256": hashlib.sha256(memoryview(raw_source)).hexdigest(), "maximum_bytes": 16 * 1024**2}
        receipt = verify_profile_converted_matrix(profile, fixture["inventory"], fixture["manifest"], payload, **keywords)
        matrix_operands[item["name"]] = (item["source_name"], raw_source, raw_converted)
        assert receipt["converted_sha256"] == hashlib.sha256(memoryview(raw_converted)).hexdigest()
        assert receipt["source_file_verification"] == receipt["converted_file_verification"] == "not_checked"
        changed = raw_converted.copy()
        changed.flat[0] ^= 0x8000
        keywords["converted_values"] = changed
        try:
            verify_profile_converted_matrix(profile, fixture["inventory"], fixture["manifest"], payload, **keywords)
            raise AssertionError("changed BF16 matrix bit was accepted")
        except ValueError as error:
            assert "byte parity" in str(error)
        matrix_count += 1
    assert matrix_count == 16
    with TemporaryDirectory(prefix="jev-matrix-gguf-") as directory:
        path = Path(directory) / "matrices.gguf"
        writer = gguf.GGUFWriter(path, "qwen35")
        for tensor_name, (_, _, values) in matrix_operands.items():
            writer.add_tensor(tensor_name, values.view(np.uint8), raw_dtype=gguf.GGMLQuantizationType.BF16)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        assert path.stat().st_size <= 16 * 1024**2
        file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        reader = gguf.GGUFReader(path)
        assert len(reader.tensors) == len(matrix_operands) == 16
        for tensor in reader.tensors:
            source_name, raw_source, raw_converted = matrix_operands[tensor.name]
            keywords = {"reader": reader, "source_name": source_name, "source_values": raw_source,
                        "expected_source_sha256": hashlib.sha256(memoryview(raw_source)).hexdigest(),
                        "expected_file_bytes": path.stat().st_size, "expected_file_sha256": file_sha256,
                        "maximum_bytes": 16 * 1024**2}
            receipt = verify_profile_gguf_matrix(profile, fixture["inventory"], fixture["manifest"], payload, **keywords)
            assert receipt["converted_sha256"] == hashlib.sha256(memoryview(raw_converted)).hexdigest()
            assert receipt["converted_file_verification"] == "caller_sha256_and_reader_range_match"
            assert receipt["source_file_verification"] == receipt["gguf_reader_verification"] == "not_checked"
            for defect in ("file-pin", "operand-budget", "copied-view", "shape"):
                changed = keywords.copy()
                original_tensors = reader.tensors
                if defect == "file-pin":
                    changed["expected_file_sha256"] = "0" * 64
                elif defect == "operand-budget":
                    changed["maximum_bytes"] = 1
                else:
                    modified = tensor._replace(data=tensor.data.copy()) if defect == "copied-view" else tensor._replace(shape=tensor.shape[::-1])
                    if defect == "shape" and np.array_equal(modified.shape, tensor.shape):
                        modified = tensor._replace(shape=tensor.shape + 1)
                    reader.tensors = [modified if item.name == tensor.name else item for item in reader.tensors]
                try:
                    verify_profile_gguf_matrix(profile, fixture["inventory"], fixture["manifest"], payload, **changed)
                    raise AssertionError("invalid BF16 file binding accepted: " + defect)
                except ValueError:
                    pass
                finally:
                    reader.tensors = original_tensors
            with path.open("r+b") as output:
                output.seek(tensor.data_offset)
                saved = output.read(1)
                output.seek(tensor.data_offset)
                output.write(bytes([saved[0] ^ 1]))
            changed = {**keywords, "expected_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            try:
                verify_profile_gguf_matrix(profile, fixture["inventory"], fixture["manifest"], payload, **changed)
                raise AssertionError("changed BF16 file payload accepted")
            except ValueError as error:
                assert "byte parity" in str(error)
            finally:
                with path.open("r+b") as output:
                    output.seek(tensor.data_offset)
                    output.write(saved)
        del tensor, reader
print(json.dumps({"qkv_shape": list(result[0][1].shape), "v_head_order": [0, 2, 1, 3], "matrix_operands": matrix_count}))
"""
    result = subprocess.run(
        [interpreter, "-c", code, str(source), json.dumps(plan)], check=True,
        capture_output=True, text=True, input=fixture, timeout=15,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(Path(__file__).resolve().parents[1]), str(source / "gguf-py")))},
    )
    assert json.loads(result.stdout) == {"qkv_shape": [16, 8], "v_head_order": [0, 2, 1, 3], "matrix_operands": 16 if vector_dtype == "BF16" else 0}


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