"""Optional pinned Prism CPU transform graph control without model weights."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


PRISM_REVISION = "842b1880415d6f508f03b789e5ce70194def7bfd"
BITNET_REVISION = "0b341e582afbf9e1011f24744b554c96a3477eb5"


@pytest.mark.parametrize("backend", ["dense", "bitnet"])
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
    writer_code = """
import json
import sys
import gguf
import numpy as np
writer = gguf.GGUFWriter(sys.argv[1], "qwen35")
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
        if sys.argv[2] == "bitnet_invalid_code":
            fields[0, 0, 0, 0] = 3
        elif sys.argv[2] == "bitnet_negative_scale":
            scales[0, 0] = -1
        elif sys.argv[2] == "bitnet_nonfinite_scale":
            scales[0, 0] = np.nan
        payload = np.bitwise_or.reduce(fields << (2 * np.arange(4, dtype=np.uint8)), axis=-1)
        encoded = np.concatenate((scales.view(np.uint8).reshape(32, 2, 2), payload), axis=-1)
        writer.add_tensor(prefix + "ffn_down.weight", encoded.reshape(32, 68),
                          raw_dtype=gguf.GGMLQuantizationType.PQ2_0)
    else:
        writer.add_tensor(prefix + "ffn_down.weight", codes.astype(np.float32) * np.repeat(scales, 128, axis=1))
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
if sys.argv[2] not in ("dense", "bitnet"):
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
    if sys.argv[2] == "bitnet":
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
                                 "nonzero" if recurrent else "zero", "nonzero" if attention else "zero"], capture_output=True, text=True,
                                timeout=30, env={**os.environ, "PYTHONPATH": str(Path(source_dir) / "gguf-py"),
                                                "PYTHONDONTWRITEBYTECODE": "1"})
        assert writer.returncode == 0, writer.stderr
        expected = json.loads(writer.stdout)
        modes = ("full", "chunked", "reset", "reordered", "parallel", "maximum")
        if backend == "bitnet":
            modes += ("kernel-recovery",)
        for mode in modes:
            control = subprocess.run([str(Path(runtime_build) / "bin" / "prism_bitnet_loader_control"),
                                      "--prefill-bitnet-control" if backend == "bitnet" else "--prefill-control",
                                      str(model_file), mode], capture_output=True, text=True, timeout=30)
            assert control.returncode == 0, control.stderr
            report = json.loads(control.stdout)
            if mode == "full":
                untraced_report = report
            token_ids = [2] * 128 if mode == "maximum" else [3, 5, 7]
            assert report["prefilled_tokens"] == len(token_ids) and report["generated_answer_tokens"] == 0
            assert report["input_tokens"] == token_ids
            assert report["backend"] == backend and report["mode"] == mode
            calls = 3 if mode == "parallel" else 2 if mode in ("chunked", "reset") else 1
            assert report["prefill_calls"] == calls
            assert report["rejected_prefill_calls"] == (mode == "kernel-recovery")
            assert report["kernel_error_status"] == (1 if mode == "kernel-recovery" else 0)
            assert report["bitnet_dispatch_calls"] == (calls if backend == "bitnet" else 0)
            assert report["weight_repacks"] == (backend == "bitnet")
            input_count = 2 if mode == "chunked" else len(token_ids)
            assert report["bitnet_last_input_tokens"] == (input_count if backend == "bitnet" else 0)
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
        command = "--prefill-bitnet-control" if backend == "bitnet" else "--prefill-control"
        trace_path = tmp_path / "trace"
        trace_command = "--prefill-bitnet-trace-control" if backend == "bitnet" else "--prefill-trace-control"
        traced = subprocess.run([str(Path(runtime_build) / "bin" / "prism_bitnet_loader_control"),
                                 trace_command, str(model_file), str(trace_path)],
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
                                  trace_command, str(model_file), str(trace_path)],
                                 capture_output=True, text=True, timeout=30)
        assert refused.returncode == 35 and refused.stdout == ""
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
        if backend == "bitnet":
            rejected = subprocess.run([str(Path(runtime_build) / "bin" / "prism_bitnet_loader_control"),
                                       command, str(model_file), "kernel-error"], capture_output=True, text=True, timeout=30)
            assert rejected.returncode == 29 and rejected.stdout == "", rejected.stderr
        rejected = subprocess.run([str(Path(runtime_build) / "bin" / "prism_bitnet_loader_control"),
                                   command, str(model_file), "unknown"], capture_output=True, text=True, timeout=30)
        assert rejected.returncode == 32 and rejected.stdout == ""
        for arguments in ([command], [command, str(model_file), "full", "extra"],
                          ["--unknown"], ["--vocab-only", str(model_file)]):
            rejected = subprocess.run([str(Path(runtime_build) / "bin" / "prism_bitnet_loader_control"),
                                       *arguments], capture_output=True, text=True, timeout=30)
            assert rejected.returncode == 2 and rejected.stdout == ""
        corruptions = (
            ("bitnet_invalid_code", "bitnet_negative_scale", "bitnet_nonfinite_scale")
            if backend == "bitnet" else ("dense_nan",)
        )
        for corruption in corruptions:
            writer = subprocess.run([converter, "-c", writer_code, str(model_file), corruption,
                                     "nonzero" if recurrent else "zero", "nonzero" if attention else "zero"],
                                    capture_output=True, text=True, timeout=30,
                                    env={**os.environ, "PYTHONPATH": str(Path(source_dir) / "gguf-py"),
                                         "PYTHONDONTWRITEBYTECODE": "1"})
            assert writer.returncode == 0, writer.stderr
            rejected = subprocess.run([str(Path(runtime_build) / "bin" / "prism_bitnet_loader_control"),
                                       command, str(model_file)], capture_output=True, text=True, timeout=30)
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
    projection_storage = native.prism_bitnet_registered_projection_storage_bytes_v1
    projection_storage.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    projection_storage.restype = ctypes.c_int

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