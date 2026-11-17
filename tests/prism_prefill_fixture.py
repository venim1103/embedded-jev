"""Generated prefill fixture helpers, importable without pytest or model weights."""

import json
from pathlib import Path


def _tiny_prefill_writer_code():
    return """
import json
import sys
import gguf
import numpy as np
owned = sys.argv[2] == "bitnet-owned-request"
typed = sys.argv[2] in ("dense-request", "bitnet-request", "bitnet-owned-request")
intermediate = 1024 if typed else 256
groups = intermediate // 128
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
if typed:
    from prism_prefill_fixture import _typed_prefill_metadata
    typed_fixture = _typed_prefill_metadata(writer, sys.argv[1])
else:
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
if owned:
    from embedded_jev.prism_codec import profile_model_specification
    expected_types = {item["name"]: item["type"] for item in profile_model_specification(*typed_fixture[:4])["tensors"]}
def rounded_bf16(values):
    encoded = gguf.quants.quantize(np.ascontiguousarray(values), gguf.GGMLQuantizationType.BF16)
    return gguf.quants.dequantize(encoded, gguf.GGMLQuantizationType.BF16)
def add_tensor(name, values, **kwargs):
    if owned:
        kind = expected_types[name]
        if kind == "BF16":
            assert not kwargs
            values = gguf.quants.quantize(values, gguf.GGMLQuantizationType.BF16)
            kwargs["raw_dtype"] = gguf.GGMLQuantizationType.BF16
        elif kind == "PQ2_0":
            assert kwargs == {"raw_dtype": gguf.GGMLQuantizationType.PQ2_0}
        else:
            assert kind == "F32" and values.dtype == np.float32 and not kwargs
    writer.add_tensor(name, values, **kwargs)
def dense_projection(weights, values):
    return weights @ (rounded_bf16(values) if owned else values)
generator = np.random.default_rng(1407)
embedding = generator.normal(0, 0.2, (64, 32)).astype(np.float32)
head = generator.normal(0, 0.1, (64, 32)).astype(np.float32)
if owned:
    embedding, head = rounded_bf16(embedding), rounded_bf16(head)
if sys.argv[2] == "dense_nan":
    head[11, 0] = np.nan
add_tensor("token_embd.weight", embedding)
add_tensor("output.weight", head)
add_tensor("output_norm.weight", np.ones(32, dtype=np.float32))
codes = ((np.arange(32 * intermediate).reshape(32, intermediate) * 7) % 3 - 1).astype(np.int8)
scales = ((np.arange(32 * groups).reshape(32, groups) % 7 + 1) / 256).astype(np.float16)
gate = np.zeros((intermediate, 32), dtype=np.float32)
gate[:, 0] = 0.25
up = np.zeros((intermediate, 32), dtype=np.float32)
up[np.arange(intermediate), (np.arange(intermediate) * 7) % 32] = np.repeat(np.tile([0.125, 0.375], groups // 2), 128)
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
    add_tensor(prefix + "attn_norm.weight", np.ones(32, dtype=np.float32))
    add_tensor(prefix + "post_attention_norm.weight", np.ones(32, dtype=np.float32))
    if layer % 2:
        for name, shape in {"attn_q.weight": (64, 32), "attn_k.weight": (16, 32),
                            "attn_v.weight": (16, 32), "attn_output.weight": (32, 32)}.items():
            values = np.zeros(shape, dtype=np.float32)
            if layer == 1 and sys.argv[4] == "nonzero":
                values = {"attn_q.weight": attention_query, "attn_k.weight": attention_key,
                          "attn_v.weight": attention_value, "attn_output.weight": attention_out}[name]
            add_tensor(prefix + name, values)
        add_tensor(prefix + "attn_q_norm.weight", np.ones(8, dtype=np.float32))
        add_tensor(prefix + "attn_k_norm.weight", np.ones(8, dtype=np.float32))
    else:
        for name, shape in {
            "attn_qkv.weight": (64, 32), "attn_gate.weight": (32, 32), "ssm_conv1d.weight": (64, 3),
            "ssm_beta.weight": (4, 32), "ssm_alpha.weight": (4, 32), "ssm_out.weight": (32, 32),
        }.items():
            values = np.zeros(shape, dtype=np.float32)
            if layer == 0 and sys.argv[3] == "nonzero":
                values = {"attn_qkv.weight": recurrent_qkv, "attn_gate.weight": recurrent_gate,
                          "ssm_conv1d.weight": convolution, "ssm_out.weight": recurrent_out}.get(name, values)
            add_tensor(prefix + name, values)
        add_tensor(prefix + "ssm_norm.weight", np.ones(8, dtype=np.float32))
        add_tensor(prefix + "ssm_dt.bias", np.zeros(4, dtype=np.float32))
        add_tensor(prefix + "ssm_a", -np.ones(4, dtype=np.float32))
    add_tensor(prefix + "ffn_gate.weight", gate if layer == 3 else np.zeros_like(gate))
    add_tensor(prefix + "ffn_up.weight", up if layer == 3 else np.zeros_like(up))
    if layer != 3:
        add_tensor(prefix + "ffn_down.weight", np.zeros((32, intermediate), dtype=np.float32))
    elif sys.argv[2].startswith("bitnet"):
        fields = (codes.reshape(32, groups, 32, 4) + 1).astype(np.uint8)
        if sys.argv[2].endswith("_invalid_code"):
            fields[0, 0, 0, 0] = 3
        elif sys.argv[2].endswith("_negative_scale"):
            scales[0, 0] = -1
        elif sys.argv[2].endswith("_nonfinite_scale"):
            scales[0, 0] = np.nan
        elif sys.argv[2].endswith("_changed_scale"):
            scales[0, 0] = np.float16(0.25)
        payload = np.bitwise_or.reduce(fields << (2 * np.arange(4, dtype=np.uint8)), axis=-1)
        encoded = np.concatenate((scales.view(np.uint8).reshape(32, groups, 2), payload), axis=-1)
        add_tensor(prefix + "ffn_down.weight", encoded.reshape(32, groups * 34),
                          raw_dtype=gguf.GGMLQuantizationType.PQ2_0)
        if sys.argv[2] in ("bitnet-profile", "bitnet-request"):
            reference = gguf.GGUFWriter(sys.argv[5], "qwen35")
            reference.add_string("jev.bitnet.execution", "group128-a8-fp32-nearest-even-identity-v1")
            reference.add_tensor(prefix + "ffn_down.weight", encoded.reshape(32, groups * 34),
                                 raw_dtype=gguf.GGMLQuantizationType.PQ2_0)
            reference.write_header_to_file()
            reference.write_kv_data_to_file()
            reference.write_tensors_to_file()
            reference.close()
    else:
        add_tensor(prefix + "ffn_down.weight", codes.astype(np.float32) * np.repeat(scales, 128, axis=1))
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
if sys.argv[2] not in ("dense", "bitnet", "bitnet-profile", "dense-request", "bitnet-request", "bitnet-owned-request"):
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
            history[-1] = dense_projection(recurrent_qkv, normalized)
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
            gating = dense_projection(recurrent_gate, normalized)
            output = output.reshape(32) * (gating / (np.float32(1) + np.exp(-gating)))
            hidden = current + dense_projection(recurrent_out, output)
            sequence[index] = hidden
        assert np.linalg.norm(hidden - embedding[token_ids[-1]]) > 0.01
    if sys.argv[4] == "nonzero":
        normalized = sequence / np.sqrt(np.mean(sequence * sequence, axis=1, keepdims=True) + np.float32(1e-5))
        query = dense_projection(attention_query, normalized.T).T[:, np.arange(4) * 16]
        query /= np.sqrt(query * query / np.float32(8) + np.float32(1e-5))
        key = dense_projection(attention_key, normalized.T).T[:, np.arange(2) * 8]
        key /= np.sqrt(key * key / np.float32(8) + np.float32(1e-5))
        positions = np.arange(len(token_ids), dtype=np.float32)
        key_cosine = (key * np.cos(positions[:, None])).astype(np.float16).astype(np.float32)
        key_sine = (key * np.sin(positions[:, None])).astype(np.float16).astype(np.float32)
        value = dense_projection(attention_value, normalized.T).T.astype(np.float16).astype(np.float32).reshape(len(token_ids), 2, 8)
        original = sequence.copy()
        for index in range(len(token_ids)):
            output = np.zeros((4, 8), dtype=np.float32)
            for head_index in range(4):
                key_head = head_index // 2
                cosine = query[index, head_index] * np.cos(positions[index])
                sine = query[index, head_index] * np.sin(positions[index])
                if owned:
                    cosine, sine = np.float32(np.float16(cosine)), np.float32(np.float16(sine))
                scores = (cosine * key_cosine[:index + 1, key_head] + sine * key_sine[:index + 1, key_head]) / np.sqrt(np.float32(8))
                scores = np.exp(scores - np.max(scores))
                scores /= np.sum(scores)
                if owned:
                    scores = scores.astype(np.float16).astype(np.float32)
                output[head_index] = np.sum(value[:index + 1, key_head] * scores[:, None], axis=0)
            sequence[index] += dense_projection(attention_out, output.reshape(32) * np.float32(0.5))
        assert np.linalg.norm(sequence - original) > 0.01
        hidden = sequence[-1]
    normalized = hidden / np.sqrt(np.mean(hidden * hidden) + np.float32(1e-5))
    gated = dense_projection(gate, normalized)
    activation = (gated / (np.float32(1) + np.exp(-gated))) * dense_projection(up, normalized)
    if sys.argv[2].startswith("bitnet"):
        activation_groups = activation.reshape(groups, 128)
        activation_scales = np.max(np.abs(activation_groups), axis=1) / np.float32(127)
        activation_scales[activation_scales == 0] = 1
        quantized = np.clip(np.rint(activation_groups / activation_scales[:, None]), -127, 127).astype(np.int32)
        partials = np.sum(codes.reshape(32, groups, 128).astype(np.int32) * quantized[None], axis=-1)
        hidden += np.sum(partials.astype(np.float32) * scales.astype(np.float32) * activation_scales, axis=1)
    else:
        hidden += (codes.astype(np.float32) * np.repeat(scales, 128, axis=1)) @ activation
    hidden = hidden / np.sqrt(np.mean(hidden * hidden) + np.float32(1e-5))
    return dense_projection(head if typed else head[[11, 17, 23]], hidden).tolist()
if typed:
    from prism_prefill_fixture import _finish_typed_prefill_fixture
    _finish_typed_prefill_fixture(sys.argv[1], typed_fixture, reference_logits)
    sys.exit(0)
print(json.dumps({"logits": reference_logits([3, 5, 7]),
                  "peer_logits": reference_logits([4, 9, 19, 3, 5, 7]),
                  "text_logits": reference_logits([4, 2]),
                  "continuation_logits": reference_logits([3, 5, 7] * 26 + [3, 5]),
                  "uncleared_reset_logits": reference_logits([4, 9, 19] * 26 + [4, 9, 3, 5, 7]),
                  "interleaved_peer_logits": reference_logits([4, 9, 19] * 26 + [4, 9, 7, 3, 5]),
                  "maximum_logits": reference_logits([2] * 128)}))
"""



def typed_prefill_tokenizer_bytes():
    tokens = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") + ["\u0120", "\u010a", ":", ".", "?", "ab", "<pad>", "<|control|>"]
    tokens += [f"token{index}" for index in range(len(tokens), 64)]
    tokens[60] = "abA"
    return json.dumps({"version": "1.0", "added_tokens": [
        {"id": tokens.index(text), "content": text, "single_word": False, "lstrip": False, "rstrip": False,
         "normalized": False, "special": True} for text in ("<pad>", "<|control|>")],
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "vocab": {token: index for index, token in enumerate(tokens)},
            "merges": [["a", "b"], ["ab", "A"]], "dropout": None, "unk_token": None,
            "continuing_subword_prefix": None, "end_of_word_suffix": None, "fuse_unk": False,
            "byte_fallback": False, "ignore_merges": False}}).encode()


def _typed_prefill_metadata(writer, path):
    import numpy as np
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    from embedded_jev.label_probe import prepare_profile_prefill_request
    from embedded_jev.model_profile import parse_model_profile
    from embedded_jev.prism_codec import write_profile_model_selected_fast_metadata_v5

    root = Path(path).parent
    fixture = json.loads((root / "fixture.json").read_bytes())
    profile = parse_model_profile(json.dumps(fixture["profile"]).encode())
    inventory, manifest = fixture["inventory"], fixture["manifest"]
    blocks = np.frombuffer(bytes.fromhex(fixture["payload_hex"]), dtype=np.uint8).reshape(32, 8, 34)
    metadata = {name: (root / "inputs" / name).read_bytes() for name in profile.packaging["metadata_files"]}
    tokenizer_bytes = (root / "inputs/tokenizer.json").read_bytes()
    source = Tokenizer.from_str(tokenizer_bytes.decode())
    template = metadata[profile.tokenization["template_file"]]
    roles = {"<pad>": "source_special", "<|control|>": "source_special"}
    arguments = dict(metadata_files=metadata, tokenizer_bytes=tokenizer_bytes, context_length=256, template_source="source",
        row_policy="pinned_prism_fast_rows_v2", merge_policy="pinned_prism_ranked_source_merges_v2", token_roles=roles,
        bootstrap_policy={"policy_version": 1, "profile_sha256": profile.sha256, "tokenizer_model": "gpt2", "pre_tokenizer_tag": "qwen35",
            "bos_token_id": 59, "eos_token_id": 59, "unknown_token_id": 58, "padding_token_id": 58,
            "add_bos_token": False, "add_eos_token": False}, maximum_seconds=12)
    emitted = write_profile_model_selected_fast_metadata_v5(profile, inventory, manifest, blocks, writer=writer, **arguments)
    assert emitted["metadata_count"] == 37
    wrapper = PreTrainedTokenizerFast(tokenizer_object=source, additional_special_tokens=list(roles), clean_up_tokenization_spaces=False)
    wrapper.chat_template = template.decode()
    assert wrapper.encode("ab", add_special_tokens=False) == [57]
    assert wrapper.encode("abA", add_special_tokens=False) == [60]
    packets = {}
    for count in (2, 3, 16):
        request = {"id": f"request-{count}", "state": "a", "question": "b",
            "options": [{"id": f"action-{index}", "description": "a" if index % 2 else "b"} for index in range(count)]}
        request["options"][0]["id"] = 'inspect"\\\u00e9'
        packets[str(count)] = prepare_profile_prefill_request(wrapper, request, profile=profile)
        if count == 3:
            request["options"].reverse()
            packets["reordered"] = prepare_profile_prefill_request(wrapper, request, profile=profile)
    for name, packet in packets.items():
        (root / f"request-{name}.json").write_text(json.dumps(packet), encoding="utf-8")
    return profile, inventory, manifest, blocks, arguments, packets



def _finish_typed_prefill_fixture(path, fixture, reference_logits):
    import hashlib

    import gguf
    from embedded_jev.prism_codec import verify_profile_gguf_selected_fast_metadata_v5

    profile, inventory, manifest, blocks, arguments, packets = fixture
    path = Path(path)
    contents = path.read_bytes()
    assert len(contents) <= 2 * 1024**2
    reader = gguf.GGUFReader(path, mode="r")
    try:
        header_bytes = int(reader.data_offset)
        header_sha256 = hashlib.sha256(contents[:header_bytes]).hexdigest()
        verified = verify_profile_gguf_selected_fast_metadata_v5(profile, inventory, manifest, blocks, reader=reader,
            expected_file_bytes=len(contents), expected_file_sha256=hashlib.sha256(contents).hexdigest(),
            maximum_file_bytes=2 * 1024**2, **arguments)
        assert verified["metadata_count"] == 38
        declarations = [{**record, "values": list(record["values"])} for record in verified["records"]]
        (path.parent / "metadata.json").write_text(json.dumps(declarations), encoding="utf-8")
        (path.parent / "oracle.json").write_text(json.dumps({
            **{name: reference_logits(packet["input_token_ids"]) for name, packet in packets.items()},
            "legacy": reference_logits([3, 5, 7])}), encoding="utf-8")
    finally:
        reader.data._mmap.close()
    print(json.dumps({"fresh_selected_records": 38, "requests": len(packets), "model_bytes": len(contents),
        "header_bytes": header_bytes, "header_sha256": header_sha256}))
