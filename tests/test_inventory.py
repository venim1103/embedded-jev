"""Offline checks for bounded, revision-pinned model inventory."""

import io
import json

import numpy as np
import pytest

from embedded_jev import inventory
from embedded_jev import weight_slice
from embedded_jev.inventory import (
    InventoryError,
    TensorHeader,
    build_inventory,
    parse_safetensors_header,
)
from embedded_jev.ternary import quantize_ternary_rtn
from embedded_jev.ternary_artifact import (
    TernaryArtifactError,
    mimo_prism_expected_widths,
    prism_v1_transform_metadata,
    save_toy_artifact,
)


def _header(specs, payload_bytes):
    header = json.dumps(specs, sort_keys=True).encode("utf-8")
    prefix = len(header).to_bytes(8, "little") + header
    return prefix, len(prefix) + payload_bytes


def test_header_counts_tensors_without_weight_data():
    prefix, shard_bytes = _header(
        {
            "model.embed_tokens.weight": {
                "dtype": "BF16",
                "shape": [2, 4],
                "data_offsets": [0, 16],
            },
            "model.layers.0.weight": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [16, 32],
            },
        },
        32,
    )
    assert parse_safetensors_header(prefix, shard_bytes) == (
        TensorHeader("model.embed_tokens.weight", (2, 4), "BF16", 8, 16),
        TensorHeader("model.layers.0.weight", (2, 2), "F32", 4, 16),
    )


def test_header_rejects_missing_or_inconsistent_data():
    prefix, shard_bytes = _header(
        {"weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}, 4
    )
    with pytest.raises(InventoryError, match="header length"):
        parse_safetensors_header(prefix[:6], shard_bytes)
    with pytest.raises(InventoryError, match="shard size"):
        parse_safetensors_header(prefix, shard_bytes + 1)
    with pytest.raises(InventoryError, match="allowed bounds"):
        parse_safetensors_header(prefix[:8], shard_bytes, max_header_bytes=1)


def test_header_rejects_unsupported_dtype_and_overlapping_offsets():
    prefix, shard_bytes = _header(
        {"weight": {"dtype": "F4", "shape": [2], "data_offsets": [0, 4]}}, 4
    )
    with pytest.raises(InventoryError, match="unsupported safetensors dtype"):
        parse_safetensors_header(prefix, shard_bytes)
    prefix, shard_bytes = _header(
        {
            "a": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
            "b": {"dtype": "BF16", "shape": [1], "data_offsets": [1, 3]},
        },
        3,
    )
    with pytest.raises(InventoryError, match="gap"):
        parse_safetensors_header(prefix, shard_bytes)
    prefix, shard_bytes = _header(
        {"weight": {"dtype": "BF16", "shape": [1] * 17, "data_offsets": [0, 2]}}, 2
    )
    with pytest.raises(InventoryError, match="invalid tensor shape"):
        parse_safetensors_header(prefix, shard_bytes)


def _model_fixture():
    text = {
        "hidden_size": 1024,
        "vocab_size": 32,
        "intermediate_size": 2048,
        "num_hidden_layers": 1,
        "layer_types": ["full_attention"],
        "num_key_value_heads": 2,
        "head_dim": 64,
        "mtp_num_hidden_layers": 1,
    }
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "tie_word_embeddings": False,
        "text_config": text,
        "vision_config": {"depth": 1},
    }
    specs = {
        "lm_head.weight": ("BF16", [32, 1024]),
        "model.language_model.embed_tokens.weight": ("BF16", [32, 1024]),
        "model.language_model.norm.weight": ("BF16", [1024]),
        "model.visual.patch_embed.proj.weight": ("BF16", [4, 4]),
        "model.visual.patch_embed.proj.bias": ("BF16", [4]),
        "model.visual.pos_embed.weight": ("BF16", [1, 4]),
    }
    for module in ("norm", "linear_fc1", "linear_fc2"):
        specs[f"model.visual.merger.{module}.weight"] = ("BF16", [4, 4])
        specs[f"model.visual.merger.{module}.bias"] = ("BF16", [4])
    for module in ("attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2", "norm1", "norm2"):
        specs[f"model.visual.blocks.0.{module}.weight"] = ("BF16", [4, 4])
        specs[f"model.visual.blocks.0.{module}.bias"] = ("BF16", [4])
    base = "model.language_model.layers.0."
    for name in ("input_layernorm", "post_attention_layernorm"):
        specs[base + name + ".weight"] = ("BF16", [1024])
    for name, shape in (
        ("mlp.down_proj.weight", [1024, 2048]),
        ("mlp.gate_proj.weight", [2048, 1024]),
        ("mlp.up_proj.weight", [2048, 1024]),
        ("self_attn.q_proj.weight", [1024, 1024]),
        ("self_attn.k_proj.weight", [128, 1024]),
        ("self_attn.v_proj.weight", [128, 1024]),
        ("self_attn.o_proj.weight", [1024, 1024]),
        ("self_attn.q_norm.weight", [64]),
        ("self_attn.k_norm.weight", [64]),
    ):
        specs[base + name] = ("BF16", shape)
    offsets = 0
    entries = {}
    for name, (dtype, shape) in sorted(specs.items()):
        count = 1
        for dimension in shape:
            count *= dimension
        end = offsets + count * 2
        entries[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offsets, end]}
        offsets = end
    prefix, shard_bytes = _header(entries, offsets)
    shard = "model-00001-of-00001.safetensors"
    metadata = {
        "config.json": json.dumps(config).encode(),
        "model.safetensors.index.json": json.dumps(
            {"metadata": {"total_size": offsets}, "weight_map": dict.fromkeys(specs, shard)}
        ).encode(),
        "tokenizer_config.json": b'{"tokenizer_class":"Qwen2Tokenizer"}',
        "processor_config.json": (
            b'{"processor_class":"Qwen3VLProcessor",'
            b'"image_processor":{"image_processor_type":"Qwen2VLImageProcessor"},'
            b'"video_processor":{"video_processor_type":"Qwen3VLVideoProcessor","fps":2}}'
        ),
        "preprocessor_config.json": b'{"image_processor_type":"Qwen2VLImageProcessor"}',
        "chat_template.jinja": b"{{ messages }}",
        "video_preprocessor_config.json": (
            b'{"video_processor_type":"Qwen3VLVideoProcessor",'
            b'"processor_class":"Qwen3VLProcessor"}'
        ),
    }
    return metadata, {shard: (prefix, shard_bytes)}


def test_inventory_reconciles_and_estimates_bytes_deterministically():
    metadata, shards = _model_fixture()
    report = build_inventory(metadata, shards)
    assert report == build_inventory(metadata, shards)
    assert report["totals"]["tensors"] == 35
    assert report["totals"]["stored_weight_bytes"] == report["accounting"]["index_total_size"]
    assert report["accounting"]["header_overhead_bytes"] == len(next(iter(shards.values()))[0])
    assert report["accounting"]["optional_mtp_configured_layers"] == 1
    assert not report["accounting"]["optional_mtp_tensors_present"]
    assert report["metadata_summary"]["processor_class"] == "Qwen3VLProcessor"
    candidate = next(t for t in report["tensors"] if t["name"].endswith("mlp.down_proj.weight"))
    assert candidate["quantization_eligible"]
    assert report["memory_estimates"]["projection_blocks_bytes"]["PTQ1_0"] == (
        report["totals"]["eligible_projection_parameters"] // 128 * 28
    )
    assert report["memory_estimates"]["vocabulary_bytes"]["selected_16_original_dtype"] == (
        16 * 1024 * 2
    )
    assert report["memory_estimates"]["vision_original_bytes"] > 0
    assert report["memory_estimates"]["rotation_sign_upper_bound_bytes_excluded_from_weight_totals"] > 0


def test_inventory_rejects_missing_inconsistent_or_unsupported_metadata():
    metadata, shards = _model_fixture()
    with pytest.raises(InventoryError, match="missing model metadata"):
        build_inventory({key: value for key, value in metadata.items() if key != "config.json"}, shards)
    wrong_index = json.loads(metadata["model.safetensors.index.json"])
    wrong_index["metadata"]["total_size"] -= 2
    with pytest.raises(InventoryError, match="index total_size"):
        build_inventory({**metadata, "model.safetensors.index.json": json.dumps(wrong_index).encode()}, shards)
    wrong_index["metadata"]["total_size"] += 2
    del wrong_index["weight_map"]["lm_head.weight"]
    with pytest.raises(InventoryError, match="index/header tensor mismatch"):
        build_inventory({**metadata, "model.safetensors.index.json": json.dumps(wrong_index).encode()}, shards)
    config = json.loads(metadata["config.json"])
    config["tie_word_embeddings"] = True
    with pytest.raises(InventoryError, match="tied or unspecified"):
        build_inventory({**metadata, "config.json": json.dumps(config).encode()}, shards)
    with pytest.raises(InventoryError, match="missing or unsupported tokenizer/processor"):
        build_inventory({**metadata, "processor_config.json": b"{}"}, shards)
    with pytest.raises(InventoryError, match="inconsistent image processor metadata"):
        build_inventory(
            {**metadata, "preprocessor_config.json": b'{"image_processor_type":"Qwen2VLImageProcessor","size":1}'},
            shards,
        )
    with pytest.raises(InventoryError, match="inconsistent video processor metadata"):
        build_inventory(
            {**metadata, "video_preprocessor_config.json": b'{"video_processor_type":"Qwen3VLVideoProcessor","processor_class":"Qwen3VLProcessor","fps":3}'},
            shards,
        )
    config = json.loads(metadata["config.json"])
    config["vision_config"]["depth"] = 2
    with pytest.raises(InventoryError, match="missing required model weights"):
        build_inventory({**metadata, "config.json": json.dumps(config).encode()}, shards)
    index = json.loads(metadata["model.safetensors.index.json"])
    index["weight_map"] = dict.fromkeys(index["weight_map"], "model-00001-of-00002.safetensors")
    with pytest.raises(InventoryError, match="incomplete or inconsistent safetensors shard numbering"):
        build_inventory(
            {**metadata, "model.safetensors.index.json": json.dumps(index).encode()},
            {"model-00001-of-00002.safetensors": next(iter(shards.values()))},
        )


class FakeResponse(io.BytesIO):
    def __init__(self, body, *, status=206, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}
        self.read_calls = 0

    def read(self, size=-1):
        self.read_calls += 1
        return super().read(size)

    def geturl(self):
        return "https://cdn.example.test/pinned-file"


def test_range_reader_refuses_full_shard_response_without_reading_body(monkeypatch):
    response = FakeResponse(b"not a shard", status=200, headers={"Content-Length": "5000000000"})
    requests = []

    def fake_urlopen(request, *, timeout):
        requests.append(request)
        assert timeout == 20
        return response

    monkeypatch.setattr(inventory, "urlopen", fake_urlopen)
    with pytest.raises(InventoryError, match="server ignored bounded range"):
        inventory._open_bounded("model-00001-of-00001.safetensors", start=0, length=8)
    assert response.read_calls == 0
    assert requests[0].get_header("Range") == "bytes=0-7"
    assert inventory.MODEL_REVISION in requests[0].full_url


def test_range_reader_validates_offsets_sizes_and_limits(monkeypatch):
    responses = iter(
        (
            FakeResponse(b"12345678", headers={"Content-Range": "bytes 0-7/100", "Content-Length": "8"}),
            FakeResponse(b"abcdefgh", headers={"Content-Range": "bytes 1-8/100"}),
            FakeResponse(b"abcdefgh", headers={"Content-Range": "bytes 0-7/100", "Content-Length": "9"}),
        )
    )
    monkeypatch.setattr(inventory, "urlopen", lambda request, *, timeout: next(responses))
    assert inventory._open_bounded("model-00001-of-00001.safetensors", start=0, length=8) == (
        b"12345678", 100
    )
    with pytest.raises(InventoryError, match="unexpected byte range"):
        inventory._open_bounded("model-00001-of-00001.safetensors", start=0, length=8)
    with pytest.raises(InventoryError, match="exceeds allowed bounds"):
        inventory._open_bounded("model-00001-of-00001.safetensors", start=0, length=8)


def test_pinned_fetch_only_reads_metadata_and_exact_header_ranges(monkeypatch):
    metadata, shards = _model_fixture()
    requests = []

    def fake_urlopen(request, *, timeout):
        assert timeout == 20
        requests.append(request)
        name = request.full_url.rsplit("/", 1)[1]
        assert request.full_url.startswith(inventory.MODEL_URL)
        byte_range = request.get_header("Range")
        if name in metadata:
            assert byte_range is None
            body = metadata[name]
            return FakeResponse(body, status=200, headers={"Content-Length": str(len(body))})
        assert name in shards and byte_range is not None
        start, end = map(int, byte_range.removeprefix("bytes=").split("-"))
        prefix, file_bytes = shards[name]
        assert (start, end) in ((0, 7), (8, len(prefix) - 1))
        body = prefix[start : end + 1]
        return FakeResponse(
            body,
            headers={
                "Content-Length": str(len(body)),
                "Content-Range": f"bytes {start}-{end}/{file_bytes}",
            },
        )

    monkeypatch.setattr(inventory, "urlopen", fake_urlopen)
    fetched_metadata, fetched_shards = inventory.fetch_pinned_headers()
    assert fetched_metadata == metadata
    assert fetched_shards == shards
    assert len(requests) == len(metadata) + 2
    report = build_inventory(fetched_metadata, fetched_shards)
    assert report["source"]["transfer_body_bytes"] == (
        sum(map(len, metadata.values())) + len(next(iter(shards.values()))[0])
    )


def test_bounded_bf16_slice_uses_exact_row_ranges_and_keeps_hashes(monkeypatch):
    metadata, headers = _model_fixture()
    shard, (prefix, file_bytes) = next(iter(headers.items()))
    name = "model.language_model.layers.0.mlp.down_proj.weight"
    tensor_start = json.loads(prefix[8:])[name]["data_offsets"][0]
    requests = []

    def fake_range(requested_shard, *, start, length):
        assert requested_shard == shard and length == 256
        requests.append((start, length))
        relative = start - len(prefix) - tensor_start
        row, column = divmod(relative // 2, 2048)
        assert column == 128 and row in (0, 1)
        value = np.float32(1.25 if row == 0 else -2.5)
        bits = (np.array([value], dtype=np.float32).view(np.uint32) >> 16).astype("<u2")
        return bits.tobytes() * 128, file_bytes

    monkeypatch.setattr(weight_slice, "_open_bounded", fake_range)
    values, source = weight_slice.fetch_bf16_projection_slice(
        metadata, headers, name=name, start_column=128, rows=2, groups=1
    )
    np.testing.assert_array_equal(values, np.array([[1.25] * 128, [-2.5] * 128]))
    assert requests == [
        (len(prefix) + tensor_start + 128 * 2, 256),
        (len(prefix) + tensor_start + (2048 + 128) * 2, 256),
    ]
    assert source["payload_bytes"] == 512 and len(source["row_sha256"]) == 2
    assert source["full_weight_hash"] == "not_checked_bounded_slice_only"


def test_bounded_bf16_slice_rejects_invalid_requests_before_fetch(monkeypatch):
    metadata, headers = _model_fixture()
    def no_fetch(*args, **kwargs):
        raise AssertionError("unexpected shard range read")

    monkeypatch.setattr(weight_slice, "_open_bounded", no_fetch)
    for request in (
        {"rows": 5}, {"groups": 3}, {"start_column": 1},
        {"start_row": 1024}, {"start_column": 1920}, {"name": "lm_head.weight"},
    ):
        with pytest.raises(InventoryError):
            weight_slice.fetch_bf16_projection_slice(metadata, headers, **request)


def test_bounded_bf16_slice_refuses_full_body_response(monkeypatch):
    metadata, headers = _model_fixture()
    response = FakeResponse(b"not a range", status=200, headers={"Content-Length": "5000000000"})
    monkeypatch.setattr(inventory, "urlopen", lambda request, *, timeout: response)
    with pytest.raises(InventoryError, match="server ignored bounded range"):
        weight_slice.fetch_bf16_projection_slice(
            metadata, headers, name="model.language_model.layers.0.mlp.down_proj.weight",
            rows=1, groups=1,
        )
    assert response.read_calls == 0


def test_synthetic_slice_screen_is_bounded_and_deterministic():
    weights = np.random.default_rng(51).normal(size=(2, 256)).astype(np.float32)
    result = weight_slice.screen_synthetic_reconstruction(weights)
    assert result == weight_slice.screen_synthetic_reconstruction(weights)
    assert result["purpose"] == "synthetic_local_reconstruction_not_model_quality"
    assert set(result["local_mse"]) == {
        "rtn_maxabs", "rtn_grid", "compensated_maxabs", "compensated_grid",
    }
    assert all(np.isfinite(list(result["local_mse"].values())))
    for invalid in (np.ones((5, 256)), np.ones((1, 257)), np.zeros((0, 256))):
        with pytest.raises(ValueError, match="at most four rows"):
            weight_slice.screen_synthetic_reconstruction(invalid)


def test_prism_candidate_widths_come_from_pinned_eligible_hf_headers():
    metadata, shards = _model_fixture()
    report = build_inventory(metadata, shards)
    names = ["blk.0.ffn_down.weight", "blk.0.ffn_gate.weight", "blk.0.attn_q.weight"]
    widths = mimo_prism_expected_widths(report, names)
    assert widths == {
        "blk.0.ffn_down.weight": 2048,
        "blk.0.ffn_gate.weight": 1024,
        "blk.0.attn_q.weight": 1024,
    }
    with pytest.raises(TernaryArtifactError, match="unverified MiMo"):
        mimo_prism_expected_widths({**report, "source": {**report["source"], "revision": "main"}}, names)
    with pytest.raises(TernaryArtifactError, match="missing or ineligible"):
        mimo_prism_expected_widths(report, ["blk.1.ffn_down.weight"])
    with pytest.raises(TernaryArtifactError, match="unsupported MiMo"):
        mimo_prism_expected_widths(report, ["blk.0.ssm_out.weight"])

    codes, scales = quantize_ternary_rtn(np.ones((2, 256), dtype=np.float32))
    payload = save_toy_artifact(
        codes, scales, algorithm="rtn-maxabs-fp16-v1",
        signs=np.ones(256, dtype=np.int8), block_size=128,
    )
    with pytest.raises(TernaryArtifactError, match="logical input width mismatch"):
        prism_v1_transform_metadata(
            {names[0]: payload}, expected_widths=mimo_prism_expected_widths(report, [names[0]])
        )