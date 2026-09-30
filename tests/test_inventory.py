"""Offline checks for bounded, revision-pinned model inventory."""

import io
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from embedded_jev import inventory
from embedded_jev import weight_slice
from embedded_jev.dense_probe import plan_streamed_text, plan_text_prefix, run_text_prefix
from embedded_jev.streamed_text import (
    run_streamed_text, score_selected_head, stream_full_vocabulary_mass,
)
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


def test_json_object_retains_metadata_cap_with_explicit_public_data_bound():
    payload = json.dumps({"text": "x" * inventory.MAX_METADATA_BYTES}).encode()
    with pytest.raises(InventoryError, match="allowed bounds"):
        inventory._json_object(payload, "model metadata")
    result = inventory._json_object(payload, "public data", max_bytes=3 << 20)
    assert len(result["text"]) == inventory.MAX_METADATA_BYTES
    for limit in (0, True):
        with pytest.raises(InventoryError, match="invalid JSON byte bound"):
            inventory._json_object(b"{}", "public data", max_bytes=limit)
    with pytest.raises(InventoryError):
        inventory._json_object(b'{"key":1,"key":2}', "public data", max_bytes=3 << 20)


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


def test_local_inventory_reads_bounded_headers_without_weight_payload(tmp_path, monkeypatch):
    metadata, shards = _model_fixture()
    for name, data in metadata.items():
        (tmp_path / name).write_bytes(data)
    for name, (prefix, size) in shards.items():
        with (tmp_path / name).open("wb") as destination:
            destination.write(prefix)
            destination.truncate(size)
    monkeypatch.setattr(inventory, "_open_bounded", lambda *args, **kwargs: pytest.fail("network read"))
    local_metadata, local_headers = inventory.read_local_headers(tmp_path)
    assert local_metadata == metadata and local_headers == shards
    assert build_inventory(local_metadata, local_headers)["totals"] == build_inventory(
        metadata, shards
    )["totals"]
    (tmp_path / "config.json").write_bytes(b"x" * (inventory.MAX_METADATA_BYTES + 1))
    with pytest.raises(InventoryError, match="metadata file exceeds allowed bounds"):
        inventory.read_local_headers(tmp_path)
    (tmp_path / "config.json").write_bytes(metadata["config.json"])
    (tmp_path / next(iter(shards))).unlink()
    with pytest.raises(InventoryError, match="unable to read local model snapshot"):
        inventory.read_local_headers(tmp_path)


def test_dense_prefix_plan_rejects_oversized_or_unsupported_layers(monkeypatch):
    metadata, headers = _model_fixture()
    plan = plan_text_prefix(metadata, headers, layers=1)
    assert plan["layers"] == 1 and plan["layer_types"] == ["full_attention"]
    assert plan["tensor_count"] == 2 + 2 + 9
    assert all(name.startswith("model.language_model.") for name in plan["parameter_names"])
    assert plan["weight_bytes"] < 4 * 1024**3
    for count in (0, 2, 5, True):
        with pytest.raises(InventoryError, match="prefix length"):
            plan_text_prefix(metadata, headers, layers=count)
    monkeypatch.setattr("embedded_jev.dense_probe.MAX_PREFIX_WEIGHT_BYTES", 10)
    with pytest.raises(InventoryError, match="BF16 budget"):
        plan_text_prefix(metadata, headers, layers=1)
    with pytest.raises(InventoryError, match="requires four dense prefix layers"):
        run_text_prefix(None, layers=1, prompt="A or B", compare_ternary=True)
    with pytest.raises(InventoryError, match="native comparison requires ternary"):
        run_text_prefix(None, layers=1, prompt="A or B", native_library=Path("kernel.so"))


def test_streamed_text_plan_bounds_each_layer_independently(monkeypatch):
    metadata, headers = _model_fixture()
    plan = plan_streamed_text(metadata, headers, layers=1)
    assert plan["layers"] == 1 and plan["layer_types"] == ["full_attention"]
    assert len(plan["layer_parameter_names"][0]) == 11
    assert plan["embedding_bytes"] > 0 and plan["max_layer_bytes"] > 0
    for count in (0, 2, True):
        with pytest.raises(InventoryError, match="layer count"):
            plan_streamed_text(metadata, headers, layers=count)
    monkeypatch.setattr("embedded_jev.dense_probe.MAX_STREAM_LAYER_BYTES", 10)
    with pytest.raises(InventoryError, match="per-layer BF16 budget"):
        plan_streamed_text(metadata, headers, layers=1)


def test_selected_lm_head_scores_only_bounded_bf16_rows(tmp_path, monkeypatch):
    metadata, headers = _model_fixture()
    for name, data in metadata.items():
        (tmp_path / name).write_bytes(data)
    shard, (prefix, file_bytes) = next(iter(headers.items()))
    with (tmp_path / shard).open("wb") as destination:
        destination.write(prefix)
        destination.truncate(file_bytes)
    row_bytes = 1024 * 2
    offset = json.loads(prefix[8:])["lm_head.weight"]["data_offsets"][0]
    with (tmp_path / shard).open("r+b") as destination:
        for token_id, value in ((0, 1.0), (1, -1.0)):
            bits = (np.array([value], dtype=np.float32).view(np.uint32) >> 16).astype("<u2")
            destination.seek(len(prefix) + offset + token_id * row_bytes)
            destination.write(bits.tobytes() * 1024)
    report = score_selected_head(tmp_path, metadata, headers, np.ones(1024), {"A": 0, "B": 1})
    assert report["head_payload_bytes"] == 2 * row_bytes
    assert report["options"]["A"]["logit"] == 1024.0
    assert report["options"]["B"]["logit"] == -1024.0
    assert report["options"]["A"]["conditional_probability"] == 1.0
    assert len(report["options"]["A"]["row_sha256"]) == 64
    assert report["full_vocabulary_mass"] == "not_computed"
    for labels in ({"A": 0, "B": 32}, {"A": 0, "B": 0}, {"B": 1, "A": 0}):
        with pytest.raises(InventoryError, match="invalid selected-head"):
            score_selected_head(tmp_path, metadata, headers, np.ones(1024), labels)
    with pytest.raises(InventoryError, match="selected label count"):
        run_streamed_text(None, prompt="A or B", label_count=1)
    with pytest.raises(InventoryError, match="fixture path and case id"):
        run_streamed_text(None, prompt="A or B", case_id="inspect-before-answer")
    with pytest.raises(InventoryError, match="requires all 32 text layers"):
        run_streamed_text(None, prompt="A or B", full_vocabulary_mass=True)
    with pytest.raises(InventoryError, match="requires the native FFN-down library"):
        run_streamed_text(None, prompt="A or B", projection_artifact=tmp_path)

    value = np.float32(1 / 512)
    bits = (np.array([value], dtype=np.float32).view(np.uint32) >> 16).astype("<u2")
    with (tmp_path / shard).open("r+b") as destination:
        for token_id, sign in ((0, 1), (1, -1)):
            destination.seek(len(prefix) + offset + token_id * row_bytes)
            signed_bits = bits if sign > 0 else bits | np.uint16(0x8000)
            destination.write(signed_bits.astype("<u2").tobytes() * 1024)
    selected = score_selected_head(tmp_path, metadata, headers, np.ones(1024), {"A": 0, "B": 1})
    mass = stream_full_vocabulary_mass(tmp_path, metadata, headers, np.ones(1024), selected)
    assert mass["vocabulary_rows"] == 32 and mass["max_token_id"] == 0
    assert mass["payload_bytes"] == 32 * row_bytes
    expected_mass = (np.exp(2) + np.exp(-2)) / (np.exp(2) + np.exp(-2) + 30)
    assert mass["selected_label_mass"] == pytest.approx(expected_mass)
    monkeypatch.setattr("embedded_jev.streamed_text.MAX_FULL_HEAD_BYTES", 10)
    with pytest.raises(InventoryError, match="full-vocabulary mass exceeds"):
        stream_full_vocabulary_mass(tmp_path, metadata, headers, np.ones(1024), selected)


def test_full_vocabulary_mass_uses_the_same_logits_for_selected_rows(tmp_path):
    metadata, headers = _model_fixture()
    shard, (prefix, file_bytes) = next(iter(headers.items()))
    offset = json.loads(prefix[8:])["lm_head.weight"]["data_offsets"][0]
    values = np.random.default_rng(0).normal(size=1024).astype(np.float32)
    bits = (values.view(np.uint32) >> 16).astype("<u2")
    vector = (bits.astype(np.uint32) << 16).view("<f4")
    with (tmp_path / shard).open("wb") as destination:
        destination.write(prefix)
        destination.truncate(file_bytes)
        destination.seek(len(prefix) + offset)
        destination.write(bits.tobytes() * 2)
    selected = score_selected_head(tmp_path, metadata, headers, vector, {"A": 0, "B": 1})
    mass = stream_full_vocabulary_mass(tmp_path, metadata, headers, vector, selected)
    assert mass["selected_label_mass"] == pytest.approx(1.0, abs=1e-12)
    assert mass["max_token_id"] in (0, 1)


@pytest.mark.parametrize("batch_rows", [1, 7, 256])
def test_full_vocabulary_mass_tracks_selected_rows_across_batches(tmp_path, monkeypatch, batch_rows):
    metadata, headers = _model_fixture()
    shard, (prefix, file_bytes) = next(iter(headers.items()))
    offset = json.loads(prefix[8:])["lm_head.weight"]["data_offsets"][0]
    with (tmp_path / shard).open("wb") as destination:
        destination.write(prefix)
        destination.truncate(file_bytes)
        for token_id, value in ((2, 1 / 512), (31, -1 / 512)):
            bits = (np.array([value], dtype=np.float32).view(np.uint32) >> 16).astype("<u2")
            destination.seek(len(prefix) + offset + token_id * 2048)
            destination.write(bits.tobytes() * 1024)
    vector = np.ones(1024, dtype=np.float32)
    selected = score_selected_head(tmp_path, metadata, headers, vector, {"A": 2, "B": 31})
    monkeypatch.setattr("embedded_jev.streamed_text.FULL_HEAD_BATCH_ROWS", batch_rows)
    mass = stream_full_vocabulary_mass(tmp_path, metadata, headers, vector, selected)
    expected = (np.exp(2) + np.exp(-2)) / (np.exp(2) + np.exp(-2) + 30)
    assert mass["selected_label_mass"] == pytest.approx(expected)
    assert mass["max_token_id"] == 2
    invalid = {"options": {"A": {"token_id": 2}, "B": {"token_id": 2}}}
    with pytest.raises(InventoryError, match="duplicate selected-label"):
        stream_full_vocabulary_mass(tmp_path, metadata, headers, vector, invalid)
    invalid["options"]["B"]["token_id"] = 32
    with pytest.raises(InventoryError, match="full-vocabulary mass exceeds"):
        stream_full_vocabulary_mass(tmp_path, metadata, headers, vector, invalid)


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


def test_local_bf16_rows_read_complete_width_with_exact_offsets(tmp_path, monkeypatch):
    metadata, headers = _model_fixture()
    for name, data in metadata.items():
        (tmp_path / name).write_bytes(data)
    shard, (prefix, file_bytes) = next(iter(headers.items()))
    name = "model.language_model.layers.0.mlp.down_proj.weight"
    offset = json.loads(prefix[8:])[name]["data_offsets"][0]
    row_bytes = 2048 * 2
    with (tmp_path / shard).open("wb") as destination:
        destination.write(prefix)
        destination.truncate(file_bytes)
    raw_rows = []
    with (tmp_path / shard).open("r+b") as destination:
        for row, value in enumerate((1.25, -2.5), start=1):
            bits = (np.array([value], dtype=np.float32).view(np.uint32) >> 16).astype("<u2")
            data = bits.tobytes() * 2048
            destination.seek(len(prefix) + offset + row * row_bytes)
            destination.write(data)
            raw_rows.append(data)
    monkeypatch.setattr(weight_slice, "_open_bounded", lambda *args, **kwargs: pytest.fail("network read"))
    values, source = weight_slice.read_local_bf16_projection_rows(
        tmp_path, name=name, start_row=1, rows=2,
    )
    np.testing.assert_array_equal(values, np.array([[1.25] * 2048, [-2.5] * 2048]))
    assert source["payload_bytes"] == 2 * row_bytes
    assert source["row_sha256"] == [hashlib.sha256(data).hexdigest() for data in raw_rows]
    assert source["full_weight_hash"] == "not_checked_local_rows_only"
    for request in ({"rows": 5}, {"start_row": 1023, "rows": 2}, {"name": "lm_head.weight"}):
        with pytest.raises(InventoryError):
            weight_slice.read_local_bf16_projection_rows(tmp_path, **request)
    monkeypatch.setattr(weight_slice, "MAX_LOCAL_PAYLOAD_BYTES", row_bytes)
    with pytest.raises(InventoryError, match="payload budget"):
        weight_slice.read_local_bf16_projection_rows(tmp_path, name=name, rows=2)


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


def test_calibration_slice_screen_preserves_exact_ternary_weights_and_limits():
    weights = np.ones((2, 256), dtype=np.float32)
    samples = np.random.default_rng(314).normal(size=(32, 256)).astype(np.float32)
    report = weight_slice.screen_calibration_reconstruction(weights, samples)
    assert report["purpose"] == "calibration_slice_reconstruction_not_validation_or_model_quality"
    assert report["calibration_rows"] == 32 and report["activation_quantized"] is False
    assert set(report["policies"]) == {"rtn_maxabs", "rtn_grid", "compensated_maxabs", "compensated_grid"}
    assert all(policy["output_mse"] == 0 and policy["scale_dtype"] == "float16" for policy in report["policies"].values())
    assert all(report["policies"][name]["damping"] > 0 for name in ("compensated_maxabs", "compensated_grid"))
    for invalid_weights, invalid_samples in (
        (np.ones((5, 256)), samples), (weights, np.ones((129, 256))),
        (weights, np.ones((32, 255))), (weights, np.full((32, 256), np.nan)),
    ):
        with pytest.raises(ValueError, match="bounded four-row"):
            weight_slice.screen_calibration_reconstruction(invalid_weights, invalid_samples)
    with pytest.raises(ValueError, match="nonzero activation curvature"):
        weight_slice.screen_calibration_reconstruction(weights, np.zeros_like(samples))
    nonternary = np.random.default_rng(315).normal(scale=0.03, size=(2, 256)).astype(np.float32)
    calibration_only = weight_slice.screen_calibration_reconstruction(nonternary, samples)
    first_validation = weight_slice.screen_calibration_reconstruction(
        nonternary, samples, validation_activations=samples[:4],
    )
    other_validation = weight_slice.screen_calibration_reconstruction(
        nonternary, samples,
        validation_activations=np.random.default_rng(316).normal(size=(4, 256)).astype(np.float32),
    )
    assert first_validation["validation_used_for_fitting"] is False
    assert first_validation["validation_rows"] == 4
    assert first_validation["purpose"] == "calibration_fit_validation_slice_not_model_quality"
    for label in report["policies"]:
        assert first_validation["policies"][label]["codes_sha256"] == calibration_only["policies"][label]["codes_sha256"]
        assert other_validation["policies"][label]["scales_sha256"] == calibration_only["policies"][label]["scales_sha256"]
        assert first_validation["policies"][label]["validation_output_mse"] != other_validation["policies"][label]["validation_output_mse"]


def test_block_diagonal_screen_preserves_full_row_bounds_and_validation_independence():
    generator = np.random.default_rng(318)
    weights = generator.normal(size=(2, 512)).astype(np.float32)
    calibration = generator.normal(size=(32, 512)).astype(np.float32)
    validation = generator.normal(size=(8, 512)).astype(np.float32)
    report = weight_slice.screen_calibration_reconstruction(
        weights, calibration, validation_activations=validation, block_diagonal=True,
    )
    repeated = weight_slice.screen_calibration_reconstruction(
        weights, calibration, validation_activations=-validation, block_diagonal=True,
    )
    assert report["columns"] == 512 and report["validation_used_for_fitting"] is False
    assert report["purpose"] == "block_diagonal_calibration_diagnostic_not_full_gptq_or_model_quality"
    assert report["policies"]["compensated_grid"]["cross_block_curvature"] is False
    for label in report["policies"]:
        assert report["policies"][label]["codes_sha256"] == repeated["policies"][label]["codes_sha256"]
    with pytest.raises(ValueError, match="bounded four-row"):
        weight_slice.screen_calibration_reconstruction(weights, calibration)


def test_full_width_synthetic_screen_checks_rotation_and_bounds():
    weights = np.random.default_rng(441).normal(size=(2, 256)).astype(np.float32)
    result = weight_slice.screen_full_width_synthetic(weights)
    assert result == weight_slice.screen_full_width_synthetic(weights)
    assert result["purpose"] == "four_full_width_rows_synthetic_inputs_not_model_quality"
    assert set(result["policies"]) == {
        "maxabs", "searched_fp16", "signed_hadamard_128_searched_fp16",
    }
    assert all(
        np.isfinite(policy["weight_mse"]) and np.isfinite(policy["output_mse"])
        for policy in result["policies"].values()
    )
    for invalid in (np.ones((5, 256)), np.ones((2, 257)), np.full((2, 256), np.inf)):
        with pytest.raises(ValueError, match="four bounded projection rows"):
            weight_slice.screen_full_width_synthetic(invalid)


def test_local_projection_streams_sparse_rows_without_copying_weights(tmp_path, monkeypatch):
    metadata, headers = _model_fixture()
    for name, data in metadata.items():
        (tmp_path / name).write_bytes(data)
    shard, (prefix, file_bytes) = next(iter(headers.items()))
    name = "model.language_model.layers.3.mlp.down_proj.weight"
    indexed_name = "model.language_model.layers.0.mlp.down_proj.weight"
    offset = json.loads(prefix[8:])[indexed_name]["data_offsets"][0]
    with (tmp_path / shard).open("wb") as destination:
        destination.write(prefix)
        destination.truncate(file_bytes)
    bits = (np.array([1.25], dtype=np.float32).view(np.uint32) >> 16).astype("<u2")
    with (tmp_path / shard).open("r+b") as destination:
        destination.seek(len(prefix) + offset)
        destination.write(bits.tobytes() * 2048)
    monkeypatch.setattr(weight_slice, "DEFAULT_TENSOR", indexed_name)
    monkeypatch.setattr(weight_slice, "_open_bounded", lambda *args, **kwargs: pytest.fail("network read"))
    report = weight_slice.screen_local_projection(tmp_path)
    assert report["tensor"] == indexed_name and report["rows"] == 1024
    assert report["columns"] == 2048 and report["rows_per_batch"] == 64
    assert report["policies"]["maxabs"]["weight_mse"] == 0.0
    assert report["policies"]["searched_fp16"]["weight_mse"] == 0.0
    assert report["policies"]["searched_fp16"]["nonzero_fraction"] == 1 / 1024
    monkeypatch.setattr(weight_slice, "MAX_STREAM_TENSOR_BYTES", 4096)
    with pytest.raises(InventoryError, match="stream screen bounds"):
        weight_slice.screen_local_projection(tmp_path)


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