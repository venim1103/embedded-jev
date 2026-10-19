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
from embedded_jev.model_profile import (
    GEOMETRY_INTS, MAX_PROFILE_BYTES, ProfileError, defiant_fable_profile, mimo_profile, parse_model_profile,
    verify_profile_files, verify_profile_headers,
)
from embedded_jev.prism_codec import create_profile_projection, pack_ternary_pq2_0, profile_projection_specification
from embedded_jev.ternary import quantize_ternary_rtn
from embedded_jev.ternary_artifact import (
    TernaryArtifactError,
    mimo_prism_expected_widths,
    profile_prism_expected_widths,
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


def _fixture_geometry():
    geometry = dict.fromkeys(GEOMETRY_INTS, 1)
    geometry.update({
        "hidden_size": 1024, "intermediate_size": 2048, "vocab_size": 32,
        "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 64,
        "linear_num_key_heads": 16, "linear_num_value_heads": 32,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4, "layer_types": ["full_attention"],
        "mamba_ssm_dtype": "float32", "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-6, "tie_word_embeddings": False, "attn_output_gate": True,
        "rope_parameters": {
            "mrope_interleaved": True, "mrope_section": [3, 3, 2],
            "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default",
        },
    })
    return geometry


def _model_fixture():
    text = {**_fixture_geometry(), "mtp_num_hidden_layers": 1}
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
        ("self_attn.o_proj.weight", [1024, 512]),
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


def _profile_fixture():
    metadata, headers = _model_fixture()
    files = {
        name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in {**metadata, "tokenizer.json": b"{}"}.items()
    }
    for name, (prefix, size) in headers.items():
        files[name] = {
            "bytes": size, "sha256": "a" * 64,
            "header_bytes": len(prefix), "header_sha256": hashlib.sha256(prefix).hexdigest(),
        }
    geometry = _fixture_geometry()
    return {
        "schema_version": 1, "profile_id": "synthetic-profile",
        "source": {
            "model": "example/Synthetic", "revision": "1" * 40, "license": "MIT",
            "format": "safetensors", "support_tier": "A",
        },
        "files": files,
        "packaging": {
            "architecture": "Qwen3_5ForConditionalGeneration", "text_config_path": ["text_config"],
            "text_prefix": "model.language_model.", "vision_prefix": "model.visual.",
            "output_head": "lm_head.weight", "shards": list(headers), "metadata_files": list(metadata),
            "mtp_prefix": "mtp.", "mtp_status": "absent", "mtp_handling": "exclude",
            "index_total_policy": "exact", "text_tensor_count": 14,
            "metadata_classes": {
                "tokenizer": "Qwen2Tokenizer", "processor": "Qwen3VLProcessor",
                "image_processor": "Qwen2VLImageProcessor", "video_processor": "Qwen3VLVideoProcessor",
            },
            "dtype_policy": dict.fromkeys((
                "input_embedding", "output_head", "language_projection", "sensitive", "vision", "optional_mtp",
            ), ["BF16"]),
        },
        "geometry": geometry,
        "tokenization": {
            "template_file": "chat_template.jinja", "gguf_template_file": None,
            "gguf_template_origin": None,
            "render_arguments": {"add_generation_prompt": True, "enable_thinking": False},
            "non_thinking_suffix": "<think></think>", "control_markers": [],
            "special_token_ids": {"<think>": 20, "</think>": 21},
            "label_token_ids": {label: number for number, label in enumerate("ABCDEFGHIJKLMNOP")},
            "prompt_formatter_version": 1,
        },
        "runtime": {
            "architecture_adapter": "qwen35-safetensors-v1", "converter_revision": "2" * 40,
            "runtime_revision": "2" * 40, "converter_flags": ["--no-nextn"], "native_policy": None,
        },
        "quantization": {
            "group_size": 128, "scale_dtype": "F16", "activation_contract": "group128-dynamic-a8-v1",
            "eligible_suffixes": [
                "mlp.down_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
                "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            ],
        },
        "evidence": [],
    }


def _packaging_fixture(*, text_only=False, shard_name="model-1-of-1.safetensors", dtype="BF16", mtp="absent", stale_total=False):
    metadata, headers = _model_fixture()
    record = _profile_fixture()
    specs = json.loads(next(iter(headers.values()))[0][8:])
    config = json.loads(metadata["config.json"])
    if text_only:
        config = {"architectures": ["Qwen3_5ForCausalLM"], **config["text_config"]}
        specs = {name.replace("model.language_model.", "model."): spec for name, spec in specs.items() if not name.startswith("model.visual.")}
        for name in ("processor_config.json", "preprocessor_config.json", "video_preprocessor_config.json"):
            metadata.pop(name)
        record["packaging"].update(architecture="Qwen3_5ForCausalLM", text_config_path=[], text_prefix="model.", vision_prefix=None)
        record["packaging"]["metadata_classes"].update(processor=None, image_processor=None, video_processor=None)
    prefix = record["packaging"]["text_prefix"]
    specs[prefix + "norm.weight"]["dtype"] = dtype
    record["packaging"]["dtype_policy"]["sensitive"] = ["BF16", "F32"]
    record["packaging"]["mtp_status"] = mtp
    if mtp != "absent":
        hidden = record["geometry"]["hidden_size"]
        shapes = {
            "mtp.fc.weight": [hidden, 2 * hidden], "mtp.norm.weight": [hidden],
            "mtp.pre_fc_norm_embedding.weight": [hidden], "mtp.pre_fc_norm_hidden.weight": [hidden],
            **{"mtp.layers.0." + name: list(shape) for name, shape in inventory._layer_shapes(record["geometry"], "full_attention").items()},
        }
        if mtp == "incomplete":
            shapes.pop("mtp.fc.weight")
        specs.update({name: {"dtype": "BF16", "shape": shape} for name, shape in shapes.items()})
    entries, weight_map, offset = {}, {}, 0
    restored = {}
    for name, spec in sorted(specs.items()):
        count = int(np.prod(spec["shape"])) * inventory.DTYPE_BYTES[spec["dtype"]]
        target = restored if name == "mtp.fc.weight" else entries
        start = 0 if target is restored else offset
        target[name] = {**spec, "data_offsets": [start, start + count]}
        weight_map[name] = "model-mtp-restored.safetensors" if target is restored else shard_name
        if target is entries:
            offset += count
    main_header = _header(entries, offset)
    headers = {shard_name: main_header}
    if restored:
        headers["model-mtp-restored.safetensors"] = _header(restored, next(iter(restored.values()))["data_offsets"][1])
    total = sum(size - len(header) for header, size in headers.values())
    if stale_total:
        total -= next(iter(restored.values()))["data_offsets"][1] if restored else 2
        record["packaging"]["index_total_policy"] = "recomputed"
    metadata["config.json"] = json.dumps(config).encode()
    metadata["model.safetensors.index.json"] = json.dumps({"metadata": {"total_size": total, "mergekit_version": "fixture"}, "weight_map": weight_map}).encode()
    record["packaging"]["metadata_files"] = list(metadata)
    record["packaging"]["shards"] = list(headers)
    record["files"] = {name: spec for name, spec in record["files"].items() if name == "tokenizer.json"}
    record["files"].update({name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in metadata.items()})
    record["files"].update({name: {"bytes": size, "sha256": "a" * 64, "header_bytes": len(header), "header_sha256": hashlib.sha256(header).hexdigest()} for name, (header, size) in headers.items()})
    return metadata, headers, record


@pytest.mark.parametrize("text_only,shard_name,dtype,mtp,stale_total", [
    (False, "model-00001-of-00001.safetensors", "BF16", "absent", False),
    (False, "model.safetensors-00001-of-00001.safetensors", "F32", "complete", True),
    (True, "model-1-of-1.safetensors", "F32", "absent", False),
    (True, "model-00001-of-00001.safetensors", "BF16", "incomplete", False),
])
def test_profile_packaging_adapters_reconcile_declared_variations(text_only, shard_name, dtype, mtp, stale_total):
    metadata, headers, record = _packaging_fixture(text_only=text_only, shard_name=shard_name, dtype=dtype, mtp=mtp, stale_total=stale_total)
    profile = parse_model_profile(json.dumps(record).encode())
    report = build_inventory(metadata, headers, profile=profile)
    assert report["source"]["model"] == "example/Synthetic"
    assert report["native_reference_plan"]["text_tensor_count"] == 14
    assert report["accounting"]["optional_mtp_status"] == mtp
    assert report["accounting"]["index_total_matches_headers"] is not stale_total
    assert (report["accounting"]["index_total_difference_bytes"] > 0) is stale_total
    if mtp == "incomplete":
        assert report["accounting"]["optional_mtp_missing_weights"] == ["mtp.fc.weight"]
    assert (report["memory_estimates"]["vision_original_bytes"] == 0) is text_only
    prefix_plan = plan_text_prefix(metadata, headers, layers=1, profile=profile)
    stream_plan = plan_streamed_text(metadata, headers, layers=1, profile=profile)
    assert prefix_plan["model"] == stream_plan["model"] == profile.model
    assert prefix_plan["profile_sha256"] == stream_plan["profile_sha256"] == profile.sha256
    assert stream_plan["embedding_name"] == profile.packaging["text_prefix"] + "embed_tokens.weight"


@pytest.mark.parametrize("defect", ["dtype", "mtp", "count", "geometry", "total"])
def test_profile_packaging_adapters_refuse_undeclared_variations(defect):
    metadata, headers, record = _packaging_fixture(dtype="F32", mtp="complete", stale_total=True)
    if defect == "dtype":
        record["packaging"]["dtype_policy"]["sensitive"] = ["BF16"]
    elif defect == "mtp":
        record["packaging"]["mtp_status"] = "incomplete"
    elif defect == "count":
        record["packaging"]["text_tensor_count"] = 15
    elif defect == "geometry":
        record["geometry"]["intermediate_size"] = 1024
    else:
        record["packaging"]["index_total_policy"] = "exact"
    with pytest.raises(InventoryError):
        build_inventory(metadata, headers, profile=parse_model_profile(json.dumps(record).encode()))


def test_profile_sidecar_headers_are_explicit_bounded_and_payload_free(tmp_path):
    metadata, headers, record = _packaging_fixture(text_only=True, mtp="complete", stale_total=True)
    profile = parse_model_profile(json.dumps(record).encode())
    header_directory = tmp_path / "headers"
    header_directory.mkdir()
    for name, data in metadata.items():
        (tmp_path / name).write_bytes(data)
    for name, (prefix, _) in headers.items():
        (header_directory / (name + ".header.json")).write_bytes(prefix[8:])
    assert not any(tmp_path.glob("*.safetensors"))
    with pytest.raises(InventoryError, match="explicit verified model profile"):
        inventory.read_local_headers(tmp_path, header_directory=header_directory)
    actual_metadata, actual_headers = inventory.read_local_headers(tmp_path, profile=profile, header_directory=header_directory)
    assert actual_metadata == metadata and actual_headers == headers
    report = build_inventory(actual_metadata, actual_headers, profile=profile)
    assert report["accounting"]["optional_mtp_status"] == "complete"
    name = next(iter(headers))
    path = header_directory / (name + ".header.json")
    raw = path.read_bytes()
    path.write_bytes(b" " + raw[1:])
    with pytest.raises(InventoryError, match="header identity"):
        inventory.read_local_headers(tmp_path, profile=profile, header_directory=header_directory)
    path.write_bytes(raw + b" ")
    with pytest.raises(InventoryError, match="profile bounds"):
        inventory.read_local_headers(tmp_path, profile=profile, header_directory=header_directory)


def test_profile_quantization_eligibility_is_explicit_not_inherited():
    metadata, headers, record = _packaging_fixture()
    record["quantization"]["eligible_suffixes"] = ["mlp.down_proj.weight"]
    report = build_inventory(metadata, headers, profile=parse_model_profile(json.dumps(record).encode()))
    eligible = [tensor["name"] for tensor in report["tensors"] if tensor["quantization_eligible"]]
    assert eligible == ["model.language_model.layers.0.mlp.down_proj.weight"]
    omitted = next(tensor for tensor in report["tensors"] if tensor["name"].endswith("mlp.up_proj.weight"))
    assert omitted["policy_reason"] == "projection_not_declared_by_profile"


def _projection_manifest_fixture():
    metadata, headers, record = _packaging_fixture(text_only=True)
    profile = parse_model_profile(json.dumps(record).encode())
    report = build_inventory(metadata, headers, profile=profile)
    codes = np.zeros((1024, 16, 128), dtype=np.int8)
    scales = np.zeros((1024, 16), dtype=np.float16)
    blocks = pack_ternary_pq2_0(codes, scales)
    manifest = {
        "schema_version": 2, "format": "jev-profile-projection-v2", "profile_id": profile.profile_id,
        "profile_sha256": profile.sha256, "tensor": "model.layers.0.mlp.down_proj.weight",
        "gguf_tensor": "blk.0.ffn_down.weight", "shape": [1024, 2048], "group_size": 128,
        "transform": "identity", "source_tensor_sha256": "f" * 64,
        "payload_bytes": blocks.nbytes, "payload_sha256": hashlib.sha256(memoryview(blocks)).hexdigest(),
    }
    return profile, report, manifest, blocks


def test_profile_projection_manifest_derives_native_geometry_and_payload_identity():
    profile, report, manifest, blocks = _projection_manifest_fixture()
    specification = profile_projection_specification(profile, report, manifest, blocks)
    assert specification["rows"] == 1024 and specification["columns"] == 2048
    assert specification["tensor_name"] == "blk.0.ffn_down.weight" and specification["abi_version"] == 2
    assert specification["profile_id"] == profile.profile_id
    assert specification["source_payload_verification"] == "declared_digest_not_rehashed_source_tensor"


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("profile_sha256", "0" * 64), ("shape", [1024, 4096]),
    ("gguf_tensor", "blk.0.ssm_out.weight"), ("tensor", "model.layers.1.mlp.down_proj.weight"),
    ("transform", "hadamard"), ("source_tensor_sha256", "unknown"), ("payload_bytes", 1),
    ("payload_sha256", "0" * 64), ("unknown_field", 1),
])
def test_profile_projection_manifest_refuses_unmatched_declarations(field, value):
    profile, report, manifest, blocks = _projection_manifest_fixture()
    manifest[field] = value
    with pytest.raises(ValueError):
        profile_projection_specification(profile, report, manifest, blocks)


def test_profile_projection_manifest_refuses_unbound_inventory_or_changed_payload():
    profile, report, manifest, blocks = _projection_manifest_fixture()
    report["source"]["profile_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="unverified profile"):
        profile_projection_specification(profile, report, manifest, blocks)
    report["source"]["profile_sha256"] = profile.sha256
    blocks.flat[-1] ^= 1
    with pytest.raises(ValueError, match="payload identity"):
        profile_projection_specification(profile, report, manifest, blocks)


def test_profile_projection_native_creation_refuses_before_dispatch():
    profile, report, manifest, blocks = _projection_manifest_fixture()

    class NeverCalled:
        def __getattr__(self, name):
            pytest.fail("invalid projection declaration reached native dispatch")

    for tokens in (0, True, 129):
        with pytest.raises(ValueError, match="token count"):
            create_profile_projection(NeverCalled(), profile, report, manifest, blocks, tokens=tokens)
    manifest["payload_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="payload identity"):
        create_profile_projection(NeverCalled(), profile, report, manifest, blocks, tokens=2)


def test_profile_projection_manifest_geometry_cannot_come_from_altered_report():
    profile, report, manifest, blocks = _projection_manifest_fixture()
    tensor = next(tensor for tensor in report["tensors"] if tensor["name"] == manifest["tensor"])
    tensor["shape"] = (1024, 1024)
    manifest["shape"] = [1024, 1024]
    with pytest.raises(ValueError, match="geometry mismatch"):
        profile_projection_specification(profile, report, manifest, blocks[:, :8].copy())


def _model_policy_fixture():
    from embedded_jev.prism_codec import PRISM_SOURCE_REVISION

    metadata, headers, record = _packaging_fixture(text_only=True)
    record["runtime"].update(converter_revision=PRISM_SOURCE_REVISION, runtime_revision=PRISM_SOURCE_REVISION)
    profile = parse_model_profile(json.dumps(record).encode())
    report = build_inventory(metadata, headers, profile=profile)
    _, _, manifest, blocks = _projection_manifest_fixture()
    manifest.update(profile_id=profile.profile_id, profile_sha256=profile.sha256)
    return profile, report, manifest, blocks


def _prefill_model_policy_fixture(target="down", *, hybrid=True, state_size=32, value_heads=4, vector_dtype="BF16", conv_dtype="BF16", kernel_size=4, pin_zero_source=False, source_shards=1, vocab_size=32, metadata_output=None, tokenizer_bytes=None, headers_output=None):
    from embedded_jev.prism_codec import PRISM_SOURCE_REVISION

    metadata, _, record = _packaging_fixture(text_only=True)
    geometry = record["geometry"]
    geometry.update(hidden_size=1024, intermediate_size=1024, num_attention_heads=4, num_key_value_heads=2, head_dim=32, vocab_size=vocab_size)
    geometry.update(num_hidden_layers=2, full_attention_interval=2 if hybrid else 1,
                    layer_types=["linear_attention", "full_attention"] if hybrid else ["full_attention", "full_attention"],
                    linear_num_key_heads=2, linear_num_value_heads=value_heads, linear_key_head_dim=state_size, linear_value_head_dim=32,
                    linear_conv_kernel_dim=kernel_size)
    if target == "o":
        geometry.update(hidden_size=128, intermediate_size=256, num_attention_heads=16, head_dim=64)
    hidden = geometry["hidden_size"]
    geometry["rope_parameters"]["mrope_section"] = [3, 3, 2] if target == "o" else [2, 1, 1]
    record["packaging"]["text_tensor_count"] = 28 if hybrid else 25
    record["tokenization"]["label_token_ids"] = dict(zip("ABCDEFGHIJKLMNOP", [11, 17, 23, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19], strict=True))
    config = json.loads(metadata["config.json"])
    config.update(geometry)
    if metadata_output is not None:
        config["max_position_embeddings"] = 512
    metadata["config.json"] = json.dumps(config, sort_keys=True).encode()
    shapes = {
        "model.embed_tokens.weight": (vocab_size, hidden), "lm_head.weight": (vocab_size, hidden), "model.norm.weight": (hidden,),
          **{f"model.layers.{layer}." + name: shape for layer, kind in enumerate(geometry["layer_types"])
              for name, shape in inventory._layer_shapes(geometry, kind).items()},
    }
    old_shards = record["packaging"]["shards"]
    shards = old_shards if source_shards == 1 else ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    record["packaging"]["shards"] = shards
    entries, offsets, weight_map = {name: {} for name in shards}, dict.fromkeys(shards, 0), {}
    for index, (name, shape) in enumerate(sorted(shapes.items())):
        dtype = vector_dtype if len(shape) == 1 else "BF16"
        if name.endswith("linear_attn.conv1d.weight"):
            dtype = conv_dtype
        count = int(np.prod(shape)) * (4 if dtype == "F32" else 2)
        shard = shards[index % len(shards)]
        entries[shard][name] = {"dtype": dtype, "shape": shape, "data_offsets": [offsets[shard], offsets[shard] + count]}
        offsets[shard] += count
        weight_map[name] = shard
    headers = {shard: _header(entries[shard], offsets[shard]) for shard in shards}
    metadata["model.safetensors.index.json"] = json.dumps({
        "metadata": {"total_size": sum(offsets.values())}, "weight_map": weight_map,
    }).encode()
    for shard in old_shards:
        record["files"].pop(shard)
    for name, data in metadata.items():
        record["files"][name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    if tokenizer_bytes is not None:
        record["files"]["tokenizer.json"] = {"bytes": len(tokenizer_bytes), "sha256": hashlib.sha256(tokenizer_bytes).hexdigest()}
    for shard, (header, size) in headers.items():
        record["files"][shard] = {"bytes": size, "sha256": "a" * 64, "header_bytes": len(header),
                                  "header_sha256": hashlib.sha256(header).hexdigest()}
        if pin_zero_source:
            digest = hashlib.sha256(header)
            remaining = size - len(header)
            while remaining:
                chunk_bytes = min(65536, remaining)
                digest.update(bytes(chunk_bytes))
                remaining -= chunk_bytes
            record["files"][shard]["sha256"] = digest.hexdigest()
    record["runtime"].update(converter_revision=PRISM_SOURCE_REVISION, runtime_revision=PRISM_SOURCE_REVISION)
    profile = parse_model_profile(json.dumps(record).encode())
    report = build_inventory(metadata, headers, profile=profile)
    attention_target = target in ("q", "k", "v", "o")
    tensor = f"model.layers.1.self_attn.{target}_proj.weight" if attention_target else f"model.layers.0.mlp.{target}_proj.weight"
    rows, columns = shapes[tensor]
    codes = ((np.arange(rows * columns).reshape(rows, columns // 128, 128) * 7) % 3 - 1).astype(np.int8)
    if target in ("q", "k", "v"):
        heads = geometry["num_attention_heads"] if target == "q" else geometry["num_key_value_heads"]
        stride = geometry["head_dim"] * (2 if target == "q" else 1)
        inactive = np.ones(rows, dtype=bool)
        inactive[np.arange(heads) * stride] = False
        codes[inactive] = 0
    scales = ((np.arange(rows * columns // 128).reshape(rows, columns // 128) % 7 + 1) / 256).astype(np.float16)
    blocks = pack_ternary_pq2_0(codes, scales)
    manifest = {
        "schema_version": 2, "format": "jev-profile-projection-v2", "profile_id": profile.profile_id,
        "profile_sha256": profile.sha256, "tensor": tensor,
        "gguf_tensor": "blk.1.attn_output.weight" if target == "o" else f"blk.1.attn_{target}.weight" if attention_target else f"blk.0.ffn_{target}.weight",
        "shape": [rows, columns], "group_size": 128, "transform": "identity", "source_tensor_sha256": "f" * 64,
        "payload_bytes": blocks.nbytes, "payload_sha256": hashlib.sha256(memoryview(blocks)).hexdigest(),
    }
    if metadata_output is not None:
        metadata_output.update(metadata)
    if headers_output is not None:
        headers_output.update(headers)
    return profile, report, manifest, blocks


@pytest.mark.parametrize("source_shards", [1, 2])
def test_profile_prefill_metadata_capture_retains_actual_pins_and_context(source_shards):
    from embedded_jev.model_profile import verify_profile_metadata
    from embedded_jev.prism_codec import prepare_profile_model_context_metadata, prepare_profile_model_template_metadata

    metadata, headers = {}, {}
    profile, report, manifest, blocks = _prefill_model_policy_fixture(source_shards=source_shards, metadata_output=metadata, headers_output=headers)
    assert set(headers) == set(profile.packaging["shards"])
    assert all(len(prefix) == profile.files[name].header_bytes and size == profile.files[name].bytes
               and hashlib.sha256(prefix).hexdigest() == profile.files[name].header_sha256 for name, (prefix, size) in headers.items())
    verify_profile_metadata(profile, metadata)
    assert report["source"]["profile_sha256"] == manifest["profile_sha256"] == profile.sha256
    receipt = prepare_profile_model_context_metadata(profile, metadata_files=metadata, tokenizer_bytes=b"{}", context_length=256)
    assert receipt["source_context_length"] == 512
    template = prepare_profile_model_template_metadata(profile, metadata_files=metadata,
        tokenizer_bytes=b"{}", context_length=256, template_source="source")
    assert template["template"].encode() == metadata[profile.tokenization["template_file"]]
    assert hashlib.sha256(memoryview(blocks)).hexdigest() == manifest["payload_sha256"]
    metadata["config.json"] += b" "
    with pytest.raises(ValueError, match="metadata identity"):
        prepare_profile_model_context_metadata(profile, metadata_files=metadata, tokenizer_bytes=b"{}", context_length=256)


@pytest.mark.parametrize("source_shards", [1, 2])
@pytest.mark.parametrize("target", ["down", "up", "gate", "q", "k", "v", "o"])
def test_profile_prefill_native_record_retains_checked_geometry_and_storage(target, source_shards):
    import ctypes
    import gc

    from embedded_jev.prism_codec import native_model_specification, profile_model_specification

    profile, report, manifest, blocks = _prefill_model_policy_fixture(target, source_shards=source_shards)
    assert len(profile.packaging["shards"]) == source_shards
    assert {item["shard"] for item in report["tensors"]} == set(profile.packaging["shards"])
    specification = profile_model_specification(profile, report, manifest, blocks)
    record = native_model_specification(specification, blocks)
    del blocks, specification
    gc.collect()
    assert record.hidden == profile.geometry["hidden_size"] and record.intermediate == profile.geometry["intermediate_size"] and record.layers == 2
    assert record.projection.tensor_name == manifest["gguf_tensor"].encode()
    assert (record.projection.rows, record.projection.columns) == tuple(manifest["shape"])
    assert record.tensor_count == 28 and record.metadata_count == 14
    assert sum(item.type == 30 for item in record.tensors[:record.tensor_count]) == 16
    assert hashlib.sha256(ctypes.string_at(record.projection.reference_pq2, record.projection.reference_bytes)).hexdigest() == manifest["payload_sha256"]
    assert ctypes.cast(record.metadata[0].values, ctypes.POINTER(ctypes.c_uint32))[0] == profile.geometry["num_attention_heads"]


def test_profile_model_policy_derives_complete_table_without_execution_approval():
    from embedded_jev.prism_codec import profile_model_specification

    profile, report, manifest, blocks = _model_policy_fixture()
    specification = profile_model_specification(profile, report, manifest, blocks)
    table = {item["name"]: item for item in specification["tensors"]}
    assert len(table) == 14
    assert table["blk.0.ffn_down.weight"] == {"name": "blk.0.ffn_down.weight", "type": "PQ2_0", "ne": [2048, 1024, 1, 1]}
    assert table["blk.0.attn_q.weight"]["ne"] == [1024, 1024, 1, 1]
    assert table["output_norm.weight"]["type"] == "F32"
    assert specification["maximum_metadata_bytes"] == 32 * 1024**2
    assert specification["maximum_metadata_keys"] == 256
    weight_bytes = sum(
        item["ne"][0] * item["ne"][1] // 128 * 34 if item["type"] == "PQ2_0"
        else item["ne"][0] * item["ne"][1] * (4 if item["type"] == "F32" else 2)
        for item in table.values()
    )
    assert specification["maximum_file_bytes"] == weight_bytes + specification["maximum_metadata_bytes"]
    assert specification["runtime_approval"] == "not_granted_by_policy_validation"
    assert specification["source_payload_verification"] == "declared_digest_not_rehashed_source_tensor"


@pytest.mark.parametrize("defect", ["missing", "duplicate", "extra", "shape", "dtype", "category", "identity"])
def test_profile_model_policy_refuses_altered_inventory_before_dispatch(defect):
    from embedded_jev.prism_codec import create_profile_model_policy

    profile, report, manifest, blocks = _model_policy_fixture()
    tensor = next(item for item in report["tensors"] if item["name"].endswith("embed_tokens.weight"))
    if defect == "missing":
        report["tensors"].remove(tensor)
    elif defect == "duplicate":
        report["tensors"].append(tensor.copy())
    elif defect == "extra":
        report["tensors"].append({**tensor, "name": "model.layers.0.control.weight"})
    elif defect == "shape":
        tensor["shape"] = [1, 1024]
    elif defect == "dtype":
        tensor["dtype"] = "F32"
    elif defect == "category":
        tensor["category"] = "sensitive"
    else:
        report["source"]["profile_sha256"] = "0" * 64

    class NeverCalled:
        def __getattr__(self, name):
            pytest.fail("altered model policy reached native dispatch")

    with pytest.raises(ValueError):
        create_profile_model_policy(NeverCalled(), "unused.gguf", profile, report, manifest, blocks)


def test_profile_model_policy_recurrent_table_matches_pinned_converter_float_policy():
    from embedded_jev.model_profile import profile_source
    from embedded_jev.prism_codec import profile_model_specification

    profile = mimo_profile()
    target = "model.language_model.layers.3.self_attn.k_proj.weight"
    tensors = []
    for name, shape in inventory._profile_tensor_shapes(profile).items():
        category, _ = inventory._classify(name, list(profile.geometry["layer_types"]), profile)
        tensors.append({"name": name, "shape": list(shape), "category": category,
                        "dtype": "BF16", "quantization_eligible": name == target})
    blocks = np.zeros((1024, 32, 34), dtype=np.uint8)
    blocks[:, :, 2:] = 0x55
    manifest = {
        "schema_version": 2, "format": "jev-profile-projection-v2", "profile_id": profile.profile_id,
        "profile_sha256": profile.sha256, "tensor": target, "gguf_tensor": "blk.3.attn_k.weight",
        "shape": [1024, 4096], "group_size": 128, "transform": "identity", "source_tensor_sha256": "f" * 64,
        "payload_bytes": blocks.nbytes, "payload_sha256": hashlib.sha256(memoryview(blocks)).hexdigest(),
    }
    specification = profile_model_specification(profile, {"source": profile_source(profile), "tensors": tensors}, manifest, blocks)
    table = {item["name"]: item for item in specification["tensors"]}
    assert len(table) == 427
    assert table["blk.0.ssm_conv1d.weight"] == {"name": "blk.0.ssm_conv1d.weight", "type": "F32", "ne": [4, 8192, 1, 1]}
    assert table["blk.0.ssm_dt.bias"]["type"] == table["blk.0.ssm_a"]["type"] == "F32"
    assert table["blk.0.attn_qkv.weight"]["ne"] == [4096, 8192, 1, 1]
    assert table["blk.3.attn_q.weight"]["ne"] == [4096, 8192, 1, 1]
    assert table["blk.3.attn_k.weight"]["type"] == "PQ2_0"


@pytest.mark.parametrize("defect", [None, "missing", "extra", "size", "digest"])
def test_profile_metadata_identity_without_weight_headers_or_execution(defect):
    from embedded_jev.model_profile import ProfileError, verify_profile_headers, verify_profile_metadata

    metadata, headers, record = _packaging_fixture(text_only=True)
    profile = parse_model_profile(json.dumps(record).encode())
    original = dict(metadata)
    if defect == "missing":
        metadata.pop("config.json")
    elif defect == "extra":
        metadata["undeclared.json"] = b"{}"
    elif defect == "size":
        metadata["config.json"] += b" "
    elif defect == "digest":
        changed = bytearray(metadata["config.json"])
        changed[0] ^= 1
        metadata["config.json"] = bytes(changed)
    if defect:
        message = "metadata file set" if defect in ("missing", "extra") else "metadata identity"
        with pytest.raises(ProfileError, match=message):
            verify_profile_metadata(profile, metadata)
        with pytest.raises(ProfileError, match=message):
            verify_profile_headers(profile, metadata, headers)
    else:
        assert verify_profile_metadata(profile, metadata) is None
        assert metadata == original
        verify_profile_headers(profile, metadata, headers)


def test_profile_metadata_extraction_preserves_existing_header_error_order():
    from embedded_jev.model_profile import ProfileError, verify_profile_headers, verify_profile_metadata

    metadata, _, record = _packaging_fixture(text_only=True)
    profile = parse_model_profile(json.dumps(record).encode())
    metadata["config.json"] += b" "
    with pytest.raises(ProfileError, match="metadata identity"):
        verify_profile_metadata(profile, metadata)
    with pytest.raises(ProfileError, match="shard file set"):
        verify_profile_headers(profile, metadata, {})
    metadata.pop("config.json")
    with pytest.raises(ProfileError, match="metadata file set"):
        verify_profile_headers(profile, metadata, {})


@pytest.mark.parametrize("text_only", [False, True])
@pytest.mark.parametrize("defect", [None, "metadata-pin", "tokenizer-pin", "missing-context", "boolean-context",
                                    "excess-context", "boolean-request", "excess-request", "mutable-metadata", "aggregate-budget"])
def test_profile_context_metadata_preparation_pins_bytes_without_tokenizer_semantic_claims(monkeypatch, text_only, defect):
    from embedded_jev import prism_codec
    from embedded_jev.prism_codec import prepare_profile_model_context_metadata

    metadata, _, record = _packaging_fixture(text_only=text_only)
    config = json.loads(metadata["config.json"])
    text = config if text_only else config["text_config"]
    text["max_position_embeddings"] = 512
    if defect == "missing-context":
        text.pop("max_position_embeddings")
    elif defect == "boolean-context":
        text["max_position_embeddings"] = True
    elif defect == "excess-context":
        text["max_position_embeddings"] = 1024**2 + 1
    metadata["config.json"] = json.dumps(config).encode()
    record["files"]["config.json"] = {"bytes": len(metadata["config.json"]), "sha256": hashlib.sha256(metadata["config.json"]).hexdigest()}
    profile = parse_model_profile(json.dumps(record).encode())
    tokenizer = b"{}"
    context_length = 256
    if defect == "metadata-pin":
        metadata["config.json"] += b" "
    elif defect == "tokenizer-pin":
        tokenizer = b"[]"
    elif defect == "boolean-request":
        context_length = True
    elif defect == "excess-request":
        context_length = 513
    elif defect == "mutable-metadata":
        metadata["config.json"] = bytearray(metadata["config.json"])
    elif defect == "aggregate-budget":
        monkeypatch.setattr(prism_codec, "MODEL_METADATA_BYTES_V2", 1)
    arguments = {"metadata_files": metadata, "tokenizer_bytes": tokenizer, "context_length": context_length}
    if defect:
        message = "metadata identity" if defect == "metadata-pin" else "tokenizer input identity" if defect == "tokenizer-pin" else "invalid source metadata" if defect in ("boolean-request", "mutable-metadata", "aggregate-budget") else "context length exceeds"
        with pytest.raises(ValueError, match=message):
            prepare_profile_model_context_metadata(profile, **arguments)
        return
    receipt = prepare_profile_model_context_metadata(profile, **arguments)
    assert receipt["context_length"] == 256 and receipt["source_context_length"] == 512
    assert receipt["tokenizer_sha256"] == profile.files["tokenizer.json"].sha256
    assert receipt["source_metadata_verification"] == "profile_pinned_caller_bytes"
    assert receipt["tokenizer_semantic_verification"] == receipt["source_weight_verification"] == "not_checked"
    assert receipt["source_template_file"] == profile.tokenization["template_file"]
    assert receipt["template_verification"] == "profile_pinned_bytes_not_rendered"
    assert receipt["gguf_template_verification"] == "not_checked"
    assert receipt["runtime_approval"] == "not_granted_by_metadata_preparation"
    with pytest.raises(TypeError):
        receipt["metadata_inputs"][0]["bytes"] = 0


def test_profile_model_conversion_plan_declares_dense_transforms_without_parity():
    from embedded_jev.prism_codec import profile_model_conversion_plan, profile_model_specification

    profile, report, manifest, blocks = _prefill_model_policy_fixture("down")
    plan = profile_model_conversion_plan(profile, report, manifest, blocks)
    table = {item["name"]: item for item in plan["tensors"]}
    assert table["blk.0.ffn_down.weight"]["transforms"] == ["frozen_projection_payload_not_dense_conversion"]
    assert table["output_norm.weight"]["transforms"] == ["add_one", "encode_f32"]
    assert table["blk.1.attn_q_norm.weight"]["transforms"] == ["add_one", "encode_f32"]
    assert table["blk.0.ssm_norm.weight"]["transforms"] == ["encode_f32"]
    assert table["blk.0.ssm_a"]["transforms"] == ["v_heads_grouped_to_tiled", "negative_exp", "encode_f32"]
    assert table["blk.0.ssm_dt.bias"]["transforms"] == ["v_heads_grouped_to_tiled", "encode_f32"]
    assert table["blk.0.ssm_conv1d.weight"]["transforms"] == ["v_rows_grouped_to_tiled", "squeeze", "encode_f32"]
    assert table["blk.0.attn_qkv.weight"]["transforms"] == ["v_rows_grouped_to_tiled", "encode_bf16"]
    assert table["blk.0.attn_gate.weight"]["transforms"] == ["v_heads_grouped_to_tiled", "encode_bf16"]
    assert table["blk.0.ssm_out.weight"]["transforms"] == ["v_columns_grouped_to_tiled", "encode_bf16"]
    assert table["blk.1.attn_q.weight"]["transforms"] == ["encode_bf16"]
    expected = profile_model_specification(profile, report, manifest, blocks)
    assert [{key: item[key] for key in ("name", "type", "ne")} for item in plan["tensors"]] == expected["tensors"]
    assert all(item["arithmetic_dtype"] == ("not_applicable" if item["type"] == "PQ2_0" else "F32") for item in plan["tensors"])
    assert plan["tensor_byte_verification"] == "not_checked"
    assert plan["source_payload_verification"] == "declared_digest_not_rehashed_source_tensor"
    assert plan["runtime_approval"] == "not_granted_by_conversion_plan"


@pytest.mark.parametrize("source_shards", [1, 2])
@pytest.mark.parametrize("vector_dtype", ["BF16", "F32"])
@pytest.mark.parametrize("vocab_size", [32, 16384])
def test_profile_model_streaming_resource_plan_v2_counts_serial_storage_without_io(monkeypatch, source_shards, vector_dtype, vocab_size):
    from embedded_jev import prism_codec

    profile, report, manifest, blocks = _prefill_model_policy_fixture(source_shards=source_shards,
        vector_dtype=vector_dtype, conv_dtype=vector_dtype, vocab_size=vocab_size)
    original_report = json.dumps(report, sort_keys=True)
    source_bytes = sum(profile.files[name].bytes for name in profile.packaging["shards"])
    metadata_input_bytes = sum(pin.bytes for name, pin in profile.files.items() if name not in profile.packaging["shards"])
    sizes, temporary_sizes = [], []
    for item in report["tensors"]:
        elements = int(np.prod(item["shape"]))
        size = blocks.nbytes if item["name"] == manifest["tensor"] else elements * (4 if len(item["shape"]) != 2 else 2)
        sizes.append(size + (-size % 32))
        if item["name"] != manifest["tensor"]:
            temporary_sizes.append(size)
    expected_peak = source_bytes + metadata_input_bytes + blocks.nbytes + 2 * sum(sizes) + prism_codec.MODEL_METADATA_BYTES_V2 + max(temporary_sizes)

    def never_open(*arguments, **keywords):
        pytest.fail("resource planning attempted file I/O")

    with monkeypatch.context() as no_io:
        no_io.setattr("builtins.open", never_open)
        no_io.setattr(prism_codec.os, "open", never_open)
        plan = prism_codec.profile_model_streaming_resource_plan_v2(profile, report, manifest, blocks,
            maximum_source_file_bytes=6 * 1024**3, maximum_operand_bytes=2 * 1024**3, maximum_storage_bytes=expected_peak)
        assert plan["resource_policy_version"] == 2
        assert plan["logical_storage_peak_bytes"] == expected_peak
        assert plan["source_storage_bytes"] == source_bytes and len(plan["source_files"]) == source_shards
        assert plan["metadata_input_storage_bytes"] == metadata_input_bytes
        assert plan["frozen_payload_input_bytes"] == blocks.nbytes
        assert plan["spool_storage_bytes"] == sum(sizes)
        assert plan["temporary_payload_peak_bytes"] == max(temporary_sizes)
        assert plan["projection_payload_sha256"] == manifest["payload_sha256"]
        assert next(item for item in plan["tensors"] if item["source_name"] == manifest["tensor"])["temporary_payload_bytes"] == 0
        assert plan["source_file_verification"] == plan["tensor_byte_verification"] == "not_checked"
        assert plan["memory_measurement"] == "not_performed"
        assert plan["runtime_approval"] == "not_granted_by_resource_plan"
        for field, value, message in (
            ("maximum_source_file_bytes", max(item["bytes"] for item in plan["source_files"]) - 1, "source file"),
            ("maximum_operand_bytes", max(item["source_bytes"] for item in plan["tensors"]) - 1, "operand"),
            ("maximum_storage_bytes", expected_peak - 1, "aggregate storage"),
        ):
            budgets = {"maximum_source_file_bytes": 6 * 1024**3, "maximum_operand_bytes": 2 * 1024**3,
                       "maximum_storage_bytes": expected_peak, field: value}
            with pytest.raises(ValueError, match=message):
                prism_codec.profile_model_streaming_resource_plan_v2(profile, report, manifest, blocks, **budgets)
        if vocab_size == 16384:
            assert expected_peak > 256 * 1024**2
            with pytest.raises(ValueError, match="aggregate storage"):
                prism_codec.profile_model_streaming_resource_plan_v2(profile, report, manifest, blocks,
                    maximum_source_file_bytes=6 * 1024**3, maximum_operand_bytes=2 * 1024**3,
                    maximum_storage_bytes=256 * 1024**2)
    plan["tensors"][0]["source_bytes"] = 0
    plan["source_files"][0]["bytes"] = 0
    assert json.dumps(report, sort_keys=True) == original_report
    assert profile.files[profile.packaging["shards"][0]].bytes > 0


@pytest.mark.parametrize("field,ceiling", [("maximum_source_file_bytes", 6 * 1024**3),
                                          ("maximum_operand_bytes", 2 * 1024**3), ("maximum_storage_bytes", 48 * 1024**3)])
@pytest.mark.parametrize("defect", [0, -1, True, 1.0, "over"])
def test_profile_model_streaming_resource_plan_v2_refuses_invalid_budgets_before_plan(monkeypatch, field, ceiling, defect):
    from embedded_jev import prism_codec

    def never_plan(*arguments, **keywords):
        pytest.fail("invalid resource budget reached conversion planning")

    monkeypatch.setattr(prism_codec, "profile_model_conversion_plan", never_plan)
    budgets = {"maximum_source_file_bytes": 6 * 1024**3, "maximum_operand_bytes": 2 * 1024**3,
               "maximum_storage_bytes": 48 * 1024**3, field: ceiling + 1 if defect == "over" else defect}
    with pytest.raises(ValueError, match="invalid v2 streaming resource budget"):
        prism_codec.profile_model_streaming_resource_plan_v2(None, {}, {}, None, **budgets)


@pytest.mark.parametrize("emit", [False, True])
@pytest.mark.parametrize("template_source,defect", [
    ("source", None), ("gguf_embedded", None),
    ("source", "unicode"), ("gguf_embedded", "unicode"),
    *[(source, defect) for source in ("source", "gguf_embedded") for defect in ("pin", "empty", "nul", "utf8", "aggregate")],
    ("gguf_embedded", "missing"), ("gguf_embedded", "unbound"), ("gguf_embedded", "mutable"),
    ("source", "unexpected-external"), ("source", "selection"),
])
def test_profile_template_preparation_is_explicit_pinned_and_never_rendered(monkeypatch, template_source, defect, emit):
    from embedded_jev import prism_codec

    metadata, _, record = _packaging_fixture(text_only=True)
    config = json.loads(metadata["config.json"])
    config["max_position_embeddings"] = 512
    metadata["config.json"] = json.dumps(config).encode()
    source_name, external_name = record["tokenization"]["template_file"], "gguf_template.jinja"
    metadata[source_name], external = b"source {{ message }}", b"gguf {{ other }}"
    selected = {"empty": b" \n", "nul": b"bad\x00template", "utf8": b"\xff", "unicode": b"text \xc3\xa9 {{ value }}"}.get(defect)
    if selected is not None:
        if template_source == "source":
            metadata[source_name] = selected
        else:
            external = selected
    for name, data in (*metadata.items(), (external_name, external)):
        record["files"][name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    if defect != "unbound":
        record["tokenization"].update(gguf_template_file=external_name, gguf_template_origin={
            "model": "example/Synthetic-GGUF", "revision": "3" * 40, "file": "toy.gguf", "file_sha256": "4" * 64,
        })
    profile = parse_model_profile(json.dumps(record).encode())
    extra = external if template_source == "gguf_embedded" or defect == "unexpected-external" else None
    if defect == "pin":
        if template_source == "source":
            metadata[source_name] += b" "
        else:
            extra += b" "
    elif defect == "missing":
        extra = None
    elif defect == "mutable":
        extra = bytearray(extra)
    elif defect == "selection":
        template_source = "automatic"
    if defect == "aggregate" or defect is None or defect == "unicode":
        total = sum(len(data) for data in metadata.values()) + len(b"{}") + len(extra or b"")
        monkeypatch.setattr(prism_codec, "MODEL_METADATA_BYTES_V2", total - (defect == "aggregate"))
    arguments = {"metadata_files": metadata, "tokenizer_bytes": b"{}", "context_length": 256,
                 "template_source": template_source, "gguf_template_bytes": extra}
    operation = prism_codec.write_profile_model_template_metadata if emit else prism_codec.prepare_profile_model_template_metadata
    writer = _profile_metadata_writer_fixture()
    if emit:
        arguments["writer"] = writer
    if defect and defect != "unicode":
        with pytest.raises(ValueError, match="metadata identity" if defect == "pin" and template_source == "source" else "template"):
            operation(profile, **arguments)
        assert writer.kv_data == [{}]
        return
    receipt = operation(profile, **arguments)
    expected = metadata[source_name] if template_source == "source" else external
    assert receipt["template"].encode() == expected
    assert receipt["selected_template_sha256"] == hashlib.sha256(expected).hexdigest()
    assert receipt["selected_template_verification"] == "profile_pinned_bytes_not_rendered"
    assert receipt["template_render_verification"] == receipt["prompt_label_boundary_verification"] == "not_checked"
    assert receipt["tokenizer_semantic_verification"] == receipt["source_weight_verification"] == "not_checked"
    assert receipt["gguf_template_verification"] == ("profile_pinned_bytes_not_rendered" if template_source == "gguf_embedded" else "not_checked")
    assert receipt["gguf_template_origin"] == (profile.tokenization["gguf_template_origin"] if template_source == "gguf_embedded" else None)
    assert receipt["gguf_template_origin_verification"] == ("profile_declared_not_source_gguf_verified" if template_source == "gguf_embedded" else "not_applicable")
    with pytest.raises(TypeError):
        receipt["template"] = "other"
    if emit:
        assert writer.kv_data == [{"tokenizer.chat_template": ("str", [expected.decode()])}]
        assert receipt["records"][0]["values"] == (expected.decode(),)
        with pytest.raises(ValueError, match="conflicting"):
            operation(profile, **arguments)
        assert writer.kv_data == [{"tokenizer.chat_template": ("str", [expected.decode()])}]


@pytest.mark.parametrize("defect", ["architecture", "missing-string", "duplicate", "split-writer"])
def test_profile_template_emission_preflights_string_protocol_without_mutation(defect):
    from embedded_jev.prism_codec import write_profile_model_template_metadata

    profile, metadata = _profile_tokenizer_preparation_fixture(b"{}")
    writer = _profile_metadata_writer_fixture()
    if defect == "architecture":
        writer.arch = "qwen3"
    elif defect == "missing-string":
        writer.add_string = None
    elif defect == "duplicate":
        writer.kv_data[0]["tokenizer.chat_template"] = ("str", ["keep"])
    else:
        writer.kv_data.append({})
    before = json.dumps(writer.kv_data, sort_keys=True)
    with pytest.raises(ValueError, match="conflicting architecture metadata writer"):
        write_profile_model_template_metadata(profile, metadata_files=metadata, tokenizer_bytes=b"{}",
            context_length=256, template_source="source", writer=writer)
    assert json.dumps(writer.kv_data, sort_keys=True) == before


def _profile_tokenizer_preparation_fixture(tokenizer_bytes):
    metadata, _, record = _packaging_fixture(text_only=True)
    config = json.loads(metadata["config.json"])
    config["max_position_embeddings"] = 512
    metadata["config.json"] = json.dumps(config).encode()
    record["files"]["config.json"] = {"bytes": len(metadata["config.json"]), "sha256": hashlib.sha256(metadata["config.json"]).hexdigest()}
    record["files"]["tokenizer.json"] = {"bytes": len(tokenizer_bytes), "sha256": hashlib.sha256(tokenizer_bytes).hexdigest()}
    return parse_model_profile(json.dumps(record).encode()), metadata


@pytest.mark.parametrize("defect", [None, "legacy", "missing-merges", "merges-type", "pair-type", "pair-length", "empty",
                                    "unknown", "joined", "duplicate", "prefix", "suffix", "base-vocabulary", "base-id",
                                    "duplicate-json", "nonfinite-json"])
def test_profile_bpe_metadata_preserves_rank_and_preprocessing_without_gguf_claims(monkeypatch, defect):
    import sys
    from types import SimpleNamespace

    from embedded_jev.prism_codec import prepare_profile_model_bpe_metadata

    vocabulary = {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4}
    document = {"model": {"type": "BPE", "vocab": vocabulary.copy(), "merges": [["a", "b"], ["ab", "c"]]},
                "normalizer": None, "pre_tokenizer": {"type": "Sequence", "pretokenizers": []},
                "decoder": None, "post_processor": None}
    model = document["model"]
    if defect == "legacy":
        model["merges"] = ["a b", "ab c"]
    elif defect == "missing-merges":
        del model["merges"]
    elif defect == "merges-type":
        model["merges"] = "a b"
    elif defect == "pair-type":
        model["merges"] = [42]
    elif defect == "pair-length":
        model["merges"] = [["a", "b", "c"]]
    elif defect == "empty":
        model["merges"] = [["a", ""]]
    elif defect == "unknown":
        model["merges"] = [["a", "missing"]]
    elif defect == "joined":
        model["merges"] = [["b", "c"]]
    elif defect == "duplicate":
        model["merges"] *= 2
    elif defect == "prefix":
        model["continuing_subword_prefix"] = "##"
    elif defect == "suffix":
        model["end_of_word_suffix"] = "</w>"
    elif defect == "base-vocabulary":
        del model["vocab"]["ab"]
    elif defect == "base-id":
        model["vocab"]["ab"] = 1
    tokenizer_bytes = json.dumps(document).encode()
    if defect == "duplicate-json":
        tokenizer_bytes = b'{"model":{},' + tokenizer_bytes[1:]
    elif defect == "nonfinite-json":
        tokenizer_bytes = tokenizer_bytes[:-1] + b',"invalid":NaN}'
    profile, metadata = _profile_tokenizer_preparation_fixture(tokenizer_bytes)
    calls = []

    def from_str(text):
        calls.append(text)
        return SimpleNamespace(get_vocab=lambda *, with_added_tokens: dict(vocabulary))

    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=SimpleNamespace(from_str=from_str)))
    if defect in ("duplicate-json", "nonfinite-json"):
        with pytest.raises(ValueError):
            prepare_profile_model_bpe_metadata(profile, metadata_files=metadata, tokenizer_bytes=tokenizer_bytes, context_length=256)
        assert calls == []
        return
    if defect not in (None, "legacy"):
        with pytest.raises(ValueError, match="BPE"):
            prepare_profile_model_bpe_metadata(profile, metadata_files=metadata, tokenizer_bytes=tokenizer_bytes, context_length=256)
        assert calls == [tokenizer_bytes.decode()]
        return
    receipt = prepare_profile_model_bpe_metadata(profile, metadata_files=metadata, tokenizer_bytes=tokenizer_bytes, context_length=256)
    assert calls == [tokenizer_bytes.decode()]
    assert receipt["bpe_merge_pairs"] == (("a", "b"), ("ab", "c")) and receipt["bpe_merge_count"] == 2
    assert receipt["bpe_base_vocabulary_size"] == 5
    assert json.loads(receipt["bpe_options_json"]) == {"type": "BPE"}
    assert {name: json.loads(data) for name, data in receipt["preprocessing_json"].items()} == {
        name: document[name] for name in ("normalizer", "pre_tokenizer", "decoder", "post_processor")}
    assert receipt["preprocessing_verification"] == "source_declarations_only_not_gguf_mapped"
    assert receipt["gguf_tokenizer_verification"] == receipt["tokenizer_semantic_verification"] == "not_checked"
    assert receipt["runtime_approval"] == "not_granted_by_bpe_metadata_preparation"
    with pytest.raises(TypeError):
        receipt["preprocessing_json"]["normalizer"] = b"{}"


@pytest.mark.parametrize("defect", [None, "empty", "rows", "row-budget", "boolean-id", "unknown-id", "duplicate", "content",
                                    "flag-type", "missing-flag", "parsed-content", "parsed-flag", "parsed-ids", "parsed-id-type", "content-budget",
                                    "row-type", "empty-content", "utf8-budget", "byte-limit", "aggregate-budget", "aggregate-limit",
                                    "parsed-type", "parsed-flag-type", "extra-parsed-id"])
def test_profile_added_token_metadata_is_bounded_pinned_and_not_a_token_type_mapping(monkeypatch, defect):
    import sys
    from types import SimpleNamespace

    from embedded_jev.prism_codec import prepare_profile_model_added_token_metadata

    vocabulary = {"a": 0, "<A>": 1}
    row = {"id": 1, "content": "<A>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}
    parsed = {1: SimpleNamespace(**{name: value for name, value in row.items() if name != "id"})}
    rows = [row.copy()]
    if defect == "empty":
        rows, parsed = [], {}
    elif defect == "rows":
        rows = {}
    elif defect == "row-budget":
        rows *= 4097
    elif defect == "boolean-id":
        rows[0]["id"] = True
    elif defect == "unknown-id":
        rows[0]["id"] = 2
    elif defect == "duplicate":
        rows *= 2
    elif defect == "content":
        rows[0]["content"] = "other"
    elif defect == "flag-type":
        rows[0]["special"] = 1
    elif defect == "missing-flag":
        del rows[0]["normalized"]
    elif defect == "parsed-content":
        parsed[1].content = "other"
    elif defect == "parsed-flag":
        parsed[1].normalized = True
    elif defect == "parsed-ids":
        parsed = {}
    elif defect == "parsed-id-type":
        parsed = {True: parsed[1]}
    elif defect == "content-budget":
        del vocabulary["<A>"]
        vocabulary["x" * 4097] = 1
        rows[0]["content"] = "x" * 4097
    elif defect == "row-type":
        rows[0] = None
    elif defect in ("empty-content", "utf8-budget", "byte-limit"):
        content = {"empty-content": "", "utf8-budget": "\u00e9" * 2049, "byte-limit": "x" * 4096}[defect]
        vocabulary = {"a": 0, content: 1}
        rows[0]["content"] = parsed[1].content = content
    elif defect in ("aggregate-budget", "aggregate-limit"):
        vocabulary, parsed, rows = {"a": 0}, {}, []
        for token_id in range(1, 18 if defect == "aggregate-budget" else 17):
            content = f"{token_id:04d}" + "x" * 4092
            vocabulary[content] = token_id
            rows.append({**row, "id": token_id, "content": content})
            parsed[token_id] = SimpleNamespace(**{name: value for name, value in rows[-1].items() if name != "id"})
    elif defect == "parsed-type":
        parsed = []
    elif defect == "parsed-flag-type":
        parsed[1].normalized = 0
    elif defect == "extra-parsed-id":
        parsed[0] = parsed[1]
    tokenizer_bytes = json.dumps({"model": {"type": "BPE"}, "added_tokens": rows}).encode()
    profile, metadata = _profile_tokenizer_preparation_fixture(tokenizer_bytes)
    calls = []

    def from_str(text):
        calls.append(text)
        return SimpleNamespace(get_vocab=lambda *, with_added_tokens: dict(vocabulary), get_added_tokens_decoder=lambda: parsed)

    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=SimpleNamespace(from_str=from_str)))
    if defect not in (None, "empty", "byte-limit", "aggregate-limit"):
        with pytest.raises(ValueError, match="added tokenizer"):
            prepare_profile_model_added_token_metadata(profile, metadata_files=metadata, tokenizer_bytes=tokenizer_bytes, context_length=256)
        assert calls == [tokenizer_bytes.decode()]
        return
    receipt = prepare_profile_model_added_token_metadata(profile, metadata_files=metadata, tokenizer_bytes=tokenizer_bytes, context_length=256)
    assert calls == [tokenizer_bytes.decode()]
    assert receipt["added_token_count"] == len(rows)
    assert receipt["added_token_content_bytes"] == sum(len(item["content"].encode()) for item in rows)
    assert receipt["added_token_round_trip_verification"] == receipt["gguf_token_type_verification"] == "not_checked"
    assert receipt["pre_tokenizer_tag_verification"] == receipt["gguf_tokenizer_verification"] == "not_checked"
    assert tuple(dict(item) for item in receipt["added_token_records"]) == tuple(
        {"token_id": item["id"], **{name: value for name, value in item.items() if name != "id"}} for item in rows)
    if rows:
        with pytest.raises(TypeError):
            receipt["added_token_records"][0]["special"] = False


@pytest.mark.parametrize("defect", [None, "policy", "capacity", "missing-row", "normalization", "user-space", "control-space",
                                    "nul", "empty", "string-budget", "content-budget"])
def test_profile_source_token_types_are_declarations_without_transforms(monkeypatch, defect):
    import sys
    from types import SimpleNamespace

    from embedded_jev.prism_codec import prepare_profile_model_token_type_metadata

    tokenizer_bytes = b'{"model":{"type":"BPE"}}'
    profile, metadata = _profile_tokenizer_preparation_fixture(tokenizer_bytes)
    tokens = [f"token{index}" for index in range(profile.geometry["vocab_size"])]
    tokens[1:5] = ["user", "special", "<|marker|>", "<pad>"]
    rows = [{"id": token_id, "content": tokens[token_id], "single_word": False, "lstrip": False, "rstrip": False,
             "normalized": True, "special": token_id == 2} for token_id in (1, 2, 3)]
    if defect in ("user-space", "control-space"):
        tokens[1] = rows[0]["content"] = "\u2581user"
        rows[0]["special"] = defect == "control-space"
    elif defect == "normalization":
        rows[0]["normalized"] = False
    elif defect in ("nul", "empty", "string-budget"):
        tokens[0] = {"nul": "base\0suffix", "empty": "", "string-budget": "x" * 4097}[defect]
    elif defect == "content-budget":
        for token_id in range(5, len(tokens)):
            tokens[token_id] = f"{token_id:04d}" + "x" * 4092
    vocabulary = {token: token_id for token_id, token in enumerate(tokens)}
    if defect == "missing-row":
        vocabulary.pop(tokens[-1])
    tokenizer_bytes = json.dumps({"model": {"type": "BPE"}, "added_tokens": rows}).encode()
    profile, metadata = _profile_tokenizer_preparation_fixture(tokenizer_bytes)
    if defect == "capacity":
        profile = SimpleNamespace(geometry={"vocab_size": 4097})
    calls = []

    def from_str(text):
        calls.append(text)
        return SimpleNamespace(get_vocab=lambda *, with_added_tokens: dict(vocabulary), get_added_tokens_decoder=lambda: {
            row["id"]: SimpleNamespace(**{name: value for name, value in row.items() if name != "id"}) for row in rows})

    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=SimpleNamespace(from_str=from_str)))
    arguments = {"metadata_files": metadata, "tokenizer_bytes": tokenizer_bytes, "context_length": 256,
                 "token_type_policy": "unsupported" if defect == "policy" else "pinned_prism_no_transform_v1"}
    if defect not in (None, "control-space"):
        with pytest.raises(ValueError, match="source token-type"):
            prepare_profile_model_token_type_metadata(profile, **arguments)
        assert len(calls) == (0 if defect in ("policy", "capacity") else 1)
        return
    prepared = prepare_profile_model_token_type_metadata(profile, **arguments)
    records = prepared["source_token_type_records"]
    assert [record["token_type"] for record in records[:5]] == ["NORMAL", "CONTROL" if defect == "control-space" else "USER_DEFINED",
                                                               "CONTROL", "CONTROL", "NORMAL"]
    assert [record["token"] for record in records] == tokens
    assert prepared["source_token_content_bytes"] == sum(len(token.encode()) for token in tokens)
    assert prepared["metadata_emission"] == "not_performed" and prepared["gguf_token_type_verification"] == "not_checked"
    assert prepared["gguf_tokenizer_verification"] == "not_checked" and prepared["pre_tokenizer_tag_verification"] == "not_checked"
    assert prepared["runtime_approval"] == "not_granted_by_source_token_type_preparation"
    with pytest.raises(TypeError):
        records[0]["token_type"] = "CONTROL"


@pytest.mark.parametrize("defect", [None, "special-id", "special-encoding", "label-id", "label-encoding", "label-special"])
def test_profile_tokenizer_contract_is_standalone_and_not_a_prompt_boundary(monkeypatch, defect):
    import sys
    from types import SimpleNamespace

    from embedded_jev.prism_codec import prepare_profile_model_tokenizer_contract

    tokenizer_bytes = b'{"model":{"type":"BPE"}}'
    profile, metadata = _profile_tokenizer_preparation_fixture(tokenizer_bytes)
    vocabulary = {f"token{index}": index for index in range(profile.geometry["vocab_size"])}
    for token, token_id in {**profile.tokenization["special_token_ids"], **profile.tokenization["label_token_ids"]}.items():
        del vocabulary[f"token{token_id}"]
        vocabulary[token] = token_id
    calls = []

    def token_to_id(token):
        if defect == "special-id" and token == "<think>" or defect == "label-id" and token == "A":
            return None
        return vocabulary.get(token)

    def encode(token, *, add_special_tokens):
        assert add_special_tokens is False
        token_ids = [vocabulary[token]]
        if defect == "special-encoding" and token == "<think>" or defect == "label-encoding" and token == "A":
            token_ids *= 2
        return SimpleNamespace(ids=token_ids)

    def from_str(text):
        calls.append(text)
        added = {vocabulary["A"]: SimpleNamespace(special=True)} if defect == "label-special" else {}
        return SimpleNamespace(get_vocab=lambda *, with_added_tokens: dict(vocabulary), token_to_id=token_to_id,
                               encode=encode, get_added_tokens_decoder=lambda: added)

    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=SimpleNamespace(from_str=from_str)))
    if defect:
        with pytest.raises(ValueError, match="special tokenizer" if defect.startswith("special") else "label tokenizer"):
            prepare_profile_model_tokenizer_contract(profile, metadata_files=metadata,
                tokenizer_bytes=tokenizer_bytes, context_length=256)
        assert calls == [tokenizer_bytes.decode()]
        return
    receipt = prepare_profile_model_tokenizer_contract(profile, metadata_files=metadata,
        tokenizer_bytes=tokenizer_bytes, context_length=256)
    assert calls == [tokenizer_bytes.decode()]
    assert receipt["label_token_ids"] == profile.tokenization["label_token_ids"]
    assert receipt["special_token_ids"] == profile.tokenization["special_token_ids"]
    assert receipt["special_token_verification"] == "declared_ids_and_standalone_encoding"
    assert receipt["label_token_verification"] == "standalone_single_non_special_ids"
    assert receipt["prompt_label_boundary_verification"] == receipt["tokenizer_semantic_verification"] == "not_checked"
    assert receipt["gguf_tokenizer_verification"] == receipt["gguf_template_verification"] == "not_checked"
    with pytest.raises(TypeError):
        receipt["label_token_ids"]["A"] = 3


@pytest.mark.parametrize("defect", [None, "metadata-pin", "tokenizer-pin", "model-kind", "parser", "empty", "duplicate-id", "gap", "boolean-id", "capacity"])
def test_profile_tokenizer_vocabulary_is_pinned_bounded_and_not_gguf_semantics(monkeypatch, defect):
    import sys
    from types import SimpleNamespace

    from embedded_jev.prism_codec import prepare_profile_model_tokenizer_vocabulary

    tokenizer_bytes = json.dumps({"model": {"type": "WordLevel" if defect == "model-kind" else "BPE"}}).encode()
    profile, metadata = _profile_tokenizer_preparation_fixture(tokenizer_bytes)
    vocabulary = {"B": 1, "A": 0}
    if defect == "metadata-pin":
        metadata["config.json"] += b" "
    elif defect == "tokenizer-pin":
        tokenizer_bytes += b" "
    elif defect == "empty":
        vocabulary = {}
    elif defect == "duplicate-id":
        vocabulary["B"] = 0
    elif defect == "gap":
        vocabulary["B"] = 2
    elif defect == "boolean-id":
        vocabulary["A"] = False
    elif defect == "capacity":
        vocabulary = {str(token_id): token_id for token_id in range(profile.geometry["vocab_size"] + 1)}
    calls = []

    def from_str(text):
        calls.append(text)
        if defect in ("metadata-pin", "tokenizer-pin", "model-kind"):
            pytest.fail("unverified tokenizer reached parser")
        if defect == "parser":
            raise ValueError("synthetic parser refusal")
        return SimpleNamespace(get_vocab=lambda *, with_added_tokens: dict(vocabulary) if with_added_tokens else {})

    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=SimpleNamespace(from_str=from_str)))
    arguments = {"metadata_files": metadata, "tokenizer_bytes": tokenizer_bytes, "context_length": 256}
    if defect:
        message = ("metadata identity" if defect == "metadata-pin" else "tokenizer input identity" if defect == "tokenizer-pin"
                   else "unsupported source tokenizer" if defect == "model-kind" else "unable to parse" if defect == "parser"
                   else "vocabulary IDs")
        with pytest.raises(ValueError, match=message):
            prepare_profile_model_tokenizer_vocabulary(profile, **arguments)
        assert len(calls) == (0 if defect in ("metadata-pin", "tokenizer-pin", "model-kind") else 1)
        return
    receipt = prepare_profile_model_tokenizer_vocabulary(profile, **arguments)
    assert calls == [tokenizer_bytes.decode()]
    assert receipt["tokens"] == ("A", "B")
    assert receipt["tokenizer_vocabulary_size"] == 2
    assert receipt["unassigned_profile_token_rows"] == profile.geometry["vocab_size"] - 2
    assert receipt["tokenizer_vocabulary_verification"] == "trusted_tokenizers_parser_contiguous_ids"
    for key in ("tokenizer_semantic_verification", "special_token_verification", "label_token_verification", "gguf_tokenizer_verification", "gguf_template_verification"):
        assert receipt[key] == "not_checked"
    with pytest.raises(TypeError):
        receipt["tokens"] = ()


@pytest.mark.parametrize("defect", [None, "source-pin", "tokenizer-pin", "context-limit", "unbound-template", "policy-manifest",
    "duplicate-policy", "duplicate-context", "duplicate-template", "missing-string", "architecture", "split-writer"])
def test_profile_source_metadata_emission_preflights_all_groups_before_mutation(defect):
    from embedded_jev.prism_codec import write_profile_model_source_metadata

    metadata = {}
    profile, report, manifest, blocks = _prefill_model_policy_fixture(metadata_output=metadata)
    writer = _profile_metadata_writer_fixture()
    tokenizer_bytes, context_length, template_source = b"{}", 256, "source"
    if defect == "source-pin":
        metadata["config.json"] += b" "
    elif defect == "tokenizer-pin":
        tokenizer_bytes += b" "
    elif defect == "context-limit":
        context_length = 513
    elif defect == "unbound-template":
        template_source = "gguf_embedded"
    elif defect == "policy-manifest":
        manifest["payload_sha256"] = "0" * 64
    elif defect and defect.startswith("duplicate"):
        name = {"duplicate-policy": "jev.model.profile_sha256", "duplicate-context": "qwen35.context_length",
                "duplicate-template": "tokenizer.chat_template"}[defect]
        writer.kv_data[0][name] = ("str", ["keep"])
    elif defect == "missing-string":
        writer.add_string = None
    elif defect == "architecture":
        writer.arch = "qwen3"
    elif defect == "split-writer":
        writer.kv_data.append({})
    before = json.dumps(writer.kv_data, sort_keys=True)
    arguments = {"metadata_files": metadata, "tokenizer_bytes": tokenizer_bytes, "context_length": context_length,
                 "template_source": template_source, "writer": writer}
    if defect:
        message = {"source-pin": "metadata identity", "tokenizer-pin": "tokenizer input identity",
                   "context-limit": "context length", "unbound-template": "missing separately pinned",
                   "policy-manifest": "payload identity"}.get(defect, "conflicting")
        with pytest.raises(ValueError, match=message):
            write_profile_model_source_metadata(profile, report, manifest, blocks, **arguments)
        assert json.dumps(writer.kv_data, sort_keys=True) == before
        return
    receipt = write_profile_model_source_metadata(profile, report, manifest, blocks, **arguments)
    assert receipt["metadata_count"] == len(writer.kv_data[0]) == 26
    assert receipt["metadata_scope"] == "policy_context_template_subset"
    assert writer.kv_data[0]["jev.model.profile_sha256"] == ("str", [profile.sha256])
    assert writer.kv_data[0]["qwen35.context_length"] == ("u32", [256])
    assert writer.kv_data[0]["tokenizer.chat_template"] == ("str", [metadata[profile.tokenization["template_file"]].decode()])
    assert receipt["source_context_length"] == 512
    assert receipt["tokenizer_semantic_verification"] == receipt["source_weight_verification"] == "not_checked"
    assert receipt["runtime_approval"] == "not_granted_by_source_metadata_emission"
    assert all(tuple(writer.kv_data[0][record["name"]][1]) == record["values"] for record in receipt["records"])


@pytest.mark.parametrize("defect", [None, "metadata-pin", "tokenizer-pin", "manifest", "empty-merges", "duplicate-policy",
                                    "duplicate-context", "duplicate-template", "duplicate-vocabulary", "duplicate-merges",
                                    "missing-array", "architecture", "split-writer", "row-budget"])
def test_profile_selected_metadata_writer_preflights_every_group_before_mutation(monkeypatch, defect):
    import sys
    from types import SimpleNamespace

    from embedded_jev.prism_codec import write_profile_model_selected_metadata

    tokens = ["a", "b", "ab"] + [f"token{token_id}" for token_id in range(3, 32)]
    vocabulary = {token: token_id for token_id, token in enumerate(tokens)}
    tokenizer_bytes = json.dumps({"model": {"type": "BPE", "vocab": vocabulary,
        "merges": [] if defect == "empty-merges" else [["a", "b"]]}}).encode()
    metadata = {}
    profile, report, manifest, blocks = _prefill_model_policy_fixture(metadata_output=metadata, tokenizer_bytes=tokenizer_bytes)
    calls = []

    def from_str(text):
        calls.append(text)
        return SimpleNamespace(get_vocab=lambda *, with_added_tokens: dict(vocabulary))

    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=SimpleNamespace(from_str=from_str)))
    writer = _profile_metadata_writer_fixture()
    arguments = {"metadata_files": metadata, "tokenizer_bytes": tokenizer_bytes, "context_length": 256,
                 "template_source": "source", "writer": writer}
    if defect == "metadata-pin":
        metadata["config.json"] += b" "
    elif defect == "tokenizer-pin":
        arguments["tokenizer_bytes"] += b" "
    elif defect == "manifest":
        manifest["payload_sha256"] = "0" * 64
    elif defect and defect.startswith("duplicate-"):
        name = {"duplicate-policy": "jev.model.profile_sha256", "duplicate-context": "qwen35.context_length",
                "duplicate-template": "tokenizer.chat_template", "duplicate-vocabulary": "tokenizer.ggml.tokens",
                "duplicate-merges": "tokenizer.ggml.merges"}[defect]
        writer.kv_data[0][name] = ("str", ["keep"])
    elif defect == "missing-array":
        writer.add_array = None
    elif defect == "architecture":
        writer.arch = "qwen3"
    elif defect == "split-writer":
        writer.kv_data.append({})
    elif defect == "row-budget":
        arguments["maximum_vocabulary_rows"] = True
    before = json.dumps(writer.kv_data, sort_keys=True)
    if defect:
        with pytest.raises(ValueError):
            write_profile_model_selected_metadata(profile, report, manifest, blocks, **arguments)
        assert json.dumps(writer.kv_data, sort_keys=True) == before
        return
    receipt = write_profile_model_selected_metadata(profile, report, manifest, blocks, **arguments)
    assert calls == [tokenizer_bytes.decode()]
    assert receipt["metadata_count"] == len(writer.kv_data[0]) == 28
    assert receipt["vocabulary_row_cap"] == 4096
    assert receipt["metadata_scope"] == "policy_context_template_tokenizer_arrays_subset"
    assert tuple(writer.kv_data[0]["tokenizer.ggml.tokens"][1]) == tuple(tokens)
    assert writer.kv_data[0]["tokenizer.ggml.merges"][1] == ["a b"]
    assert receipt["metadata_verification"] == "declared_records_emitted_not_file_verified"
    assert receipt["gguf_tokenizer_verification"] == receipt["gguf_token_type_verification"] == receipt["pre_tokenizer_tag_verification"] == "not_checked"


@pytest.mark.parametrize("defect", [None, "metadata-pin", "tokenizer-pin", "manifest", "selection", "file-budget", "record-budget",
                                    "row-budget", "hash", "seconds", "preparation-deadline", "verification-deadline"])
def test_profile_selected_metadata_file_proof_prepares_all_groups_under_one_deadline(monkeypatch, defect):
    import sys
    from types import SimpleNamespace

    from embedded_jev import prism_codec

    tokens = ["a", "b", "ab"] + [f"token{token_id}" for token_id in range(3, 32)]
    vocabulary = {token: token_id for token_id, token in enumerate(tokens)}
    tokenizer_bytes = json.dumps({"model": {"type": "BPE", "vocab": vocabulary, "merges": [["a", "b"]]}}).encode()
    metadata = {}
    profile, report, manifest, blocks = _prefill_model_policy_fixture(metadata_output=metadata, tokenizer_bytes=tokenizer_bytes)
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=SimpleNamespace(from_str=lambda text:
        SimpleNamespace(get_vocab=lambda *, with_added_tokens: dict(vocabulary)))))
    calls = []

    def verify(actual_profile, **kwargs):
        assert actual_profile is profile and kwargs["model_owner"] is kwargs["descriptor"] is None
        records = kwargs["declarations"]
        assert len(records) == 29 and records[0]["name"] == "general.architecture"
        assert tuple(item["name"] for item in records[-4:]) == (
            "qwen35.context_length", "tokenizer.chat_template", "tokenizer.ggml.tokens", "tokenizer.ggml.merges")
        assert records[-2]["values"] == tuple(tokens) and records[-1]["values"] == ("a b",)
        assert kwargs["maximum_metadata_bytes"] == 8192 and kwargs["policy_version"] == 2
        calls.append(kwargs)
        return {"metadata_names": tuple(item["name"] for item in records)}

    monkeypatch.setattr(prism_codec, "_verify_profile_gguf_metadata_records", verify)
    arguments = {"reader": object(), "metadata_files": metadata, "tokenizer_bytes": tokenizer_bytes, "context_length": 256,
                 "template_source": "source", "expected_file_bytes": 100, "expected_file_sha256": "1" * 64}
    if defect == "metadata-pin":
        metadata["config.json"] += b" "
    elif defect == "tokenizer-pin":
        arguments["tokenizer_bytes"] += b" "
    elif defect == "manifest":
        manifest["payload_sha256"] = "0" * 64
    elif defect == "selection":
        arguments["template_source"] = "guess"
    elif defect == "file-budget":
        arguments["maximum_file_bytes"] = True
    elif defect == "record-budget":
        arguments["maximum_metadata_bytes"] = 128 * 1024 + 1
    elif defect == "row-budget":
        arguments["maximum_vocabulary_rows"] = 16385
    elif defect == "hash":
        arguments["expected_file_sha256"] = "bad"
    elif defect == "seconds":
        arguments["maximum_seconds"] = float("nan")
    elif defect in ("preparation-deadline", "verification-deadline"):
        clock = iter([5, 40] if defect == "preparation-deadline" else [5, 6, 40])
        monkeypatch.setattr(prism_codec.time, "monotonic", lambda: next(clock))
    if defect:
        with pytest.raises(ValueError):
            prism_codec.verify_profile_gguf_selected_metadata_v2(profile, report, manifest, blocks, **arguments)
        assert len(calls) == (1 if defect == "verification-deadline" else 0)
        return
    receipt = prism_codec.verify_profile_gguf_selected_metadata_v2(profile, report, manifest, blocks, **arguments)
    assert len(calls) == 1 and receipt["metadata_scope"] == "policy_context_template_tokenizer_arrays_subset"
    assert receipt["metadata_verification"] == "selected_typed_policy_source_array_records_match_file"
    assert receipt["gguf_token_type_verification"] == receipt["gguf_tokenizer_verification"] == receipt["pre_tokenizer_tag_verification"] == "not_checked"


@pytest.mark.parametrize("defect", [None, "full", "leading", "collision", "policy", "policy-type", "capacity", "metadata-pin",
                                    "tokenizer-pin", "duplicate", "boolean", "negative", "out-of-range", "parser"])
def test_profile_unused_token_rows_require_explicit_bounded_noncolliding_declarations(monkeypatch, defect):
    import sys
    from types import SimpleNamespace

    from embedded_jev.prism_codec import prepare_profile_model_unused_token_rows

    tokenizer_bytes = b'{"model":{"type":"BPE"}}'
    profile, metadata = _profile_tokenizer_preparation_fixture(tokenizer_bytes)
    capacity = profile.geometry["vocab_size"]
    vocabulary = {"a": 0, "b": 3}
    if defect == "full":
        vocabulary = {f"token{token_id}": token_id for token_id in range(capacity)}
    elif defect == "leading":
        vocabulary["a"] = 1
    elif defect == "collision":
        vocabulary = {"[PAD1]": 0, "b": 3}
    elif defect in ("duplicate", "boolean", "negative", "out-of-range"):
        vocabulary["b"] = {"duplicate": 0, "boolean": True, "negative": -1, "out-of-range": capacity}[defect]
    calls = []

    def from_str(text):
        calls.append(text)
        if defect == "parser":
            raise ValueError("synthetic parser refusal")
        return SimpleNamespace(get_vocab=lambda *, with_added_tokens: dict(vocabulary))

    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=SimpleNamespace(from_str=from_str)))
    policy = "pinned_prism_pad_unused_v1"
    if defect == "policy":
        policy = "guess"
    elif defect == "policy-type":
        policy = True
    elif defect == "capacity":
        profile = SimpleNamespace(geometry={"vocab_size": 4097})
    elif defect == "metadata-pin":
        metadata["config.json"] += b" "
    elif defect == "tokenizer-pin":
        tokenizer_bytes += b" "
    arguments = {"metadata_files": metadata, "tokenizer_bytes": tokenizer_bytes, "context_length": 256, "unused_policy": policy}
    if defect not in (None, "full", "leading"):
        with pytest.raises(ValueError):
            prepare_profile_model_unused_token_rows(profile, **arguments)
        assert len(calls) == (0 if defect in ("policy", "policy-type", "capacity", "metadata-pin", "tokenizer-pin") else 1)
        return
    receipt = prepare_profile_model_unused_token_rows(profile, **arguments)
    assert calls == [tokenizer_bytes.decode()]
    missing = sorted(set(range(capacity)) - set(vocabulary.values()))
    assert tuple(dict(row) for row in receipt["unused_token_rows"]) == tuple(
        {"token_id": token_id, "token": f"[PAD{token_id}]", "token_type": "UNUSED"} for token_id in missing)
    assert len(receipt["unused_token_rows"]) == receipt["unassigned_profile_token_rows"]
    assert receipt["metadata_emission"] == "not_performed"
    assert receipt["source_token_type_verification"] == receipt["gguf_token_type_verification"] == receipt["gguf_tokenizer_verification"] == "not_checked"
    if missing:
        with pytest.raises(TypeError):
            receipt["unused_token_rows"][0]["token"] = "other"


@pytest.mark.parametrize("defect", [None, "contiguous", "full", "leading", "empty", "duplicate", "boolean", "negative", "out-of-range",
                                    "non-string", "metadata-pin", "tokenizer-pin", "model-kind", "parser", "range-budget", "range-tail-budget"])
def test_profile_tokenizer_inventory_reports_missing_ranges_without_padding_or_relaxing_emission(monkeypatch, defect):
    import sys
    from types import SimpleNamespace

    from embedded_jev import prism_codec

    tokenizer_bytes = json.dumps({"model": {"type": "Unigram" if defect == "model-kind" else "BPE"}}).encode()
    profile, metadata = _profile_tokenizer_preparation_fixture(tokenizer_bytes)
    capacity = profile.geometry["vocab_size"]
    vocabulary = {"a": 0, "b": 3}
    if defect == "contiguous":
        vocabulary["b"] = 1
    elif defect == "full":
        vocabulary = {f"token{token_id}": token_id for token_id in range(capacity)}
    elif defect == "leading":
        vocabulary["a"] = 1
    elif defect == "empty":
        vocabulary = {}
    elif defect in ("duplicate", "boolean", "negative", "out-of-range"):
        vocabulary["b"] = {"duplicate": 0, "boolean": True, "negative": -1, "out-of-range": capacity}[defect]
    elif defect == "non-string":
        vocabulary = {1: 0}
    calls = []

    def from_str(text):
        calls.append(text)
        if defect == "parser":
            raise ValueError("synthetic parser refusal")
        return SimpleNamespace(get_vocab=lambda *, with_added_tokens: dict(vocabulary))

    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=SimpleNamespace(from_str=from_str)))
    if defect == "metadata-pin":
        metadata["config.json"] += b" "
    elif defect == "tokenizer-pin":
        tokenizer_bytes += b" "
    elif defect in ("range-budget", "range-tail-budget"):
        capacity = 8194 if defect == "range-tail-budget" else 8196
        vocabulary = {f"token{token_id}": token_id for token_id in range(0, capacity, 2)}
        monkeypatch.setattr(prism_codec, "_parse_profile_model_tokenizer_vocabulary", lambda *args, **kwargs:
            ({}, None, vocabulary))
        profile = SimpleNamespace(geometry={"vocab_size": capacity})
    arguments = {"metadata_files": metadata, "tokenizer_bytes": tokenizer_bytes, "context_length": 256}
    if defect not in (None, "contiguous", "full", "leading"):
        with pytest.raises(ValueError):
            prism_codec.prepare_profile_model_tokenizer_inventory(profile, **arguments)
        assert len(calls) == (0 if defect in ("metadata-pin", "tokenizer-pin", "model-kind", "range-budget", "range-tail-budget") else 1)
        return
    receipt = prism_codec.prepare_profile_model_tokenizer_inventory(profile, **arguments)
    assert calls == [tokenizer_bytes.decode()]
    expected = () if defect == "full" else ((2, capacity),) if defect == "contiguous" else ((0, 1), (2, 3), (4, capacity)) if defect == "leading" else ((1, 3), (4, capacity))
    assert receipt["missing_token_id_ranges"] == expected
    assert receipt["unassigned_profile_token_rows"] == sum(end - first for first, end in expected) == capacity - len(vocabulary)
    assert receipt["contiguous_source_ids"] is (defect in ("contiguous", "full"))
    assert "tokens" not in receipt
    assert receipt["padding_policy_verification"] == receipt["gguf_token_type_verification"] == receipt["gguf_tokenizer_verification"] == "not_checked"
    with pytest.raises(TypeError):
        receipt["missing_token_id_ranges"] = ()
    if defect in (None, "leading"):
        with pytest.raises(ValueError, match="vocabulary IDs"):
            prism_codec.prepare_profile_model_tokenizer_vocabulary(profile, **arguments)


@pytest.mark.parametrize("defect", [None, "metadata-pin", "tokenizer-pin", "unassigned", "empty-merges", "bad-merge", "duplicate-vocabulary",
                                    "duplicate-merges", "architecture", "split-writer", "missing-array"])
def test_profile_tokenizer_array_emission_preflights_both_fields_with_one_parser(monkeypatch, defect):
    import sys
    from types import SimpleNamespace

    from embedded_jev.prism_codec import write_profile_model_tokenizer_arrays

    count = _profile_tokenizer_preparation_fixture(b"{}")[0].geometry["vocab_size"]
    tokens = ["a", "b", "c", "ab", "abc"] + [f"token{index}" for index in range(5, count)]
    if defect == "unassigned":
        tokens.pop()
    vocabulary = {token: index for index, token in enumerate(tokens)}
    tokenizer_bytes = json.dumps({"model": {"type": "BPE", "vocab": vocabulary,
        "merges": [] if defect == "empty-merges" else [["a", "b"], ["ab", "missing" if defect == "bad-merge" else "c"]]}}).encode()
    profile, metadata = _profile_tokenizer_preparation_fixture(tokenizer_bytes)
    calls = []

    def from_str(text):
        calls.append(text)
        return SimpleNamespace(get_vocab=lambda *, with_added_tokens: dict(vocabulary))

    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=SimpleNamespace(from_str=from_str)))
    writer = _profile_metadata_writer_fixture()
    writer.add_array = lambda name, values: writer.kv_data[0].update({name: ("str_array", list(values))})
    if defect == "metadata-pin":
        metadata["config.json"] += b" "
    elif defect == "tokenizer-pin":
        tokenizer_bytes += b" "
    elif defect and defect.startswith("duplicate-"):
        writer.kv_data[0]["tokenizer.ggml.tokens" if defect == "duplicate-vocabulary" else "tokenizer.ggml.merges"] = ("str_array", ["keep"])
    elif defect == "architecture":
        writer.arch = "qwen3"
    elif defect == "split-writer":
        writer.kv_data.append({})
    elif defect == "missing-array":
        writer.add_array = None
    before = json.dumps(writer.kv_data, sort_keys=True)
    arguments = {"metadata_files": metadata, "tokenizer_bytes": tokenizer_bytes, "context_length": 256, "writer": writer}
    if defect:
        with pytest.raises(ValueError):
            write_profile_model_tokenizer_arrays(profile, **arguments)
        assert json.dumps(writer.kv_data, sort_keys=True) == before
        assert len(calls) == (0 if defect in ("metadata-pin", "tokenizer-pin") else 1)
        return
    receipt = write_profile_model_tokenizer_arrays(profile, **arguments)
    assert calls == [tokenizer_bytes.decode()]
    assert writer.kv_data == [{"tokenizer.ggml.tokens": ("str_array", tokens), "tokenizer.ggml.merges": ("str_array", ["a b", "ab c"])}]
    assert receipt["metadata_count"] == len(receipt["records"]) == 2
    assert tuple(record["name"] for record in receipt["records"]) == ("tokenizer.ggml.tokens", "tokenizer.ggml.merges")
    assert receipt["metadata_scope"] == "tokenizer_vocabulary_merge_subset" and receipt["merge_content_bytes"] == 7
    assert receipt["metadata_verification"] == "declared_records_emitted_not_file_verified"
    assert receipt["gguf_tokenizer_verification"] == receipt["pre_tokenizer_tag_verification"] == receipt["gguf_token_type_verification"] == "not_checked"
    with pytest.raises(TypeError):
        receipt["records"][0]["values"] = ()


@pytest.mark.parametrize("budget", [True, 0, 16385])
def test_profile_tokenizer_row_budgets_refuse_before_source_preparation(monkeypatch, budget):
    import sys

    from embedded_jev import prism_codec

    monkeypatch.setitem(sys.modules, "tokenizers", None)
    arguments = {"metadata_files": {}, "tokenizer_bytes": b"", "context_length": 256, "maximum_vocabulary_rows": budget}
    for emit in (prism_codec.write_profile_model_tokenizer_vocabulary, prism_codec.write_profile_model_tokenizer_arrays):
        with pytest.raises(ValueError, match="row budget"):
            emit(object(), **arguments, writer=object())
    with pytest.raises(ValueError, match="budget"):
        prism_codec.verify_profile_gguf_tokenizer_arrays_v2(object(), **arguments, reader=object(),
            expected_file_bytes=100, expected_file_sha256="1" * 64)


def test_profile_tokenizer_row_opt_in_preserves_defaults_and_content_limits():
    from embedded_jev.prism_codec import _profile_tokenizer_vocabulary_record

    prepared = {"tokens": tuple(f"t{token_id:04x}" for token_id in range(8193)), "unassigned_profile_token_rows": 0}
    with pytest.raises(ValueError, match="vocabulary count"):
        _profile_tokenizer_vocabulary_record(prepared)
    record, content = _profile_tokenizer_vocabulary_record(prepared, maximum_vocabulary_rows=16384)
    assert len(record["values"]) == 8193 and content == 8193 * 5
    with pytest.raises(ValueError, match="vocabulary count"):
        _profile_tokenizer_vocabulary_record(prepared, maximum_vocabulary_rows=8192)
    prepared["tokens"] = tuple(f"t{token_id:04x}" for token_id in range(16384))
    with pytest.raises(ValueError, match="content budget"):
        _profile_tokenizer_vocabulary_record(prepared, maximum_vocabulary_rows=16384)


@pytest.mark.parametrize("budget", [True, 0, 128 * 1024 + 1])
def test_profile_tokenizer_record_budgets_refuse_before_source_preparation(monkeypatch, budget):
    import sys

    from embedded_jev.prism_codec import verify_profile_gguf_tokenizer_arrays_v2

    monkeypatch.setitem(sys.modules, "tokenizers", None)
    with pytest.raises(ValueError, match="budget"):
        verify_profile_gguf_tokenizer_arrays_v2(object(), reader=object(), metadata_files={}, tokenizer_bytes=b"",
            context_length=256, expected_file_bytes=100, expected_file_sha256="1" * 64, maximum_metadata_bytes=budget)


@pytest.mark.parametrize("defect", [None, "extended", "metadata-pin", "tokenizer-pin", "unassigned", "empty-merges", "file-budget", "file-cap",
                                    "expected-size", "expected-cap", "file-hash", "seconds-bool", "seconds-nan", "preparation-deadline", "verification-deadline"])
def test_profile_tokenizer_array_file_verification_prepares_fresh_bounded_declarations(monkeypatch, defect):
    import sys
    from types import SimpleNamespace

    from embedded_jev import prism_codec

    count = _profile_tokenizer_preparation_fixture(b"{}")[0].geometry["vocab_size"]
    tokens = ["a", "b", "c", "ab", "abc"] + [f"token{index}" for index in range(5, count)]
    if defect == "unassigned":
        tokens.pop()
    vocabulary = {token: index for index, token in enumerate(tokens)}
    tokenizer_bytes = json.dumps({"model": {"type": "BPE", "vocab": vocabulary,
        "merges": [] if defect == "empty-merges" else [["a", "b"], ["ab", "c"]]}}).encode()
    profile, metadata = _profile_tokenizer_preparation_fixture(tokenizer_bytes)
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=SimpleNamespace(from_str=lambda text:
        SimpleNamespace(get_vocab=lambda *, with_added_tokens: dict(vocabulary)))))
    calls = []

    def verify(actual_profile, **kwargs):
        assert actual_profile is profile and kwargs["descriptor"] is kwargs["model_owner"] is None
        assert kwargs["policy_version"] == 2 and kwargs["header_cap"] == 1024**2 and kwargs["file_cap"] == 24 * 1024**3
        assert kwargs["maximum_metadata_bytes"] == (128 * 1024 if defect == "extended" else 8192)
        assert tuple(item["name"] for item in kwargs["declarations"]) == ("tokenizer.ggml.tokens", "tokenizer.ggml.merges")
        assert all(item["kind"] == "str_array" for item in kwargs["declarations"])
        assert tuple(kwargs["declarations"][0]["values"]) == tuple(tokens)
        assert tuple(kwargs["declarations"][1]["values"]) == ("a b", "ab c")
        calls.append(kwargs)
        return {"policy_metadata_verification": "must_not_leak_policy_claim", "metadata_names": tuple(item["name"] for item in kwargs["declarations"])}

    monkeypatch.setattr(prism_codec, "_verify_profile_gguf_metadata_records", verify)
    arguments = {"reader": object(), "metadata_files": metadata, "tokenizer_bytes": tokenizer_bytes, "context_length": 256,
                 "expected_file_bytes": 100, "expected_file_sha256": "1" * 64}
    if defect == "extended":
        arguments.update(maximum_vocabulary_rows=16384, maximum_metadata_bytes=128 * 1024)
    elif defect == "metadata-pin":
        metadata["config.json"] += b" "
    elif defect == "tokenizer-pin":
        arguments["tokenizer_bytes"] += b" "
    elif defect in ("file-budget", "file-cap"):
        arguments["maximum_file_bytes"] = True if defect == "file-budget" else 24 * 1024**3 + 1
    elif defect in ("expected-size", "expected-cap"):
        arguments["expected_file_bytes"] = 0 if defect == "expected-size" else 16 * 1024**2 + 1
    elif defect == "file-hash":
        arguments["expected_file_sha256"] = "bad"
    elif defect in ("seconds-bool", "seconds-nan"):
        arguments["maximum_seconds"] = True if defect == "seconds-bool" else float("nan")
    elif defect in ("preparation-deadline", "verification-deadline"):
        clock = iter([5, 40] if defect == "preparation-deadline" else [5, 6, 40])
        monkeypatch.setattr(prism_codec.time, "monotonic", lambda: next(clock))
    if defect not in (None, "extended"):
        with pytest.raises(ValueError):
            prism_codec.verify_profile_gguf_tokenizer_arrays_v2(profile, **arguments)
        assert len(calls) == (1 if defect == "verification-deadline" else 0)
        return
    receipt = prism_codec.verify_profile_gguf_tokenizer_arrays_v2(profile, **arguments)
    assert len(calls) == 1 and "policy_metadata_verification" not in receipt
    assert receipt["metadata_scope"] == "tokenizer_vocabulary_merge_subset" and receipt["merge_content_bytes"] == 7
    assert receipt["metadata_verification"] == "selected_typed_tokenizer_array_records_match_file"
    assert receipt["gguf_tokenizer_verification"] == receipt["gguf_token_type_verification"] == receipt["pre_tokenizer_tag_verification"] == "not_checked"
    if defect == "extended":
        assert receipt["vocabulary_row_cap"] == 16384 and receipt["selected_metadata_cap_bytes"] == 128 * 1024
        assert receipt["tokenizer_array_policy_version"] == 2
    else:
        assert not {"vocabulary_row_cap", "selected_metadata_cap_bytes", "tokenizer_array_policy_version"}.intersection(receipt)


@pytest.mark.parametrize("defect", [None, "source-pin", "tokenizer-pin", "architecture", "duplicate", "split-writer", "missing-array",
                                    "unassigned", "empty-token", "nul", "item-budget", "utf8-budget", "byte-limit", "utf8-limit",
                                    "count-budget", "aggregate-budget", "aggregate-limit"])
def test_profile_vocabulary_emission_requires_exact_source_rows_and_bounded_content(monkeypatch, defect):
    import sys
    from types import SimpleNamespace

    from embedded_jev import prism_codec

    tokenizer_bytes = b'{"model":{"type":"BPE"}}'
    profile, metadata = _profile_tokenizer_preparation_fixture(tokenizer_bytes)
    tokens = [f"token{index}" for index in range(profile.geometry["vocab_size"])]
    if defect == "unassigned":
        tokens.pop()
    elif defect in ("empty-token", "nul", "item-budget", "utf8-budget", "byte-limit", "utf8-limit"):
        tokens[0] = {"empty-token": "", "nul": "a\x00b", "item-budget": "x" * 4097, "utf8-budget": "\u00e9" * 2049,
                     "byte-limit": "x" * 4096, "utf8-limit": "\u00e9" * 2048}[defect]
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=SimpleNamespace(from_str=lambda text:
        SimpleNamespace(get_vocab=lambda *, with_added_tokens: {token: index for index, token in enumerate(tokens)}))))
    writer = _profile_metadata_writer_fixture()
    writer.add_array = lambda name, values: writer.kv_data[0].update({name: ("str_array", list(values))})
    if defect == "source-pin":
        metadata["config.json"] += b" "
    elif defect == "tokenizer-pin":
        tokenizer_bytes += b" "
    elif defect == "architecture":
        writer.arch = "qwen3"
    elif defect == "duplicate":
        writer.kv_data[0]["tokenizer.ggml.tokens"] = ("str_array", ["keep"])
    elif defect == "split-writer":
        writer.kv_data.append({})
    elif defect == "missing-array":
        writer.add_array = None
    elif defect in ("count-budget", "aggregate-budget", "aggregate-limit"):
        prepared = prism_codec.prepare_profile_model_tokenizer_vocabulary(profile, metadata_files=metadata,
            tokenizer_bytes=tokenizer_bytes, context_length=256)
        tokens = [f"token{index}" for index in range(4097)] if defect == "count-budget" else [
            f"{index:04d}" + "x" * 4092 for index in range(16 if defect == "aggregate-limit" else 17)]
        monkeypatch.setattr(prism_codec, "prepare_profile_model_tokenizer_vocabulary", lambda *args, **kwargs:
            {**prepared, "tokens": tuple(tokens), "tokenizer_vocabulary_size": len(tokens), "unassigned_profile_token_rows": 0})
    before = json.dumps(writer.kv_data, sort_keys=True)
    arguments = {"metadata_files": metadata, "tokenizer_bytes": tokenizer_bytes, "context_length": 256, "writer": writer}
    if defect not in (None, "byte-limit", "utf8-limit", "aggregate-limit"):
        with pytest.raises(ValueError):
            prism_codec.write_profile_model_tokenizer_vocabulary(profile, **arguments)
        assert json.dumps(writer.kv_data, sort_keys=True) == before
        return
    receipt = prism_codec.write_profile_model_tokenizer_vocabulary(profile, **arguments)
    expected = [{"tokenizer.ggml.tokens": ("str_array", tokens)}]
    assert writer.kv_data == expected
    assert receipt["records"][0]["values"] == tuple(tokens) and receipt["metadata_count"] == 1
    assert receipt["vocabulary_content_bytes"] == sum(len(token.encode()) for token in tokens)
    assert receipt["metadata_scope"] == "tokenizer_vocabulary_subset"
    assert receipt["metadata_verification"] == "declared_records_emitted_not_file_verified"
    assert receipt["gguf_tokenizer_verification"] == receipt["pre_tokenizer_tag_verification"] == receipt["gguf_token_type_verification"] == "not_checked"
    with pytest.raises(TypeError):
        receipt["records"][0]["values"] = ()
    with pytest.raises(ValueError, match="conflicting"):
        prism_codec.write_profile_model_tokenizer_vocabulary(profile, **arguments)
    assert writer.kv_data == expected


@pytest.mark.parametrize("defect", [None, "source-pin", "tokenizer-pin", "architecture", "duplicate", "split-writer", "missing-array",
                                    "empty", "space", "nul", "item-budget", "count-budget", "aggregate-budget",
                                    "utf8-budget", "byte-limit", "utf8-limit", "aggregate-limit"])
def test_profile_bpe_merge_emission_preflights_pins_format_budgets_and_writer(monkeypatch, defect):
    import sys
    from types import SimpleNamespace

    from embedded_jev import prism_codec

    vocabulary = {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4}
    merges = [["a", "b"], ["ab", "c"]]
    if defect == "empty":
        merges = []
    elif defect in ("space", "nul", "item-budget", "utf8-budget", "byte-limit", "utf8-limit"):
        piece = {"space": "a b", "nul": "a\x00b", "item-budget": "a" * 4096, "utf8-budget": "\u00e9" * 2048,
                 "byte-limit": "a" * 4094, "utf8-limit": "\u00e9" * 2047}[defect]
        vocabulary, merges = {piece: 0, "c": 1, piece + "c": 2}, [[piece, "c"]]
    expected_values = tuple(" ".join(pair) for pair in merges)
    tokenizer_bytes = json.dumps({"model": {"type": "BPE", "vocab": vocabulary, "merges": merges}}).encode()
    profile, metadata = _profile_tokenizer_preparation_fixture(tokenizer_bytes)
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=SimpleNamespace(from_str=lambda text:
        SimpleNamespace(get_vocab=lambda *, with_added_tokens: dict(vocabulary)))))
    writer = _profile_metadata_writer_fixture()
    writer.add_array = lambda name, values: writer.kv_data[0].update({name: ("str_array", list(values))})
    if defect == "source-pin":
        metadata["config.json"] += b" "
    elif defect == "tokenizer-pin":
        tokenizer_bytes += b" "
    elif defect == "architecture":
        writer.arch = "qwen3"
    elif defect == "duplicate":
        writer.kv_data[0]["tokenizer.ggml.merges"] = ("str_array", ["keep"])
    elif defect == "split-writer":
        writer.kv_data.append({})
    elif defect == "missing-array":
        writer.add_array = None
    elif defect in ("count-budget", "aggregate-budget", "aggregate-limit"):
        prepared = prism_codec.prepare_profile_model_bpe_metadata(profile, metadata_files=metadata,
            tokenizer_bytes=tokenizer_bytes, context_length=256)
        pairs = (("a", "b"),) * 4097 if defect == "count-budget" else (("a" * 2048, "b" * 2047),) * (16 if defect == "aggregate-limit" else 17)
        expected_values = tuple(" ".join(pair) for pair in pairs)
        monkeypatch.setattr(prism_codec, "prepare_profile_model_bpe_metadata", lambda *args, **kwargs:
            {**prepared, "bpe_merge_pairs": pairs, "bpe_merge_count": len(pairs)})
    before = json.dumps(writer.kv_data, sort_keys=True)
    arguments = {"metadata_files": metadata, "tokenizer_bytes": tokenizer_bytes, "context_length": 256, "writer": writer}
    if defect not in (None, "byte-limit", "utf8-limit", "aggregate-limit"):
        with pytest.raises(ValueError):
            prism_codec.write_profile_model_bpe_merge_metadata(profile, **arguments)
        assert json.dumps(writer.kv_data, sort_keys=True) == before
        return
    receipt = prism_codec.write_profile_model_bpe_merge_metadata(profile, **arguments)
    expected = [{"tokenizer.ggml.merges": ("str_array", list(expected_values))}]
    assert writer.kv_data == expected
    assert receipt["metadata_count"] == 1 and receipt["metadata_scope"] == "bpe_merge_subset"
    assert receipt["records"][0]["values"] == expected_values
    assert receipt["bpe_merge_count"] == len(expected_values) and receipt["merge_content_bytes"] == sum(len(value.encode()) for value in expected_values)
    assert receipt["gguf_tokenizer_verification"] == receipt["pre_tokenizer_tag_verification"] == receipt["gguf_token_type_verification"] == "not_checked"
    assert receipt["metadata_verification"] == "declared_records_emitted_not_file_verified"
    with pytest.raises(TypeError):
        receipt["records"][0]["values"] = ()
    with pytest.raises(ValueError, match="conflicting"):
        prism_codec.write_profile_model_bpe_merge_metadata(profile, **arguments)
    assert writer.kv_data == expected


@pytest.mark.parametrize("defect", [None, "duplicate-policy", "duplicate-geometry", "missing-string", "architecture", "split-writer"])
def test_profile_policy_metadata_emission_preflights_complete_declarations(defect):
    from embedded_jev import prism_codec

    profile, report, manifest, blocks = _prefill_model_policy_fixture()
    writer = _profile_metadata_writer_fixture()
    if defect == "duplicate-policy":
        writer.kv_data[0]["jev.model.profile_sha256"] = ("str", ["keep"])
    elif defect == "duplicate-geometry":
        writer.kv_data[0]["qwen35.block_count"] = ("u32", [1])
    elif defect == "missing-string":
        writer.add_string = None
    elif defect == "architecture":
        writer.arch = "qwen3"
    elif defect == "split-writer":
        writer.kv_data.append({})
    before = json.dumps(writer.kv_data, sort_keys=True)
    if defect:
        with pytest.raises(ValueError, match="conflicting"):
            prism_codec.write_profile_model_policy_metadata(profile, report, manifest, blocks, writer=writer)
        assert json.dumps(writer.kv_data, sort_keys=True) == before
        return
    receipt = prism_codec.write_profile_model_policy_metadata(profile, report, manifest, blocks, writer=writer)
    assert receipt["metadata_count"] == len(writer.kv_data[0]) == 24
    assert writer.kv_data[0]["jev.model.profile_sha256"] == ("str", [profile.sha256])
    assert writer.kv_data[0]["jev.bitnet.source_tensor_sha256"] == ("str", [manifest["source_tensor_sha256"]])
    assert writer.kv_data[0]["jev.bitnet.execution"] == ("str", ["group128-a8-fp32-nearest-even-identity-v1"])
    assert writer.kv_data[0]["qwen35.block_count"] == ("u32", [2])
    assert "general.architecture" not in writer.kv_data[0]
    assert receipt["source_file_verification"] == receipt["tokenizer_verification"] == "not_checked"
    assert receipt["runtime_approval"] == "not_granted_by_policy_metadata_emission"
    assert all(tuple(writer.kv_data[0][record["name"]][1]) == record["values"] for record in receipt["records"])
    with pytest.raises(ValueError, match="conflicting"):
        prism_codec.write_profile_model_policy_metadata(profile, report, manifest, blocks, writer=writer)


def _profile_metadata_writer_fixture():
    class Writer:
        arch = "qwen35"

        def __init__(self):
            self.kv_data = [{}]

        def add_uint32(self, name, value):
            self.kv_data[0][name] = ("u32", [value])

        def add_float32(self, name, value):
            self.kv_data[0][name] = ("f32", [value])

        def add_array(self, name, values):
            self.kv_data[0][name] = ("i32_array", list(values))

        def add_string(self, name, value):
            self.kv_data[0][name] = ("str", [value])

    return Writer()


@pytest.mark.parametrize("defect", [None, "metadata-pin", "tokenizer-pin", "architecture", "duplicate", "split-writer"])
def test_profile_context_metadata_emission_verifies_source_before_writer_mutation(defect):
    from embedded_jev.prism_codec import write_profile_model_context_metadata

    metadata, _, record = _packaging_fixture(text_only=True)
    config = json.loads(metadata["config.json"])
    config["max_position_embeddings"] = 512
    metadata["config.json"] = json.dumps(config).encode()
    record["files"]["config.json"] = {"bytes": len(metadata["config.json"]), "sha256": hashlib.sha256(metadata["config.json"]).hexdigest()}
    profile = parse_model_profile(json.dumps(record).encode())
    tokenizer = b"{}"
    writer = _profile_metadata_writer_fixture()
    writer.add_string = None
    if defect == "metadata-pin":
        metadata["config.json"] += b" "
    elif defect == "tokenizer-pin":
        tokenizer = b"[]"
    elif defect == "architecture":
        writer.arch = "qwen3"
    elif defect == "duplicate":
        writer.kv_data[0]["qwen35.context_length"] = ("u32", [1])
    elif defect == "split-writer":
        writer.kv_data.append({})
    before = json.dumps(writer.kv_data, sort_keys=True)
    arguments = {"metadata_files": metadata, "tokenizer_bytes": tokenizer, "context_length": 256, "writer": writer}
    if defect:
        message = "metadata identity" if defect == "metadata-pin" else "tokenizer input identity" if defect == "tokenizer-pin" else "conflicting architecture metadata writer"
        with pytest.raises(ValueError, match=message):
            write_profile_model_context_metadata(profile, **arguments)
        assert json.dumps(writer.kv_data, sort_keys=True) == before
        return
    receipt = write_profile_model_context_metadata(profile, **arguments)
    assert writer.kv_data == [{"qwen35.context_length": ("u32", [256])}]
    assert receipt["records"][0]["values"] == (256,) and receipt["metadata_count"] == 1
    assert receipt["source_metadata_verification"] == "profile_pinned_caller_bytes"
    assert receipt["tokenizer_semantic_verification"] == receipt["gguf_template_verification"] == "not_checked"
    assert receipt["runtime_approval"] == "not_granted_by_context_emission"
    with pytest.raises(ValueError, match="conflicting"):
        write_profile_model_context_metadata(profile, **arguments)
    assert writer.kv_data == [{"qwen35.context_length": ("u32", [256])}]


@pytest.mark.parametrize("target", ["down", "o"])
@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("defect", [None, "architecture", "duplicate", "missing-method", "split-writer"])
def test_profile_architecture_metadata_emission_is_declared_and_preflighted(target, hybrid, defect):
    from embedded_jev import prism_codec

    profile, report, manifest, blocks = _prefill_model_policy_fixture(target, hybrid=hybrid)
    specification = prism_codec.profile_model_specification(profile, report, manifest, blocks)
    writer = _profile_metadata_writer_fixture()
    writer.add_string = None
    if defect == "architecture":
        writer.arch = "qwen3"
    elif defect == "duplicate":
        writer.kv_data[0]["qwen35.embedding_length"] = ("u32", [1])
    elif defect == "missing-method":
        writer.add_array = None
    elif defect == "split-writer":
        writer.kv_data.append({})
    before = json.dumps(writer.kv_data, sort_keys=True)
    if defect:
        with pytest.raises(ValueError, match="invalid or conflicting architecture metadata writer"):
            prism_codec.write_profile_model_architecture_metadata(profile, report, manifest, blocks, writer=writer)
        assert json.dumps(writer.kv_data, sort_keys=True) == before
        return
    receipt = prism_codec.write_profile_model_architecture_metadata(profile, report, manifest, blocks, writer=writer)
    expected = {record["name"]: (record["kind"], record["values"]) for record in specification["metadata"]}
    expected.update({"qwen35.block_count": ("u32", [specification["layers"]]),
                     "qwen35.embedding_length": ("u32", [specification["hidden"]]),
                     "qwen35.feed_forward_length": ("u32", [specification["intermediate"]])})
    assert writer.kv_data[0] == expected and receipt["metadata_count"] == len(expected)
    assert receipt["tokenizer_verification"] == receipt["source_file_verification"] == "not_checked"
    assert receipt["metadata_verification"] == "declared_records_emitted_not_file_verified"
    assert receipt["runtime_approval"] == "not_granted_by_metadata_emission"
    with pytest.raises(TypeError):
        receipt["records"][0]["values"] = (0,)
    with pytest.raises(ValueError, match="conflicting"):
        prism_codec.write_profile_model_architecture_metadata(profile, report, manifest, blocks, writer=writer)
    assert writer.kv_data[0] == expected


@pytest.mark.parametrize("target", ["down", "up", "gate", "q", "k", "v", "o"])
def test_profile_model_conversion_plan_equal_heads_preserves_frozen_projection(target):
    from embedded_jev.prism_codec import profile_model_conversion_plan

    profile, report, manifest, blocks = _prefill_model_policy_fixture(target, value_heads=2)
    plan = profile_model_conversion_plan(profile, report, manifest, blocks)
    assert plan["linear_head_layout"]["linear_num_key_heads"] == plan["linear_head_layout"]["linear_num_value_heads"] == 2
    assert plan["projection_payload_sha256"] == manifest["payload_sha256"]
    assert plan["source_revision"] == profile.revision
    assert len({item["source_name"] for item in plan["tensors"]}) == len(plan["tensors"]) == 28
    target_row = next(item for item in plan["tensors"] if item["name"] == manifest["gguf_tensor"])
    assert target_row["source_name"] == manifest["tensor"]
    assert target_row["transforms"] == ["frozen_projection_payload_not_dense_conversion"]
    assert not any("grouped_to_tiled" in step for item in plan["tensors"] for step in item["transforms"])
    source_row = next(item for item in report["tensors"] if item["name"] == plan["tensors"][0]["source_name"])
    old_shape = tuple(source_row["shape"])
    plan["tensors"][0]["source_shape"][0] += 1
    assert tuple(source_row["shape"]) == old_shape


@pytest.mark.parametrize("defect", ["digest", "budget", "boolean-budget", "matrix", "dtype", "shape", "nonfinite"])
def test_profile_converted_vector_refuses_invalid_operands_before_torch(tmp_path, monkeypatch, defect):
    import sys

    from embedded_jev.prism_codec import verify_profile_converted_vector

    profile, report, manifest, blocks = _prefill_model_policy_fixture("down")
    source = np.zeros(profile.geometry["hidden_size"], dtype="<u2")
    converted = np.ones(source.size, dtype="<f4")
    arguments = {"source_name": "model.norm.weight", "source_values": source, "converted_values": converted,
                 "expected_source_sha256": hashlib.sha256(memoryview(source)).hexdigest()}
    if defect == "digest":
        arguments["expected_source_sha256"] = "0" * 64
    elif defect == "budget":
        arguments["maximum_bytes"] = 1
    elif defect == "boolean-budget":
        arguments["maximum_bytes"] = True
    elif defect == "matrix":
        arguments["source_name"] = manifest["tensor"]
    elif defect == "dtype":
        arguments["source_values"] = source.astype(np.float32)
    elif defect == "shape":
        arguments["converted_values"] = converted[:-1]
    elif defect == "nonfinite":
        converted[0] = np.inf
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError):
        verify_profile_converted_vector(profile, report, manifest, blocks, **arguments)


@pytest.mark.parametrize("defect", ["size", "budget", "boolean-budget", "digest", "deadline", "boolean-deadline", "matrix", "operand-budget"])
@pytest.mark.parametrize("kind", ["vector", "matrix", "convolution"])
def test_profile_gguf_vector_refuses_invalid_file_budget_before_reader_import(monkeypatch, defect, kind):
    import sys

    from embedded_jev.prism_codec import verify_profile_gguf_vector, verify_profile_gguf_matrix, verify_profile_gguf_convolution

    profile, report, manifest, blocks = _prefill_model_policy_fixture("down")
    name = {"matrix": "model.layers.1.self_attn.k_proj.weight", "vector": "model.norm.weight",
            "convolution": "model.layers.0.linear_attn.conv1d.weight"}[kind]
    shape = next(item["shape"] for item in report["tensors"] if item["name"] == name)
    arguments = {"reader": object(), "source_name": name,
                 "source_values": np.zeros(shape, dtype="<u2"),
                 "expected_source_sha256": "0" * 64, "expected_file_bytes": 1, "expected_file_sha256": "0" * 64}
    if defect == "size":
        arguments["expected_file_bytes"] = True
    elif defect == "budget":
        arguments["maximum_file_bytes"] = 16 * 1024**2 + 1
    elif defect == "boolean-budget":
        arguments["maximum_file_bytes"] = True
    elif defect == "digest":
        arguments["expected_file_sha256"] = "invalid"
    elif defect == "deadline":
        arguments["maximum_seconds"] = float("nan")
    elif defect == "boolean-deadline":
        arguments["maximum_seconds"] = True
    elif defect == "matrix":
        arguments["source_name"] = manifest["tensor"]
    elif defect == "operand-budget":
        if kind != "vector":
            arguments["maximum_bytes"] = 1
        else:
            arguments["source_values"] = np.zeros(64 * 1024, dtype="<u2")
    monkeypatch.setitem(sys.modules, "gguf", None)
    with pytest.raises(ValueError):
        verifier = {"matrix": verify_profile_gguf_matrix, "vector": verify_profile_gguf_vector,
                "convolution": verify_profile_gguf_convolution}[kind]
        verifier(profile, report, manifest, blocks, **arguments)


@pytest.mark.parametrize("defect", [None, "digest", "budget", "boolean-budget", "frozen", "vector", "convolution", "dtype", "shape", "nonfinite", "bits"])
def test_profile_converted_matrix_identity_and_refusals(defect):
    from embedded_jev.prism_codec import iter_profile_bf16_matrix_chunks, verify_profile_converted_matrix

    profile, report, manifest, blocks = _prefill_model_policy_fixture("down")
    source = (np.arange(64 * 1024).reshape(64, 1024) % 17).astype("<u2")
    converted = source.copy()
    arguments = {"source_name": "model.layers.1.self_attn.k_proj.weight", "source_values": source,
                 "converted_values": converted, "expected_source_sha256": hashlib.sha256(memoryview(source)).hexdigest()}
    if defect == "digest":
        arguments["expected_source_sha256"] = "0" * 64
    elif defect == "budget":
        arguments["maximum_bytes"] = 1
    elif defect == "boolean-budget":
        arguments["maximum_bytes"] = True
    elif defect == "frozen":
        arguments["source_name"] = manifest["tensor"]
    elif defect == "vector":
        arguments["source_name"] = "model.norm.weight"
    elif defect == "convolution":
        arguments["source_name"] = "model.layers.0.linear_attn.conv1d.weight"
    elif defect == "dtype":
        arguments["source_values"] = source.astype(np.float32)
    elif defect == "shape":
        arguments["converted_values"] = converted[:-1]
    elif defect == "nonfinite":
        converted[0, 0] = 0x7F80
    elif defect == "bits":
        converted[0, 0] ^= 0x8000
    if defect is None:
        receipt = verify_profile_converted_matrix(profile, report, manifest, blocks, **arguments)
        assert receipt["converted_sha256"] == hashlib.sha256(memoryview(converted)).hexdigest()
        assert receipt["source_file_verification"] == receipt["converted_file_verification"] == "not_checked"
        chunks = list(iter_profile_bf16_matrix_chunks(profile, report, manifest, blocks,
                      source_name=arguments["source_name"], source_values=source, maximum_chunk_bytes=3 * source.shape[1] * 2))
        assert [first for first, _ in chunks] == list(range(0, source.shape[0], 3))
        assert np.array_equal(np.concatenate([chunk for _, chunk in chunks]), converted)
        assert all(chunk.flags.owndata and chunk.flags.c_contiguous and not chunk.flags.writeable and
                   not np.shares_memory(chunk, source) and chunk.nbytes <= 3 * source.shape[1] * 2 for _, chunk in chunks)
        with pytest.raises(TypeError):
            receipt["converted_sha256"] = "0" * 64
    else:
        with pytest.raises(ValueError):
            verify_profile_converted_matrix(profile, report, manifest, blocks, **arguments)


@pytest.mark.parametrize("value_heads", [2, 4])
@pytest.mark.parametrize("suffix", ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"])
def test_profile_converted_matrix_reorders_exact_head_indices(value_heads, suffix, tmp_path):
    from embedded_jev.prism_codec import iter_profile_bf16_matrix_chunks, temporary_profile_bf16_matrix_payload, verify_profile_converted_matrix

    profile, report, manifest, blocks = _prefill_model_policy_fixture("down", value_heads=value_heads)
    name = "model.layers.0.linear_attn." + suffix + ".weight"
    shape = next(item["shape"] for item in report["tensors"] if item["name"] == name)
    source = (np.arange(int(np.prod(shape))).reshape(shape) % 257).astype("<u2")
    order = [0, 1] if value_heads == 2 else [0, 2, 1, 3]
    width = 1 if suffix in ("in_proj_a", "in_proj_b") else 32
    selected = (np.array(order)[:, None] * width + np.arange(width)).reshape(-1)
    if suffix == "out_proj":
        converted = source[:, selected].copy()
    elif suffix == "in_proj_qkv":
        rows = np.concatenate((np.arange(128), 128 + selected))
        converted = source[rows].copy()
    else:
        converted = source[selected].copy()
    arguments = {"source_name": name, "source_values": source, "converted_values": converted,
                 "expected_source_sha256": hashlib.sha256(memoryview(source)).hexdigest(), "maximum_bytes": 16 * 1024**2}
    receipt = verify_profile_converted_matrix(profile, report, manifest, blocks, **arguments)
    assert receipt["converted_sha256"] == hashlib.sha256(memoryview(converted)).hexdigest()
    row_count = min(7, max(1, source.shape[0] // 2))
    chunks = list(iter_profile_bf16_matrix_chunks(profile, report, manifest, blocks,
                  source_name=name, source_values=source, maximum_chunk_bytes=row_count * source.shape[1] * 2))
    assert [first for first, _ in chunks] == list(range(0, source.shape[0], row_count))
    assert np.array_equal(np.concatenate([chunk for _, chunk in chunks]), converted)
    assert all(chunk.flags.owndata and chunk.flags.c_contiguous and not chunk.flags.writeable and
               not np.shares_memory(chunk, source) and chunk.nbytes <= row_count * source.shape[1] * 2 for _, chunk in chunks)
    path = tmp_path / "source.bf16"
    source.tofile(path)
    mapped = np.memmap(path, mode="r", dtype="<u2", shape=source.shape)
    mapped_chunks = list(iter_profile_bf16_matrix_chunks(profile, report, manifest, blocks,
                         source_name=name, source_values=mapped, maximum_chunk_bytes=row_count * source.shape[1] * 2))
    assert np.array_equal(np.concatenate([chunk for _, chunk in mapped_chunks]), converted)
    assert all(chunk.flags.owndata and chunk.flags.c_contiguous and not chunk.flags.writeable and
               not np.shares_memory(chunk, mapped) for _, chunk in mapped_chunks)
    with temporary_profile_bf16_matrix_payload(profile, report, manifest, blocks, source_name=name,
             source_values=mapped, expected_source_sha256=arguments["expected_source_sha256"], directory=tmp_path,
             maximum_chunk_bytes=row_count * source.shape[1] * 2) as (payload, payload_receipt):
        assert payload.read_bytes() == converted.tobytes()
        assert payload_receipt["converted_sha256"] == receipt["converted_sha256"]
        assert payload_receipt["converted_bytes"] == converted.nbytes
        assert payload_receipt["source_file_verification"] == "not_checked"
        with pytest.raises(TypeError):
            payload_receipt["converted_sha256"] = "0" * 64
    assert not payload.exists() and not payload.parent.exists() and path.exists()
    del mapped
    if value_heads == 4:
        assert not np.array_equal(source, converted)
        arguments["converted_values"] = source
        with pytest.raises(ValueError, match="byte parity"):
            verify_profile_converted_matrix(profile, report, manifest, blocks, **arguments)


@pytest.mark.parametrize("defect", ["size", "file-cap", "default-file-budget", "boolean-file-budget", "digest", "source-digest",
                                    "deadline", "boolean-deadline", "operand-cap", "boolean-operand-budget", "operand-budget",
                                    "chunk-cap", "boolean-chunk-budget", "row-budget", "frozen", "dtype", "shape", "layout"])
def test_profile_gguf_matrix_v2_refuses_invalid_inputs_before_reader_import(monkeypatch, defect):
    import sys

    from embedded_jev.prism_codec import verify_profile_gguf_matrix_v2

    profile, report, manifest, blocks = _prefill_model_policy_fixture("down")
    source = np.zeros((64, 1024), dtype="<u2")
    arguments = {"reader": object(), "source_name": "model.layers.1.self_attn.k_proj.weight", "source_values": source,
                 "expected_source_sha256": hashlib.sha256(memoryview(source)).hexdigest(),
                 "expected_file_bytes": 1, "expected_file_sha256": "0" * 64}
    if defect == "size":
        arguments["expected_file_bytes"] = True
    elif defect == "file-cap":
        arguments["maximum_file_bytes"] = 24 * 1024**3 + 1
    elif defect == "default-file-budget":
        arguments["expected_file_bytes"] = 16 * 1024**2 + 1
    elif defect == "boolean-file-budget":
        arguments["maximum_file_bytes"] = True
    elif defect == "digest":
        arguments["expected_file_sha256"] = "invalid"
    elif defect == "source-digest":
        arguments["expected_source_sha256"] = "0" * 64
    elif defect == "deadline":
        arguments["maximum_seconds"] = float("nan")
    elif defect == "boolean-deadline":
        arguments["maximum_seconds"] = True
    elif defect == "operand-cap":
        arguments["maximum_bytes"] = 2 * 1024**3 + 1
    elif defect == "boolean-operand-budget":
        arguments["maximum_bytes"] = True
    elif defect == "operand-budget":
        arguments["maximum_bytes"] = 1
    elif defect == "chunk-cap":
        arguments["maximum_chunk_bytes"] = 1024**2 + 1
    elif defect == "boolean-chunk-budget":
        arguments["maximum_chunk_bytes"] = True
    elif defect == "row-budget":
        arguments["maximum_chunk_bytes"] = 2047
    elif defect == "frozen":
        arguments["source_name"] = manifest["tensor"]
    elif defect == "dtype":
        arguments["source_values"] = source.astype(np.float32)
    elif defect == "shape":
        arguments["source_values"] = source[:-1]
    elif defect == "layout":
        arguments["source_values"] = np.asfortranarray(source)
    monkeypatch.setitem(sys.modules, "gguf", None)
    with pytest.raises(ValueError):
        verify_profile_gguf_matrix_v2(profile, report, manifest, blocks, **arguments)


@pytest.mark.parametrize("defect", ["zero-budget", "boolean-budget", "large-budget", "row-budget", "frozen", "vector",
                                    "convolution", "dtype", "endian", "shape", "layout", "array", "infinity", "nan", "late-nonfinite"])
def test_profile_bf16_matrix_chunks_refuse_unsupported_or_nonfinite_values(monkeypatch, defect):
    import sys

    from embedded_jev.prism_codec import iter_profile_bf16_matrix_chunks

    profile, report, manifest, blocks = _prefill_model_policy_fixture("down")
    source = np.zeros((64, 1024), dtype="<u2")
    arguments = {"source_name": "model.layers.1.self_attn.k_proj.weight", "source_values": source,
                 "maximum_chunk_bytes": 2 * source.shape[1] * 2}
    if defect == "zero-budget":
        arguments["maximum_chunk_bytes"] = 0
    elif defect == "boolean-budget":
        arguments["maximum_chunk_bytes"] = True
    elif defect == "large-budget":
        arguments["maximum_chunk_bytes"] = 1024**2 + 1
    elif defect == "row-budget":
        arguments["maximum_chunk_bytes"] = source.shape[1] * 2 - 1
    elif defect == "frozen":
        arguments["source_name"] = manifest["tensor"]
    elif defect == "vector":
        arguments["source_name"] = "model.norm.weight"
    elif defect == "convolution":
        arguments["source_name"] = "model.layers.0.linear_attn.conv1d.weight"
    elif defect == "dtype":
        arguments["source_values"] = source.astype(np.float32)
    elif defect == "endian":
        arguments["source_values"] = source.astype(">u2")
    elif defect == "shape":
        arguments["source_values"] = source[:-1]
    elif defect == "layout":
        arguments["source_values"] = np.asfortranarray(source)
    elif defect == "array":
        arguments["source_values"] = source.tolist()
    elif defect == "infinity":
        source[0, 0] = 0xFF80
    elif defect == "nan":
        source[0, 0] = 0x7FC1
    elif defect == "late-nonfinite":
        source[2, 0] = 0x7F80
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "gguf", None)
    chunks = iter_profile_bf16_matrix_chunks(profile, report, manifest, blocks, **arguments)
    if defect == "late-nonfinite":
        first, chunk = next(chunks)
        assert first == 0 and chunk.shape == (2, 1024) and not chunk.flags.writeable
    with pytest.raises(ValueError):
        next(chunks)


def test_profile_bf16_matrix_chunks_borrow_source_but_own_each_result(monkeypatch):
    import sys

    from embedded_jev.prism_codec import iter_profile_bf16_matrix_chunks

    profile, report, manifest, blocks = _prefill_model_policy_fixture("down")
    source = np.zeros((64, 1024), dtype="<u2")
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "gguf", None)
    chunks = iter_profile_bf16_matrix_chunks(profile, report, manifest, blocks,
                source_name="model.layers.1.self_attn.k_proj.weight", source_values=source,
                maximum_chunk_bytes=2 * source.shape[1] * 2)
    first, chunk = next(chunks)
    source[:] = 0x8000
    assert first == 0 and np.all(chunk == 0)
    second, later = next(chunks)
    assert second == 2 and np.all(later == 0x8000)
    assert not np.shares_memory(chunk, later) and not np.shares_memory(later, source)
    chunks.close()


@pytest.mark.parametrize("policy_version", [1, 2])
@pytest.mark.parametrize("defect", ["digest", "boolean-budget", "large-budget", "small-budget", "deadline", "boolean-deadline",
                                    "chunk-budget", "frozen", "nonfinite", "late-nonfinite", "mid-write", "source-mutation",
                                    "payload-mutation", "payload-restored", "payload-replaced", "payload-deleted", "exit-deadline", "body-error"])
def test_temporary_profile_bf16_payload_refuses_and_cleans_partial_files(tmp_path, monkeypatch, defect, policy_version):
    import os

    from embedded_jev import prism_codec

    profile, report, manifest, blocks = _prefill_model_policy_fixture("down")
    source = np.zeros((64, 1024), dtype="<u2")
    arguments = {"source_name": "model.layers.1.self_attn.k_proj.weight", "source_values": source,
                 "directory": tmp_path, "maximum_chunk_bytes": 2 * source.shape[1] * 2}
    if defect == "nonfinite":
        source[0, 0] = 0x7F80
    elif defect == "late-nonfinite":
        source[2, 0] = 0x7FC0
    arguments["expected_source_sha256"] = hashlib.sha256(memoryview(source)).hexdigest()
    if defect == "digest":
        arguments["expected_source_sha256"] = "0" * 64
    elif defect == "boolean-budget":
        arguments["maximum_bytes"] = True
    elif defect == "large-budget":
        arguments["maximum_bytes"] = (2 * 1024**3 if policy_version == 2 else 16 * 1024**2) + 1
    elif defect == "small-budget":
        arguments["maximum_bytes"] = source.nbytes - 1
    elif defect == "deadline":
        arguments["maximum_seconds"] = float("nan")
    elif defect == "boolean-deadline":
        arguments["maximum_seconds"] = True
    elif defect == "chunk-budget":
        arguments["maximum_chunk_bytes"] = True
    elif defect == "frozen":
        arguments["source_name"] = manifest["tensor"]
    elif defect == "mid-write":
        original = prism_codec.iter_profile_bf16_matrix_chunks
        def interrupted(*args, **kwargs):
            chunks = original(*args, **kwargs)
            yield next(chunks)
            raise RuntimeError("injected chunk failure")
        monkeypatch.setattr(prism_codec, "iter_profile_bf16_matrix_chunks", interrupted)
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("unrelated fixture")
    failure = RuntimeError if defect in ("mid-write", "body-error") else ValueError
    owner = prism_codec.temporary_profile_bf16_matrix_payload_v2 if policy_version == 2 else prism_codec.temporary_profile_bf16_matrix_payload
    with monkeypatch.context() as patch:
        with pytest.raises(failure):
            with owner(profile, report, manifest, blocks, **arguments) as (path, receipt):
                if policy_version == 2:
                    assert receipt["payload_policy_version"] == 2 and receipt["payload_cap_bytes"] == 2 * 1024**3
                else:
                    assert "payload_policy_version" not in receipt
                if defect == "source-mutation":
                    source[0, 0] ^= 1
                elif defect == "payload-mutation":
                    with path.open("ab") as output:
                        output.write(b"x")
                elif defect == "payload-restored":
                    before = path.stat()
                    with path.open("r+b") as output:
                        output.write(b"x")
                        output.seek(0)
                        output.write(bytes(1))
                    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
                elif defect == "payload-replaced":
                    saved = path.read_bytes()
                    path.rename(path.with_suffix(".old"))
                    path.write_bytes(saved)
                elif defect == "payload-deleted":
                    path.unlink()
                elif defect == "exit-deadline":
                    clock = prism_codec.time.monotonic
                    patch.setattr(prism_codec.time, "monotonic", lambda: clock() + 31)
                elif defect == "body-error":
                    raise RuntimeError("injected consumer failure")
                else:
                    pytest.fail("invalid temporary payload was yielded")
    assert list(tmp_path.iterdir()) == [sentinel] and sentinel.read_text() == "unrelated fixture"


@pytest.mark.parametrize("source_dtype", ["BF16", "F32"])
@pytest.mark.parametrize("value_heads", [2, 4])
def test_profile_converted_convolution_exact_squeeze_and_channel_order(source_dtype, value_heads):
    from embedded_jev.prism_codec import verify_profile_converted_convolution

    profile, report, manifest, blocks = _prefill_model_policy_fixture("down", value_heads=value_heads, conv_dtype=source_dtype)
    name = "model.layers.0.linear_attn.conv1d.weight"
    shape = next(item["shape"] for item in report["tensors"] if item["name"] == name)
    values = (np.arange(int(np.prod(shape))).reshape(shape) % 17 - 8).astype("<f4") / 16
    source = (values.view(np.uint32) >> 16).astype("<u2") if source_dtype == "BF16" else values
    order = np.arange(shape[0]) if value_heads == 2 else np.concatenate((np.arange(128),
             128 + (np.array([0, 2, 1, 3])[:, None] * 32 + np.arange(32)).reshape(-1)))
    converted = values[:, 0, :][order].copy()
    arguments = {"source_name": name, "source_values": source, "converted_values": converted,
                 "expected_source_sha256": hashlib.sha256(memoryview(source)).hexdigest()}
    receipt = verify_profile_converted_convolution(profile, report, manifest, blocks, **arguments)
    assert receipt["converted_sha256"] == hashlib.sha256(memoryview(converted)).hexdigest()
    assert receipt["source_file_verification"] == receipt["converted_file_verification"] == "not_checked"
    converted.flat[0] = np.nextafter(converted.flat[0], np.float32(1))
    with pytest.raises(ValueError, match="byte parity"):
        verify_profile_converted_convolution(profile, report, manifest, blocks, **arguments)


@pytest.mark.parametrize("defect", ["wrong-name", "kernel", "digest", "budget", "boolean-budget", "dtype", "shape", "nonfinite", "contiguity"])
def test_profile_converted_convolution_refuses_invalid_operands(defect):
    from embedded_jev.prism_codec import verify_profile_converted_convolution

    profile, report, manifest, blocks = _prefill_model_policy_fixture("down", kernel_size=1 if defect == "kernel" else 4)
    name = "model.layers.0.linear_attn.conv1d.weight"
    shape = next(item["shape"] for item in report["tensors"] if item["name"] == name)
    source = np.zeros(shape, dtype="<u2")
    converted = np.zeros((shape[0], shape[2]), dtype="<f4")
    arguments = {"source_name": name, "source_values": source, "converted_values": converted,
                 "expected_source_sha256": hashlib.sha256(memoryview(source)).hexdigest()}
    if defect == "wrong-name":
        arguments["source_name"] = manifest["tensor"]
    elif defect == "digest":
        arguments["expected_source_sha256"] = "0" * 64
    elif defect == "budget":
        arguments["maximum_bytes"] = 1
    elif defect == "boolean-budget":
        arguments["maximum_bytes"] = True
    elif defect == "dtype":
        arguments["source_values"] = source.astype("<f4")
    elif defect == "shape":
        arguments["converted_values"] = converted[:-1]
    elif defect == "nonfinite":
        source.flat[0] = 0x7F80
        arguments["expected_source_sha256"] = hashlib.sha256(memoryview(source)).hexdigest()
    elif defect == "contiguity":
        arguments["source_values"] = source[:, :, ::-1]
    with pytest.raises(ValueError):
        verify_profile_converted_convolution(profile, report, manifest, blocks, **arguments)


@pytest.mark.parametrize("defect", ["missing", "extra", "pins", "source-digest", "source-dtype", "source-shape", "budget", "boolean", "file", "file-extent", "kernel", "owner", "deadline"])
def test_profile_gguf_tensor_table_refuses_invalid_inputs_before_import(monkeypatch, defect):
    import sys

    from embedded_jev.prism_codec import profile_model_conversion_plan, verify_profile_gguf_tensor_table

    profile, report, manifest, blocks = _prefill_model_policy_fixture("down", kernel_size=1 if defect == "kernel" else 4)
    plan = profile_model_conversion_plan(profile, report, manifest, blocks)
    sources = {item["source_name"]: np.zeros(item["source_shape"], dtype="<u2") for item in plan["tensors"] if item["type"] != "PQ2_0"}
    pins = {name: hashlib.sha256(memoryview(values)).hexdigest() for name, values in sources.items()}
    arguments = {"reader": object(), "source_values": sources, "expected_source_sha256": pins,
                 "expected_file_bytes": 16 * 1024**2, "expected_file_sha256": "0" * 64}
    name = next(iter(sources))
    if defect == "missing":
        sources.pop(name)
    elif defect == "extra":
        sources["unknown"] = np.zeros(1, dtype="<u2")
    elif defect == "pins":
        pins[name] = "invalid"
    elif defect == "source-digest":
        pins[name] = "0" * 64
    elif defect == "source-dtype":
        sources[name] = sources[name].astype("<f4")
    elif defect == "source-shape":
        sources[name] = sources[name].reshape(-1)
    elif defect == "budget":
        arguments["maximum_bytes"] = 1
    elif defect == "boolean":
        arguments["maximum_bytes"] = True
    elif defect == "file":
        arguments["maximum_file_bytes"] = 16 * 1024**2 + 1
    elif defect == "file-extent":
        arguments["expected_file_bytes"] = 1
    elif defect == "owner":
        arguments["model_owner"] = object()
    elif defect == "deadline":
        arguments["maximum_seconds"] = float("nan")
    monkeypatch.setitem(sys.modules, "gguf", None)
    with pytest.raises(ValueError):
        verify_profile_gguf_tensor_table(profile, report, manifest, blocks, **arguments)


@pytest.mark.parametrize("kind,defect", [
    (kind, None) for kind in ("vector", "matrix", "convolution", "f32-vector", "f32-convolution")
] + [(kind, defect) for kind in ("vector", "matrix") for defect in (
    "frozen", "missing", "dtype", "shape", "layout", "array-pin", "file-pin", "file-size", "budget", "boolean", "file-budget", "nan",
    "array-change", "file-change", "header-change", "mid-read", "restored-write", "deadline", "symlink", "fifo",
)])
def test_profile_safetensors_operand_matches_pinned_zero_file_ranges(tmp_path, monkeypatch, kind, defect):
    from embedded_jev import prism_codec
    from embedded_jev.prism_codec import profile_model_conversion_plan, verify_profile_safetensors_operand

    profile, report, manifest, blocks = _prefill_model_policy_fixture(
        pin_zero_source=True, vector_dtype="F32" if kind == "f32-vector" else "BF16", conv_dtype="F32" if kind == "f32-convolution" else "BF16",
    )
    entries, payload_bytes = {}, 0
    for item in sorted(report["tensors"], key=lambda item: item["name"]):
        size = int(np.prod(item["shape"])) * (4 if item["dtype"] == "F32" else 2)
        entries[item["name"]] = {"dtype": item["dtype"], "shape": item["shape"], "data_offsets": [payload_bytes, payload_bytes + size]}
        payload_bytes += size
    header, size = _header(entries, payload_bytes)
    plan = profile_model_conversion_plan(profile, report, manifest, blocks)
    item = next(item for item in plan["tensors"] if item["type"] != "PQ2_0" and (
        len(item["source_shape"]) == (2 if kind == "matrix" else 3 if "convolution" in kind else 1)
    ))
    values = np.zeros(item["source_shape"], dtype="<f4" if item["source_dtype"] == "F32" else "<u2")
    path = tmp_path / profile.packaging["shards"][0]
    actual_path = tmp_path / "actual-source.safetensors" if defect == "symlink" else path
    if defect == "fifo":
        prism_codec.os.mkfifo(path)
    else:
        actual_path.write_bytes(header + bytes(payload_bytes))
    if defect == "symlink":
        path.symlink_to(actual_path)
    arguments = {
        "source_path": path, "source_name": item["source_name"], "source_values": values,
        "expected_source_sha256": hashlib.sha256(memoryview(values)).hexdigest(), "expected_file_bytes": size,
        "expected_file_sha256": profile.files[path.name].sha256,
    }
    before_open = {"frozen", "missing", "dtype", "shape", "layout", "array-pin", "file-pin", "file-size", "budget", "boolean", "file-budget", "nan"}
    expected_error = "source operand"
    if defect == "frozen":
        arguments["source_name"] = manifest["tensor"]
    elif defect == "missing":
        arguments["source_name"] = "missing-source-tensor"
    elif defect == "dtype":
        arguments["source_values"] = values.astype(np.float64)
    elif defect == "shape":
        arguments["source_values"] = values[:-1]
    elif defect == "layout":
        arguments["source_values"] = values[::-1]
    elif defect == "array-pin":
        arguments["expected_source_sha256"] = "0" * 64
    elif defect == "file-pin":
        arguments["expected_file_sha256"] = "0" * 64
    elif defect == "file-size":
        arguments["expected_file_bytes"] -= 1
    elif defect == "budget":
        arguments["maximum_bytes"] = 1
    elif defect == "boolean":
        arguments["maximum_bytes"] = True
    elif defect == "file-budget":
        arguments["maximum_file_bytes"] = 16 * 1024**2 + 1
    elif defect == "nan":
        arguments["maximum_seconds"] = float("nan")
    elif defect == "array-change":
        values.flat[0] = 1
        arguments["expected_source_sha256"] = hashlib.sha256(memoryview(values)).hexdigest()
        expected_error = "file range byte mismatch"
    elif defect == "file-change":
        with path.open("r+b") as output:
            output.seek(len(header) + entries[item["source_name"]]["data_offsets"][0])
            output.write(b"\x01")
        expected_error = "file identity mismatch"
    elif defect in ("header-change", "mid-read", "restored-write"):
        original_pread = prism_codec.os.pread
        def read(descriptor, count, position):
            if defect == "restored-write":
                result = original_pread(descriptor, count, position)
                if position == len(header) + entries[item["source_name"]]["data_offsets"][0] and count == values.nbytes:
                    with path.open("r+b") as output:
                        output.seek(size - 1)
                        output.write(b"\x01")
                        output.seek(size - 1)
                        output.write(b"\x00")
                return result
            if position == 0 and count == len(header):
                with path.open("r+b") as output:
                    output.seek(8 if defect == "header-change" else len(header) + entries[item["source_name"]]["data_offsets"][0])
                    output.write(b"\x01")
            return original_pread(descriptor, count, position)

        monkeypatch.setattr(prism_codec.os, "pread", read)
        expected_error = "file changed during verification" if defect == "restored-write" else "header identity mismatch" if defect == "header-change" else "file range byte mismatch"
    elif defect == "deadline":
        monkeypatch.setattr(prism_codec.time, "monotonic", iter((0, 31)).__next__)
        expected_error = "deadline exceeded"
    elif defect == "fifo":
        expected_error = "not regular"
    if defect in before_open:
        def never_open(*arguments, **keywords):
            pytest.fail("invalid source operand reached filesystem open")

        monkeypatch.setattr(prism_codec.os, "open", never_open)
    try:
        if defect:
            if kind == "matrix":
                if defect == "nan":
                    expected_error = "invalid source-bound BF16 payload deadline"
                elif defect == "symlink":
                    expected_error = "source file unavailable"
                with pytest.raises(ValueError, match=expected_error):
                    with prism_codec.temporary_profile_safetensors_bf16_payload(profile, report, manifest, blocks,
                             **arguments, directory=tmp_path):
                        pytest.fail("invalid source-bound operand yielded a payload")
                assert set(tmp_path.iterdir()) == {path, actual_path}
            else:
                with pytest.raises(OSError if defect == "symlink" else ValueError, match=None if defect == "symlink" else expected_error):
                    verify_profile_safetensors_operand(profile, report, manifest, blocks, **arguments)
            return
        receipt = verify_profile_safetensors_operand(profile, report, manifest, blocks, **arguments)
        assert receipt["source_file_verification"] == "profile_pinned_shard_and_header_range_match_caller_operand"
        assert receipt["source_range_bytes"] == values.nbytes
        assert receipt["source_range_offset"] == len(header) + entries[item["source_name"]]["data_offsets"][0]
        assert receipt["converted_file_verification"] == receipt["pin_issuer_verification"] == "not_checked"
        assert receipt["runtime_approval"] == "not_granted_by_source_operand_verification"
        with pytest.raises(TypeError):
            receipt["source_range_offset"] = 0
        if kind == "matrix":
            from embedded_jev import streamed_text

            records = []
            original_record = streamed_text.bounded_native_descriptor_record

            def record(descriptor, **keywords):
                records.append(descriptor)
                return original_record(descriptor, **keywords)

            with monkeypatch.context() as shard_patch:
                shard_patch.setattr(streamed_text, "bounded_native_descriptor_record", record)
                with prism_codec.mapped_profile_safetensors_bf16_shard(profile, report, manifest, blocks,
                         source_path=path, source_shard=path.name, expected_file_bytes=size,
                         expected_file_sha256=arguments["expected_file_sha256"]) as (matrix, shard_receipt):
                    assert shard_receipt["source_file_identity"] == receipt["source_file_identity"]
                    for source_name in (item["source_name"], "model.layers.0.mlp.up_proj.weight"):
                        matrix_bytes = int(np.prod(entries[source_name]["shape"])) * 2
                        route = next(tensor for tensor in report["tensors"] if tensor["name"] == source_name)
                        original_shard = route["shard"]
                        route["shard"] = "caller-mutated-routing"
                        try:
                            with matrix(source_name, hashlib.sha256(bytes(matrix_bytes)).hexdigest()) as (mapped_source, _):
                                assert mapped_source.nbytes == matrix_bytes and not np.any(mapped_source)
                                with pytest.raises(ValueError, match="already has a live matrix"):
                                    with matrix(source_name, hashlib.sha256(bytes(matrix_bytes)).hexdigest()):
                                        pytest.fail("overlapping shard consumers were accepted")
                        finally:
                            route["shard"] = original_shard
                        assert mapped_source._mmap.closed
                assert len(records) == 2 and records[0] == records[1]
                with pytest.raises(ValueError, match="owner is closed"):
                    with matrix(item["source_name"], arguments["expected_source_sha256"]):
                        pytest.fail("closed shard owner yielded a mapping")
            mapped_arguments = {key: value for key, value in arguments.items() if key != "source_values"}
            with prism_codec.mapped_profile_safetensors_bf16_matrix(profile, report, manifest, blocks,
                     **mapped_arguments) as (mapped_source, mapped_receipt):
                assert isinstance(mapped_source, np.memmap)
                assert mapped_source.dtype == np.dtype("<u2") and mapped_source.shape == values.shape
                assert not mapped_source.flags.owndata and not mapped_source.flags.writeable
                assert np.array_equal(mapped_source, values)
                assert mapped_receipt["source_range_offset"] == receipt["source_range_offset"]
                with pytest.raises(ValueError):
                    mapped_source.flags.writeable = True
                with pytest.raises(TypeError):
                    mapped_receipt["source_range_offset"] = 0
            assert mapped_source._mmap.closed
            with prism_codec.temporary_profile_safetensors_bf16_payload(profile, report, manifest, blocks,
                     **arguments, directory=tmp_path) as (payload, payload_receipt):
                assert payload.read_bytes() == values.tobytes()
                assert payload_receipt["source_file"] == receipt
                assert payload_receipt["source_file_verification"] == "profile_pinned_shard_header_range_entry_and_exit_checks_required"
                with pytest.raises(TypeError):
                    payload_receipt["source_file"]["source_range_offset"] = 0
            assert not payload.exists() and not payload.parent.exists() and path.exists()
            for change in ("content", "restored", "replacement", "deleted"):
                try:
                    with pytest.raises(ValueError):
                        with prism_codec.temporary_profile_safetensors_bf16_payload(profile, report, manifest, blocks,
                                 **arguments, directory=tmp_path) as (payload, _):
                            before = path.stat()
                            if change in ("content", "restored"):
                                with path.open("r+b") as output:
                                    output.seek(size - 1)
                                    output.write(bytes([1]))
                                    if change == "restored":
                                        output.seek(size - 1)
                                        output.write(bytes(1))
                                if change == "restored":
                                    prism_codec.os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
                            elif change == "replacement":
                                path.unlink()
                                path.write_bytes(header + bytes(payload_bytes))
                            else:
                                path.unlink()
                finally:
                    path.write_bytes(header + bytes(payload_bytes))
                assert not payload.exists() and not payload.parent.exists()
            with monkeypatch.context() as clock_patch:
                with pytest.raises(ValueError, match="deadline exceeded"):
                    with prism_codec.temporary_profile_safetensors_bf16_payload(profile, report, manifest, blocks,
                             **arguments, directory=tmp_path) as (payload, _):
                        later = prism_codec.time.monotonic() + 31
                        clock_patch.setattr(prism_codec.time, "monotonic", lambda: later)
                assert not payload.exists() and not payload.parent.exists()
            with pytest.raises(RuntimeError, match="source-bound consumer failed"):
                with prism_codec.temporary_profile_safetensors_bf16_payload(profile, report, manifest, blocks,
                         **arguments, directory=tmp_path) as (payload, _):
                    raise RuntimeError("source-bound consumer failed")
            assert not payload.exists() and not payload.parent.exists()
            shard_arguments = {"source_path": path, "source_shard": path.name, "expected_file_bytes": size,
                               "expected_file_sha256": arguments["expected_file_sha256"]}
            with pytest.raises(ValueError, match="failed or unfinished matrix consumers"):
                with prism_codec.mapped_profile_safetensors_bf16_shard(profile, report, manifest, blocks,
                         **shard_arguments) as (matrix, _):
                    with pytest.raises(RuntimeError, match="shared consumer failed"):
                        with matrix(item["source_name"], arguments["expected_source_sha256"]) as (mapped_source, _):
                            raise RuntimeError("shared consumer failed")
                    assert mapped_source._mmap.closed
            with pytest.raises(ValueError, match="failed or unfinished matrix consumers"):
                with prism_codec.mapped_profile_safetensors_bf16_shard(profile, report, manifest, blocks,
                         **shard_arguments) as (matrix, _):
                    unfinished = matrix(item["source_name"], arguments["expected_source_sha256"])
                    mapped_source, _ = unfinished.__enter__()
            assert mapped_source._mmap.closed
            with pytest.raises(ValueError, match="owner is closed"):
                unfinished.__exit__(None, None, None)
            try:
                with pytest.raises(ValueError, match="file identity changed"):
                    with prism_codec.mapped_profile_safetensors_bf16_shard(profile, report, manifest, blocks,
                             **shard_arguments) as (matrix, _):
                        with matrix(item["source_name"], arguments["expected_source_sha256"]):
                            pass
                        with path.open("r+b") as output:
                            output.seek(size - 1)
                            output.write(bytes([1]))
                        with matrix(item["source_name"], arguments["expected_source_sha256"]):
                            pytest.fail("between-matrix source mutation yielded a mapping")
            finally:
                path.write_bytes(header + bytes(payload_bytes))
    finally:
        path.unlink()
        if actual_path != path:
            actual_path.unlink()


@pytest.mark.parametrize("source_dtype", ["BF16", "F32"])
@pytest.mark.parametrize("include_frozen_source", [False, True])
def test_profile_conversion_shard_v2_maps_auxiliaries_and_matrices_serially(tmp_path, monkeypatch, source_dtype, include_frozen_source):
    from embedded_jev import prism_codec, streamed_text

    profile, report, manifest, blocks = _prefill_model_policy_fixture(pin_zero_source=True,
        vector_dtype=source_dtype, conv_dtype=source_dtype)
    frozen_source_pin = hashlib.sha256(bytes(int(np.prod(manifest["shape"])) * 2)).hexdigest()
    manifest["source_tensor_sha256"] = frozen_source_pin
    frozen_payload_pin = hashlib.sha256(memoryview(blocks)).hexdigest()
    plan = prism_codec.profile_model_conversion_plan(profile, report, manifest, blocks)
    monkeypatch.setattr(prism_codec, "profile_model_conversion_plan", lambda *arguments, **keywords: plan)
    entries, payload_bytes = {}, 0
    for item in sorted(report["tensors"], key=lambda item: item["name"]):
        size = int(np.prod(item["shape"])) * (4 if item["dtype"] == "F32" else 2)
        entries[item["name"]] = {"dtype": item["dtype"], "shape": item["shape"], "data_offsets": [payload_bytes, payload_bytes + size]}
        payload_bytes += size
    header, file_bytes = _header(entries, payload_bytes)
    path = tmp_path / profile.packaging["shards"][0]
    with path.open("xb") as output:
        output.write(header)
        output.truncate(file_bytes)
    hashes = []
    original_record = streamed_text.bounded_source_descriptor_record_v2

    def record(descriptor, **keywords):
        hashes.append(descriptor)
        return original_record(descriptor, **keywords)

    monkeypatch.setattr(streamed_text, "bounded_source_descriptor_record_v2", record)
    try:
        with prism_codec.mapped_profile_safetensors_conversion_shard_v2(profile, report, manifest, blocks,
                 source_path=path, source_shard=path.name, expected_file_bytes=file_bytes,
                 expected_file_sha256=profile.files[path.name].sha256,
                 include_frozen_source=include_frozen_source) as (operand, shard_receipt):
            assert shard_receipt["source_auxiliary_cap_bytes"] == 1 << 20
            for source_name in ("model.norm.weight", "model.layers.0.linear_attn.conv1d.weight", "model.layers.1.self_attn.k_proj.weight"):
                entry = entries[source_name]
                storage = entry["data_offsets"][1] - entry["data_offsets"][0]
                source_sha256 = hashlib.sha256(bytes(storage)).hexdigest()
                with operand(source_name, source_sha256, maximum_bytes=storage) as (values, receipt):
                    assert values.shape == tuple(entry["shape"]) and values.dtype == np.dtype("<f4" if entry["dtype"] == "F32" else "<u2")
                    assert not values.flags.writeable and not values.flags.owndata
                    assert receipt["source_dtype"] == entry["dtype"] and receipt["source_shape"] == values.shape
                    assert receipt["source_operand_kind"] == ("matrix" if len(values.shape) == 2 else "auxiliary")
                    assert hashlib.sha256(memoryview(values)).hexdigest() == source_sha256
                    with pytest.raises(ValueError, match="already has a live matrix"):
                        with operand(source_name, source_sha256):
                            pytest.fail("overlapping conversion source was mapped")
                assert values._mmap.closed and len(hashes) == 1
            if include_frozen_source:
                manifest["source_tensor_sha256"] = "0" * 64
                try:
                    with operand(manifest["tensor"], frozen_source_pin) as (frozen, receipt):
                        assert receipt["source_operand_kind"] == "frozen_source"
                        assert receipt["frozen_payload_sha256"] == frozen_payload_pin
                        assert receipt["source_to_frozen_quantization_verification"] == "not_checked"
                        assert hashlib.sha256(memoryview(frozen)).hexdigest() == frozen_source_pin
                        with pytest.raises(ValueError):
                            next(prism_codec.iter_profile_bf16_matrix_chunks(profile, report, manifest, blocks,
                                source_name=manifest["tensor"], source_values=frozen))
                    assert frozen._mmap.closed
                    with pytest.raises(ValueError, match="frozen source digest differs from manifest"):
                        with operand(manifest["tensor"], "0" * 64):
                            pytest.fail("caller digest replaced the frozen manifest binding")
                finally:
                    manifest["source_tensor_sha256"] = frozen_source_pin
            else:
                with pytest.raises(ValueError, match="requires an unfrozen BF16 matrix"):
                    with operand(manifest["tensor"], frozen_source_pin):
                        pytest.fail("default conversion mapping admitted the frozen replacement")
            vector_entry = next(item for item in plan["tensors"] if item["source_name"] == "model.norm.weight")
            original_shape = vector_entry["source_shape"]
            vector_entry["source_shape"] = [(1 << 20) // (4 if source_dtype == "F32" else 2) + 1]

            def never_map(*arguments, **keywords):
                pytest.fail("oversized auxiliary reached mapping")

            try:
                with monkeypatch.context() as bounded:
                    bounded.setattr(prism_codec.np, "memmap", never_map)
                    with pytest.raises(ValueError, match="file pin or operand budget mismatch"):
                        with operand("model.norm.weight", "0" * 64, maximum_bytes=2 * 1024**3):
                            pytest.fail("conversion mapping admitted an oversized auxiliary")
            finally:
                vector_entry["source_shape"] = original_shape
        assert len(hashes) == 2 and hashes[0] == hashes[1]
        assert hashlib.sha256(memoryview(blocks)).hexdigest() == frozen_payload_pin
        if include_frozen_source:
            original_layout = blocks.shape, blocks.dtype
            for change in ("bytes", "shape", "dtype"):
                changed = False
                try:
                    with pytest.raises(ValueError, match="frozen payload changed during source ownership"):
                        with prism_codec.mapped_profile_safetensors_conversion_shard_v2(profile, report, manifest, blocks,
                                 source_path=path, source_shard=path.name, expected_file_bytes=file_bytes,
                                 expected_file_sha256=profile.files[path.name].sha256, include_frozen_source=True):
                            if change == "bytes":
                                blocks.flat[0] ^= 1
                            elif change == "shape":
                                blocks.shape = (blocks.size,)
                            else:
                                blocks.dtype = np.uint16
                            changed = True
                finally:
                    blocks.dtype = original_layout[1]
                    blocks.shape = original_layout[0]
                    if changed and change == "bytes":
                        blocks.flat[0] ^= 1
                assert hashlib.sha256(memoryview(blocks)).hexdigest() == frozen_payload_pin
            wrong_manifest = {**manifest, "source_tensor_sha256": "0" * 64}
            with pytest.raises(ValueError, match="mapped source tensor identity mismatch"):
                with prism_codec.mapped_profile_safetensors_conversion_shard_v2(profile, report, wrong_manifest, blocks,
                         source_path=path, source_shard=path.name, expected_file_bytes=file_bytes,
                         expected_file_sha256=profile.files[path.name].sha256, include_frozen_source=True) as (operand, _):
                    with operand(wrong_manifest["tensor"], wrong_manifest["source_tensor_sha256"]):
                        pytest.fail("declared frozen digest did not match actual pinned source bytes")
    finally:
        path.unlink()


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_profile_conversion_shard_v2_refuses_nonboolean_frozen_flag_before_plan(monkeypatch, value):
    from embedded_jev import prism_codec

    def never_plan(*arguments, **keywords):
        pytest.fail("invalid frozen-source opt-in reached planning")

    monkeypatch.setattr(prism_codec, "profile_model_conversion_plan", never_plan)
    with pytest.raises(ValueError, match="invalid mapped source identity or budget"):
        with prism_codec.mapped_profile_safetensors_conversion_shard_v2(None, {}, {}, None,
                 source_path="unused.safetensors", source_shard="unused.safetensors",
                 expected_file_bytes=1, expected_file_sha256="0" * 64, include_frozen_source=value):
            pytest.fail("nonboolean frozen-source opt-in was accepted")


@pytest.mark.parametrize("consumer", ["mapping", "payload", "shared", "shared-v2", "conversion-v2"])
@pytest.mark.parametrize("defect", [
    None, "allocation", "peak-scaling", "source-pin", "file-pin", "file-size", "operand-budget", "boolean-budget", "file-budget",
    "deadline-budget", "frozen", "vector", "missing", "symlink", "fifo", "before-yield", "changed", "restored",
    "replacement", "deleted", "deadline", "body-error", "cwd",
])
def test_profile_mapped_bf16_source_bounds_lifetime_and_allocation(tmp_path, monkeypatch, defect, consumer):
    import errno
    import gc
    import tracemalloc
    from contextlib import contextmanager

    from embedded_jev import prism_codec

    profile, inventory, manifest, blocks = _prefill_model_policy_fixture(pin_zero_source=True)
    entries, payload_bytes = {}, 0
    for item in sorted(inventory["tensors"], key=lambda item: item["name"]):
        size = int(np.prod(item["shape"])) * 2
        entries[item["name"]] = {"dtype": item["dtype"], "shape": item["shape"], "data_offsets": [payload_bytes, payload_bytes + size]}
        payload_bytes += size
    header, file_bytes = _header(entries, payload_bytes)
    path = tmp_path / profile.packaging["shards"][0]
    path.write_bytes(header + bytes(payload_bytes))
    source_name = "model.layers.0.mlp.up_proj.weight"
    shape = tuple(entries[source_name]["shape"])
    operand_bytes = int(np.prod(shape)) * 2
    arguments = {"source_path": path, "source_name": source_name,
                 "expected_source_sha256": hashlib.sha256(bytes(operand_bytes)).hexdigest(),
                 "expected_file_bytes": file_bytes, "expected_file_sha256": profile.files[path.name].sha256}
    opened, mappings = [], []
    original_map, original_pread = prism_codec.np.memmap, prism_codec.os.pread

    def mapping(source, **keywords):
        opened.append(source.fileno())
        result = original_map(source, **keywords)
        mappings.append(result)
        if defect == "before-yield":
            with path.open("r+b") as output:
                output.seek(file_bytes - 1)
                output.write(bytes([1]))
        return result

    def bounded_read(descriptor, count, offset):
        assert 0 < count <= 65536
        return original_pread(descriptor, count, offset)

    monkeypatch.setattr(prism_codec.np, "memmap", mapping)
    monkeypatch.setattr(prism_codec.os, "pread", bounded_read)
    if defect == "source-pin":
        arguments["expected_source_sha256"] = "0" * 64
    elif defect == "file-pin":
        arguments["expected_file_sha256"] = "0" * 64
    elif defect == "file-size":
        arguments["expected_file_bytes"] -= 1
    elif defect == "operand-budget":
        arguments["maximum_bytes"] = operand_bytes - 1
    elif defect == "boolean-budget":
        arguments["maximum_bytes"] = True
    elif defect == "file-budget":
        arguments["maximum_file_bytes"] = (6 * 1024**3 if consumer in ("shared-v2", "conversion-v2") else 16 * 1024**2) + 1
    elif defect == "deadline-budget":
        arguments["maximum_seconds"] = float("nan")
    elif defect in ("frozen", "vector", "missing"):
        arguments["source_name"] = manifest["tensor"] if defect == "frozen" else "model.norm.weight" if defect == "vector" else "absent"
    elif defect in ("symlink", "fifo"):
        alternate = tmp_path / "alternate-source"
        if defect == "symlink":
            alternate.symlink_to(path)
        else:
            prism_codec.os.mkfifo(alternate)
        arguments["source_path"] = alternate
    expected_failure = defect not in (None, "allocation", "peak-scaling", "cwd")
    was_tracing = tracemalloc.is_tracing()

    @contextmanager
    def shared_matrix():
        file_arguments = {key: value for key, value in arguments.items() if key not in ("source_name", "expected_source_sha256", "maximum_bytes")}
        owner = prism_codec.mapped_profile_safetensors_bf16_shard_v2 if consumer == "shared-v2" else prism_codec.mapped_profile_safetensors_bf16_shard
        if consumer == "conversion-v2":
            owner = prism_codec.mapped_profile_safetensors_conversion_shard_v2
        with owner(profile, inventory, manifest, blocks, source_shard=path.name, **file_arguments) as (matrix, shard_receipt):
            if consumer in ("shared-v2", "conversion-v2"):
                assert shard_receipt["source_policy_version"] == 2
                assert shard_receipt["source_file_cap_bytes"] == 6 * 1024**3
                assert shard_receipt["source_operand_cap_bytes"] == 2 * 1024**3
            with matrix(arguments["source_name"], arguments["expected_source_sha256"],
                     maximum_bytes=arguments.get("maximum_bytes", 16 * 1024**2)) as result:
                if consumer in ("shared-v2", "conversion-v2"):
                    assert result[1]["source_policy_version"] == 2
                yield result

    def ownership():
        if consumer == "payload":
            return prism_codec.temporary_profile_mapped_bf16_payload(profile, inventory, manifest, blocks, **arguments, directory=tmp_path)
        if consumer in ("shared", "shared-v2", "conversion-v2"):
            return shared_matrix()
        return prism_codec.mapped_profile_safetensors_bf16_matrix(profile, inventory, manifest, blocks, **arguments)

    try:
        if defect == "peak-scaling":
            if was_tracing:
                pytest.skip("requires isolated ownership of the Python allocation tracer")

            def peak_allocations(check):
                check()
                gc.collect()
                tracemalloc.start()
                try:
                    check()
                    return tracemalloc.get_traced_memory()[1]
                finally:
                    tracemalloc.stop()

            peaks = []
            for selected_name in ("lm_head.weight", source_name):
                selected_bytes = int(np.prod(entries[selected_name]["shape"])) * 2
                arguments.update(source_name=selected_name, expected_source_sha256=hashlib.sha256(bytes(selected_bytes)).hexdigest())
                with prism_codec.mapped_profile_safetensors_bf16_matrix(profile, inventory, manifest, blocks, **arguments) as (supplied, _):
                    def copied_source():
                        prism_codec.verify_profile_safetensors_operand(profile, inventory, manifest, blocks, source_values=supplied, **arguments)

                    def mapped_source():
                        with ownership():
                            pass

                    peaks.append((peak_allocations(copied_source), peak_allocations(mapped_source)))
            assert selected_bytes == 2 * 1024**2
            assert peaks[1][0] >= 2 * selected_bytes
            assert peaks[1][1] < 3 * 1024**2
            assert peaks[1][1] <= peaks[0][1] + 256 * 1024
            assert all(mapped._mmap.closed for mapped in mappings)
            assert not list(tmp_path.glob("jev-bf16-payload-*"))
            for descriptor in opened:
                with pytest.raises(OSError):
                    prism_codec.os.fstat(descriptor)
            return
        if defect == "allocation":
            with ownership():
                pass
            if not was_tracing:
                tracemalloc.start()
            baseline = tracemalloc.get_traced_memory()[0]
        try:
            with ownership() as (values, receipt):
                if defect == "allocation":
                    retained = tracemalloc.get_traced_memory()[0] - baseline
                    assert operand_bytes == 2 * 1024**2 and retained < operand_bytes // 2
                if consumer != "payload":
                    assert values.shape == shape and values.dtype == np.dtype("<u2")
                    assert not values.flags.owndata and not values.flags.writeable
                    assert receipt["source_range_bytes"] == operand_bytes
                else:
                    assert values.stat().st_size == operand_bytes
                    assert receipt["source_file"]["source_range_bytes"] == operand_bytes
                    assert receipt["converted_sha256"] == arguments["expected_source_sha256"]
                if defect in ("changed", "restored"):
                    before = path.stat()
                    with path.open("r+b") as output:
                        output.seek(file_bytes - 1)
                        output.write(bytes([1]))
                        if defect == "restored":
                            output.seek(file_bytes - 1)
                            output.write(bytes(1))
                    if defect == "restored":
                        prism_codec.os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
                elif defect == "replacement":
                    path.unlink()
                    path.write_bytes(header + bytes(payload_bytes))
                elif defect == "deleted":
                    path.unlink()
                elif defect == "deadline":
                    later = prism_codec.time.monotonic() + 31
                    monkeypatch.setattr(prism_codec.time, "monotonic", lambda: later)
                elif defect == "body-error":
                    raise RuntimeError("mapped consumer failed")
                else:
                    if defect == "cwd":
                        elsewhere = tmp_path / "elsewhere"
                        elsewhere.mkdir()
                        monkeypatch.chdir(elsewhere)
                    if consumer != "payload":
                        for _, chunk in prism_codec.iter_profile_bf16_matrix_chunks(profile, inventory, manifest, blocks,
                               source_name=source_name, source_values=values):
                           assert chunk.nbytes <= 65536 and chunk.flags.owndata and not np.any(chunk)
            assert not expected_failure
        except (ValueError, OSError, RuntimeError) as error:
            assert expected_failure, str(error)
            if defect == "body-error":
                assert isinstance(error, RuntimeError) and str(error) == "mapped consumer failed"
            elif defect == "symlink":
                assert isinstance(error, OSError) and error.errno == errno.ELOOP
            elif defect == "vector" and consumer == "conversion-v2":
                assert isinstance(error, ValueError) and str(error) == "mapped source tensor identity mismatch"
            else:
                expected_message = (
                    "tensor identity mismatch" if defect == "source-pin" else
                    "file pin or operand budget mismatch" if defect in ("file-pin", "file-size", "operand-budget") else
                    "invalid mapped BF16 payload deadline" if defect == "deadline-budget" and consumer == "payload" else
                    "invalid mapped source identity or budget" if defect in ("boolean-budget", "file-budget", "deadline-budget") else
                    "requires an unfrozen BF16 matrix" if defect in ("frozen", "vector", "missing") else
                    "not regular" if defect == "fifo" else
                    "deadline exceeded" if defect == "deadline" else "file identity changed"
                )
                assert isinstance(error, ValueError) and expected_message in str(error)
        assert all(mapped._mmap.closed for mapped in mappings)
        assert not list(tmp_path.glob("jev-bf16-payload-*"))
        for descriptor in opened:
            with pytest.raises(OSError):
                prism_codec.os.fstat(descriptor)
    finally:
        if defect == "allocation" and not was_tracing and tracemalloc.is_tracing():
            tracemalloc.stop()
        path.unlink(missing_ok=True)
        if defect in ("symlink", "fifo"):
            alternate.unlink()


def test_profile_mapped_bf16_shard_v2_streams_large_synthetic_ranges(tmp_path, monkeypatch):
    import tracemalloc

    from embedded_jev import prism_codec, streamed_text

    profile, inventory, manifest, blocks = _prefill_model_policy_fixture(pin_zero_source=True, vocab_size=16384)
    entries, payload_bytes = {}, 0
    for item in sorted(inventory["tensors"], key=lambda item: item["name"]):
        storage = int(np.prod(item["shape"])) * 2
        entries[item["name"]] = {"dtype": item["dtype"], "shape": item["shape"], "data_offsets": [payload_bytes, payload_bytes + storage]}
        payload_bytes += storage
    header, file_bytes = _header(entries, payload_bytes)
    source_name = "model.embed_tokens.weight"
    operand_bytes = 32 * 1024**2
    assert 64 * 1024**2 < file_bytes <= 128 * 1024**2
    assert file_bytes + operand_bytes <= 256 * 1024**2
    assert entries[source_name]["data_offsets"][0] > 16 * 1024**2
    path = tmp_path / profile.packaging["shards"][0]
    with path.open("xb") as output:
        output.write(header)
        output.truncate(file_bytes)
    zero_chunk = bytes(65536)
    source_digest = hashlib.sha256()
    for _ in range(operand_bytes // len(zero_chunk)):
        source_digest.update(zero_chunk)
    arguments = {"source_path": path, "source_shard": path.name,
                 "expected_file_bytes": file_bytes, "expected_file_sha256": profile.files[path.name].sha256}
    original_pread = prism_codec.os.pread
    original_record = streamed_text.bounded_source_descriptor_record_v2
    hashes = []

    def bounded_read(descriptor, count, offset):
        assert 0 < count <= 65536
        return original_pread(descriptor, count, offset)

    def record(descriptor, **keywords):
        hashes.append(descriptor)
        return original_record(descriptor, **keywords)

    def never_open(*arguments, **keywords):
        pytest.fail("invalid file policy reached source open")

    try:
        with monkeypatch.context() as preflight:
            preflight.setattr(prism_codec.os, "open", never_open)
            with pytest.raises(ValueError, match="invalid mapped source identity or budget"):
                with prism_codec.mapped_profile_safetensors_bf16_shard(profile, inventory, manifest, blocks,
                         **arguments, maximum_file_bytes=file_bytes):
                    pytest.fail("legacy shard policy accepted a large source")
            with pytest.raises(ValueError, match="invalid mapped source identity or budget"):
                with prism_codec.mapped_profile_safetensors_bf16_shard_v2(profile, inventory, manifest, blocks, **arguments):
                    pytest.fail("v2 default file budget silently expanded")
            for maximum_file_bytes in (6 * 1024**3 + 1, True):
                with pytest.raises(ValueError, match="invalid mapped source identity or budget"):
                    with prism_codec.mapped_profile_safetensors_bf16_shard_v2(profile, inventory, manifest, blocks,
                             **arguments, maximum_file_bytes=maximum_file_bytes):
                        pytest.fail("invalid v2 file ceiling reached source open")
        monkeypatch.setattr(prism_codec.os, "pread", bounded_read)
        monkeypatch.setattr(streamed_text, "bounded_source_descriptor_record_v2", record)
        with prism_codec.mapped_profile_safetensors_bf16_shard_v2(profile, inventory, manifest, blocks,
                 **arguments, maximum_file_bytes=6 * 1024**3) as (matrix, receipt):
            assert receipt["source_policy_version"] == 2 and len(hashes) == 1
            with pytest.raises(ValueError, match="operand budget mismatch"):
                with matrix(source_name, source_digest.hexdigest()):
                    pytest.fail("v2 default operand budget silently expanded")
            with monkeypatch.context() as preflight:
                preflight.setattr(prism_codec.np, "memmap", never_open)
                for maximum_bytes in (2 * 1024**3 + 1, True):
                    with pytest.raises(ValueError, match="invalid mapped source identity or budget"):
                        with matrix(source_name, source_digest.hexdigest(), maximum_bytes=maximum_bytes):
                            pytest.fail("invalid v2 operand ceiling reached mapping")
            with matrix(source_name, source_digest.hexdigest(), maximum_bytes=2 * 1024**3) as (values, source_receipt):
                assert values.nbytes == operand_bytes and not values.flags.owndata and not values.flags.writeable
                assert source_receipt["source_range_offset"] == len(header) + entries[source_name]["data_offsets"][0]
                converted_digest = hashlib.sha256()
                for _, chunk in prism_codec.iter_profile_bf16_matrix_chunks(profile, inventory, manifest, blocks,
                         source_name=source_name, source_values=values):
                    assert chunk.nbytes <= 65536 and chunk.flags.owndata
                    converted_digest.update(memoryview(chunk))
                assert converted_digest.hexdigest() == source_digest.hexdigest()
                payload_arguments = {"source_name": source_name, "source_values": values,
                                     "expected_source_sha256": source_digest.hexdigest(), "directory": tmp_path}
                with monkeypatch.context() as preflight:
                    preflight.setattr(prism_codec.os, "open", never_open)
                    for owner in (prism_codec.temporary_profile_bf16_matrix_payload, prism_codec.temporary_profile_bf16_matrix_payload_v2):
                        with pytest.raises(ValueError, match="invalid temporary BF16 payload source or resource budget"):
                            with owner(profile, inventory, manifest, blocks, **payload_arguments):
                                pytest.fail("default payload budget silently expanded")
                    with pytest.raises(ValueError, match="invalid temporary BF16 payload source or resource budget"):
                        with prism_codec.temporary_profile_bf16_matrix_payload_v2(profile, inventory, manifest, blocks,
                                 **payload_arguments, maximum_bytes=2 * 1024**3 + 1):
                            pytest.fail("invalid payload ceiling reached file open")
                    tracemalloc.start()
                with prism_codec.temporary_profile_bf16_matrix_payload_v2(profile, inventory, manifest, blocks,
                         **payload_arguments, maximum_bytes=2 * 1024**3) as (payload_path, payload_receipt):
                    assert payload_receipt["payload_policy_version"] == 2
                    assert payload_receipt["payload_cap_bytes"] == 2 * 1024**3
                    assert payload_receipt["converted_bytes"] == payload_path.stat().st_size == operand_bytes
                    assert file_bytes + payload_path.stat().st_size <= 256 * 1024**2
                    payload_digest = hashlib.sha256()
                    with payload_path.open("rb", buffering=0) as output:
                        while data := output.read(65536):
                            payload_digest.update(data)
                    assert payload_digest.hexdigest() == payload_receipt["converted_sha256"] == source_digest.hexdigest()
                _, payload_peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                assert payload_peak < 3 * 1024**2
                assert not payload_path.exists() and not payload_path.parent.exists()
                original_chunks = prism_codec.iter_profile_bf16_matrix_chunks

                def interrupted_chunks(*arguments, **keywords):
                    chunks = original_chunks(*arguments, **keywords)
                    try:
                        yield next(chunks)
                        raise RuntimeError("injected large payload failure")
                    finally:
                        chunks.close()

                with monkeypatch.context() as interruption:
                    interruption.setattr(prism_codec, "iter_profile_bf16_matrix_chunks", interrupted_chunks)
                    with pytest.raises(RuntimeError, match="injected large payload failure"):
                        with prism_codec.temporary_profile_bf16_matrix_payload_v2(profile, inventory, manifest, blocks,
                                 **payload_arguments, maximum_bytes=operand_bytes):
                            pytest.fail("partial large payload was yielded")
                assert not list(tmp_path.glob("jev-bf16-payload-*"))
            assert values._mmap.closed and len(hashes) == 1
        assert len(hashes) == 2 and hashes[0] == hashes[1]
    finally:
        tracemalloc.stop()
        path.unlink()


def test_profile_mapped_bf16_shard_refuses_cross_shard_matrix(tmp_path):
    from embedded_jev.prism_codec import mapped_profile_safetensors_bf16_shard, profile_model_conversion_plan

    profile, inventory, manifest, blocks = _prefill_model_policy_fixture(pin_zero_source=True, source_shards=2)
    shard = profile.packaging["shards"][0]
    entries, payload_bytes = {}, 0
    for item in sorted(inventory["tensors"], key=lambda item: item["name"]):
        if item["shard"] != shard:
            continue
        size = int(np.prod(item["shape"])) * 2
        entries[item["name"]] = {"dtype": item["dtype"], "shape": item["shape"], "data_offsets": [payload_bytes, payload_bytes + size]}
        payload_bytes += size
    header, file_bytes = _header(entries, payload_bytes)
    path = tmp_path / shard
    path.write_bytes(header + bytes(payload_bytes))
    plan = profile_model_conversion_plan(profile, inventory, manifest, blocks)
    wrong = next(item for item in plan["tensors"] if item["type"] == "BF16" and item["source_name"] not in entries)
    right = next(item for item in plan["tensors"] if item["type"] == "BF16" and item["source_name"] in entries)
    arguments = {"source_path": path, "source_shard": shard, "expected_file_bytes": file_bytes,
                 "expected_file_sha256": profile.files[shard].sha256}
    try:
        with mapped_profile_safetensors_bf16_shard(profile, inventory, manifest, blocks, **arguments) as (matrix, _):
            wrong_bytes = int(np.prod(wrong["source_shape"])) * 2
            with pytest.raises(ValueError, match="belongs to another shard"):
                with matrix(wrong["source_name"], hashlib.sha256(bytes(wrong_bytes)).hexdigest()):
                    pytest.fail("a matrix from another declared shard was accepted")
            right_bytes = int(np.prod(right["source_shape"])) * 2
            with matrix(right["source_name"], hashlib.sha256(bytes(right_bytes)).hexdigest()) as (values, receipt):
                assert values.nbytes == right_bytes and not np.any(values)
                assert receipt["source_file_sha256"] == arguments["expected_file_sha256"]
            assert values._mmap.closed
        with pytest.raises(ValueError, match="shard is not declared"):
            with mapped_profile_safetensors_bf16_shard(profile, inventory, manifest, blocks,
                     **{**arguments, "source_shard": "undeclared.safetensors"}):
                pytest.fail("an undeclared shard was accepted")
    finally:
        path.unlink()


@pytest.mark.parametrize("defect", [None, "manifest-digest", "manifest-schema", "manifest-type", "array", "file", "dtype", "budget"])
def test_profile_frozen_source_tensor_matches_manifest_without_refitting(tmp_path, defect):
    from embedded_jev.prism_codec import verify_profile_frozen_source_tensor

    profile, report, manifest, blocks = _prefill_model_policy_fixture(pin_zero_source=True)
    entries, payload_bytes = {}, 0
    for item in sorted(report["tensors"], key=lambda item: item["name"]):
        size = int(np.prod(item["shape"])) * (4 if item["dtype"] == "F32" else 2)
        entries[item["name"]] = {"dtype": item["dtype"], "shape": item["shape"], "data_offsets": [payload_bytes, payload_bytes + size]}
        payload_bytes += size
    header, size = _header(entries, payload_bytes)
    path = tmp_path / profile.packaging["shards"][0]
    path.write_bytes(header + bytes(payload_bytes))
    values = np.zeros(manifest["shape"], dtype="<u2")
    manifest = {**manifest, "source_tensor_sha256": hashlib.sha256(memoryview(values)).hexdigest()}
    before = blocks.tobytes()
    arguments = {"source_path": path, "source_values": values, "expected_file_bytes": size, "expected_file_sha256": profile.files[path.name].sha256}
    if defect == "manifest-digest":
        manifest["source_tensor_sha256"] = "f" * 64
    elif defect == "manifest-schema":
        manifest.pop("source_tensor_sha256")
    elif defect == "manifest-type":
        manifest = None
    elif defect == "array":
        values.flat[0] = 1
    elif defect == "file":
        with path.open("r+b") as output:
            output.seek(len(header) + entries[manifest["tensor"]]["data_offsets"][0])
            output.write(b"\x01")
    elif defect == "dtype":
        arguments["source_values"] = values.astype(np.float32)
    elif defect == "budget":
        arguments["maximum_bytes"] = True
    try:
        if defect:
            with pytest.raises(ValueError, match="source operand"):
                verify_profile_frozen_source_tensor(profile, report, manifest, blocks, **arguments)
        else:
            receipt = verify_profile_frozen_source_tensor(profile, report, manifest, blocks, **arguments)
            assert receipt["source_sha256"] == manifest["source_tensor_sha256"]
            assert receipt["frozen_payload_sha256"] == manifest["payload_sha256"]
            assert receipt["frozen_source_verification"] == "manifest_source_digest_matches_profile_pinned_source_range"
            assert receipt["source_to_frozen_quantization_verification"] == receipt["pin_issuer_verification"] == "not_checked"
            assert receipt["runtime_approval"] == "not_granted_by_frozen_source_verification"
            with pytest.raises(TypeError):
                receipt["frozen_payload_sha256"] = "0" * 64
        assert blocks.tobytes() == before
    finally:
        path.unlink()


@pytest.mark.parametrize("defect", [None, "missing", "extra", "shards", "pin", "frozen-pin", "dtype", "shape", "layout", "array-type", "payload-type", "budget", "file-budget", "boolean", "deadline", "digest"])
def test_profile_source_tensor_table_preflights_complete_maps(monkeypatch, defect):
    from types import MappingProxyType

    from embedded_jev import prism_codec as codec

    profile, report, manifest, blocks = _prefill_model_policy_fixture(pin_zero_source=True)
    plan = codec.profile_model_conversion_plan(profile, report, manifest, blocks)
    sources = {item["source_name"]: np.zeros(item["source_shape"], dtype="<u2") for item in plan["tensors"]}
    pins = {name: hashlib.sha256(memoryview(values)).hexdigest() for name, values in sources.items()}
    manifest = {**manifest, "source_tensor_sha256": pins[manifest["tensor"]]}
    arguments = {"source_paths": {profile.packaging["shards"][0]: "unused"}, "source_values": sources, "expected_source_sha256": pins}
    name = next(iter(sources))
    if defect == "missing":
        sources.pop(name)
    elif defect == "extra":
        sources["unknown"] = np.zeros(1, dtype="<u2")
    elif defect == "shards":
        arguments["source_paths"] = {}
    elif defect == "pin":
        pins[name] = "invalid"
    elif defect == "frozen-pin":
        pins[manifest["tensor"]] = "f" * 64
    elif defect == "dtype":
        sources[name] = sources[name].astype("<f4")
    elif defect == "shape":
        sources[name] = sources[name].reshape(-1)
    elif defect == "layout":
        sources[name] = sources[name][::-1]
    elif defect == "array-type":
        sources[name] = None
    elif defect == "payload-type":
        blocks = blocks.tolist()
    elif defect == "budget":
        arguments["maximum_bytes"] = 1
    elif defect == "file-budget":
        arguments["maximum_file_bytes"] = 1
    elif defect == "boolean":
        arguments["maximum_bytes"] = True
    elif defect == "deadline":
        arguments["maximum_seconds"] = float("nan")
    elif defect == "digest":
        pins[name] = "0" * 64
    calls = []
    def verify(*received, **keywords):
        assert defect is None, "invalid table reached source file dispatch"
        values = keywords["source_values"]
        assert not values.flags.writeable
        assert not np.shares_memory(values, sources[keywords["source_name"]])
        calls.append(keywords["source_name"])
        return MappingProxyType({"source_file_identity": (1, 2, 3, 4, 5)})

    monkeypatch.setattr(codec, "verify_profile_safetensors_operand", verify)
    def verify_frozen(*received, **keywords):
        return verify(*received, source_name=manifest["tensor"], **keywords)

    monkeypatch.setattr(codec, "verify_profile_frozen_source_tensor", verify_frozen)
    def never_open(*received, **keywords):
        raise ValueError("source table final check reached") if defect is None else AssertionError("invalid table opened source file")

    monkeypatch.setattr(codec.os, "open", never_open)
    with pytest.raises(ValueError, match="source table"):
        codec.verify_profile_safetensors_tensor_table(profile, report, manifest, blocks, **arguments)
    assert len(calls) == (len(plan["tensors"]) if defect is None else 0)


@pytest.mark.parametrize("defect", [None, "f32", "caller-change", "restored-between", "restored-final", "deadline"])
def test_profile_source_tensor_table_matches_complete_pinned_shard(tmp_path, monkeypatch, defect):
    from embedded_jev import prism_codec as codec

    profile, report, manifest, blocks = _prefill_model_policy_fixture(
        pin_zero_source=True, vector_dtype="F32" if defect == "f32" else "BF16", conv_dtype="F32" if defect == "f32" else "BF16",
    )
    entries, payload_bytes = {}, 0
    for item in sorted(report["tensors"], key=lambda item: item["name"]):
        size = int(np.prod(item["shape"])) * (4 if item["dtype"] == "F32" else 2)
        entries[item["name"]] = {"dtype": item["dtype"], "shape": item["shape"], "data_offsets": [payload_bytes, payload_bytes + size]}
        payload_bytes += size
    header, size = _header(entries, payload_bytes)
    path = tmp_path / profile.packaging["shards"][0]
    path.write_bytes(header + bytes(payload_bytes))
    plan = codec.profile_model_conversion_plan(profile, report, manifest, blocks)
    sources = {item["source_name"]: np.zeros(item["source_shape"], dtype="<f4" if item["source_dtype"] == "F32" else "<u2") for item in plan["tensors"]}
    pins = {name: hashlib.sha256(memoryview(values)).hexdigest() for name, values in sources.items()}
    manifest = {**manifest, "source_tensor_sha256": pins[manifest["tensor"]]}
    paths = {path.name: path}
    count = 0
    original = codec._verify_profile_safetensors_operand
    def verify(*received, **keywords):
        nonlocal count
        result = original(*received, **keywords)
        count += 1
        if defect == "caller-change" and count == 1:
            for values in sources.values():
                values.flat[0] = 1
            paths[path.name] = tmp_path / "not-the-source"
            pins.clear()
        elif (defect == "restored-between" and count == 1) or (defect == "restored-final" and count == len(plan["tensors"])):
            with path.open("r+b") as output:
                output.seek(size - 1)
                output.write(b"\x01")
                output.seek(size - 1)
                output.write(b"\x00")
        elif defect == "deadline" and count == 1:
            monkeypatch.setattr(codec.time, "monotonic", lambda: float("inf"))
        return result

    monkeypatch.setattr(codec, "_verify_profile_safetensors_operand", verify)
    before = blocks.tobytes()
    try:
        if defect in ("restored-between", "restored-final", "deadline"):
            with pytest.raises(ValueError, match="source table .*" + ("deadline" if defect == "deadline" else "identity changed")):
                codec.verify_profile_safetensors_tensor_table(profile, report, manifest, blocks, source_paths=paths, source_values=sources, expected_source_sha256=pins)
        else:
            receipt = codec.verify_profile_safetensors_tensor_table(profile, report, manifest, blocks, source_paths=paths, source_values=sources, expected_source_sha256=pins)
            assert receipt["tensor_count"] == count == len(plan["tensors"]) == 28
            assert receipt["source_shard_count"] == 1 and receipt["source_file_bytes"] == size
            assert receipt["source_table_verification"] == "all_selected_source_ranges_match_profile_pins_and_caller_operands"
            assert receipt["source_to_frozen_quantization_verification"] == receipt["converted_file_verification"] == receipt["pin_issuer_verification"] == "not_checked"
            assert receipt["cross_file_atomic_snapshot"] == "not_guaranteed"
            frozen = next(item for item in receipt["tensors"] if item["source_name"] == manifest["tensor"])
            assert frozen["source_sha256"] == manifest["source_tensor_sha256"] and frozen["frozen_payload_sha256"] == manifest["payload_sha256"]
            assert len({item["source_file_identity"] for item in receipt["tensors"]}) == 1
            with pytest.raises(TypeError):
                receipt["tensor_count"] = 0
        assert blocks.tobytes() == before
    finally:
        path.unlink()


@pytest.mark.parametrize("defect", [None, "caller-change", "source-change", "deadline", "missing", "owner", "budget", "extent", "pin", "reader", "frozen-pin"])
def test_profile_source_converted_table_shares_snapshots_and_rechecks(monkeypatch, defect):
    import sys
    from types import MappingProxyType, SimpleNamespace

    from embedded_jev import prism_codec as codec

    profile, report, manifest, blocks = _prefill_model_policy_fixture(pin_zero_source=True)
    plan = codec.profile_model_conversion_plan(profile, report, manifest, blocks)
    sources = {item["source_name"]: np.zeros(item["source_shape"], dtype="<u2") for item in plan["tensors"]}
    pins = {name: hashlib.sha256(memoryview(values)).hexdigest() for name, values in sources.items()}
    manifest = {**manifest, "source_tensor_sha256": pins[manifest["tensor"]]}
    class Reader:
        tensors = [SimpleNamespace(name=item["name"]) for item in plan["tensors"]]

    monkeypatch.setitem(sys.modules, "gguf", SimpleNamespace(GGUFReader=Reader))
    arguments = {"reader": Reader(), "source_paths": {profile.packaging["shards"][0]: "unused"}, "source_values": sources,
                 "expected_source_sha256": pins, "expected_file_bytes": 16 * 1024**2, "expected_file_sha256": "0" * 64}
    if defect == "missing":
        sources.pop(next(iter(sources)))
    elif defect == "owner":
        arguments["model_owner"] = object()
    elif defect == "budget":
        arguments["maximum_bytes"] = 1
    elif defect == "extent":
        arguments["expected_file_bytes"] = 1
    elif defect == "pin":
        arguments["expected_file_sha256"] = "invalid"
    elif defect == "reader":
        arguments["reader"] = object()
    elif defect == "frozen-pin":
        pins[manifest["tensor"]] = "f" * 64
    calls = []
    snapshots = {}
    def source(*received, **keywords):
        assert defect in (None, "caller-change", "source-change", "deadline"), "invalid composed table reached source dispatch"
        assert manifest["tensor"] in keywords["source_values"]
        for name, values in keywords["source_values"].items():
            assert not values.flags.writeable and not np.shares_memory(values, sources[name])
            assert hashlib.sha256(memoryview(values)).hexdigest() == keywords["expected_source_sha256"][name]
        if not calls:
            snapshots.update(keywords["source_values"])
        else:
            assert all(values is snapshots[name] for name, values in keywords["source_values"].items())
        calls.append(("source", keywords["maximum_seconds"]))
        return MappingProxyType({"identity": 2 if defect == "source-change" and len(calls) == 3 else 1})

    def converted(*received, **keywords):
        assert manifest["tensor"] not in keywords["source_values"]
        assert all(values is snapshots[name] for name, values in keywords["source_values"].items())
        calls.append(("converted", keywords["maximum_seconds"]))
        if defect == "caller-change":
            for values in sources.values():
                values.flat[0] = 1
            pins.clear()
            arguments["source_paths"].clear()
        elif defect == "deadline":
            monkeypatch.setattr(codec.time, "monotonic", lambda: float("inf"))
        return MappingProxyType({"source_file_verification": "not_checked"})

    monkeypatch.setattr(codec, "verify_profile_safetensors_tensor_table", source)
    monkeypatch.setattr(codec, "verify_profile_gguf_tensor_table", converted)
    if defect not in (None, "caller-change"):
        with pytest.raises(ValueError):
            codec.verify_profile_gguf_source_tensor_table(profile, report, manifest, blocks, **arguments)
    else:
        receipt = codec.verify_profile_gguf_source_tensor_table(profile, report, manifest, blocks, **arguments)
        assert receipt["tensor_count"] == 28 and receipt["source_file_rechecked_after_conversion"] is True
        assert receipt["source_to_frozen_quantization_verification"] == receipt["model_metadata_verification"] == "not_checked"
        assert receipt["converted"]["source_file_verification"] == "not_checked"
        assert receipt["cross_file_atomic_snapshot"] == "not_guaranteed"
        with pytest.raises(TypeError):
            receipt["tensor_count"] = 0
    assert [name for name, _ in calls] == (["source", "converted", "source"] if defect in (None, "caller-change", "source-change") else ["source", "converted"] if defect == "deadline" else [])
    if calls:
        assert all(0 < duration <= 30 for _, duration in calls)
        assert all(after <= before for (_, before), (_, after) in zip(calls, calls[1:]))


@pytest.mark.parametrize("defect", [None, "caller-change", "source-change", "deadline", "budget", "file", "digest", "frozen", "owner", "dtype", "shape"])
def test_profile_gguf_source_operand_shares_snapshot_digest_and_deadline(monkeypatch, defect):
    from types import MappingProxyType

    from embedded_jev import prism_codec

    profile, report, manifest, blocks = _prefill_model_policy_fixture()
    plan = prism_codec.profile_model_conversion_plan(profile, report, manifest, blocks)
    entry = next(item for item in plan["tensors"] if item["type"] == "F32" and len(item["source_shape"]) == 1)
    values = np.zeros(entry["source_shape"], dtype="<u2")
    source_receipt = MappingProxyType({"source_sha256": hashlib.sha256(memoryview(values)).hexdigest()})
    converted_receipt = MappingProxyType({"source_sha256": source_receipt["source_sha256"]})
    events, snapshots = [], []
    clock = [0]
    monkeypatch.setattr(prism_codec.time, "monotonic", lambda: clock[0])

    def source(*arguments, **keywords):
        events.append("source")
        snapshots.append(keywords["source_values"])
        assert not keywords["source_values"].flags.writeable and not np.shares_memory(keywords["source_values"], values)
        assert keywords["expected_source_sha256"] == source_receipt["source_sha256"]
        assert keywords["maximum_seconds"] <= 30
        if defect == "source-change" and len(events) == 3:
            raise ValueError("source operand file identity mismatch")
        clock[0] += 2
        return source_receipt

    def converted(*arguments, **keywords):
        events.append("converted")
        assert keywords["source_values"] is snapshots[0] and keywords["kind"] == "vector"
        assert keywords["expected_source_sha256"] == source_receipt["source_sha256"]
        assert keywords["maximum_seconds"] == 28
        if defect == "caller-change":
            values.flat[0] = 1
        elif defect == "deadline":
            clock[0] = 31
        return converted_receipt

    monkeypatch.setattr(prism_codec, "verify_profile_safetensors_operand", source)
    monkeypatch.setattr(prism_codec, "_verify_profile_gguf_operand", converted)
    arguments = {
        "reader": object(), "source_path": "unused", "source_name": entry["source_name"], "source_values": values,
        "expected_source_sha256": source_receipt["source_sha256"], "expected_source_file_bytes": 1,
        "expected_source_file_sha256": "a" * 64, "expected_file_bytes": 1, "expected_file_sha256": "b" * 64,
    }
    if defect == "budget":
        arguments["maximum_bytes"] = True
    elif defect == "file":
        arguments["expected_source_file_bytes"] = 16 * 1024**2 + 1
    elif defect == "digest":
        arguments["expected_source_sha256"] = "0" * 64
    elif defect == "frozen":
        arguments["source_name"] = manifest["tensor"]
    elif defect == "owner":
        arguments["model_owner"] = object()
    elif defect == "dtype":
        arguments["source_values"] = values.astype(np.float64)
    elif defect == "shape":
        arguments["source_values"] = values[:-1]
    if defect not in (None, "caller-change"):
        with pytest.raises(ValueError):
            prism_codec.verify_profile_gguf_source_operand(profile, report, manifest, blocks, **arguments)
        assert events == (["source", "converted", "source"] if defect == "source-change" else ["source", "converted"] if defect == "deadline" else [])
        return
    receipt = prism_codec.verify_profile_gguf_source_operand(profile, report, manifest, blocks, **arguments)
    assert events == ["source", "converted", "source"] and snapshots[0] is snapshots[1]
    assert receipt["source"] is source_receipt and receipt["converted"] is converted_receipt
    assert receipt["source_file_rechecked_after_conversion"] is True and receipt["cross_file_atomic_snapshot"] == "not_guaranteed"
    with pytest.raises(TypeError):
        receipt["source_sha256"] = "0" * 64


@pytest.mark.parametrize("defect", [None, "ceilings", "converted-failure", "source-exit", "deadline", "source-cap", "gguf-cap",
                                    "operand-cap", "chunk-cap", "source-budget", "digest", "frozen", "vector", "shard", "row-budget", "owner"])
def test_profile_gguf_source_matrix_v2_shares_mapping_deadline_and_owner_exit(monkeypatch, defect):
    from contextlib import contextmanager
    from types import MappingProxyType

    from embedded_jev import prism_codec

    profile, report, manifest, blocks = _prefill_model_policy_fixture()
    source_name = "model.layers.1.self_attn.k_proj.weight"
    values = np.zeros((64, 1024), dtype="<u2")
    values.flags.writeable = False
    source_pin = hashlib.sha256(memoryview(values)).hexdigest()
    source_shard = profile.packaging["shards"][0]
    source_receipt = MappingProxyType({"source_sha256": source_pin})
    converted_receipt = MappingProxyType({"source_sha256": source_pin})
    clock = [0]
    events = []
    monkeypatch.setattr(prism_codec.time, "monotonic", lambda: clock[0])

    @contextmanager
    def matrix(name, pin, *, maximum_bytes):
        assert name == source_name and pin == source_pin
        assert maximum_bytes == arguments.get("maximum_bytes", 16 * 1024**2)
        events.append("matrix-enter")
        clock[0] += 2
        try:
            yield values, source_receipt
        finally:
            events.append("matrix-exit")
            clock[0] += 2

    @contextmanager
    def owner(*unused, **keywords):
        assert keywords["source_shard"] == source_shard and keywords["maximum_seconds"] == 30
        assert keywords["maximum_file_bytes"] == arguments.get("maximum_source_file_bytes", 16 * 1024**2)
        events.append("source-enter")
        clock[0] += 2
        try:
            yield matrix, source_receipt
            if defect == "source-exit":
                raise ValueError("source exit refused")
        finally:
            events.append("source-exit")
            clock[0] += 2

    def converted(*unused, **keywords):
        assert keywords["source_values"] is values and not values.flags.writeable
        assert keywords["expected_source_sha256"] == source_pin and keywords["maximum_seconds"] == 26
        assert keywords["maximum_file_bytes"] == arguments.get("maximum_file_bytes", 16 * 1024**2)
        assert keywords["maximum_bytes"] == arguments.get("maximum_bytes", 16 * 1024**2)
        assert keywords["maximum_chunk_bytes"] == arguments.get("maximum_chunk_bytes", 65536)
        events.append("converted")
        if defect == "converted-failure":
            raise ValueError("converted range refused")
        clock[0] = 31 if defect == "deadline" else 6
        return converted_receipt

    monkeypatch.setattr(prism_codec, "mapped_profile_safetensors_bf16_shard_v2", owner)
    monkeypatch.setattr(prism_codec, "verify_profile_gguf_matrix_v2", converted)
    arguments = {"reader": object(), "source_path": "unused", "source_shard": source_shard, "source_name": source_name,
                 "expected_source_sha256": source_pin, "expected_source_file_bytes": profile.files[source_shard].bytes,
                 "expected_source_file_sha256": profile.files[source_shard].sha256,
                 "expected_file_bytes": 16 * 1024**2, "expected_file_sha256": "a" * 64}
    if defect == "ceilings":
        arguments.update(maximum_source_file_bytes=6 * 1024**3, maximum_file_bytes=24 * 1024**3,
                         maximum_bytes=2 * 1024**3, maximum_chunk_bytes=1024**2)
    elif defect == "source-cap":
        arguments["maximum_source_file_bytes"] = 6 * 1024**3 + 1
    elif defect == "gguf-cap":
        arguments["maximum_file_bytes"] = 24 * 1024**3 + 1
    elif defect == "operand-cap":
        arguments["maximum_bytes"] = 2 * 1024**3 + 1
    elif defect == "chunk-cap":
        arguments["maximum_chunk_bytes"] = 1024**2 + 1
    elif defect == "source-budget":
        arguments["expected_source_file_bytes"] = 16 * 1024**2 + 1
    elif defect == "digest":
        arguments["expected_source_sha256"] = "invalid"
    elif defect in ("frozen", "vector"):
        arguments["source_name"] = manifest["tensor"] if defect == "frozen" else "model.norm.weight"
    elif defect == "shard":
        arguments["source_shard"] = "undeclared.safetensors"
    elif defect == "row-budget":
        arguments["maximum_chunk_bytes"] = 2047
    elif defect == "owner":
        arguments["model_owner"] = object()
    if defect not in (None, "ceilings"):
        with pytest.raises(TypeError if defect == "owner" else ValueError):
            prism_codec.verify_profile_gguf_source_matrix_v2(profile, report, manifest, blocks, **arguments)
        assert events == (["source-enter", "matrix-enter", "converted", "matrix-exit", "source-exit"]
                          if defect in ("converted-failure", "source-exit", "deadline") else [])
        return
    receipt = prism_codec.verify_profile_gguf_source_matrix_v2(profile, report, manifest, blocks, **arguments)
    assert events == ["source-enter", "matrix-enter", "converted", "matrix-exit", "source-exit"]
    assert receipt["source"] is source_receipt and receipt["source_shard"] is source_receipt
    assert receipt["converted"] is converted_receipt
    assert receipt["source_file_rechecked_after_conversion"] is receipt["source_mapping_closed_after_conversion"] is True
    assert receipt["cross_file_atomic_snapshot"] == "not_guaranteed"
    with pytest.raises(TypeError):
        receipt["source_sha256"] = "0" * 64


@pytest.mark.parametrize("defect", ["digest", "bytes", "boolean", "file-budget", "seconds", "nan", "owner"])
def test_profile_gguf_policy_metadata_refuses_invalid_inputs_before_import(monkeypatch, defect):
    import sys

    from embedded_jev.prism_codec import verify_profile_gguf_policy_metadata

    profile, report, manifest, blocks = _prefill_model_policy_fixture()
    arguments = {"reader": object(), "expected_file_bytes": 1, "expected_file_sha256": "f" * 64}
    if defect == "digest":
        arguments["expected_file_sha256"] = "not-a-digest"
    elif defect == "bytes":
        arguments["expected_file_bytes"] = 16 * 1024**2 + 1
    elif defect == "boolean":
        arguments["expected_file_bytes"] = True
    elif defect == "file-budget":
        arguments["maximum_file_bytes"] = 16 * 1024**2 + 1
    elif defect == "seconds":
        arguments["maximum_seconds"] = True
    elif defect == "nan":
        arguments["maximum_seconds"] = float("nan")
    else:
        arguments["model_owner"] = object()
    monkeypatch.setitem(sys.modules, "gguf", None)
    with pytest.raises(ValueError):
        verify_profile_gguf_policy_metadata(profile, report, manifest, blocks, **arguments)


@pytest.mark.parametrize("defect", ["digest", "bytes", "boolean", "file-budget", "boolean-budget", "default-budget",
                                    "seconds", "nan", "infinite", "zero", "owner"])
def test_profile_gguf_policy_metadata_v2_refuses_invalid_inputs_before_import(monkeypatch, defect):
    import sys

    from embedded_jev.prism_codec import verify_profile_gguf_policy_metadata_v2

    profile, report, manifest, blocks = _prefill_model_policy_fixture()
    arguments = {"reader": object(), "expected_file_bytes": 1, "expected_file_sha256": "f" * 64}
    if defect == "digest":
        arguments["expected_file_sha256"] = "invalid"
    elif defect == "bytes":
        arguments["expected_file_bytes"] = 0
    elif defect == "boolean":
        arguments["expected_file_bytes"] = True
    elif defect == "file-budget":
        arguments["maximum_file_bytes"] = 24 * 1024**3 + 1
    elif defect == "boolean-budget":
        arguments["maximum_file_bytes"] = True
    elif defect == "default-budget":
        arguments["expected_file_bytes"] = 16 * 1024**2 + 1
    elif defect in ("seconds", "nan", "infinite", "zero"):
        arguments["maximum_seconds"] = {"seconds": True, "nan": float("nan"), "infinite": float("inf"), "zero": 0}[defect]
    else:
        arguments["model_owner"] = object()
    monkeypatch.setitem(sys.modules, "gguf", None)
    with pytest.raises(TypeError if defect == "owner" else ValueError):
        verify_profile_gguf_policy_metadata_v2(profile, report, manifest, blocks, **arguments)


@pytest.mark.parametrize("defect", [None, "file-budget", "file-bytes", "file-pin", "seconds", "context", "template-selection", "embedded-missing", "template-budget",
                                    "metadata-pin", "tokenizer-pin", "mutable", "prepare-deadline", "verify-deadline", "owner"])
def test_profile_gguf_source_metadata_v2_prepares_pins_before_file_verification(monkeypatch, defect):
    from types import MappingProxyType

    from embedded_jev import prism_codec

    metadata = {}
    profile, report, manifest, blocks = _prefill_model_policy_fixture(metadata_output=metadata)
    clock = [0]
    events = []
    monkeypatch.setattr(prism_codec.time, "monotonic", lambda: clock[0])
    original_prepare = prism_codec.prepare_profile_model_template_metadata

    def prepare(*arguments, **keywords):
        result = original_prepare(*arguments, **keywords)
        events.append("prepared")
        clock[0] = 31 if defect == "prepare-deadline" else 2
        return result

    def verify(*arguments, **keywords):
        prepared = keywords["source_metadata"]
        events.append("verified")
        assert prepared["context_length"] == 256 and prepared["source_context_length"] == 512
        assert prepared["template"].encode() == metadata[profile.tokenization["template_file"]]
        assert keywords["policy_version"] == 2 and keywords["model_owner"] is None and keywords["maximum_seconds"] == 28
        clock[0] = 31 if defect == "verify-deadline" else 4
        return MappingProxyType({"metadata_names": (prepared["context_metadata_name"], prepared["template_metadata_name"])})

    monkeypatch.setattr(prism_codec, "prepare_profile_model_template_metadata", prepare)
    monkeypatch.setattr(prism_codec, "_verify_profile_gguf_policy_metadata", verify)
    arguments = {"reader": object(), "metadata_files": metadata, "tokenizer_bytes": b"{}", "context_length": 256,
                 "template_source": "source", "expected_file_bytes": 1, "expected_file_sha256": "f" * 64}
    if defect == "file-budget":
        arguments["maximum_file_bytes"] = 24 * 1024**3 + 1
    elif defect == "file-bytes":
        arguments["expected_file_bytes"] = True
    elif defect == "file-pin":
        arguments["expected_file_sha256"] = "invalid"
    elif defect == "seconds":
        arguments["maximum_seconds"] = True
    elif defect == "context":
        arguments["context_length"] = 513
    elif defect == "template-selection":
        arguments["template_source"] = "automatic"
    elif defect == "embedded-missing":
        arguments["template_source"] = "gguf_embedded"
    elif defect == "template-budget":
        from dataclasses import replace

        name = profile.tokenization["template_file"]
        profile = replace(profile, files={**profile.files, name: replace(profile.files[name], bytes=8193)})
    elif defect == "metadata-pin":
        arguments["metadata_files"] = {**metadata, "config.json": metadata["config.json"] + b" "}
    elif defect == "tokenizer-pin":
        arguments["tokenizer_bytes"] = b"{ }"
    elif defect == "mutable":
        arguments["metadata_files"] = {name: bytearray(data) for name, data in metadata.items()}
    elif defect == "owner":
        arguments["model_owner"] = object()
    if defect is not None:
        with pytest.raises(TypeError if defect == "owner" else ValueError):
            prism_codec.verify_profile_gguf_source_metadata_v2(profile, report, manifest, blocks, **arguments)
        assert events == (["prepared"] if defect == "prepare-deadline" else ["prepared", "verified"] if defect == "verify-deadline" else [])
        return
    receipt = prism_codec.verify_profile_gguf_source_metadata_v2(profile, report, manifest, blocks, **arguments)
    assert events == ["prepared", "verified"]
    assert receipt["context_length"] == 256 and receipt["template_render_verification"] == "not_checked"
    assert receipt["metadata_verification"] == "selected_policy_context_template_records_match_file"
    assert receipt["runtime_approval"] == "not_granted_by_source_metadata_verification"
    with pytest.raises(TypeError):
        receipt["context_length"] = 512


def test_profile_model_policy_refuses_unpinned_runtime():
    from embedded_jev.prism_codec import profile_model_specification

    profile, report, manifest, blocks = _projection_manifest_fixture()
    with pytest.raises(ValueError, match="runtime pin"):
        profile_model_specification(profile, report, manifest, blocks)


@pytest.mark.parametrize("defect", ["digest", "size", "budget", "boolean", "deadline", "scorer-pins", "table-reader", "table-sources", "table-pins", "table-boolean", "table-budget", "table-file", "metadata-reader", "metadata-bool", "source-reader", "source-paths", "source-frozen", "source-budget", "source-extra", "source-clock", "exit-source", "exit-boolean", "exit-null"])
def test_verified_model_policy_refuses_file_identity_before_native_dispatch(tmp_path, defect, monkeypatch):
    from embedded_jev import prism_codec

    profile, report, manifest, blocks = _model_policy_fixture()
    path = tmp_path / "model.gguf"
    path.write_bytes(b"synthetic file identity fixture")
    arguments = {"expected_bytes": path.stat().st_size, "expected_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if defect == "digest":
        arguments["expected_sha256"] = "0" * 64
    elif defect == "size":
        arguments["expected_bytes"] += 1
    elif defect == "budget":
        arguments["maximum_bytes"] = arguments["expected_bytes"] - 1
    elif defect == "boolean":
        arguments["expected_bytes"] = True
    elif defect == "scorer-pins":
        arguments["runtime_scorer"] = object()
    elif defect.startswith("exit-"):
        arguments["verify_source_files_on_exit"] = True if defect == "exit-source" else 1 if defect == "exit-boolean" else None
    elif defect.startswith("metadata-"):
        arguments["verify_policy_metadata"] = True if defect == "metadata-reader" else 1
    elif defect.startswith("source-"):
        arguments.update(source_paths={}, frozen_source_values=np.zeros(manifest["shape"], dtype="<u2"))
        if defect == "source-paths":
            arguments.pop("source_paths")
        elif defect == "source-frozen":
            arguments.pop("frozen_source_values")
        elif defect == "source-budget":
            arguments.update(tensor_reader=object(), source_values={}, expected_source_sha256={}, maximum_source_file_bytes=True)
        elif defect == "source-extra":
            arguments.update(tensor_reader=object(), source_values={manifest["tensor"]: np.zeros(1, dtype="<u2")}, expected_source_sha256={})
        elif defect == "source-clock":
            arguments.update(tensor_reader=object(), source_values={}, expected_source_sha256={})
            monkeypatch.setattr(prism_codec.time, "monotonic", lambda: 0)
            def expired_snapshot(*received, **keywords):
                monkeypatch.setattr(prism_codec.time, "monotonic", lambda: 31)
                return None, {}, {}, {}, None, 0

            monkeypatch.setattr(prism_codec, "_profile_source_table_snapshot", expired_snapshot)
    elif defect.startswith("table-"):
        arguments.update(tensor_reader=object(), source_values={}, expected_source_sha256={})
        if defect == "table-reader":
            arguments.pop("tensor_reader")
        elif defect == "table-sources":
            arguments.pop("source_values")
        elif defect == "table-pins":
            arguments.pop("expected_source_sha256")
        elif defect == "table-boolean":
            arguments["maximum_tensor_bytes"] = True
        elif defect == "table-budget":
            arguments["maximum_tensor_bytes"] = 64 * 1024**2 + 1
        elif defect == "table-file":
            arguments.update(expected_bytes=16 * 1024**2 + 1, maximum_bytes=16 * 1024**2 + 1)
    else:
        monkeypatch.setattr(prism_codec.time, "monotonic", iter((0, 31)).__next__)

    class NeverCalled:
        def __getattr__(self, name):
            pytest.fail("unverified model file reached native dispatch")

    with pytest.raises(ValueError):
        with prism_codec.verified_profile_model_policy(NeverCalled(), path, profile, report, manifest, blocks, **arguments):
            pytest.fail("unverified model policy was yielded")


@pytest.mark.parametrize("defect", ["caller", "change", "replace", "factory-change", "record", "runtime-change", "runtime-deadline", "runtime-pins", "table-failure", "table-change", "table-deadline", "table-snapshot", "metadata-failure", "metadata-change", "metadata-deadline", "metadata-snapshot", "source-failure", "source-change", "source-deadline", "source-snapshot"])
def test_verified_model_policy_retains_storage_and_frees_before_file_close(tmp_path, monkeypatch, defect):
    import ctypes
    from types import MappingProxyType, SimpleNamespace

    from embedded_jev import prism_codec

    profile, report, manifest, blocks = _model_policy_fixture()
    path = tmp_path / "model.gguf"
    payload = b"synthetic file identity fixture"
    path.write_bytes(payload)
    events = []
    descriptors = []
    handles = ctypes.c_void_p(123), ctypes.c_void_p(456)

    def factory(native, descriptor, *arguments):
        assert Path(descriptor).read_bytes() == payload
        descriptors.append(descriptor)
        events.append("create")
        if defect == "factory-change":
            path.write_bytes(payload + b"changed")
        if defect in ("table-snapshot", "metadata-snapshot", "source-snapshot"):
            table_pins.clear()
            table_sources.clear()
        return *handles, prism_codec.profile_model_specification(*arguments)

    class Free:
        def __call__(self, handle):
            assert handle.value == 123 and Path(descriptors[0]).is_file()
            events.append("free")

    monkeypatch.setattr(prism_codec, "create_profile_model_policy", factory)
    arguments = {}
    if defect.startswith(("table-", "metadata-", "source-")):
        table_sources = {"source": np.ones(1, dtype="<u2")}
        table_pins = {"source": hashlib.sha256(memoryview(table_sources["source"])).hexdigest()}
        expected_pins = table_pins.copy()
        table_owner = []
        table_receipt = MappingProxyType({"tensor_table_verification": "unit_test_table_gate", "tensor_count": 1})

        def verify_table(*received, **keywords):
            events.append("table")
            table_owner.append(keywords["model_owner"])
            assert Path(keywords["model_owner"].model_path).read_bytes() == payload
            assert keywords["expected_source_sha256"] == expected_pins and keywords["expected_source_sha256"] is not table_pins
            assert set(keywords["source_values"]) == {"source"} and keywords["source_values"] is not table_sources
            if defect == "table-failure":
                raise ValueError("tensor parity failure")
            if defect == "table-change":
                path.write_bytes(payload + b"changed")
            if defect == "table-deadline":
                monkeypatch.setattr(prism_codec.time, "monotonic", lambda: 31)
            return table_receipt

        if defect == "table-deadline":
            monkeypatch.setattr(prism_codec.time, "monotonic", lambda: 0)
        monkeypatch.setattr(prism_codec, "verify_profile_gguf_tensor_table", verify_table)
        arguments.update(tensor_reader=object(), source_values=table_sources, expected_source_sha256=table_pins)
        if defect.startswith("source-"):
            def snapshot(*received, **keywords):
                return None, table_sources.copy(), table_pins.copy(), {}, None, 0

            monkeypatch.setattr(prism_codec, "_profile_source_table_snapshot", snapshot)
            source_receipt = MappingProxyType({"source_table_verification": "unit_test_source_gate"})
            evidence = MappingProxyType({"converted": table_receipt, "source": source_receipt,
                                         "source_conversion_table_verification": "unit_test_composed_gate"})
            def verify_source(*received, **keywords):
                result = verify_table(*received, **keywords)
                assert result is table_receipt
                events[-1] = "source"
                if defect == "source-failure":
                    raise ValueError("source range mismatch")
                if defect == "source-change":
                    path.write_bytes(payload + b"changed")
                if defect == "source-deadline":
                    monkeypatch.setattr(prism_codec.time, "monotonic", lambda: 31)
                return evidence

            if defect == "source-deadline":
                monkeypatch.setattr(prism_codec.time, "monotonic", lambda: 0)
            monkeypatch.setattr(prism_codec, "verify_profile_gguf_source_tensor_table", verify_source)
            arguments.update(source_paths={}, frozen_source_values=np.zeros(1, dtype="<u2"))
        if defect.startswith("metadata-"):
            metadata_receipt = MappingProxyType({"policy_metadata_verification": "unit_test_metadata_gate", "metadata_names": ("typed_field",)})

            def verify_metadata(*received, **keywords):
                events.append("metadata")
                assert keywords["model_owner"] is table_owner[0]
                assert Path(keywords["model_owner"].model_path).read_bytes() == payload
                if defect == "metadata-failure":
                    raise ValueError("policy metadata declaration mismatch")
                if defect == "metadata-change":
                    path.write_bytes(payload + b"changed")
                if defect == "metadata-deadline":
                    monkeypatch.setattr(prism_codec.time, "monotonic", lambda: 31)
                return metadata_receipt

            if defect == "metadata-deadline":
                monkeypatch.setattr(prism_codec.time, "monotonic", lambda: 0)
            monkeypatch.setattr(prism_codec, "verify_profile_gguf_policy_metadata", verify_metadata)
            arguments["verify_policy_metadata"] = True
    if defect in ("runtime-change", "runtime-deadline", "runtime-pins"):
        runtime_calls = []
        pins = {"bridge": {"sha256": "f" * 64}}

        def verify_runtime(native, received, **keywords):
            assert received["bridge"]["sha256"] == "f" * 64 and received is not pins
            runtime_calls.append(1)
            if defect == "runtime-change" and len(runtime_calls) == 3:
                raise ValueError("runtime file identity mismatch")
            if defect == "runtime-deadline" and len(runtime_calls) == 3:
                raise InventoryError("native dependency hashing deadline exceeded")
            return {"bridge": {"sha256": "f" * 64}}

        monkeypatch.setattr(prism_codec, "verify_model_runtime_files", verify_runtime)
        arguments["expected_runtime_files"] = pins
    if defect == "record":
        def record_failure(*arguments):
            raise RuntimeError("record failure")

        monkeypatch.setattr(prism_codec, "native_model_specification", record_failure)
    native = SimpleNamespace(prism_bitnet_cpu_model_policy_free_v2=Free())
    with pytest.raises(RuntimeError if defect in ("caller", "record", "runtime-pins", "table-snapshot", "metadata-snapshot", "source-snapshot") else ValueError):
        with prism_codec.verified_profile_model_policy(
            native, path, profile, report, manifest, blocks,
            expected_bytes=len(payload), expected_sha256=hashlib.sha256(payload).hexdigest(),
            **arguments,
        ) as owner:
            if defect.startswith(("table-", "metadata-", "source-")):
                assert defect in ("table-snapshot", "metadata-snapshot", "source-snapshot")
                assert owner.verification["tensor_table"] is table_receipt
                assert owner.verification["tensor_table_verification"] == "unit_test_table_gate"
                with pytest.raises(TypeError):
                    owner.verification["tensor_table"]["tensor_count"] = 0
            if defect == "source-snapshot":
                assert owner.verification["source_conversion_table"] is evidence
                assert owner.verification["source_table_verification"] == "unit_test_source_gate"
                with pytest.raises(TypeError):
                    owner.verification["source_conversion_table"]["source_conversion_table_verification"] = "changed"
            if defect == "metadata-snapshot":
                assert owner.verification["policy_metadata"] is metadata_receipt
                with pytest.raises(TypeError):
                    owner.verification["policy_metadata"]["metadata_names"] = ()
            saved_bytes = ctypes.string_at(owner.record.projection.reference_pq2, owner.record.projection.reference_bytes)
            blocks.flat[-1] ^= 1
            assert ctypes.string_at(owner.record.projection.reference_pq2, owner.record.projection.reference_bytes) == saved_bytes
            assert owner.verification["runtime_approval"] == "not_granted_by_file_verification"
            with pytest.raises(TypeError):
                owner.verification["model_sha256"] = "0" * 64
            if defect == "change":
                path.write_bytes(payload + b"changed")
            elif defect == "replace":
                path.unlink()
                path.write_bytes(b"replacement")
                assert Path(owner.model_path).read_bytes() == payload
            elif defect not in ("runtime-change", "runtime-deadline"):
                if defect == "runtime-pins":
                    pins["bridge"]["sha256"] = "0" * 64
                raise RuntimeError("caller failure")
    expected_events = ["create", "table", "metadata", "free"] if defect.startswith("metadata-") else ["create", "source", "free"] if defect.startswith("source-") else ["create", "table", "free"] if defect.startswith("table-") else ["create", "free"]
    assert events == expected_events and handles[0].value is handles[1].value is None
    assert not Path(descriptors[0]).exists()
    if defect not in ("factory-change", "record", "table-failure", "table-change", "table-deadline", "metadata-failure", "metadata-change", "metadata-deadline", "source-failure", "source-change", "source-deadline"):
        for attribute in ("model_path", "record"):
            with pytest.raises(ValueError, match="closed"):
                getattr(owner, attribute)
    if defect.startswith(("table-", "metadata-", "source-")):
        with pytest.raises(ValueError, match="closed"):
            _ = table_owner[0].model_path


@pytest.mark.parametrize("shard_count", [1, 2])
@pytest.mark.parametrize("defect", ["ok", "default-change", "change", "first-change", "restored", "replace", "symlink", "missing", "deadline", "shared-deadline", "caller", "caller-change", "paths", "relative-path", "inventory"])
def test_verified_model_policy_source_exit_recheck_is_opt_in_and_frees(tmp_path, monkeypatch, defect, shard_count):
    import ctypes
    import os
    from contextlib import nullcontext
    from dataclasses import replace
    from types import MappingProxyType, SimpleNamespace

    from embedded_jev import prism_codec, streamed_text
    from embedded_jev.model_profile import ProfileFile

    profile, report, manifest, blocks = _model_policy_fixture()
    model_path, source_path = tmp_path / "model.gguf", tmp_path / "source.safetensors"
    model_bytes, source_bytes = b"synthetic model lifetime fixture", b"synthetic source lifetime fixture" * 3000
    model_path.write_bytes(model_bytes)
    source_path.write_bytes(source_bytes)
    source_name, shard = report["tensors"][0]["name"], report["tensors"][0]["shard"]
    profile = replace(profile, files=MappingProxyType({**profile.files, shard: ProfileFile(len(source_bytes), hashlib.sha256(source_bytes).hexdigest())}))
    observed = source_path.stat()
    source_identity = observed.st_dev, observed.st_ino, observed.st_size, observed.st_mtime_ns, observed.st_ctime_ns
    first_source_path = source_path
    source_paths = {shard: source_path}
    source_receipts = [MappingProxyType({"source_name": source_name, "source_file_identity": source_identity})]
    if shard_count == 2:
        shard = shard + ".second"
        source_name = report["tensors"][1]["name"]
        report["tensors"][1]["shard"] = shard
        source_path = tmp_path / "second-source.safetensors"
        source_path.write_bytes(source_bytes)
        source_paths[shard] = source_path
        profile = replace(profile, files=MappingProxyType({**profile.files, shard: ProfileFile(len(source_bytes), hashlib.sha256(source_bytes).hexdigest())}))
        observed = source_path.stat()
        source_identity = observed.st_dev, observed.st_ino, observed.st_size, observed.st_mtime_ns, observed.st_ctime_ns
        source_receipts.append(MappingProxyType({"source_name": source_name, "source_file_identity": source_identity}))
    table = MappingProxyType({"tensor_table_verification": "unit_test_table_gate"})
    source = MappingProxyType({"source_table_verification": "unit_test_source_gate", "tensors": tuple(source_receipts)})
    evidence = MappingProxyType({"converted": table, "source": source, "source_conversion_table_verification": "unit_test_composed_gate"})
    path_value = [str(source_path)]
    class MutablePath:
        def __fspath__(self):
            return path_value[0]

    source_paths[shard] = MutablePath()
    if defect == "relative-path":
        monkeypatch.chdir(tmp_path)
        path_value[0] = source_path.name
    events, descriptors, reads = [], [], []
    handles = ctypes.c_void_p(123), ctypes.c_void_p(456)
    original_pread = os.pread
    def pread(descriptor, size, offset):
        assert size <= 65536
        reads.append(size)
        return original_pread(descriptor, size, offset)

    def snapshot(*arguments, **keywords):
        return None, {}, {}, keywords["source_paths"].copy(), None, 0

    def factory(native, descriptor, *arguments):
        descriptors.append(descriptor)
        events.append("create")
        return *handles, prism_codec.profile_model_specification(*arguments)

    def verify_source(*arguments, **keywords):
        events.append("source")
        return evidence

    class Free:
        def __call__(self, handle):
            assert handle.value == 123 and Path(descriptors[0]).is_file()
            events.append("free")

    monkeypatch.setattr(os, "pread", pread)
    monkeypatch.setattr(prism_codec, "_profile_source_table_snapshot", snapshot)
    monkeypatch.setattr(prism_codec, "create_profile_model_policy", factory)
    monkeypatch.setattr(prism_codec, "verify_profile_gguf_source_tensor_table", verify_source)
    native = SimpleNamespace(prism_bitnet_cpu_model_policy_free_v2=Free())
    expected_error = None if defect in ("ok", "default-change", "paths", "relative-path", "inventory") or (defect == "shared-deadline" and shard_count == 1) else OSError if defect in ("symlink", "missing") else RuntimeError if defect == "caller" else ValueError
    with (nullcontext() if expected_error is None else pytest.raises(expected_error)):
        with prism_codec.verified_profile_model_policy(
            native, model_path, profile, report, manifest, blocks,
            expected_bytes=len(model_bytes), expected_sha256=hashlib.sha256(model_bytes).hexdigest(),
            tensor_reader=object(), source_values={}, expected_source_sha256={}, source_paths=source_paths,
            frozen_source_values=np.zeros(1, dtype="<u2"), verify_source_files_on_exit=defect != "default-change",
        ) as owner:
            assert owner.verification["source_file_lifetime_verification"] == ("pre_yield_only" if defect == "default-change" else "entry_and_exit_checks_required")
            if defect in ("change", "default-change", "caller-change"):
                source_path.write_bytes(b"X" + source_bytes[1:])
            elif defect == "first-change":
                first_source_path.write_bytes(b"X" + source_bytes[1:])
            elif defect == "restored":
                source_path.write_bytes(b"X" + source_bytes[1:])
                source_path.write_bytes(source_bytes)
                os.utime(source_path, ns=(observed.st_atime_ns, observed.st_mtime_ns))
            elif defect in ("replace", "symlink"):
                replacement = tmp_path / "replacement"
                replacement.write_bytes(source_bytes)
                source_path.unlink()
                if defect == "replace":
                    replacement.rename(source_path)
                else:
                    source_path.symlink_to(replacement)
            elif defect == "missing":
                source_path.unlink()
            elif defect == "paths":
                source_paths.clear()
                path_value[0] = str(tmp_path / "nonexistent")
            elif defect == "relative-path":
                other_directory = tmp_path / "other-cwd"
                other_directory.mkdir()
                monkeypatch.chdir(other_directory)
            elif defect == "inventory":
                report["tensors"].clear()
            elif defect == "deadline":
                clock = [0]
                def expired():
                    clock[0] += 31
                    return clock[0]

                monkeypatch.setattr(prism_codec.time, "monotonic", expired)
            elif defect == "shared-deadline":
                clock, deadlines = [0], []
                original_record = streamed_text.bounded_native_descriptor_record
                def timed_record(descriptor, *, maximum_bytes, deadline):
                    deadlines.append(deadline)
                    result = original_record(descriptor, maximum_bytes=maximum_bytes, deadline=deadline)
                    clock[0] += 20
                    return result

                monkeypatch.setattr(prism_codec.time, "monotonic", lambda: clock[0])
                monkeypatch.setattr(streamed_text, "bounded_native_descriptor_record", timed_record)
            if defect in ("caller", "caller-change"):
                raise RuntimeError("caller failed")
    assert events == ["create", "source", "free"] and handles[0].value is handles[1].value is None
    assert not Path(descriptors[0]).exists()
    with pytest.raises(ValueError, match="closed"):
        _ = owner.record
    if defect == "default-change":
        assert reads == []
    elif defect in ("ok", "paths", "relative-path", "inventory", "caller"):
        assert len(reads) >= 2 * shard_count
    if defect == "shared-deadline":
        assert deadlines == [30] * shard_count and clock[0] == 20 * shard_count


@pytest.mark.parametrize("kind", ["directory", "fifo", "symlink"])
def test_verified_model_policy_refuses_nonregular_or_symlink_inputs(tmp_path, kind):
    import os

    from embedded_jev.prism_codec import verified_profile_model_policy

    profile, report, manifest, blocks = _model_policy_fixture()
    path = tmp_path / "model.gguf"
    if kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        source = tmp_path / "source.gguf"
        source.write_bytes(b"x")
        path.symlink_to(source)

    class NeverCalled:
        def __getattr__(self, name):
            pytest.fail("nonregular or symlink model reached native dispatch")

    with pytest.raises((ValueError, OSError)):
        with verified_profile_model_policy(NeverCalled(), path, profile, report, manifest, blocks,
                                           expected_bytes=1, expected_sha256=hashlib.sha256(b"x").hexdigest()):
            pytest.fail("invalid model file was accepted")


@pytest.mark.parametrize("defect", [None, "digest", "size", "budget", "missing", "path", "requested-alias", "pin-set",
                                   "deadline", "seconds-zero", "seconds-boolean", "seconds-infinite", "seconds-nan", "seconds-cap"])
@pytest.mark.parametrize("with_scorer", [False, True])
def test_model_runtime_file_verification_binds_resolved_dependencies(tmp_path, monkeypatch, defect, with_scorer):
    from types import SimpleNamespace

    from embedded_jev.prism_codec import verify_model_runtime_files
    from embedded_jev import prism_codec

    names = ("bridge", "ggml_cpu", "ggml_base") + (() if not with_scorer else ("scorer", "llama", "ggml_registry"))
    paths = {name: tmp_path / (name + ".so") for name in names}
    for name, path in paths.items():
        path.write_bytes(name.encode())
    expected = {name: {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for name, path in paths.items()}
    class Locator:
        def __call__(self, index):
            if defect == "missing":
                return None
            if defect == "path" and index == 0:
                return str(tmp_path / "other.so").encode()
            return str(paths[("ggml_cpu", "ggml_base")[index]]).encode()

    native = SimpleNamespace(_name=str(paths["bridge"]), prism_group_scale_library_path=Locator())
    monkeypatch.setattr(prism_codec, "_model_runtime_library_paths", lambda *arguments: paths.copy(), raising=False)
    scorer = SimpleNamespace() if with_scorer else None
    maximum_bytes = 1 if defect == "budget" else 64 * 1024**2
    maximum_seconds = {"seconds-zero": 0, "seconds-boolean": True, "seconds-infinite": float("inf"),
                       "seconds-nan": float("nan"), "seconds-cap": 601}.get(defect, 30)
    if defect == "deadline":
        monkeypatch.setattr(prism_codec.time, "monotonic", iter((0, 0, 0, 31)).__next__)
    if defect == "digest":
        expected["bridge"]["sha256"] = "0" * 64
    elif defect == "size":
        expected["ggml_cpu"]["bytes"] += 1
    elif defect == "requested-alias":
        native._name = str(tmp_path / "not-the-loaded-bridge.so")
    elif defect == "pin-set":
        expected.pop("llama" if with_scorer else "ggml_base")
    if defect not in (None, "requested-alias"):
        with pytest.raises(ValueError):
            verify_model_runtime_files(native, expected, maximum_bytes=maximum_bytes, maximum_seconds=maximum_seconds, scorer=scorer)
    else:
        result = verify_model_runtime_files(native, expected, maximum_bytes=maximum_bytes, maximum_seconds=maximum_seconds, scorer=scorer)
        assert result == {name: {"path": str(paths[name]), **expected[name]} for name in paths}


@pytest.mark.parametrize("defect", [None, "short-read", "grow", "truncate", "replace", "deadline", "pre-deadline", "final-deadline", "budget", "fifo", "symlink"])
def test_bounded_native_file_record_limits_reads_and_refuses_changes(tmp_path, monkeypatch, defect):
    from embedded_jev import streamed_text

    path = tmp_path / "native.so"
    payload = b"a" * (2 * 65536 + 17)
    path.write_bytes(payload)
    if defect == "fifo":
        path.unlink()
        streamed_text.os.mkfifo(path)
    elif defect == "symlink":
        original = tmp_path / "original.so"
        path.rename(original)
        path.symlink_to(original)
    reads = []
    received = []
    original_fdopen = streamed_text.os.fdopen

    class Reader:
        def __init__(self, *arguments, **keywords):
            assert keywords["buffering"] == 0
            self.source = original_fdopen(*arguments, **keywords)

        def __enter__(self):
            return self

        def __exit__(self, *arguments):
            return self.source.__exit__(*arguments)

        def fileno(self):
            return self.source.fileno()

        def read(self, size):
            reads.append(size)
            chunk = self.source.read(min(size, 32768) if defect == "short-read" else size)
            received.append(len(chunk))
            if len(reads) == 1:
                if defect == "grow":
                    with path.open("ab") as output:
                        output.write(b"growth")
                elif defect == "truncate":
                    path.write_bytes(b"")
                elif defect == "replace":
                    path.unlink()
                    path.write_bytes(payload)
            return chunk

    monkeypatch.setattr(streamed_text.os, "fdopen", Reader)
    times = {"pre-deadline": iter((31,)), "deadline": iter((0, 31)), "final-deadline": iter((0, 0, 0, 31))}.get(defect)
    monkeypatch.setattr(streamed_text.time, "monotonic", times.__next__ if times else lambda: 0)
    arguments = {"maximum_bytes": len(payload) - (defect == "budget"), "deadline": 30}
    if defect in (None, "short-read"):
        assert streamed_text.bounded_native_file_record(path, **arguments) == {
            "path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
        }
    else:
        with pytest.raises((ValueError, OSError)):
            streamed_text.bounded_native_file_record(path, **arguments)
    assert max(reads, default=0) <= 65536 and sum(received) <= len(payload)
    if defect in ("budget", "fifo", "symlink", "pre-deadline"):
        assert reads == []


@pytest.mark.parametrize("hasher", ["bounded_native_descriptor_record", "bounded_source_descriptor_record_v2", "bounded_gguf_descriptor_record_v2"])
@pytest.mark.parametrize("defect", [None, "exact-ceiling", "budget", "boolean", "ceiling", "descriptor-boolean", "deadline", "nan", "infinite", "deadline-boolean", "grow", "truncate", "pipe"])
def test_bounded_native_descriptor_preserves_position_and_refuses_changes(tmp_path, monkeypatch, defect, hasher):
    from embedded_jev import streamed_text

    path = tmp_path / "descriptor.so"
    payload = b"a" * (65536 + 17)
    path.write_bytes(payload)
    calls = []
    original_pread = streamed_text.os.pread

    def read(descriptor, size, position):
        calls.append((size, position))
        chunk = original_pread(descriptor, size, position)
        if len(calls) == 1 and defect == "grow":
            with path.open("ab") as output:
                output.write(b"growth")
        elif len(calls) == 1 and defect == "truncate":
            path.write_bytes(b"")
        return chunk

    monkeypatch.setattr(streamed_text.os, "pread", read)
    monkeypatch.setattr(streamed_text.time, "monotonic", lambda: 31 if defect == "deadline" else 0)
    with path.open("rb", buffering=0) as source:
        source.seek(7)
        descriptor = source.fileno()
        pipe = None
        if defect == "pipe":
            pipe = streamed_text.os.pipe()
            descriptor = pipe[0]
        arguments = {"maximum_bytes": True if defect == "boolean" else len(payload) - (defect == "budget"), "deadline": 30}
        ceiling = {"bounded_native_descriptor_record": 64 * 1024**2,
                   "bounded_source_descriptor_record_v2": 6 * 1024**3,
                   "bounded_gguf_descriptor_record_v2": 24 * 1024**3}[hasher]
        if defect in ("ceiling", "exact-ceiling"):
            arguments["maximum_bytes"] = ceiling + (defect == "ceiling")
        elif defect == "descriptor-boolean":
            descriptor = True
        if defect in ("nan", "infinite", "deadline-boolean"):
            arguments["deadline"] = {"nan": float("nan"), "infinite": float("inf"), "deadline-boolean": True}[defect]
        try:
            if defect in (None, "exact-ceiling"):
                assert getattr(streamed_text, hasher)(descriptor, **arguments) == {
                    "path": f"/proc/self/fd/{descriptor}", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                }
            else:
                with pytest.raises(ValueError):
                    getattr(streamed_text, hasher)(descriptor, **arguments)
            assert source.tell() == 7
        finally:
            if pipe:
                for endpoint in pipe:
                    streamed_text.os.close(endpoint)
    assert max((size for size, _ in calls), default=0) <= 65536
    assert sum(size for size, _ in calls) <= len(payload)
    if defect in ("budget", "boolean", "ceiling", "descriptor-boolean", "deadline", "nan", "infinite", "deadline-boolean", "pipe"):
        assert calls == []


@pytest.mark.parametrize("maximum_bytes", [0, True, 24 * 1024**3 + 1])
def test_bounded_gguf_descriptor_v2_invalid_budget_refuses_before_descriptor_allocation(monkeypatch, maximum_bytes):
    from embedded_jev import streamed_text

    def never_duplicate(descriptor):
        pytest.fail("invalid GGUF budget reached descriptor allocation")

    monkeypatch.setattr(streamed_text.os, "dup", never_duplicate)
    with pytest.raises(ValueError, match="invalid native descriptor or file budget"):
        streamed_text.bounded_gguf_descriptor_record_v2(0, maximum_bytes=maximum_bytes, deadline=30)


def test_bounded_source_descriptor_v2_crosses_legacy_ceiling_without_large_buffers(tmp_path, monkeypatch):
    import time
    import tracemalloc

    from embedded_jev import streamed_text

    path = tmp_path / "synthetic-source-v2.bin"
    size = 64 * 1024**2 + 17
    with path.open("xb") as output:
        output.truncate(size)
    block = bytes(65536)
    expected = hashlib.sha256()
    for _ in range(size // len(block)):
        expected.update(block)
    expected.update(bytes(size % len(block)))
    original_pread = streamed_text.os.pread
    counts = []

    def bounded_read(descriptor, count, offset):
        assert 0 < count <= 65536
        counts.append(count)
        return original_pread(descriptor, count, offset)

    monkeypatch.setattr(streamed_text.os, "pread", bounded_read)
    was_tracing = tracemalloc.is_tracing()
    try:
        with path.open("rb", buffering=0) as source:
            source.seek(7)
            deadline = time.monotonic() + 30
            with pytest.raises(ValueError, match="invalid native descriptor or file budget"):
                streamed_text.bounded_native_descriptor_record(source.fileno(), maximum_bytes=size, deadline=deadline)
            assert counts == []
            if not was_tracing:
                tracemalloc.start()
            record = streamed_text.bounded_source_descriptor_record_v2(source.fileno(), maximum_bytes=size, deadline=deadline)
            if not was_tracing:
                assert tracemalloc.get_traced_memory()[1] < 1024**2
            assert source.tell() == 7
            assert record == {"path": f"/proc/self/fd/{source.fileno()}", "bytes": size, "sha256": expected.hexdigest()}
            assert sum(counts) == size and counts[-1] == 17
    finally:
        if not was_tracing and tracemalloc.is_tracing():
            tracemalloc.stop()
        path.unlink()


def test_profile_record_is_strict_immutable_and_canonically_hashed():
    record = _profile_fixture()
    profile = parse_model_profile(json.dumps(record).encode())
    assert profile.profile_id == "synthetic-profile"
    assert profile.model == "example/Synthetic" and profile.revision == "1" * 40
    assert profile.geometry["hidden_size"] == 1024
    assert profile.geometry["layer_types"] == ("full_attention",)
    assert profile.sha256 == parse_model_profile(json.dumps(record, sort_keys=True, indent=4).encode()).sha256
    with pytest.raises(TypeError):
        profile.geometry["hidden_size"] = 4096
    with pytest.raises(TypeError):
        profile.tokenization["render_arguments"]["enable_thinking"] = True
    with pytest.raises(TypeError):
        profile.files["config.json"] = profile.files["tokenizer.json"]


@pytest.mark.parametrize("section,field,value", [
    (None, "schema_version", True), (None, "schema_version", 2), (None, "unknown", 1),
    ("source", "revision", "main"), ("source", "unknown", "value"),
    ("source", "support_tier", "C"), ("geometry", "hidden_size", True),
    ("geometry", "rms_norm_eps", float("nan")), ("geometry", "rms_norm_eps", 10**400),
    ("geometry", "unknown", 1),
    ("geometry", "layer_types", ["linear_attention"]),
    ("packaging", "unknown", "value"), ("packaging", "mtp_handling", "include"),
    ("packaging", "shards", ["../model.safetensors"]),
    ("tokenization", "render_arguments", {"add_generation_prompt": True, "enable_thinking": True}),
    ("tokenization", "template_file", []), ("tokenization", "gguf_template_file", {}),
    ("runtime", "converter_flags", []), ("quantization", "group_size", True),
])
def test_profile_record_refuses_invalid_or_unknown_fields(section, field, value):
    record = _profile_fixture()
    target = record if section is None else record[section]
    target[field] = value
    with pytest.raises(ProfileError):
        parse_model_profile(json.dumps(record).encode())


def test_profile_record_refuses_duplicate_keys_paths_labels_and_oversized_input():
    for payload in (b'{"schema_version":1,"schema_version":1}', b"x" * (MAX_PROFILE_BYTES + 1), b"[]"):
        with pytest.raises(ProfileError):
            parse_model_profile(payload)
    record = _profile_fixture()
    record["files"]["../config.json"] = record["files"]["config.json"]
    with pytest.raises(ProfileError, match="unsafe profile file path"):
        parse_model_profile(json.dumps(record).encode())
    record = _profile_fixture()
    record["tokenization"]["label_token_ids"]["B"] = 0
    with pytest.raises(ProfileError, match="distinct non-special"):
        parse_model_profile(json.dumps(record).encode())
    record = _profile_fixture()
    record["files"]["config.json"]["extra"] = 1
    with pytest.raises(ProfileError, match="unknown profile fields"):
        parse_model_profile(json.dumps(record).encode())


def test_profile_header_identity_refuses_another_checkpoint():
    metadata, headers = _model_fixture()
    profile = parse_model_profile(json.dumps(_profile_fixture()).encode())
    verify_profile_headers(profile, metadata, headers)
    changed = {**metadata, "config.json": metadata["config.json"] + b" "}
    with pytest.raises(ProfileError, match="metadata identity"):
        verify_profile_headers(profile, changed, headers)
    name, (prefix, size) = next(iter(headers.items()))
    with pytest.raises(ProfileError, match="shard header identity"):
        verify_profile_headers(profile, metadata, {name: (prefix, size + 1)})
    with pytest.raises(ProfileError, match="shard header identity"):
        verify_profile_headers(profile, metadata, {name: (prefix[:-1] + b" ", size)})
    with pytest.raises(ProfileError, match="metadata file set"):
        verify_profile_headers(profile, {}, headers)
    with pytest.raises(ProfileError, match="shard file set"):
        verify_profile_headers(profile, metadata, {})


def test_profile_nonweight_identity_never_hashes_weight_payloads(tmp_path):
    profile = parse_model_profile(json.dumps(_profile_fixture()).encode())
    metadata, headers = _model_fixture()
    for name, data in {**metadata, "tokenizer.json": b"{}"}.items():
        (tmp_path / name).write_bytes(data)
    verify_profile_files(profile, tmp_path)
    with pytest.raises(ProfileError, match="nonweight identity check"):
        verify_profile_files(profile, tmp_path, headers)
    (tmp_path / "tokenizer.json").write_bytes(b"[]")
    with pytest.raises(ProfileError, match="file identity"):
        verify_profile_files(profile, tmp_path)
    with pytest.raises(ProfileError, match="not declared"):
        verify_profile_files(profile, tmp_path, ("other.json",))


def test_mimo_profile_preserves_existing_source_and_geometry_pins():
    profile = mimo_profile()
    assert profile.model == inventory.MODEL_ID and profile.revision == inventory.MODEL_REVISION
    assert profile.license == "MIT" and len(profile.files) == 17
    assert profile.geometry["num_hidden_layers"] == 32
    assert profile.geometry["hidden_size"] == 4096
    assert profile.geometry["intermediate_size"] == 12288
    assert profile.geometry["vocab_size"] == 248320
    assert profile.packaging["text_tensor_count"] == 427
    assert profile.runtime["native_policy"] == "mimo-qwen35-layer3-ffn-down-v1"
    assert tuple(profile.packaging["metadata_files"]) == inventory.REQUIRED_METADATA + inventory.OPTIONAL_METADATA
    assert profile.tokenization["label_token_ids"] == dict(zip("ABCDEFGHIJKLMNOP", range(32, 48), strict=True))
    assert tuple(profile.files[name].sha256 for name in profile.packaging["shards"]) == (
        "aab052180118aee34abc3029b54eaa49096aac606b97d703866b420dceb703c3",
        "7a0486565f06d25ac4628e9dba470dc3f604353471d240d5a0bf7128f64df396",
        "6c73207563d1879bfd6c143a028cc70be68edff56280458c71a43df4240300f4",
        "1379a7cf8c0b8555a39ab65a47e830e0eb45e776b46045734da3afd73c09eea2",
    )
    assert sum(profile.files[name].bytes for name in profile.packaging["shards"]) == 18819720848
    assert sum(profile.files[name].header_bytes for name in profile.packaging["shards"]) == 93360


def test_defiant_fable_record_is_separate_metadata_evidence_not_runtime_support():
    profile = defiant_fable_profile()
    assert profile.model.startswith("DavidAU/") and profile.model != mimo_profile().model
    assert profile.revision == "7af0a9c4e221e01b246b3c577fbb7110b79823e8"
    assert profile.license == "Apache-2.0" and len(profile.files) == 17
    assert profile.packaging["mtp_status"] == "complete"
    assert profile.packaging["index_total_policy"] == "recomputed"
    assert "processor_config.json" not in profile.files and "merges.txt" not in profile.files
    assert profile.files["model-mtp-restored.safetensors"].header_bytes == 96
    assert profile.runtime["native_policy"] is None
    assert profile.tokenization["non_thinking_suffix"] != mimo_profile().tokenization["non_thinking_suffix"]
    assert sum(profile.files[name].bytes for name in profile.packaging["shards"]) == 19306304048


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
    native_plan = report["native_reference_plan"]
    assert native_plan["scope"] == "metadata_only_text_reference_no_conversion"
    assert native_plan["text_tensor_count"] == 14
    assert native_plan["text_source_weight_bytes"] == (
        report["totals"]["stored_weight_bytes"] - report["memory_estimates"]["vision_original_bytes"]
    )
    assert native_plan["full_vocabulary_weight_bytes"] == 2 * 32 * 1024 * 2
    assert native_plan["largest_source_tensor_bytes"] == 2048 * 1024 * 2
    assert native_plan["largest_fp32_tensor_bytes"] == 2048 * 1024 * 4
    assert native_plan["largest_source_plus_fp32_tensor_bytes"] == 2048 * 1024 * 6
    assert native_plan["target_generated_answer_tokens"] == 0


def test_unverified_inventory_never_claims_pinned_model_identity():
    metadata, headers = _model_fixture()
    report = build_inventory(metadata, headers)
    assert report["source"]["model"] is None and report["source"]["revision"] is None
    assert report["source"]["profile_id"] is None
    assert report["source"]["identity_verification"] == "unbound_header_inputs"


def test_native_reference_plan_excludes_optional_payloads_without_shrinking_head():
    metadata, shards = _model_fixture()
    report = build_inventory(metadata, shards)
    original = report["native_reference_plan"]
    tensors = report["tensors"] + [{"category": "optional_mtp", "storage_bytes": 10**12}]
    plan = inventory._native_reference_plan(tensors)
    assert plan["text_source_weight_bytes"] == original["text_source_weight_bytes"]
    assert plan["full_vocabulary_weight_bytes"] == original["full_vocabulary_weight_bytes"]
    assert plan["largest_source_plus_fp32_tensor_bytes"] == original["largest_source_plus_fp32_tensor_bytes"]
    assert plan["excluded_source_weight_bytes"]["optional_mtp"] == 10**12


def test_local_inventory_reads_bounded_headers_without_weight_payload(tmp_path, monkeypatch):
    metadata, shards = _model_fixture()
    for name, data in metadata.items():
        (tmp_path / name).write_bytes(data)
    for name, (prefix, size) in shards.items():
        with (tmp_path / name).open("wb") as destination:
            destination.write(prefix)
            destination.truncate(size)
    monkeypatch.setattr(inventory, "_open_bounded", lambda *args, **kwargs: pytest.fail("network read"))
    profile = parse_model_profile(json.dumps(_profile_fixture()).encode())
    with pytest.raises(InventoryError, match="no verified model profile"):
        inventory.read_local_headers(tmp_path)
    local_metadata, local_headers = inventory.read_local_headers(tmp_path, profile=profile)
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


def test_local_inventory_binds_a_second_profile_without_mimo_identity(tmp_path):
    metadata, headers = _model_fixture()
    record = _profile_fixture()
    record["profile_id"] = "synthetic-second"
    record["source"].update(model="example/Second", revision="3" * 40)
    metadata["config.json"] += b"\n"
    record["files"]["config.json"] = {
        "bytes": len(metadata["config.json"]), "sha256": hashlib.sha256(metadata["config.json"]).hexdigest(),
    }
    profile = parse_model_profile(json.dumps(record).encode())
    for name, data in metadata.items():
        (tmp_path / name).write_bytes(data)
    for name, (prefix, size) in headers.items():
        with (tmp_path / name).open("wb") as destination:
            destination.write(prefix)
            destination.truncate(size)
    with pytest.raises(InventoryError, match="no verified model profile"):
        inventory.read_local_headers(tmp_path)
    local_metadata, local_headers = inventory.read_local_headers(tmp_path, profile=profile)
    report = build_inventory(local_metadata, local_headers, profile=profile)
    assert report["source"]["model"] == "example/Second"
    assert report["source"]["revision"] == "3" * 40
    assert report["source"]["profile_id"] == "synthetic-second"
    assert report["source"]["profile_sha256"] == profile.sha256
    with pytest.raises(InventoryError, match="shard file set"):
        build_inventory(local_metadata, local_headers, profile=mimo_profile())
    original_profile = parse_model_profile(json.dumps(_profile_fixture()).encode())
    with pytest.raises(InventoryError, match="metadata identity"):
        build_inventory(local_metadata, local_headers, profile=original_profile)


@pytest.mark.parametrize("runner", [run_text_prefix, run_streamed_text])
def test_unknown_scoring_source_is_refused_before_model_import(tmp_path, monkeypatch, runner):
    import sys

    metadata, headers = _model_fixture()
    for name, data in metadata.items():
        (tmp_path / name).write_bytes(data)
    for name, (prefix, size) in headers.items():
        with (tmp_path / name).open("wb") as destination:
            destination.write(prefix)
            destination.truncate(size)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(InventoryError, match="shard file set"):
        runner(tmp_path, layers=1, prompt="never scored")


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
    for backend in ("unsupported", "prism_ggml", "prism_ggml_f32", "prism_ggml_hadamard128"):
        with pytest.raises(InventoryError, match="native graph backend requires"):
            run_streamed_text(None, prompt="A or B", native_ffn_backend=backend)
    library = tmp_path / "kernel.so"
    library.write_bytes(b"placeholder not loaded")
    with pytest.raises(InventoryError, match="cannot reinterpret"):
        run_streamed_text(
            None, prompt="A or B", native_ffn_backend="prism_ggml_hadamard128",
            native_ffn_library=library, projection_artifact=tmp_path / "identity-artifact",
        )

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
    profile = parse_model_profile(json.dumps(_profile_fixture()).encode())
    monkeypatch.setattr("embedded_jev.model_profile.mimo_profile", lambda: profile)
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
    profile = parse_model_profile(json.dumps(_profile_fixture()).encode())
    values, source = weight_slice.read_local_bf16_projection_rows(
        tmp_path, name=name, start_row=1, rows=2, profile=profile,
    )
    np.testing.assert_array_equal(values, np.array([[1.25] * 2048, [-2.5] * 2048]))
    assert source["payload_bytes"] == 2 * row_bytes
    assert source["row_sha256"] == [hashlib.sha256(data).hexdigest() for data in raw_rows]
    assert source["full_weight_hash"] == "not_checked_local_rows_only"
    for request in ({"rows": 5}, {"start_row": 1023, "rows": 2}, {"name": "lm_head.weight"}):
        with pytest.raises(InventoryError):
            weight_slice.read_local_bf16_projection_rows(tmp_path, profile=profile, **request)
    monkeypatch.setattr(weight_slice, "MAX_LOCAL_PAYLOAD_BYTES", row_bytes)
    with pytest.raises(InventoryError, match="payload budget"):
        weight_slice.read_local_bf16_projection_rows(tmp_path, name=name, rows=2, profile=profile)


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
    profile = parse_model_profile(json.dumps(_profile_fixture()).encode())
    report = weight_slice.screen_local_projection(tmp_path, profile=profile)
    assert report["tensor"] == indexed_name and report["rows"] == 1024
    assert report["columns"] == 2048 and report["rows_per_batch"] == 64
    assert report["policies"]["maxabs"]["weight_mse"] == 0.0
    assert report["policies"]["searched_fp16"]["weight_mse"] == 0.0
    assert report["policies"]["searched_fp16"]["nonzero_fraction"] == 1 / 1024
    monkeypatch.setattr(weight_slice, "MAX_STREAM_TENSOR_BYTES", 4096)
    with pytest.raises(InventoryError, match="stream screen bounds"):
        weight_slice.screen_local_projection(tmp_path, profile=profile)


def test_prism_candidate_widths_come_from_profile_verified_eligible_hf_headers():
    metadata, shards = _model_fixture()
    profile = parse_model_profile(json.dumps(_profile_fixture()).encode())
    report = build_inventory(metadata, shards, profile=profile)
    names = ["blk.0.ffn_down.weight", "blk.0.ffn_gate.weight", "blk.0.attn_q.weight"]
    widths = profile_prism_expected_widths(report, names, profile=profile)
    assert widths == {
        "blk.0.ffn_down.weight": 2048,
        "blk.0.ffn_gate.weight": 1024,
        "blk.0.attn_q.weight": 1024,
    }
    with pytest.raises(TernaryArtifactError, match="unverified MiMo"):
        mimo_prism_expected_widths(report, names)
    with pytest.raises(TernaryArtifactError, match="unverified profile"):
        profile_prism_expected_widths(
            {**report, "source": {**report["source"], "revision": "main"}}, names, profile=profile,
        )
    with pytest.raises(TernaryArtifactError, match="missing or ineligible"):
        profile_prism_expected_widths(report, ["blk.1.ffn_down.weight"], profile=profile)
    with pytest.raises(TernaryArtifactError, match="unsupported profile"):
        profile_prism_expected_widths(report, ["blk.0.ssm_out.weight"], profile=profile)

    codes, scales = quantize_ternary_rtn(np.ones((2, 256), dtype=np.float32))
    payload = save_toy_artifact(
        codes, scales, algorithm="rtn-maxabs-fp16-v1",
        signs=np.ones(256, dtype=np.int8), block_size=128,
    )
    with pytest.raises(TernaryArtifactError, match="logical input width mismatch"):
        prism_v1_transform_metadata(
            {names[0]: payload}, expected_widths=profile_prism_expected_widths(report, [names[0]], profile=profile)
        )