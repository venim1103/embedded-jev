"""Bounded, offline safetensors-header inspection for the pinned MiMo model."""

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from math import prod
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


MODEL_ID = "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B"
MODEL_REVISION = "2367e865d009c13ac81713a2878291d33ab28177"
REQUIRED_METADATA = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "chat_template.jinja",
)
OPTIONAL_METADATA = ("video_preprocessor_config.json",)
MAX_METADATA_BYTES = 1 << 20
MAX_SHARDS = 16
GROUP_SIZE = 128
ROTATION_SIZE = 1024
SHARD_NAME = re.compile(r"model-([0-9]{5})-of-([0-9]{5})\.safetensors\Z")
LAYER_NAME = re.compile(r"model\.language_model\.layers\.(0|[1-9][0-9]*)\.(.*)\Z")
CONTENT_RANGE = re.compile(r"bytes ([0-9]{1,20})-([0-9]{1,20})/([0-9]{1,20})\Z")
MODEL_URL = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/"


DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


class InventoryError(ValueError):
    """Malformed or unsupported model metadata."""


@dataclass(frozen=True)
class TensorHeader:
    name: str
    shape: tuple[int, ...]
    dtype: str
    parameters: int
    storage_bytes: int


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InventoryError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise InventoryError(f"invalid JSON constant: {value}")


def parse_safetensors_header(
    prefix: bytes, shard_bytes: int, *, max_header_bytes: int = 1 << 20
) -> tuple[TensorHeader, ...]:
    """Validate an eight-byte length, JSON header, and complete shard byte layout.

    ``prefix`` contains only the length and header, not the weight payload.
    ``shard_bytes`` is the independently reported total file size.
    """
    if len(prefix) < 8:
        raise InventoryError("missing safetensors header length")
    if type(shard_bytes) is not int or shard_bytes < 0:
        raise InventoryError("invalid safetensors shard size")
    header_bytes = int.from_bytes(prefix[:8], "little")
    if header_bytes == 0 or header_bytes > max_header_bytes:
        raise InventoryError("safetensors header exceeds allowed bounds")
    if len(prefix) != 8 + header_bytes or shard_bytes < len(prefix):
        raise InventoryError("safetensors header length disagrees with shard size")
    try:
        header = json.loads(
            prefix[8:].decode("utf-8"),
            object_pairs_hook=_unique_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InventoryError("invalid safetensors header JSON") from exc
    if not isinstance(header, dict):
        raise InventoryError("safetensors header must be an object")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(
        not isinstance(value, str) for value in metadata.values()
    ):
        raise InventoryError("invalid safetensors metadata")
    if len(header) > 4096:
        raise InventoryError("too many tensors in safetensors header")

    tensors = []
    spans = []
    for name, spec in header.items():
        if not isinstance(spec, dict) or set(spec) != {"dtype", "shape", "data_offsets"}:
            raise InventoryError(f"invalid tensor header: {name}")
        dtype, shape, offsets = spec["dtype"], spec["shape"], spec["data_offsets"]
        if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
            raise InventoryError(f"unsupported safetensors dtype: {dtype}")
        if not isinstance(shape, list) or len(shape) > 16 or any(
            type(dimension) is not int or dimension < 0 or dimension > shard_bytes
            for dimension in shape
        ):
            raise InventoryError(f"invalid tensor shape: {name}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int or offset < 0 for offset in offsets)
            or offsets[0] > offsets[1]
        ):
            raise InventoryError(f"invalid data offsets: {name}")
        parameters = prod(shape)
        storage_bytes = parameters * DTYPE_BYTES[dtype]
        if offsets[1] - offsets[0] != storage_bytes:
            raise InventoryError(f"tensor offset size mismatch: {name}")
        tensors.append(TensorHeader(name, tuple(shape), dtype, parameters, storage_bytes))
        spans.append((offsets[0], offsets[1]))

    end = 0
    for start, stop in sorted(spans):
        if start != end:
            raise InventoryError("safetensors data offsets overlap or leave a gap")
        end = stop
    if end != shard_bytes - len(prefix):
        raise InventoryError("safetensors offsets disagree with shard size")
    return tuple(sorted(tensors, key=lambda tensor: tensor.name))


def _json_object(data: bytes, name: str) -> dict:
    if len(data) > MAX_METADATA_BYTES:
        raise InventoryError(f"metadata file exceeds allowed bounds: {name}")
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InventoryError(f"invalid JSON metadata: {name}") from exc
    if not isinstance(value, dict):
        raise InventoryError(f"metadata must be a JSON object: {name}")
    return value


def _positive_int(value, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise InventoryError(f"invalid config field: {name}")
    return value


def _model_config(config: dict) -> tuple[dict, dict | None, list[str]]:
    if config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise InventoryError("unsupported model architecture")
    if config.get("tie_word_embeddings") is not False:
        raise InventoryError("tied or unspecified embeddings require separate accounting")
    text = config.get("text_config")
    vision = config.get("vision_config")
    if not isinstance(text, dict) or (vision is not None and not isinstance(vision, dict)):
        raise InventoryError("missing or unsupported text/vision config")
    layers = _positive_int(text.get("num_hidden_layers"), "num_hidden_layers")
    layer_types = text.get("layer_types")
    if (
        not isinstance(layer_types, list)
        or len(layer_types) != layers
        or any(layer not in ("full_attention", "linear_attention") for layer in layer_types)
    ):
        raise InventoryError("missing or unsupported layer types")
    for key in ("hidden_size", "vocab_size", "intermediate_size"):
        _positive_int(text.get(key), key)
    if vision is not None:
        _positive_int(vision.get("depth"), "vision depth")
    return text, vision, layer_types


def _classify(name: str, layer_types: list[str]) -> tuple[str, str]:
    if name == "model.language_model.embed_tokens.weight":
        return "input_embedding", "vocabulary_matrix_retained"
    if name == "lm_head.weight":
        return "output_head", "vocabulary_matrix_retained"
    if name.startswith("model.visual."):
        return "vision", "vision_requires_separate_validation"
    if re.search(r"(^|\.)(mtp|mtp_layers)(\.|$)", name):
        return "optional_mtp", "optional_mtp_retained"
    if name == "model.language_model.norm.weight":
        return "sensitive", "normalization_retained"

    match = LAYER_NAME.fullmatch(name)
    if match is None or int(match[1]) >= len(layer_types):
        raise InventoryError(f"unsupported tensor path: {name}")
    suffix = match[2]
    layer_type = layer_types[int(match[1])]
    if suffix in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        return "sensitive", "normalization_retained"
    if suffix in ("mlp.down_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight"):
        return "language_projection", "candidate"
    if layer_type == "full_attention":
        if suffix in (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
        ):
            return "language_projection", "candidate"
        if suffix in ("self_attn.q_norm.weight", "self_attn.k_norm.weight"):
            return "sensitive", "normalization_retained"
    if layer_type == "linear_attention":
        if suffix in (
            "linear_attn.in_proj_a.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.out_proj.weight",
        ):
            return "language_projection", "recurrent_or_fused_projection_needs_validation"
        if suffix in (
            "linear_attn.A_log",
            "linear_attn.dt_bias",
            "linear_attn.conv1d.weight",
            "linear_attn.norm.weight",
        ):
            return "sensitive", "recurrent_or_normalization_retained"
    raise InventoryError(f"unsupported tensor path: {name}")


def _policy(tensor: TensorHeader, layer_types: list[str]) -> tuple[str, bool, str]:
    category, reason = _classify(tensor.name, layer_types)
    if reason != "candidate":
        return category, False, reason
    if len(tensor.shape) != 2 or tensor.dtype not in ("BF16", "F16", "F32"):
        return category, False, "candidate_requires_float_matrix"
    if tensor.shape[1] % GROUP_SIZE:
        return category, False, "input_width_not_divisible_by_group_128"
    if tensor.shape[1] % ROTATION_SIZE:
        return category, False, "input_width_not_divisible_by_rotation_1024"
    return category, True, "initial_group_128_rotation_1024_candidate"


def _required_weights(text: dict, vision: dict | None, layer_types: list[str]) -> set[str]:
    required = {
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    }
    for number, layer_type in enumerate(layer_types):
        base = f"model.language_model.layers.{number}."
        required.update(
            base + suffix
            for suffix in (
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "mlp.down_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
            )
        )
        if layer_type == "full_attention":
            required.update(
                base + "self_attn." + suffix
                for suffix in (
                    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                    "q_norm.weight", "k_norm.weight",
                )
            )
        else:
            required.update(
                base + "linear_attn." + suffix
                for suffix in (
                    "in_proj_a.weight", "in_proj_b.weight", "in_proj_qkv.weight",
                    "in_proj_z.weight", "out_proj.weight", "A_log", "dt_bias",
                    "conv1d.weight", "norm.weight",
                )
            )
    if vision is not None:
        required.update(
            (
                "model.visual.patch_embed.proj.weight",
                "model.visual.patch_embed.proj.bias",
                "model.visual.pos_embed.weight",
            )
        )
        required.update(
            f"model.visual.merger.{module}.{parameter}"
            for module in ("norm", "linear_fc1", "linear_fc2")
            for parameter in ("weight", "bias")
        )
        for number in range(vision["depth"]):
            required.update(
                f"model.visual.blocks.{number}.{module}.{parameter}"
                for module in (
                    "attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2",
                    "norm1", "norm2",
                )
                for parameter in ("weight", "bias")
            )
    return required


def _q8_bytes(tensor: TensorHeader) -> int:
    if len(tensor.shape) != 2 or tensor.shape[1] % 32:
        raise InventoryError(f"Q8_0 requires 32-wide input blocks: {tensor.name}")
    return tensor.parameters // 32 * 34


def _open_bounded(name: str, *, start: int | None = None, length: int = 0) -> tuple[bytes, int | None]:
    headers = {"Accept-Encoding": "identity"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{start + length - 1}"
    request = Request(MODEL_URL + name, headers=headers)
    try:
        with urlopen(request, timeout=20) as response:
            if urlsplit(response.geturl()).scheme != "https":
                raise InventoryError(f"non-HTTPS response: {name}")
            expected_status = 206 if start is not None else 200
            if response.status != expected_status:
                raise InventoryError(f"server ignored bounded range or returned unexpected status: {name}")
            if response.headers.get("Content-Encoding", "identity") != "identity":
                raise InventoryError(f"encoded HTTP response not supported: {name}")
            content_length = response.headers.get("Content-Length")
            if start is None:
                limit = MAX_METADATA_BYTES
                total = None
            else:
                limit = length
                match = CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
                if match is None:
                    raise InventoryError(f"missing or invalid Content-Range: {name}")
                actual_start, actual_end, total = map(int, match.groups())
                if (actual_start, actual_end) != (start, start + length - 1) or total <= actual_end:
                    raise InventoryError(f"unexpected byte range: {name}")
            if content_length is not None and (
                not content_length.isdecimal() or int(content_length) > limit
            ):
                raise InventoryError(f"HTTP response exceeds allowed bounds: {name}")
            data = response.read(limit + 1)
            if len(data) > limit or (content_length is not None and len(data) != int(content_length)):
                raise InventoryError(f"HTTP response exceeds allowed bounds: {name}")
            if start is not None and len(data) != length:
                raise InventoryError(f"short safetensors range response: {name}")
            return data, total
    except (HTTPError, URLError) as exc:
        raise InventoryError(f"unable to retrieve pinned model file: {name}: {exc}") from exc


def fetch_pinned_headers() -> tuple[dict[str, bytes], dict[str, tuple[bytes, int]]]:
    """Fetch only small pinned metadata files and exact safetensors header ranges."""
    metadata = {name: _open_bounded(name)[0] for name in REQUIRED_METADATA}
    for name in OPTIONAL_METADATA:
        try:
            metadata[name] = _open_bounded(name)[0]
        except InventoryError as exc:
            if not isinstance(exc.__cause__, HTTPError) or exc.__cause__.code != 404:
                raise
    index = _json_object(metadata["model.safetensors.index.json"], "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map or any(
        not isinstance(shard, str) or SHARD_NAME.fullmatch(shard) is None
        for shard in weight_map.values()
    ):
        raise InventoryError("unsupported safetensors shard names in index")
    shards = set(weight_map.values())
    if len(shards) > MAX_SHARDS:
        raise InventoryError("too many indexed safetensors shards")
    headers = {}
    for shard in sorted(shards):
        length_prefix, file_bytes = _open_bounded(shard, start=0, length=8)
        header_bytes = int.from_bytes(length_prefix, "little")
        if header_bytes == 0 or header_bytes > MAX_METADATA_BYTES:
            raise InventoryError(f"safetensors header exceeds allowed bounds: {shard}")
        header, second_file_bytes = _open_bounded(shard, start=8, length=header_bytes)
        if second_file_bytes != file_bytes:
            raise InventoryError(f"shard size changed between range requests: {shard}")
        headers[shard] = (length_prefix + header, file_bytes)
    return metadata, headers


def build_inventory(
    metadata_files: Mapping[str, bytes], shard_headers: Mapping[str, tuple[bytes, int]]
) -> dict:
    """Reconcile pinned index, complete shard headers, and explicit policy offline."""
    missing = set(REQUIRED_METADATA) - metadata_files.keys()
    if missing:
        raise InventoryError(f"missing model metadata files: {sorted(missing)}")
    if set(metadata_files) - set(REQUIRED_METADATA) - set(OPTIONAL_METADATA):
        raise InventoryError("unsupported model metadata file")
    parsed = {}
    for name, data in sorted(metadata_files.items()):
        if len(data) > MAX_METADATA_BYTES:
            raise InventoryError(f"metadata file exceeds allowed bounds: {name}")
        if name.endswith(".json"):
            parsed[name] = _json_object(data, name)
        else:
            try:
                data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise InventoryError(f"invalid chat template: {name}") from exc

    config = parsed["config.json"]
    text, vision, layer_types = _model_config(config)
    tokenizer_config = parsed["tokenizer_config.json"]
    processor_config = parsed["processor_config.json"]
    image_config = parsed["preprocessor_config.json"]
    video_config = parsed.get("video_preprocessor_config.json")
    if (
        tokenizer_config.get("tokenizer_class") != "Qwen2Tokenizer"
        or processor_config.get("processor_class") != "Qwen3VLProcessor"
        or image_config.get("image_processor_type") != "Qwen2VLImageProcessor"
        or (video_config is not None and video_config.get("video_processor_type") != "Qwen3VLVideoProcessor")
        or not metadata_files["chat_template.jinja"].strip()
    ):
        raise InventoryError("missing or unsupported tokenizer/processor metadata")
    index = parsed["model.safetensors.index.json"]
    if set(index) != {"metadata", "weight_map"}:
        raise InventoryError("unsupported safetensors index")
    weight_map = index["weight_map"]
    index_metadata = index["metadata"]
    if (
        not isinstance(weight_map, dict)
        or not weight_map
        or not isinstance(index_metadata, dict)
        or type(index_metadata.get("total_size")) is not int
        or index_metadata["total_size"] < 0
    ):
        raise InventoryError("invalid safetensors index")
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(shard, str)
        or SHARD_NAME.fullmatch(shard) is None
        for name, shard in weight_map.items()
    ):
        raise InventoryError("unsupported tensor or shard name in index")
    shards = set(weight_map.values())
    if len(shards) > MAX_SHARDS or shards != set(shard_headers):
        raise InventoryError("missing, excess, or too many safetensors shard headers")
    shard_numbers = {int(SHARD_NAME.fullmatch(shard)[1]) for shard in shards}
    shard_totals = {int(SHARD_NAME.fullmatch(shard)[2]) for shard in shards}
    if shard_totals != {len(shards)} or shard_numbers != set(range(1, len(shards) + 1)):
        raise InventoryError("incomplete or inconsistent safetensors shard numbering")

    headers_by_name = {}
    shard_sources = []
    shard_size_sum = 0
    for shard in sorted(shards):
        prefix, file_bytes = shard_headers[shard]
        headers = parse_safetensors_header(prefix, file_bytes)
        for tensor in headers:
            if tensor.name in headers_by_name:
                raise InventoryError(f"duplicate tensor in shards: {tensor.name}")
            headers_by_name[tensor.name] = (tensor, shard)
        shard_size_sum += file_bytes
        shard_sources.append(
            {
                "name": shard,
                "file_bytes": file_bytes,
                "header_bytes": len(prefix),
                "header_sha256": hashlib.sha256(prefix).hexdigest(),
                "full_weight_hash": "not_checked_header_only",
            }
        )
    if set(headers_by_name) != set(weight_map):
        raise InventoryError(
            f"index/header tensor mismatch: missing={sorted(set(weight_map) - set(headers_by_name))[:5]}, "
            f"unindexed={sorted(set(headers_by_name) - set(weight_map))[:5]}"
        )
    for name, (_, shard) in headers_by_name.items():
        if shard != weight_map[name]:
            raise InventoryError(f"index/header shard mismatch: {name}")
    required = _required_weights(text, vision, layer_types)
    if missing := required - headers_by_name.keys():
        raise InventoryError(f"missing required model weights: {sorted(missing)[:5]}")
    embedding = headers_by_name["model.language_model.embed_tokens.weight"][0]
    head = headers_by_name["lm_head.weight"][0]
    expected_vocab_shape = (text["vocab_size"], text["hidden_size"])
    if embedding.shape != expected_vocab_shape or head.shape != expected_vocab_shape:
        raise InventoryError("vocabulary weight shapes disagree with config")

    tensors = []
    categories = {}
    retained_by_dtype = {}
    eligible_parameters = 0
    eligible_bytes = 0
    rotation_sign_upper_bound_bytes = 0
    ptq1_bytes = 0
    pq2_bytes = 0
    for name, (tensor, shard) in sorted(headers_by_name.items()):
        category, eligible, reason = _policy(tensor, layer_types)
        tensors.append(
            {
                "name": name,
                "shard": shard,
                "shape": tensor.shape,
                "dtype": tensor.dtype,
                "parameters": tensor.parameters,
                "storage_bytes": tensor.storage_bytes,
                "category": category,
                "quantization_eligible": eligible,
                "policy_reason": reason,
            }
        )
        counter = categories.setdefault(category, {"parameters": 0, "storage_bytes": 0})
        counter["parameters"] += tensor.parameters
        counter["storage_bytes"] += tensor.storage_bytes
        if eligible:
            eligible_parameters += tensor.parameters
            eligible_bytes += tensor.storage_bytes
            rotation_sign_upper_bound_bytes += tensor.shape[1]
            ptq1_bytes += tensor.parameters // GROUP_SIZE * 28
            pq2_bytes += tensor.parameters // GROUP_SIZE * 34
        else:
            retained_by_dtype[tensor.dtype] = (
                retained_by_dtype.get(tensor.dtype, 0) + tensor.storage_bytes
            )

    weight_bytes = sum(tensor["storage_bytes"] for tensor in tensors)
    if weight_bytes != index_metadata["total_size"]:
        raise InventoryError(
            f"index total_size {index_metadata['total_size']} != header tensor bytes {weight_bytes}"
        )
    q8_embedding = _q8_bytes(embedding)
    q8_head = _q8_bytes(head)
    ternary_ptq1 = weight_bytes - eligible_bytes + ptq1_bytes
    ternary_pq2 = weight_bytes - eligible_bytes + pq2_bytes
    if text["vocab_size"] < 16 or head.dtype not in ("BF16", "F16", "F32"):
        raise InventoryError("selected-label head requires at least 16 float vocabulary rows")
    selected_head_bytes = 16 * text["hidden_size"] * DTYPE_BYTES[head.dtype]
    full_layers = layer_types.count("full_attention")
    linear_layers = layer_types.count("linear_attention")
    kv_bytes_per_token = (
        full_layers * 2 * _positive_int(text.get("num_key_value_heads"), "num_key_value_heads")
        * _positive_int(text.get("head_dim"), "head_dim") * 2
    )
    recurrent_bytes = 0
    if linear_layers:
        for key in ("linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim"):
            _positive_int(text.get(key), key)
        if text.get("mamba_ssm_dtype") != "float32":
            raise InventoryError("unsupported recurrent-state dtype for cache estimate")
        recurrent_bytes = (
            linear_layers * text["linear_num_value_heads"] * text["linear_key_head_dim"]
            * text["linear_value_head_dim"] * 4
        )

    return {
        "schema_version": 1,
        "metadata_summary": {
            "architecture": config["architectures"][0],
            "tokenizer_class": tokenizer_config["tokenizer_class"],
            "processor_class": processor_config["processor_class"],
            "image_processor_type": image_config["image_processor_type"],
            "video_processor_type": (
                video_config["video_processor_type"] if video_config is not None else None
            ),
            "chat_template_file": "chat_template.jinja",
        },
        "source": {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "metadata_files": {
                name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                for name, data in sorted(metadata_files.items())
            },
            "shards": shard_sources,
            "transfer_body_bytes": sum(map(len, metadata_files.values()))
            + sum(len(prefix) for prefix, _ in shard_headers.values()),
            "tool": {"name": "embedded_jev.inventory", "version": 1, "python": sys.version.split()[0]},
        },
        "accounting": {
            "index_total_size": index_metadata["total_size"],
            "header_tensor_bytes": weight_bytes,
            "shard_file_bytes": shard_size_sum,
            "header_overhead_bytes": shard_size_sum - weight_bytes,
            "tied_embeddings_declared": False,
            "in_shard_shared_offsets": False,
            "cross_shard_shared_payloads": "not_checkable_from_headers",
            "optional_mtp_configured_layers": text.get("mtp_num_hidden_layers", 0),
            "optional_mtp_tensors_present": "optional_mtp" in categories,
        },
        "tensors": tensors,
        "totals": {
            "tensors": len(tensors),
            "parameters": sum(tensor["parameters"] for tensor in tensors),
            "stored_weight_bytes": weight_bytes,
            "by_category": categories,
            "eligible_projection_parameters": eligible_parameters,
            "eligible_projection_storage_bytes": eligible_bytes,
            "retained_bytes_by_dtype": retained_by_dtype,
        },
        "memory_estimates": {
            "weights_only_bytes": {
                "original": weight_bytes,
                "ptq1_group128_bf16_vocab": ternary_ptq1,
                "pq2_group128_bf16_vocab": ternary_pq2,
                "ptq1_group128_q8_vocab": (
                    ternary_ptq1 - embedding.storage_bytes - head.storage_bytes
                    + q8_embedding + q8_head
                ),
                "pq2_group128_q8_vocab": (
                    ternary_pq2 - embedding.storage_bytes - head.storage_bytes
                    + q8_embedding + q8_head
                ),
                "ptq1_group128_q8_input_selected16_head": (
                    ternary_ptq1 - embedding.storage_bytes - head.storage_bytes
                    + q8_embedding + selected_head_bytes
                ),
            },
            "projection_blocks_bytes": {"PTQ1_0": ptq1_bytes, "PQ2_0": pq2_bytes},
            "vocabulary_bytes": {
                "input_original": embedding.storage_bytes,
                "head_original": head.storage_bytes,
                "input_Q8_0": q8_embedding,
                "head_Q8_0": q8_head,
                "selected_16_original_dtype": selected_head_bytes,
            },
            "vision_original_bytes": categories.get("vision", {}).get("storage_bytes", 0),
            "optional_mtp_original_bytes": categories.get("optional_mtp", {}).get("storage_bytes", 0),
            "rotation_sign_upper_bound_bytes_excluded_from_weight_totals": rotation_sign_upper_bound_bytes,
            "runtime_cache_example_not_in_weight_totals": {
                "context_tokens": 4096,
                "sequences": 1,
                "full_attention_fp16_kv_bytes": kv_bytes_per_token * 4096,
                "linear_attention_fp32_recurrent_bytes": recurrent_bytes,
                "convolution_state_and_scratch_bytes": "unknown_until_native_measurement",
            },
        },
        "assumptions": [
            "Exact header counts and index totals are verified; quantized bytes are format estimates, not artifacts.",
            "Only named FFN and full-attention projections with group-128 and rotation-1024 widths are eligible.",
            "PTQ1_0 estimates 28 bytes/128 values; PQ2_0 34 bytes/128; Q8_0 34 bytes/32.",
            "Q8 vocabulary conversion and selected-16 head require validated runtime converter and custom loader respectively.",
            "Transform-sign bound assumes one byte per input feature per eligible tensor; metadata overhead remains unknown.",
            "Tokenizer and processor identities are metadata checks, not template rendering or tokenization parity.",
            "Cache example excludes convolution state, scratch, allocator overhead, and snapshot duplication.",
            "Header inspection cannot verify identical data stored in different shards.",
        ],
    }


def main() -> None:
    metadata, headers = fetch_pinned_headers()
    print(json.dumps(build_inventory(metadata, headers), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()