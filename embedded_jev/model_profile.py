"""Strict, immutable source records for offline Qwen3.5 profile onboarding."""

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping


MAX_PROFILE_BYTES = 1 << 20
MAX_PROFILE_FILE_BYTES = 32 << 20
MIMO_PROFILE_PATH = Path(__file__).with_name("profiles") / "mimo.json"
DEFIANT_FABLE_PROFILE_PATH = Path(__file__).with_name("profiles") / "defiant-fable.json"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REVISION = re.compile(r"[0-9a-f]{40}\Z")
PROFILE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
GEOMETRY_INTS = (
    "hidden_size", "intermediate_size", "vocab_size", "num_hidden_layers",
    "full_attention_interval", "num_attention_heads", "num_key_value_heads",
    "head_dim", "linear_num_key_heads", "linear_num_value_heads",
    "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim",
)
TENSOR_CLASSES = (
    "input_embedding", "output_head", "language_projection", "sensitive",
    "vision", "optional_mtp",
)


class ProfileError(ValueError):
    """An invalid record or a source that does not match its declared profile."""


@dataclass(frozen=True)
class ProfileFile:
    bytes: int
    sha256: str
    header_bytes: int | None = None
    header_sha256: str | None = None


@dataclass(frozen=True)
class ModelProfile:
    profile_id: str
    model: str
    revision: str
    license: str
    files: Mapping[str, ProfileFile]
    packaging: Mapping
    geometry: Mapping
    tokenization: Mapping
    runtime: Mapping
    quantization: Mapping
    evidence: tuple[str, ...]
    sha256: str


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProfileError(f"duplicate profile JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ProfileError(f"invalid profile JSON constant: {value}")


def _fields(value, names, where):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ProfileError(f"missing or unknown profile fields: {where}")
    return value


def _string(value, where):
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ProfileError(f"invalid profile string: {where}")
    return value


def _integer(value, where, *, minimum=1):
    if type(value) is not int or not minimum <= value < 1 << 63:
        raise ProfileError(f"invalid profile integer: {where}")
    return value


def _path(value):
    _string(value, "file path")
    path = PurePosixPath(value)
    if (
        path.is_absolute() or "\\" in value or any(part in ("", ".", "..") for part in value.split("/"))
        or any(ord(character) < 32 for character in value)
    ):
        raise ProfileError(f"unsafe profile file path: {value}")
    return value


def _strings(value, where, *, allow_empty=False):
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ProfileError(f"invalid profile list: {where}")
    for item in value:
        _string(item, where)
    if len(set(value)) != len(value):
        raise ProfileError(f"duplicate profile list entry: {where}")
    return value


def _hash(value, where):
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ProfileError(f"invalid profile SHA-256: {where}")
    return value


def _positive_number(value, where):
    if (
        type(value) not in (int, float) or value <= 0
        or (type(value) is int and value >= 1 << 63) or not math.isfinite(value)
    ):
        raise ProfileError(f"invalid profile number: {where}")
    return value


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def parse_model_profile(data: bytes) -> ModelProfile:
    """Parse a bounded v1 record without accessing model files or running code."""
    if not isinstance(data, bytes) or not data or len(data) > MAX_PROFILE_BYTES:
        raise ProfileError("profile record exceeds allowed byte bounds")
    try:
        record = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_keys, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ProfileError("invalid profile JSON") from exc
    _fields(record, (
        "schema_version", "profile_id", "source", "files", "packaging", "geometry",
        "tokenization", "runtime", "quantization", "evidence",
    ), "record")
    if type(record["schema_version"]) is not int or record["schema_version"] != 1:
        raise ProfileError("unsupported profile schema version")
    profile_id = _string(record["profile_id"], "profile_id")
    if PROFILE_ID.fullmatch(profile_id) is None:
        raise ProfileError("invalid profile ID")
    source = _fields(record["source"], ("model", "revision", "license", "format", "support_tier"), "source")
    for name in ("model", "revision", "license"):
        _string(source[name], f"source.{name}")
    if REVISION.fullmatch(source["revision"]) is None:
        raise ProfileError("profile revision must be a pinned Git commit")
    if source["format"] != "safetensors" or source["support_tier"] != "A":
        raise ProfileError("unsupported profile source format or support tier")
    if not isinstance(record["files"], dict) or not record["files"]:
        raise ProfileError("missing profile file identities")
    files = {}
    for name, spec in record["files"].items():
        _path(name)
        has_header = isinstance(spec, dict) and bool({"header_bytes", "header_sha256"} & spec.keys())
        _fields(spec, ("bytes", "sha256", "header_bytes", "header_sha256") if has_header else ("bytes", "sha256"), name)
        size = _integer(spec["bytes"], name)
        digest = _hash(spec["sha256"], name)
        header_bytes = header_sha256 = None
        if has_header:
            header_bytes = _integer(spec["header_bytes"], f"{name}.header_bytes", minimum=9)
            if header_bytes > min(size, (16 << 20) + 8):
                raise ProfileError(f"profile header exceeds allowed bounds: {name}")
            header_sha256 = _hash(spec["header_sha256"], f"{name}.header_sha256")
        files[name] = ProfileFile(size, digest, header_bytes, header_sha256)

    packaging = _fields(record["packaging"], (
        "architecture", "text_config_path", "text_prefix", "vision_prefix", "output_head",
        "shards", "metadata_files", "mtp_prefix", "mtp_status", "mtp_handling",
        "index_total_policy", "text_tensor_count", "dtype_policy", "metadata_classes",
    ), "packaging")
    architecture = packaging["architecture"]
    if architecture not in ("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM"):
        raise ProfileError("unsupported profile architecture")
    expected_path = ["text_config"] if architecture == "Qwen3_5ForConditionalGeneration" else []
    if packaging["text_config_path"] != expected_path:
        raise ProfileError("unsupported profile text configuration path")
    for name in ("text_prefix", "mtp_prefix"):
        if not _string(packaging[name], name).endswith("."):
            raise ProfileError(f"profile prefix must end with a dot: {name}")
    if packaging["vision_prefix"] is not None and not _string(packaging["vision_prefix"], "vision_prefix").endswith("."):
        raise ProfileError("profile vision prefix must end with a dot")
    _string(packaging["output_head"], "output_head")
    shards = _strings(packaging["shards"], "shards")
    metadata = _strings(packaging["metadata_files"], "metadata_files")
    if len(shards) > 16 or set(shards) != {name for name, spec in files.items() if spec.header_bytes is not None}:
        raise ProfileError("profile shard list disagrees with file identities")
    if not {"config.json", "model.safetensors.index.json", "tokenizer_config.json"} <= set(metadata):
        raise ProfileError("missing required profile metadata declarations")
    if set(shards) & set(metadata) or not set(metadata) <= files.keys():
        raise ProfileError("profile metadata list disagrees with file identities")
    if any(not name.endswith(".safetensors") for name in shards):
        raise ProfileError("unsupported profile shard extension")
    if packaging["mtp_status"] not in ("absent", "complete", "incomplete") or packaging["mtp_handling"] != "exclude":
        raise ProfileError("unsupported profile MTP policy")
    if packaging["index_total_policy"] not in ("exact", "recomputed"):
        raise ProfileError("unsupported profile index total policy")
    _integer(packaging["text_tensor_count"], "text_tensor_count")
    classes = _fields(packaging["metadata_classes"], ("tokenizer", "processor", "image_processor", "video_processor"), "metadata_classes")
    supported_classes = {
        "tokenizer": ("Qwen2Tokenizer", "TokenizersBackend"),
        "processor": (None, "Qwen3VLProcessor"),
        "image_processor": (None, "Qwen2VLImageProcessor", "Qwen2VLImageProcessorFast"),
        "video_processor": (None, "Qwen3VLVideoProcessor"),
    }
    for name, declared in classes.items():
        if declared not in supported_classes[name]:
            raise ProfileError(f"unsupported profile metadata class: {name}")
    dtypes = _fields(packaging["dtype_policy"], TENSOR_CLASSES, "dtype_policy")
    for category, allowed in dtypes.items():
        if not set(_strings(allowed, category)) <= {"BF16", "F16", "F32"}:
            raise ProfileError(f"unsupported profile tensor dtype: {category}")

    geometry = _fields(record["geometry"], (*GEOMETRY_INTS, "layer_types", "mamba_ssm_dtype", "partial_rotary_factor", "rms_norm_eps", "rope_parameters", "tie_word_embeddings", "attn_output_gate"), "geometry")
    for name in GEOMETRY_INTS:
        _integer(geometry[name], name)
    layers = geometry["layer_types"]
    if (
        not isinstance(layers, list) or len(layers) != geometry["num_hidden_layers"] or len(layers) > 256
        or any(layer not in ("linear_attention", "full_attention") for layer in layers)
        or layers != ["full_attention" if (number + 1) % geometry["full_attention_interval"] == 0 else "linear_attention" for number in range(len(layers))]
    ):
        raise ProfileError("profile layer types disagree with the regular attention interval")
    if geometry["tie_word_embeddings"] is not False or type(geometry["attn_output_gate"]) is not bool or geometry["mamba_ssm_dtype"] != "float32":
        raise ProfileError("unsupported profile embedding, gate or recurrent-state policy")
    if geometry["num_attention_heads"] % geometry["num_key_value_heads"] or geometry["linear_num_value_heads"] % geometry["linear_num_key_heads"]:
        raise ProfileError("profile head counts are incompatible")
    _positive_number(geometry["rms_norm_eps"], "rms_norm_eps")
    factor = _positive_number(geometry["partial_rotary_factor"], "partial_rotary_factor")
    rope = _fields(geometry["rope_parameters"], ("mrope_interleaved", "mrope_section", "partial_rotary_factor", "rope_theta", "rope_type"), "rope_parameters")
    _positive_number(rope["partial_rotary_factor"], "rope_parameters.partial_rotary_factor")
    if factor > 1 or rope["partial_rotary_factor"] != factor or rope["mrope_interleaved"] is not True or rope["rope_type"] != "default":
        raise ProfileError("unsupported profile rotary policy")
    _positive_number(rope["rope_theta"], "rope_theta")
    if not isinstance(rope["mrope_section"], list) or len(rope["mrope_section"]) != 3:
        raise ProfileError("invalid profile MRoPE sections")
    for section in rope["mrope_section"]:
        _integer(section, "mrope_section", minimum=0)
    if sum(rope["mrope_section"]) != geometry["head_dim"] * factor / 2:
        raise ProfileError("profile MRoPE sections disagree with rotary width")

    tokenization = _fields(record["tokenization"], (
        "template_file", "gguf_template_file", "render_arguments", "non_thinking_suffix",
        "control_markers", "special_token_ids", "label_token_ids", "prompt_formatter_version",
    ), "tokenization")
    _path(tokenization["template_file"])
    if tokenization["template_file"] not in files or tokenization["template_file"] not in metadata or "tokenizer.json" not in files:
        raise ProfileError("missing profile tokenizer or template identity")
    if tokenization["gguf_template_file"] is not None:
        _path(tokenization["gguf_template_file"])
        if tokenization["gguf_template_file"] not in files:
            raise ProfileError("missing profile GGUF template identity")
    arguments = tokenization["render_arguments"]
    if not isinstance(arguments, dict) or set(arguments) not in ({"add_generation_prompt", "enable_thinking"}, {"add_generation_prompt", "enable_thinking", "reasoning_effort"}):
        raise ProfileError("unknown or missing profile rendering arguments")
    if arguments["add_generation_prompt"] is not True or arguments["enable_thinking"] is not False:
        raise ProfileError("profile must declare prompt-only non-thinking rendering")
    if "reasoning_effort" in arguments:
        _string(arguments["reasoning_effort"], "reasoning_effort")
    _string(tokenization["non_thinking_suffix"], "non_thinking_suffix")
    _strings(tokenization["control_markers"], "control_markers", allow_empty=True)
    _integer(tokenization["prompt_formatter_version"], "prompt_formatter_version")
    special_ids = tokenization["special_token_ids"]
    label_ids = tokenization["label_token_ids"]
    if not isinstance(special_ids, dict) or not special_ids or not isinstance(label_ids, dict) or set(label_ids) != set("ABCDEFGHIJKLMNOP"):
        raise ProfileError("missing profile special tokens or A-P labels")
    for token_ids in (special_ids, label_ids):
        for name, token_id in token_ids.items():
            _string(name, "token name")
            if _integer(token_id, "token ID", minimum=0) >= geometry["vocab_size"]:
                raise ProfileError("profile token ID exceeds vocabulary")
    if len(set(label_ids.values())) != 16 or set(label_ids.values()) & set(special_ids.values()):
        raise ProfileError("profile label IDs must be distinct non-special tokens")

    runtime = _fields(record["runtime"], ("architecture_adapter", "converter_revision", "runtime_revision", "converter_flags", "native_policy"), "runtime")
    if runtime["architecture_adapter"] != "qwen35-safetensors-v1":
        raise ProfileError("unsupported profile architecture adapter")
    for name in ("converter_revision", "runtime_revision"):
        if not isinstance(runtime[name], str) or REVISION.fullmatch(runtime[name]) is None:
            raise ProfileError(f"invalid profile runtime revision: {name}")
    if "--no-nextn" not in _strings(runtime["converter_flags"], "converter_flags"):
        raise ProfileError("profile converter must explicitly exclude MTP")
    if runtime["native_policy"] is not None:
        _string(runtime["native_policy"], "native_policy")
    quantization = _fields(record["quantization"], ("group_size", "scale_dtype", "activation_contract", "eligible_suffixes"), "quantization")
    if type(quantization["group_size"]) is not int or quantization["group_size"] != 128 or quantization["scale_dtype"] != "F16" or quantization["activation_contract"] != "group128-dynamic-a8-v1":
        raise ProfileError("unsupported profile quantization contract")
    _strings(quantization["eligible_suffixes"], "eligible_suffixes")
    evidence = _strings(record["evidence"], "evidence", allow_empty=True)
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return ModelProfile(
        profile_id, source["model"], source["revision"], source["license"], MappingProxyType(files),
        _freeze(packaging), _freeze(geometry), _freeze(tokenization), _freeze(runtime),
        _freeze(quantization), tuple(evidence), hashlib.sha256(canonical).hexdigest(),
    )


def load_model_profile(path: Path) -> ModelProfile:
    """Read one bounded local record; never discover or download model files."""
    try:
        with path.open("rb") as source:
            return parse_model_profile(source.read(MAX_PROFILE_BYTES + 1))
    except OSError as exc:
        raise ProfileError(f"unable to read model profile: {exc}") from exc


def mimo_profile() -> ModelProfile:
    """Return the checked-in record of the previously verified MiMo snapshot."""
    return load_model_profile(MIMO_PROFILE_PATH)


def defiant_fable_profile() -> ModelProfile:
    """Return the selected metadata-only profile, not runtime compatibility."""
    return load_model_profile(DEFIANT_FABLE_PROFILE_PATH)


def profile_text_config(profile: ModelProfile, config: dict) -> dict:
    """Read a declared static configuration path and verify its geometry."""
    text = config
    for name in profile.packaging["text_config_path"]:
        text = text.get(name) if isinstance(text, dict) else None
    if not isinstance(text, dict):
        raise ProfileError("missing profile text configuration")
    for name, expected in profile.geometry.items():
        actual = text.get(name)
        if _freeze(actual) != expected or (type(expected) is int and type(actual) is not int):
            raise ProfileError(f"configuration geometry does not match the model profile: {name}")
    return text


def verify_profile_headers(
    profile: ModelProfile, metadata_files: Mapping[str, bytes],
    shard_headers: Mapping[str, tuple[bytes, int]],
) -> None:
    """Verify cheap source identity; full weight payloads are not authenticated."""
    if set(metadata_files) != set(profile.packaging["metadata_files"]):
        raise ProfileError("metadata file set does not match the model profile")
    if set(shard_headers) != set(profile.packaging["shards"]):
        raise ProfileError("shard file set does not match the model profile")
    for name, data in metadata_files.items():
        expected = profile.files[name]
        if len(data) != expected.bytes or hashlib.sha256(data).hexdigest() != expected.sha256:
            raise ProfileError(f"metadata identity does not match the model profile: {name}")
    for name, (prefix, size) in shard_headers.items():
        expected = profile.files[name]
        if (
            size != expected.bytes or len(prefix) != expected.header_bytes
            or hashlib.sha256(prefix).hexdigest() != expected.header_sha256
        ):
            raise ProfileError(f"shard header identity does not match the model profile: {name}")


def verify_profile_files(profile: ModelProfile, directory: Path, names=None) -> None:
    """Hash bounded nonweight files only; never read a complete weight shard."""
    names = tuple(names) if names is not None else tuple(
        name for name, spec in profile.files.items() if spec.header_bytes is None
    )
    if not set(names) <= profile.files.keys():
        raise ProfileError("file is not declared by the model profile")
    for name in names:
        expected = profile.files[name]
        if expected.header_bytes is not None or expected.bytes > MAX_PROFILE_FILE_BYTES:
            raise ProfileError(f"file exceeds bounded nonweight identity check: {name}")
        try:
            path = directory / name
            if path.stat().st_size != expected.bytes:
                raise ProfileError(f"file size does not match the model profile: {name}")
            digest = hashlib.sha256()
            count = 0
            with path.open("rb") as source:
                while chunk := source.read(min(1 << 16, expected.bytes - count + 1)):
                    count += len(chunk)
                    if count > expected.bytes:
                        raise ProfileError(f"file grew beyond the model profile size: {name}")
                    digest.update(chunk)
            if count != expected.bytes or digest.hexdigest() != expected.sha256:
                raise ProfileError(f"file identity does not match the model profile: {name}")
        except OSError as exc:
            raise ProfileError(f"unable to verify model profile file: {name}: {exc}") from exc


def match_model_profile(
    metadata_files: Mapping[str, bytes], shard_headers: Mapping[str, tuple[bytes, int]],
    *, profile: ModelProfile | None = None,
) -> ModelProfile | None:
    """Identify verified headers, never infer identity from a directory name."""
    if profile is not None:
        verify_profile_headers(profile, metadata_files, shard_headers)
        return profile
    for candidate in (mimo_profile(),):
        try:
            verify_profile_headers(candidate, metadata_files, shard_headers)
        except ProfileError:
            continue
        return candidate
    return None


def require_model_profile(
    metadata_files: Mapping[str, bytes], shard_headers: Mapping[str, tuple[bytes, int]],
    *, profile: ModelProfile | None = None,
) -> ModelProfile:
    """Refuse unknown sources before a local model or payload reader can run."""
    identified = match_model_profile(metadata_files, shard_headers, profile=profile)
    if identified is None:
        raise ProfileError("no verified model profile matches the local files")
    return identified


def profile_source(profile: ModelProfile | None) -> dict:
    """Report verified identity or explicitly unbound analytical header inputs."""
    return {
        "model": profile.model if profile is not None else None,
        "revision": profile.revision if profile is not None else None,
        "profile_id": profile.profile_id if profile is not None else None,
        "profile_sha256": profile.sha256 if profile is not None else None,
        "license": profile.license if profile is not None else None,
        "identity_verification": "metadata_and_shard_headers" if profile is not None else "unbound_header_inputs",
    }