"""Run bounded pinned MiMo text layers without materializing the full model."""

import argparse
import gc
import hashlib
import json
import resource
import weakref
from pathlib import Path

import numpy as np

from embedded_jev.decision_dataset import SPLITS, load_decision_dataset, save_calibration_capture
from embedded_jev.dense_probe import MAX_PREFIX_TOKENS, plan_streamed_text
from embedded_jev.inventory import InventoryError, _json_object, build_inventory, read_local_headers
from embedded_jev.label_probe import (
    LABELS, decision_case_messages, load_decision_fixture, probe_decision_cases,
    probe_label_boundary,
)


MAX_SELECTED_HEAD_BYTES = 256 * 1024
MAX_FULL_HEAD_BYTES = 3 * 1024**3
FULL_HEAD_BATCH_ROWS = 256


def native_backend_dependencies(library: Path, backend: str) -> dict:
    """Record actual dynamically loaded GGML dependencies for the graph bridge."""
    if backend == "direct":
        return {}
    if backend not in ("prism_ggml", "prism_ggml_f32", "prism_ggml_hadamard128", "prism_ggml_registered"):
        raise InventoryError("unsupported native FFN-down backend")
    import ctypes

    try:
        locator = ctypes.CDLL(str(library)).prism_group_scale_library_path
        locator.argtypes = [ctypes.c_int]
        locator.restype = ctypes.c_char_p
        dependencies = {}
        for index, name in enumerate(("ggml_cpu", "ggml_base")):
            raw_path = locator(index)
            if not raw_path:
                raise InventoryError("native graph library did not identify its loaded GGML dependency")
            path = Path(raw_path.decode("utf-8")).resolve()
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            dependencies[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}
        return dependencies
    except (OSError, AttributeError, UnicodeDecodeError) as exc:
        raise InventoryError(f"unable to identify native graph dependencies: {exc}") from exc


def make_native_ffn_down(weight, library: Path, artifact: Path | None = None, *, backend: str = "direct"):
    """Substitute one in-memory group-128 BitNet-derived AVX2 projection."""
    import ctypes

    import torch

    from embedded_jev.activation import quantize_a8_per_group
    from embedded_jev.activation import rotate_signed_hadamard
    from embedded_jev.ternary import (
        pack_group128_codes, quantize_ternary_rtn, unpack_group128_codes,
    )

    rotated_backend = backend == "prism_ggml_hadamard128"
    registered_backend = backend == "prism_ggml_registered"
    signs = None
    original_probe = None
    rotated_probe = None
    if rotated_backend and artifact is not None:
        raise InventoryError("signed rotation cannot reinterpret an identity projection artifact")
    if artifact is None:
        if weight is None or tuple(weight.shape) != (4096, 12288) or weight.dtype != torch.bfloat16:
            raise InventoryError("native FFN-down requires the pinned BF16 projection")
        if rotated_backend:
            signs = np.random.default_rng(773).choice([-1, 1], size=12288).astype(np.float32)
            codes = np.empty((4096, 12288), dtype=np.int8)
            scales = np.empty((4096, 96), dtype=np.float16)
            for first in range(0, 4096, 64):
                values = weight[first:first + 64].detach().float().cpu().numpy()
                transformed = rotate_signed_hadamard(values, signs, 128)
                if first == 0:
                    original_probe = values[:4].copy()
                    rotated_probe = transformed[:4].copy()
                codes[first:first + 64], scales[first:first + 64] = quantize_ternary_rtn(transformed, scale_search=True)
        else:
            codes, scales = quantize_ternary_rtn(weight.detach().float().cpu().numpy(), scale_search=True)
        grouped_codes = codes.reshape(4096, 96, 128)
        packed = np.ascontiguousarray(pack_group128_codes(grouped_codes))
        origin = "in_memory_signed_hadamard_rtn" if rotated_backend else "in_memory_rtn"
    else:
        from embedded_jev.projection_artifact import (
            PINNED_SHARD_SHA256, load_projection_artifact,
        )

        packed, scales, manifest = load_projection_artifact(artifact)
        if manifest["shape"] != [4096, 12288] or manifest["source_shard_sha256"] != PINNED_SHARD_SHA256:
            raise InventoryError("saved projection does not match pinned BF16 source")
        grouped_codes = unpack_group128_codes(packed)
        packed = np.ascontiguousarray(packed)
        origin = "saved_hash_checked_native_fixture"
    weight_scales = np.ascontiguousarray(scales.astype(np.float32))
    if backend not in ("direct", "prism_ggml", "prism_ggml_f32", "prism_ggml_hadamard128", "prism_ggml_registered"):
        raise InventoryError("unsupported native FFN-down backend")
    native_a8 = backend in ("prism_ggml_f32", "prism_ggml_hadamard128", "prism_ggml_registered")
    pq2_blocks = None
    if registered_backend:
        from embedded_jev.prism_codec import pack_ternary_pq2_0

        pq2_blocks = pack_ternary_pq2_0(grouped_codes, scales)
    symbol = (
        "prism_bitnet_registered_projection_compute" if registered_backend else
        "prism_bitnet_group_scale_matmul_hadamard128" if rotated_backend else
        "prism_bitnet_group_scale_matmul_f32" if native_a8 else
        "prism_bitnet_group_scale_matmul_avx2" if backend == "prism_ggml" else "bitnet_group_scale_matmul_avx2"
    )
    try:
        native = ctypes.CDLL(str(library))
        function = getattr(native, symbol)
        if registered_backend:
            create_projection = native.prism_bitnet_registered_projection_create
            free_projection = native.prism_bitnet_registered_projection_free
    except (OSError, AttributeError) as exc:
        raise InventoryError(f"unable to load native FFN-down backend {backend}: {exc}") from exc
    if registered_backend:
        create_projection.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                                     ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
        create_projection.restype = ctypes.c_int
        free_projection.argtypes = [ctypes.c_void_p]
        free_projection.restype = None
        function.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                             ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    else:
        function.argtypes = [
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float) if native_a8 else ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
        ]
        if rotated_backend:
            function.argtypes = function.argtypes + [ctypes.POINTER(ctypes.c_float)]
    function.restype = ctypes.c_int
    diagnostics = {
        "calls": 0, "max_native_reference_error": 0.0,
        "packed_bytes": packed.nbytes, "candidate_origin": origin,
        "bf16_projection_materialized": artifact is None,
        "backend": backend, "graph_op": "mul_mat" if registered_backend else "map_custom2" if backend != "direct" else None,
        "activation_preparation": "native_tensor_trait" if registered_backend else "native_graph_callback" if native_a8 else "python",
        "backend_dependencies": native_backend_dependencies(library, backend),
        "transform": {
            "kind": "signed_normalized_hadamard", "block_size": 128, "sign_seed": 773,
            "signs_sha256": hashlib.sha256(signs.tobytes()).hexdigest(),
            "input_transform": "native_signs_then_fwht_then_a8",
        } if rotated_backend else {"kind": "identity"},
    }
    if registered_backend:
        diagnostics.update({
            "native_dispatch_calls": 0, "weight_repacks": 0, "weight_uploads": 0,
            "native_handle_released": False, "weight_tensor_storage": "pq2_0_ternary_subset",
            "weight_tensor_bytes": pq2_blocks.nbytes, "kernel": "bitnet_group_scale_matmul_avx2",
        })

    class NativeFFNDown(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._native_handle = ctypes.c_void_p()
            self._native_tokens = None
            self._finalizer = None
            self._closed = False

        def close(self):
            if registered_backend:
                if self._finalizer is not None:
                    self._finalizer()
                self._native_handle = ctypes.c_void_p()
                self._closed = True
                diagnostics["native_handle_released"] = True

        def forward(self, features):
            if registered_backend and self._closed:
                raise InventoryError("registered native FFN-down is closed")
            if features.device.type != "cpu" or features.dtype != torch.bfloat16 or features.shape[-1] != 12288:
                raise InventoryError("native FFN-down received incompatible activations")
            shape = features.shape
            inputs = np.ascontiguousarray(features.detach().float().numpy().reshape(-1, 12288))
            if rotated_backend:
                rotated_input = rotate_signed_hadamard(inputs[-1:], signs, 128)
                dense_original = inputs[-1:] @ original_probe.T
                dense_rotated = rotated_input @ rotated_probe.T
                if not np.allclose(dense_rotated, dense_original, rtol=1e-5, atol=1e-4):
                    raise InventoryError("signed weight/input rotation changed the dense FP32 projection probe")
                diagnostics["dense_rotation_max_abs_error"] = float(np.max(np.abs(dense_rotated - dense_original)))
            if native_a8:
                input_pointer = inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                scales_pointer = None
            else:
                activations, activation_scales = quantize_a8_per_group(inputs)
                activations = np.ascontiguousarray(activations)
                activation_scales = np.ascontiguousarray(activation_scales)
                input_pointer = activations.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
                scales_pointer = activation_scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            outputs = np.empty((inputs.shape[0], 4096), dtype=np.float32)
            if registered_backend:
                if self._native_handle.value and self._native_tokens != inputs.shape[0]:
                    raise InventoryError("registered native FFN-down token shape changed")
                if not self._native_handle.value:
                    status = create_projection(pq2_blocks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                                               inputs.shape[0], 4096, 96, ctypes.byref(self._native_handle))
                    if status != 0:
                        raise InventoryError(f"registered native FFN-down creation failed with status {status}")
                    self._native_tokens = inputs.shape[0]
                    self._finalizer = weakref.finalize(self, free_projection, self._native_handle)
                    diagnostics["weight_uploads"] += 1
                calls, repacks = ctypes.c_size_t(0), ctypes.c_size_t(0)
                status = function(self._native_handle, input_pointer,
                                  outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                  ctypes.byref(calls), ctypes.byref(repacks))
                diagnostics["native_dispatch_calls"] = calls.value
                diagnostics["weight_repacks"] = repacks.value
            else:
                arguments = [
                    packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                    weight_scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    input_pointer, scales_pointer,
                    inputs.shape[0], 4096, 96, outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ]
                if rotated_backend:
                    arguments.append(signs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
                status = function(*arguments)
            if status != 0 or not np.isfinite(outputs).all():
                raise InventoryError(f"native FFN-down failed with status {status}")
            reference_inputs = rotate_signed_hadamard(inputs[-1:], signs, 128) if rotated_backend else inputs[-1:]
            reference_codes, reference_scales = quantize_a8_per_group(reference_inputs)
            last_codes = reference_codes[0].reshape(96, 128).astype(np.int32)
            partial = np.einsum("rgi,gi->rg", grouped_codes.astype(np.int32), last_codes)
            expected = (partial * weight_scales * reference_scales[0]).sum(axis=1)
            error = float(np.max(np.abs(outputs[-1] - expected)))
            if not np.isfinite(error) or error > 0.005:
                raise InventoryError("native FFN-down disagrees with integer group reference")
            diagnostics["calls"] += 1
            diagnostics["max_native_reference_error"] = max(
                diagnostics["max_native_reference_error"], error,
            )
            return torch.from_numpy(outputs.reshape(*shape[:-1], 4096)).to(dtype=features.dtype)

    return NativeFFNDown(), diagnostics


def score_selected_head(directory: Path, metadata_files, shard_headers, hidden, label_ids: dict) -> dict:
    """Compute only conditional label scores from bounded untied BF16 head rows."""
    report = build_inventory(metadata_files, shard_headers)
    head = next((tensor for tensor in report["tensors"] if tensor["name"] == "lm_head.weight"), None)
    vector = np.asarray(hidden, dtype=np.float32)
    if (
        head is None or head["dtype"] != "BF16" or len(head["shape"]) != 2
        or vector.shape != (head["shape"][1],) or not np.isfinite(vector).all()
        or not isinstance(label_ids, dict) or not 2 <= len(label_ids) <= len(LABELS)
        or list(label_ids) != list(LABELS[:len(label_ids)])
        or any(type(token_id) is not int or not 0 <= token_id < head["shape"][0]
               for token_id in label_ids.values())
        or len(set(label_ids.values())) != len(label_ids)
        or len(label_ids) * head["shape"][1] * 2 > MAX_SELECTED_HEAD_BYTES
    ):
        raise InventoryError("invalid selected-head labels, shape, or byte budget")
    index = _json_object(metadata_files["model.safetensors.index.json"], "model.safetensors.index.json")
    shard = index["weight_map"]["lm_head.weight"]
    header, file_bytes = shard_headers[shard]
    offsets = _json_object(header[8:], shard)["lm_head.weight"]["data_offsets"]
    row_bytes = head["shape"][1] * 2
    logits = []
    digests = []
    try:
        with (directory / shard).open("rb") as source:
            for token_id in label_ids.values():
                position = len(header) + offsets[0] + token_id * row_bytes
                if position + row_bytes > len(header) + offsets[1] or position + row_bytes > file_bytes:
                    raise InventoryError("selected LM-head row exceeds pinned bounds")
                source.seek(position)
                data = source.read(row_bytes)
                if len(data) != row_bytes:
                    raise InventoryError("short selected LM-head row")
                row = (np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16).view("<f4")
                logits.append(float(np.dot(vector, row)))
                digests.append(hashlib.sha256(data).hexdigest())
    except OSError as exc:
        raise InventoryError(f"unable to read selected LM-head rows: {exc}") from exc
    if not np.isfinite(logits).all():
        raise InventoryError("nonfinite selected label logits")
    shifted = np.exp(np.asarray(logits, dtype=np.float64) - max(logits))
    probabilities = shifted / shifted.sum()
    return {
        "scope": "conditional_among_selected_labels_not_calibrated_confidence",
        "full_vocabulary_mass": "not_computed",
        "head_payload_bytes": len(label_ids) * row_bytes,
        "options": {
            label: {
                "token_id": token_id, "logit": logit,
                "conditional_probability": float(probability), "row_sha256": digest,
            }
            for (label, token_id), logit, probability, digest in zip(
                label_ids.items(), logits, probabilities, digests, strict=True
            )
        },
    }


def stream_full_vocabulary_mass(directory: Path, metadata_files, shard_headers, hidden, selected: dict) -> dict:
    """Bound the full-vocabulary normalizer while retaining only one head row batch."""
    report = build_inventory(metadata_files, shard_headers)
    head = next((tensor for tensor in report["tensors"] if tensor["name"] == "lm_head.weight"), None)
    vector = np.asarray(hidden, dtype=np.float32)
    options = selected.get("options") if isinstance(selected, dict) else None
    if (
        head is None or head["dtype"] != "BF16" or len(head["shape"]) != 2
        or head["storage_bytes"] > MAX_FULL_HEAD_BYTES
        or vector.shape != (head["shape"][1],) or not np.isfinite(vector).all()
        or not isinstance(options, dict) or not 2 <= len(options) <= len(LABELS)
        or any(not isinstance(option, dict) or type(option.get("token_id")) is not int
               or not 0 <= option["token_id"] < head["shape"][0] for option in options.values())
    ):
        raise InventoryError("full-vocabulary mass exceeds head or vector bounds")
    selected_ids = [option["token_id"] for option in options.values()]
    if len(set(selected_ids)) != len(selected_ids):
        raise InventoryError("duplicate selected-label token IDs")
    index = _json_object(metadata_files["model.safetensors.index.json"], "model.safetensors.index.json")
    shard = index["weight_map"]["lm_head.weight"]
    header, file_bytes = shard_headers[shard]
    offsets = _json_object(header[8:], shard)["lm_head.weight"]["data_offsets"]
    if offsets[1] - offsets[0] != head["storage_bytes"] or len(header) + offsets[1] > file_bytes:
        raise InventoryError("full-vocabulary LM-head byte span mismatch")
    rows, width = head["shape"]
    normalizer = -np.inf
    selected_normalizer = -np.inf
    max_logit = -np.inf
    max_token_id = None
    try:
        with (directory / shard).open("rb") as source:
            source.seek(len(header) + offsets[0])
            for first_row in range(0, rows, FULL_HEAD_BATCH_ROWS):
                count = min(FULL_HEAD_BATCH_ROWS, rows - first_row)
                data = source.read(count * width * 2)
                if len(data) != count * width * 2:
                    raise InventoryError("short full-vocabulary LM-head batch")
                weights = (np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16).view("<f4")
                logits = weights.reshape(count, width) @ vector
                if not np.isfinite(logits).all():
                    raise InventoryError("nonfinite full-vocabulary logits")
                normalizer = float(np.logaddexp(normalizer, np.logaddexp.reduce(logits.astype(np.float64))))
                selected_offsets = [token_id - first_row for token_id in selected_ids
                                    if first_row <= token_id < first_row + count]
                if selected_offsets:
                    selected_normalizer = float(np.logaddexp(
                        selected_normalizer,
                        np.logaddexp.reduce(logits[selected_offsets].astype(np.float64)),
                    ))
                winner = int(np.argmax(logits))
                if float(logits[winner]) > max_logit:
                    max_logit = float(logits[winner])
                    max_token_id = first_row + winner
    except OSError as exc:
        raise InventoryError(f"unable to stream full LM head: {exc}") from exc
    selected_mass = float(np.exp(selected_normalizer - normalizer))
    if not np.isfinite(selected_mass) or not 0 <= selected_mass <= 1:
        raise InventoryError("invalid selected-label full-vocabulary mass")
    return {
        "vocabulary_rows": rows,
        "payload_bytes": head["storage_bytes"],
        "rows_per_batch": FULL_HEAD_BATCH_ROWS,
        "logsumexp": normalizer,
        "max_token_id": max_token_id,
        "max_logit": max_logit,
        "selected_label_mass": selected_mass,
        "scope": "diagnostic_full_vocabulary_mass_not_confidence_calibration",
    }


def run_streamed_text(
    directory: Path, *, layers: int = 4, prompt: str, label_count: int | None = None,
    fixture_path: Path | None = None, case_id: str | None = None,
    native_ffn_library: Path | None = None,
    full_vocabulary_mass: bool = False,
    projection_artifact: Path | None = None,
    dataset_path: Path | None = None,
    split: str | None = None,
    calibration_output: Path | None = None,
    activation_observer=None,
    native_ffn_backend: str = "direct",
) -> dict:
    """Execute at most one verified BF16 decoder layer at a time on CPU."""
    if label_count is not None and (type(label_count) is not int or not 2 <= label_count <= len(LABELS)):
        raise InventoryError("selected label count must be between 2 and 16")
    if dataset_path is not None and fixture_path is not None:
        raise InventoryError("choose either a decision dataset or a synthetic fixture")
    if (dataset_path is None) != (split is None):
        raise InventoryError("decision dataset requires an explicit split")
    if dataset_path is not None and (split not in SPLITS or case_id is None):
        raise InventoryError("decision dataset requires a valid split and case id")
    if dataset_path is None and (fixture_path is None) != (case_id is None):
        raise InventoryError("fixture path and case id must be provided together")
    if native_ffn_library is not None and (layers < 4 or not native_ffn_library.is_file()):
        raise InventoryError("native FFN-down requires four layers and an existing library")
    if native_ffn_backend not in ("direct", "prism_ggml", "prism_ggml_f32", "prism_ggml_hadamard128", "prism_ggml_registered") or (native_ffn_backend != "direct" and native_ffn_library is None):
        raise InventoryError("native graph backend requires a supported mode and native library")
    if native_ffn_backend == "prism_ggml_hadamard128" and projection_artifact is not None:
        raise InventoryError("signed rotation cannot reinterpret an identity projection artifact")
    if projection_artifact is not None and native_ffn_library is None:
        raise InventoryError("projection artifact requires the native FFN-down library")
    if full_vocabulary_mass and layers != 32:
        raise InventoryError("full-vocabulary mass requires all 32 text layers")
    if calibration_output is not None:
        if dataset_path is None or split != "calibration" or layers < 4 or native_ffn_library is not None:
            raise InventoryError("activation capture requires a BF16 calibration split and at least four layers")
        if calibration_output.exists():
            raise InventoryError("calibration capture already exists")
    if activation_observer is not None and (
        not callable(activation_observer) or dataset_path is None
        or split not in ("calibration", "validation") or layers < 4 or native_ffn_library is not None
    ):
        raise InventoryError("activation observer requires a BF16 calibration or validation dataset prefix")
    import torch
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5TextModel, create_causal_mask,
    )

    metadata, headers = read_local_headers(directory)
    plan = plan_streamed_text(metadata, headers, layers=layers)
    config = AutoConfig.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    fixture_case = None
    dataset = None
    if fixture_path is not None or dataset_path is not None:
        if dataset_path is not None:
            dataset, dataset_digest = load_decision_dataset(dataset_path)
            cases = dataset["splits"][split]
        else:
            fixture, fixture_digest = load_decision_fixture(fixture_path)
            cases = fixture["cases"]
        case_reports = probe_decision_cases(tokenizer, cases)
        matching = [
            (case, case_report)
            for case, case_report in zip(cases, case_reports, strict=True)
            if case["id"] == case_id
        ]
        if len(matching) != 1:
            raise InventoryError("decision case id not found in requested input or split")
        fixture_case, case_report = matching[0]
        messages = decision_case_messages(fixture_case)
        if label_count is not None and label_count != len(fixture_case["options"]):
            raise InventoryError("label count disagrees with fixture options")
        label_count = len(fixture_case["options"])
    else:
        messages = [{"role": "user", "content": prompt}]
        label_count = label_count or 2
    boundary = probe_label_boundary(tokenizer, messages, LABELS[:label_count])
    if not 1 <= boundary["prompt_token_count"] <= MAX_PREFIX_TOKENS:
        raise InventoryError("streamed text prompt exceeds token budget")
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )
    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long)
    weight_map = _json_object(metadata["model.safetensors.index.json"], "model.safetensors.index.json")["weight_map"]
    with init_empty_weights():
        model = Qwen3_5TextModel(config.text_config)
    model.eval()

    def materialize(target_names):
        parameters = dict(model.named_parameters())
        for shard in sorted({weight_map[name] for name in target_names}):
            with safe_open(directory / shard, framework="pt", device="cpu") as source:
                for name in target_names:
                    if weight_map[name] != shard:
                        continue
                    target_name = name.removeprefix("model.language_model.")
                    value = source.get_tensor(name)
                    if value.dtype != torch.bfloat16 or tuple(value.shape) != tuple(parameters[target_name].shape):
                        raise InventoryError(f"streamed BF16 tensor shape mismatch: {name}")
                    set_module_tensor_to_device(
                        model, target_name, "cpu", value=value, dtype=torch.bfloat16,
                    )

    materialize([plan["embedding_name"]])
    with torch.inference_mode():
        hidden = model.embed_tokens(input_ids)
    model.embed_tokens = torch.nn.Identity()
    gc.collect()
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(hidden.shape[1], device=hidden.device)
    position_ids = position_ids.view(1, 1, -1).expand(4, hidden.shape[0], -1)
    text_position_ids = position_ids[0]
    causal_mask = create_causal_mask(
        config=model.config, inputs_embeds=hidden, attention_mask=attention_mask,
        past_key_values=None, position_ids=text_position_ids,
    )
    linear_mask = model._update_linear_attn_mask(attention_mask, None)
    position_embeddings = model.rotary_emb(hidden, position_ids[1:])
    captured = []
    native_diagnostics = None

    for layer_index, names in enumerate(plan["layer_parameter_names"]):
        if set(names) != {
            "model.language_model." + name
            for name, _ in model.layers[layer_index].named_parameters(prefix=f"layers.{layer_index}")
        }:
            raise InventoryError(f"layer {layer_index} parameter names disagree with pinned checkpoint")
        selected_names = names
        if layer_index == 3 and projection_artifact is not None:
            selected_names = [name for name in names if not name.endswith(".mlp.down_proj.weight")]
        materialize(selected_names)
        decoder = model.layers[layer_index]
        if layer_index == 3:
            if native_ffn_library is not None:
                decoder.mlp.down_proj, native_diagnostics = make_native_ffn_down(
                    None if projection_artifact is not None else decoder.mlp.down_proj.weight,
                    native_ffn_library, projection_artifact, backend=native_ffn_backend,
                )

            def capture_ffn_input(_module, args):
                captured.append(args[0].detach().float().cpu().clone())

            hook = decoder.mlp.down_proj.register_forward_pre_hook(capture_ffn_input)
        if any(parameter.is_meta or parameter.dtype != torch.bfloat16 for parameter in decoder.parameters()):
            raise InventoryError(f"incomplete BF16 layer {layer_index} materialization")
        layer_mask = linear_mask if plan["layer_types"][layer_index] == "linear_attention" else causal_mask
        try:
            with torch.inference_mode():
                hidden = decoder(
                    hidden, position_embeddings=position_embeddings,
                    attention_mask=layer_mask, position_ids=text_position_ids,
                    past_key_values=None, use_cache=False,
                )
        finally:
            if layer_index == 3 and native_ffn_backend == "prism_ggml_registered":
                decoder.mlp.down_proj.close()
        if layer_index == 3:
            hook.remove()
        if not torch.isfinite(hidden).all():
            raise InventoryError(f"nonfinite streamed text layer {layer_index}")
        model.layers[layer_index] = torch.nn.Identity()
        del decoder
        gc.collect()

    materialize([plan["final_norm_name"]])
    with torch.inference_mode():
        hidden = model.norm(hidden)
    if not torch.isfinite(hidden).all():
        raise InventoryError("nonfinite streamed final norm")
    summary = {
        "model": plan["model"],
        "revision": plan["revision"],
        "layers": layers,
        "tokens": input_ids.shape[1],
        "output_shape": list(hidden.shape),
        "output_dtype": str(hidden.dtype),
        "last_token_sha256": hashlib.sha256(hidden[0, -1].float().numpy().tobytes()).hexdigest(),
        "prompt_sha256": boundary["prompt_sha256"],
        "generated_tokens": 0,
        "max_layer_bytes": plan["max_layer_bytes"],
        "embedding_bytes": plan["embedding_bytes"],
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "purpose": "streamed_text_layers_not_model_logits_or_quality",
    }
    if fixture_case is not None:
        case_metadata = {
            "case_id": case_id,
            "expected_option_id": fixture_case["expected_option_id"],
            "prompt_token_count": case_report["prompt_token_count"],
        }
        if dataset is not None:
            summary["dataset"] = {
                **case_metadata, "purpose": dataset["purpose"], "sha256": dataset_digest,
                "provenance": dataset["provenance"], "split": split,
            }
        else:
            summary["fixture"] = {
                **case_metadata, "purpose": fixture["purpose"], "sha256": fixture_digest,
            }
    if layers >= 4:
        if len(captured) != 1 or tuple(captured[0].shape) != (1, input_ids.shape[1], 12288):
            raise InventoryError("missing streamed FFN-down input")
        summary["ffn_down_input_sha256"] = hashlib.sha256(captured[0].numpy().tobytes()).hexdigest()
    if activation_observer is not None:
        observed = captured[0].numpy().reshape(-1, 12288).copy()
        observed.setflags(write=False)
        activation_observer(observed)
    if calibration_output is not None:
        capture_manifest = save_calibration_capture(
            calibration_output, captured[0].numpy().reshape(-1, 12288),
            dataset_path=dataset_path, dataset_sha256=dataset_digest, case_id=case_id,
        )
        summary["calibration_capture"] = {"path": str(calibration_output), "manifest": capture_manifest}
    if native_diagnostics is not None:
        if native_diagnostics["calls"] != 1:
            raise InventoryError("native FFN-down was not executed exactly once")
        summary["native_ffn_down"] = {
            **native_diagnostics,
            "purpose": "one_in_memory_bitnet_derived_projection_not_loadable_model",
            "activation_quantized": True,
        }
    if layers == config.text_config.num_hidden_layers:
        selected = {label: boundary["label_token_ids"][label] for label in LABELS[:label_count]}
        scored = score_selected_head(
            directory, metadata, headers, hidden[0, -1].float().numpy(), selected,
        )
        shard = weight_map["lm_head.weight"]
        with safe_open(directory / shard, framework="pt", device="cpu") as source:
            head = source.get_slice("lm_head.weight")
            head_rows = torch.stack([head[token_id] for token_id in selected.values()])
        if any(
            hashlib.sha256(row.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()
            != scored["options"][label]["row_sha256"]
            for label, row in zip(selected, head_rows, strict=True)
        ):
            raise InventoryError("selected LM-head rows disagree with safetensors")
        torch_logits = torch.nn.functional.linear(hidden[0, -1].unsqueeze(0), head_rows)[0].float()
        scored_logits = torch.tensor(
            [scored["options"][label]["logit"] for label in selected], dtype=torch.float32,
        )
        if not torch.allclose(torch_logits, scored_logits.to(torch.bfloat16).float(), rtol=0.01, atol=0.125):
            raise InventoryError("selected FP32 logits disagree with BF16 head rounding")
        scored["max_fp32_to_bf16_logit_gap"] = float(torch.max(torch.abs(torch_logits - scored_logits)))
        summary["selected_head"] = scored
        if full_vocabulary_mass:
            summary["vocabulary_mass"] = stream_full_vocabulary_mass(
                directory, metadata, headers, hidden[0, -1].float().numpy(), scored,
            )
            scored["full_vocabulary_mass"] = summary["vocabulary_mass"]["selected_label_mass"]
        if fixture_case is not None:
            options = [
                {
                    "id": option["id"], "description": option["description"], "label": label,
                    **scored["options"][label],
                }
                for label, option in zip(LABELS[:len(fixture_case["options"])], fixture_case["options"], strict=True)
            ]
            summary["decision"] = {
                "scope": (
                    "dataset_observation_not_calibrated_confidence" if dataset is not None
                    else "synthetic_fixture_observation_not_quality_or_calibration"
                ),
                "chosen_option_id": max(options, key=lambda option: option["conditional_probability"])["id"],
                "expected_option_id": fixture_case["expected_option_id"],
                "options": options,
            }
    summary["peak_process_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pinned MiMo text layers one at a time")
    parser.add_argument("--local-dir", required=True, type=Path)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--label-count", type=int)
    parser.add_argument("--prompt", default="Choose A or B. A: pause. B: continue.")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--calibration-output", type=Path)
    parser.add_argument("--case-id")
    parser.add_argument("--native-ffn-library", type=Path)
    parser.add_argument("--native-ffn-backend", choices=("direct", "prism_ggml", "prism_ggml_f32", "prism_ggml_hadamard128", "prism_ggml_registered"), default="direct")
    parser.add_argument("--projection-artifact", type=Path)
    parser.add_argument("--full-vocabulary-mass", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_streamed_text(
        args.local_dir, layers=args.layers, prompt=args.prompt, label_count=args.label_count,
        fixture_path=args.fixture, case_id=args.case_id,
        native_ffn_library=args.native_ffn_library,
        full_vocabulary_mass=args.full_vocabulary_mass,
        projection_artifact=args.projection_artifact,
        dataset_path=args.dataset,
        split=args.split,
        calibration_output=args.calibration_output,
        native_ffn_backend=args.native_ffn_backend,
    ), sort_keys=True))


if __name__ == "__main__":
    main()