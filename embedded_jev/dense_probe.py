"""Bound a text-only dense prefix before loading pinned MiMo weights."""

import argparse
import hashlib
import json
from pathlib import Path

from embedded_jev.inventory import (
    InventoryError, _json_object, build_inventory, read_local_headers,
)


MAX_PREFIX_LAYERS = 4
MAX_PREFIX_WEIGHT_BYTES = 4 * 1024**3
MAX_PREFIX_TOKENS = 32


def plan_text_prefix(metadata_files, shard_headers, *, layers: int = 4) -> dict:
    """Select only the embedding, first layers, and text final norm."""
    report = build_inventory(metadata_files, shard_headers)
    text = _json_object(metadata_files["config.json"], "config.json")["text_config"]
    if type(layers) is not int or not 1 <= layers <= min(MAX_PREFIX_LAYERS, len(text["layer_types"])):
        raise InventoryError("unsupported dense text prefix length")
    names = {
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
    }
    names.update(
        tensor["name"] for tensor in report["tensors"]
        if tensor["name"].startswith("model.language_model.layers.")
        and int(tensor["name"].split(".")[3]) < layers
    )
    selected = [tensor for tensor in report["tensors"] if tensor["name"] in names]
    total_bytes = sum(tensor["storage_bytes"] for tensor in selected)
    if (
        len(selected) != len(names)
        or any(tensor["dtype"] != "BF16" for tensor in selected)
        or total_bytes > MAX_PREFIX_WEIGHT_BYTES
    ):
        raise InventoryError("dense text prefix is missing or exceeds BF16 budget")
    return {
        "model": report["source"]["model"],
        "revision": report["source"]["revision"],
        "layers": layers,
        "layer_types": text["layer_types"][:layers],
        "tensor_count": len(selected),
        "weight_bytes": total_bytes,
        "parameter_names": sorted(names),
    }


def run_text_prefix(
    directory: Path, *, layers: int = 1, prompt: str, compare_ternary: bool = False,
    native_library: Path | None = None, chat_template: bool = False,
) -> dict:
    """Run a small text-only dense prefix without generating answer tokens."""
    if compare_ternary and layers != MAX_PREFIX_LAYERS:
        raise InventoryError("ternary comparison requires four dense prefix layers")
    if native_library is not None and not compare_ternary:
        raise InventoryError("native comparison requires ternary comparison")
    import torch
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    from embedded_jev.label_probe import probe_label_boundary

    metadata, headers = read_local_headers(directory)
    plan = plan_text_prefix(metadata, headers, layers=layers)
    config = AutoConfig.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    if chat_template:
        messages = [{"role": "user", "content": prompt}]
        boundary = probe_label_boundary(tokenizer, messages)
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
        )
        input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    if not 1 <= input_ids.shape[1] <= MAX_PREFIX_TOKENS:
        raise InventoryError("dense prefix prompt exceeds token budget")
    with init_empty_weights():
        model = Qwen3_5TextModel(config.text_config)
    model.layers = torch.nn.ModuleList(list(model.layers[:layers]))
    parameters = dict(model.named_parameters())
    source_names = {
        name.removeprefix("model.language_model."): name for name in plan["parameter_names"]
    }
    if set(parameters) != set(source_names):
        raise InventoryError("dense prefix parameter names disagree with pinned checkpoint")
    weight_map = _json_object(metadata["model.safetensors.index.json"], "model.safetensors.index.json")["weight_map"]
    for shard in sorted({weight_map[name] for name in source_names.values()}):
        with safe_open(directory / shard, framework="pt", device="cpu") as source:
            for target_name, full_name in source_names.items():
                if weight_map[full_name] != shard:
                    continue
                value = source.get_tensor(full_name)
                if value.dtype != torch.bfloat16 or tuple(value.shape) != tuple(parameters[target_name].shape):
                    raise InventoryError(f"dense prefix tensor shape or dtype mismatch: {full_name}")
                set_module_tensor_to_device(
                    model, target_name, "cpu", value=value, dtype=torch.bfloat16,
                )
    if any(parameter.is_meta or parameter.dtype != torch.bfloat16 for parameter in model.parameters()):
        raise InventoryError("incomplete BF16 dense prefix materialization")
    if any(buffer.is_meta for buffer in model.buffers()):
        raise InventoryError("dense prefix has an unmaterialized buffer")

    captured = []
    dense_outputs = []
    if layers == MAX_PREFIX_LAYERS:
        def capture_ffn_input(_module, args):
            captured.append(args[0].detach().float().cpu().clone())

        hook = model.layers[3].mlp.down_proj.register_forward_pre_hook(capture_ffn_input)
        if compare_ternary:
            def capture_ffn_output(_module, _args, output):
                dense_outputs.append(output.detach().float().cpu().clone())

            output_hook = model.layers[3].mlp.down_proj.register_forward_hook(capture_ffn_output)
    model.eval()
    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
    if layers == MAX_PREFIX_LAYERS:
        hook.remove()
        if compare_ternary:
            output_hook.remove()
    if not torch.isfinite(output.last_hidden_state).all():
        raise InventoryError("nonfinite dense prefix output")
    summary = {
        "model": plan["model"],
        "revision": plan["revision"],
        "layers": layers,
        "weight_bytes": plan["weight_bytes"],
        "tokens": input_ids.shape[1],
        "output_shape": list(output.last_hidden_state.shape),
        "output_dtype": str(output.last_hidden_state.dtype),
        "last_token_sha256": hashlib.sha256(
            output.last_hidden_state[0, -1].float().numpy().tobytes()
        ).hexdigest(),
        "purpose": "text_prefix_activation_smoke_not_model_logits_or_quality",
    }
    if chat_template:
        summary["prompt_sha256"] = boundary["prompt_sha256"]
        summary["label_token_ids"] = boundary["label_token_ids"]
        summary["generated_tokens"] = 0
    if layers == MAX_PREFIX_LAYERS:
        if len(captured) != 1 or tuple(captured[0].shape) != (1, input_ids.shape[1], 12288):
            raise InventoryError("missing layer-3 FFN-down input activation")
        if not torch.isfinite(captured[0]).all():
            raise InventoryError("nonfinite FFN-down input activation")
        summary["ffn_down_input_shape"] = list(captured[0].shape)
        summary["ffn_down_input_sha256"] = hashlib.sha256(captured[0].numpy().tobytes()).hexdigest()
        summary["ffn_down_input_rms"] = float(torch.sqrt(torch.mean(captured[0] ** 2)))
    if compare_ternary:
        import numpy as np

        from embedded_jev.ternary import quantize_ternary_rtn, reconstruct_ternary

        if len(dense_outputs) != 1 or tuple(dense_outputs[0].shape) != (1, input_ids.shape[1], 4096):
            raise InventoryError("missing dense FFN-down output")
        weights = model.layers[3].mlp.down_proj.weight.detach().float().cpu().numpy()
        codes, scales = quantize_ternary_rtn(weights, scale_search=True)
        reconstructed = reconstruct_ternary(codes, scales)
        reference = dense_outputs[0].numpy().reshape(-1, 4096)
        candidate = captured[0].numpy().reshape(-1, 12288) @ reconstructed.T
        reference_energy = float(np.mean(reference.astype(np.float64) ** 2))
        error = float(np.mean((candidate.astype(np.float64) - reference) ** 2))
        if not np.isfinite(error) or reference_energy <= 0:
            raise InventoryError("nonfinite or zero dense FFN-down output")
        summary["ternary_comparison"] = {
            "policy": "unrotated_searched_fp16_group128",
            "activation_source": "one_local_text_prompt_not_representative_calibration",
            "activation_quantized": False,
            "relative_output_rmse": float(np.sqrt(error / reference_energy)),
            "weight_mse": float(np.mean((weights - reconstructed) ** 2)),
        }
        if native_library is not None:
            import ctypes

            from embedded_jev.activation import quantize_a8_per_group
            from embedded_jev.ternary import pack_group128_codes

            groups = weights.shape[1] // 128
            packed = pack_group128_codes(codes.reshape(weights.shape[0], groups, 128))
            weight_scales = np.ascontiguousarray(scales.astype(np.float32))
            activations, activation_scales = quantize_a8_per_group(captured[0].numpy()[0, -1:])
            grouped_activations = np.ascontiguousarray(activations.reshape(groups, 128))
            native_dot = ctypes.CDLL(str(native_library)).bitnet_group_scale_matvec_avx2
            native_dot.argtypes = [
                ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_float),
            ]
            native_dot.restype = ctypes.c_int
            actual = np.empty(weights.shape[0], dtype=np.float32)
            status = native_dot(
                packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                weight_scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                grouped_activations.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                activation_scales[0].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                weights.shape[0], groups, actual.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            if status != 0:
                raise InventoryError(f"native grouped-dot status {status}")
            group_dots = np.einsum(
                "rgi,gi->rg", codes.reshape(weights.shape[0], groups, 128).astype(np.int32),
                grouped_activations.astype(np.int32),
            )
            expected = (group_dots * weight_scales * activation_scales[0]).sum(axis=1)
            dense_last = reference[-1]
            dense_energy = float(np.mean(dense_last.astype(np.float64) ** 2))
            native_error = float(np.mean((actual.astype(np.float64) - dense_last) ** 2))
            if not np.isfinite(actual).all() or not np.isfinite(expected).all() or dense_energy <= 0:
                raise InventoryError("nonfinite or zero native comparison output")
            summary["native_comparison"] = {
                "purpose": "one_model_path_activation_bitnet_derived_dot_not_model_inference",
                "groups": groups,
                "packed_bytes": packed.nbytes,
                "max_native_reference_error": float(np.max(np.abs(actual - expected))),
                "relative_dense_output_rmse": float(np.sqrt(native_error / dense_energy)),
                "activation_quantized": True,
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a bounded pinned MiMo text prefix")
    parser.add_argument("--local-dir", required=True, type=Path)
    parser.add_argument("--layers", type=int, choices=range(1, MAX_PREFIX_LAYERS + 1), default=1)
    parser.add_argument("--prompt", default="Choose A or B. A: pause. B: continue.")
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--compare-ternary", action="store_true")
    parser.add_argument("--native-library", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_text_prefix(
        args.local_dir, layers=args.layers, prompt=args.prompt,
        compare_ternary=args.compare_ternary, native_library=args.native_library,
        chat_template=args.chat_template,
    ), sort_keys=True))


if __name__ == "__main__":
    main()