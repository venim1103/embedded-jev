"""Run bounded pinned MiMo text layers without materializing the full model."""

import argparse
import gc
import hashlib
import json
import resource
from pathlib import Path

import numpy as np

from embedded_jev.dense_probe import MAX_PREFIX_TOKENS, plan_streamed_text
from embedded_jev.inventory import InventoryError, _json_object, build_inventory, read_local_headers
from embedded_jev.label_probe import (
    LABELS, decision_case_messages, load_decision_fixture, probe_decision_cases,
    probe_label_boundary,
)


MAX_SELECTED_HEAD_BYTES = 256 * 1024


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


def run_streamed_text(
    directory: Path, *, layers: int = 4, prompt: str, label_count: int | None = None,
    fixture_path: Path | None = None, case_id: str | None = None,
) -> dict:
    """Execute at most one verified BF16 decoder layer at a time on CPU."""
    if label_count is not None and (type(label_count) is not int or not 2 <= label_count <= len(LABELS)):
        raise InventoryError("selected label count must be between 2 and 16")
    if (fixture_path is None) != (case_id is None):
        raise InventoryError("fixture path and case id must be provided together")
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
    if fixture_path is not None:
        fixture, fixture_digest = load_decision_fixture(fixture_path)
        case_reports = probe_decision_cases(tokenizer, fixture["cases"])
        matching = [
            (case, case_report)
            for case, case_report in zip(fixture["cases"], case_reports, strict=True)
            if case["id"] == case_id
        ]
        if len(matching) != 1:
            raise InventoryError("fixture case id not found")
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

    for layer_index, names in enumerate(plan["layer_parameter_names"]):
        if set(names) != {
            "model.language_model." + name
            for name, _ in model.layers[layer_index].named_parameters(prefix=f"layers.{layer_index}")
        }:
            raise InventoryError(f"layer {layer_index} parameter names disagree with pinned checkpoint")
        materialize(names)
        decoder = model.layers[layer_index]
        if any(parameter.is_meta or parameter.dtype != torch.bfloat16 for parameter in decoder.parameters()):
            raise InventoryError(f"incomplete BF16 layer {layer_index} materialization")
        if layer_index == 3:
            def capture_ffn_input(_module, args):
                captured.append(args[0].detach().float().cpu().clone())

            hook = decoder.mlp.down_proj.register_forward_pre_hook(capture_ffn_input)
        layer_mask = linear_mask if plan["layer_types"][layer_index] == "linear_attention" else causal_mask
        with torch.inference_mode():
            hidden = decoder(
                hidden, position_embeddings=position_embeddings,
                attention_mask=layer_mask, position_ids=text_position_ids,
                past_key_values=None, use_cache=False,
            )
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
        summary["fixture"] = {
            "purpose": fixture["purpose"],
            "sha256": fixture_digest,
            "case_id": case_id,
            "expected_option_id": fixture_case["expected_option_id"],
            "prompt_token_count": case_report["prompt_token_count"],
        }
    if layers >= 4:
        if len(captured) != 1 or tuple(captured[0].shape) != (1, input_ids.shape[1], 12288):
            raise InventoryError("missing streamed FFN-down input")
        summary["ffn_down_input_sha256"] = hashlib.sha256(captured[0].numpy().tobytes()).hexdigest()
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
        if fixture_case is not None:
            options = [
                {
                    "id": option["id"], "description": option["description"], "label": label,
                    **scored["options"][label],
                }
                for label, option in zip(LABELS[:len(fixture_case["options"])], fixture_case["options"], strict=True)
            ]
            summary["decision"] = {
                "scope": "synthetic_fixture_observation_not_quality_or_calibration",
                "chosen_option_id": max(options, key=lambda option: option["conditional_probability"])["id"],
                "expected_option_id": fixture_case["expected_option_id"],
                "options": options,
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pinned MiMo text layers one at a time")
    parser.add_argument("--local-dir", required=True, type=Path)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--label-count", type=int)
    parser.add_argument("--prompt", default="Choose A or B. A: pause. B: continue.")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--case-id")
    args = parser.parse_args()
    print(json.dumps(run_streamed_text(
        args.local_dir, layers=args.layers, prompt=args.prompt, label_count=args.label_count,
        fixture_path=args.fixture, case_id=args.case_id,
    ), sort_keys=True))


if __name__ == "__main__":
    main()