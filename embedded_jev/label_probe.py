"""Check fixed-label token boundaries without loading model weights."""

import argparse
import hashlib
import json
from pathlib import Path

from embedded_jev.inventory import InventoryError, MODEL_ID, MODEL_REVISION, _json_object


LABELS = tuple("ABCDEFGHIJKLMNOP")
NON_THINKING_SUFFIX = "<|im_start|>assistant\n<think></think>"
ALLOWED_FILES = {
    "chat_template.jinja": 1 << 20,
    "config.json": 1 << 20,
    "tokenizer.json": 25 << 20,
    "tokenizer_config.json": 1 << 20,
}
PROCESSOR_FILES = {
    "preprocessor_config.json": 1 << 20,
    "processor_config.json": 1 << 20,
    "video_preprocessor_config.json": 1 << 20,
}


class LabelProbeError(ValueError):
    """The pinned prompt and tokenizer cannot safely score fixed labels."""


def probe_label_boundary(tokenizer, messages: list[dict], labels: tuple[str, ...] = LABELS) -> dict:
    """Check label IDs as continuations of the actual non-thinking chat prompt."""
    if not 2 <= len(labels) <= len(LABELS) or labels != LABELS[: len(labels)]:
        raise LabelProbeError("labels must be a prefix of A-P with 2-16 options")
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise LabelProbeError("missing model chat template")
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    if not isinstance(prompt, str) or not prompt.endswith(NON_THINKING_SUFFIX):
        raise LabelProbeError("unexpected non-thinking assistant prefix")
    prefix_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prefix_ids or any(type(token_id) is not int for token_id in prefix_ids):
        raise LabelProbeError("invalid prompt token IDs")
    tokenized = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    if not hasattr(tokenized, "get") or tokenized.get("input_ids") != prefix_ids:
        raise LabelProbeError("rendered prompt disagrees with template tokenization")

    label_ids = {}
    for label in labels:
        candidate_ids = tokenizer.encode(prompt + label, add_special_tokens=False)
        if candidate_ids[:-1] != prefix_ids or len(candidate_ids) != len(prefix_ids) + 1:
            raise LabelProbeError(f"{label} does not extend the exact prompt by one token")
        token_id = candidate_ids[-1]
        if token_id in tokenizer.all_special_ids or token_id in label_ids.values():
            raise LabelProbeError(f"{label} is special or shares a token ID")
        label_ids[label] = token_id

    return {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_token_count": len(prefix_ids),
        "label_token_ids": label_ids,
        "tokenizer_class": type(tokenizer).__name__,
        "generated_tokens": 0,
    }


def probe_text_processor(
    processor, tokenizer, messages: list[dict], labels: tuple[str, ...] = LABELS
) -> dict:
    """Require processor text inputs to match the validated prompt tokenization."""
    report = probe_label_boundary(tokenizer, messages, labels)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    encoded = processor(text=prompt, return_tensors=None)
    expected_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if (
        not hasattr(encoded, "keys")
        or not {"input_ids", "attention_mask"} <= set(encoded)
        or set(encoded) - {"input_ids", "attention_mask", "mm_token_type_ids"}
        or encoded["input_ids"] != [expected_ids]
        or encoded["attention_mask"] != [[1] * len(expected_ids)]
        or (
            "mm_token_type_ids" in encoded
            and encoded["mm_token_type_ids"] != [[0] * len(expected_ids)]
        )
    ):
        raise LabelProbeError("processor text inputs disagree with tokenizer or contain media")
    report["processor_class"] = type(processor).__name__
    report["processor_text_only"] = True
    report["processor_mm_token_type_ids"] = (
        "all_text" if "mm_token_type_ids" in encoded else "absent"
    )
    return report


def decision_case_messages(case: dict) -> list[dict]:
    """Render a validated synthetic fixture case with stable A-P labels."""
    labels = LABELS[: len(case["options"])]
    choices = "\n".join(
        f"{label}. {option['description']}"
        for label, option in zip(labels, case["options"], strict=True)
    )
    return [{
        "role": "user",
        "content": (
            f"State: {case['state']}\nQuestion: {case['question']}\n"
            f"Options:\n{choices}\nAnswer:"
        ),
    }]


def validate_decision_cases(cases: list[dict]) -> None:
    """Check bounded labeled cases independently of tokenizer or model loading."""
    if not isinstance(cases, list) or not 1 <= len(cases) <= 32:
        raise LabelProbeError("decision fixture must contain 1-32 cases")
    seen_ids = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {
            "id", "group", "state", "question", "options", "expected_option_id"
        }:
            raise LabelProbeError("unsupported decision fixture case")
        case_id = case["id"]
        if not isinstance(case_id, str) or not case_id or case_id in seen_ids:
            raise LabelProbeError("duplicate or invalid decision fixture id")
        seen_ids.add(case_id)
        if any(
            not isinstance(case[key], str) or not case[key] or len(case[key]) > 2000
            for key in ("group", "state", "question")
        ):
            raise LabelProbeError(f"invalid decision fixture text: {case_id}")
        options = case["options"]
        if not isinstance(options, list) or not 2 <= len(options) <= len(LABELS):
            raise LabelProbeError(f"invalid decision fixture options: {case_id}")
        option_ids = []
        for option in options:
            if not isinstance(option, dict) or set(option) != {"id", "description"} or any(
                not isinstance(option[key], str) or not option[key] or "\n" in option[key]
                for key in ("id", "description")
            ):
                raise LabelProbeError(f"invalid decision fixture option: {case_id}")
            option_ids.append(option["id"])
        if len(set(option_ids)) != len(option_ids) or case["expected_option_id"] not in option_ids:
            raise LabelProbeError(f"duplicate options or unknown expected choice: {case_id}")


def probe_decision_cases(tokenizer, cases: list[dict], processor=None) -> list[dict]:
    """Check a small labeled fixture's option mapping and prompt boundaries."""
    validate_decision_cases(cases)
    reports = []
    for case in cases:
        options = case["options"]
        option_ids = [option["id"] for option in options]
        labels = LABELS[: len(options)]
        messages = decision_case_messages(case)
        report = (
            probe_text_processor(processor, tokenizer, messages, labels)
            if processor is not None else probe_label_boundary(tokenizer, messages, labels)
        )
        case_report = {
            "id": case["id"],
            "group": case["group"],
            "expected_option_id": case["expected_option_id"],
            "expected_label": labels[option_ids.index(case["expected_option_id"])],
            "prompt_sha256": report["prompt_sha256"],
            "prompt_token_count": report["prompt_token_count"],
            "label_token_ids": report["label_token_ids"],
        }
        if processor is not None:
            case_report["processor_mm_token_type_ids"] = report["processor_mm_token_type_ids"]
        reports.append(case_report)
    return reports


def load_decision_fixture(path: Path) -> tuple[dict, str]:
    """Read only the versioned, bounded synthetic engineering fixture schema."""
    if path.stat().st_size > 64 << 10:
        raise LabelProbeError("decision fixture exceeds 64 KiB")
    data = path.read_bytes()
    if len(data) > 64 << 10:
        raise LabelProbeError("decision fixture exceeds 64 KiB")
    try:
        fixture = _json_object(data, str(path))
    except InventoryError as exc:
        raise LabelProbeError("invalid decision fixture JSON") from exc
    if (
        set(fixture) != {"schema_version", "purpose", "cases"}
        or type(fixture["schema_version"]) is not int or fixture["schema_version"] != 1
        or fixture["purpose"] != "synthetic_engineering_smoke_not_calibration_or_benchmark"
    ):
        raise LabelProbeError("unsupported decision fixture schema or purpose")
    return fixture, hashlib.sha256(data).hexdigest()


def _bounded_size(name: str, size: int | None, *, processor: bool = False) -> int:
    allowed = ALLOWED_FILES | (PROCESSOR_FILES if processor else {})
    if name not in allowed or type(size) is not int or not 0 < size <= allowed[name]:
        raise LabelProbeError(f"missing or oversized pinned tokenizer file: {name}")
    return size


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect pinned MiMo A-P label tokenization")
    parser.add_argument("--fetch", action="store_true", help="fetch bounded tokenizer-only files")
    parser.add_argument("--processor", action="store_true", help="check text-only processor parity")
    parser.add_argument("--fixture", type=Path, help="check a bounded synthetic decision fixture")
    args = parser.parse_args()

    import huggingface_hub
    import jinja2
    import tokenizers
    import transformers

    paths = {}
    sources = {}
    for name in sorted(ALLOWED_FILES | (PROCESSOR_FILES if args.processor else {})):
        if args.fetch:
            metadata = huggingface_hub.get_hf_file_metadata(
                huggingface_hub.hf_hub_url(MODEL_ID, name, revision=MODEL_REVISION)
            )
            _bounded_size(name, metadata.size, processor=args.processor)
        path = Path(
            huggingface_hub.hf_hub_download(
                MODEL_ID, name, revision=MODEL_REVISION, local_files_only=not args.fetch
            )
        )
        size = _bounded_size(name, path.stat().st_size, processor=args.processor)
        with path.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        paths[name] = path
        sources[name] = {"bytes": size, "sha256": digest}

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False,
        use_fast=True, local_files_only=True,
    )
    if tokenizer.chat_template != paths["chat_template.jinja"].read_text(encoding="utf-8"):
        raise LabelProbeError("loaded tokenizer template differs from pinned file")
    messages = [{
        "role": "user",
        "content": (
            "State: No temperature measurement is available.\n"
            "Question: Is the room too hot?\n"
            "Options:\nA. Yes\nB. No\nC. Insufficient evidence\nAnswer:"
        ),
    }]
    processor = (
        transformers.AutoProcessor.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False, local_files_only=True
        ) if args.processor else None
    )
    if args.fixture is not None:
        fixture, digest = load_decision_fixture(args.fixture)
        report = {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "fixture_purpose": fixture["purpose"],
            "fixture_sha256": digest,
            "cases": probe_decision_cases(tokenizer, fixture["cases"], processor),
            "generated_tokens": 0,
        }
        if processor is not None:
            report["processor_class"] = type(processor).__name__
            report["processor_text_only"] = True
    elif processor is not None:
        report = probe_text_processor(processor, tokenizer, messages)
    else:
        report = probe_label_boundary(tokenizer, messages)
    report["source_files"] = sources
    report["tool_versions"] = {
        "huggingface_hub": huggingface_hub.__version__,
        "jinja2": jinja2.__version__,
        "tokenizers": tokenizers.__version__,
        "transformers": transformers.__version__,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()