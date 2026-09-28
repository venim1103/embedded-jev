"""Check fixed-label token boundaries without loading model weights."""

import argparse
import hashlib
import json
from pathlib import Path

from embedded_jev.inventory import MODEL_ID, MODEL_REVISION


LABELS = tuple("ABCDEFGHIJKLMNOP")
NON_THINKING_SUFFIX = "<|im_start|>assistant\n<think></think>"
ALLOWED_FILES = {
    "chat_template.jinja": 1 << 20,
    "config.json": 1 << 20,
    "tokenizer.json": 25 << 20,
    "tokenizer_config.json": 1 << 20,
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


def _bounded_size(name: str, size: int | None) -> int:
    if name not in ALLOWED_FILES or type(size) is not int or not 0 < size <= ALLOWED_FILES[name]:
        raise LabelProbeError(f"missing or oversized pinned tokenizer file: {name}")
    return size


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect pinned MiMo A-P label tokenization")
    parser.add_argument("--fetch", action="store_true", help="fetch bounded tokenizer-only files")
    args = parser.parse_args()

    import huggingface_hub
    import jinja2
    import tokenizers
    import transformers

    paths = {}
    sources = {}
    for name in sorted(ALLOWED_FILES):
        if args.fetch:
            metadata = huggingface_hub.get_hf_file_metadata(
                huggingface_hub.hf_hub_url(MODEL_ID, name, revision=MODEL_REVISION)
            )
            _bounded_size(name, metadata.size)
        path = Path(
            huggingface_hub.hf_hub_download(
                MODEL_ID, name, revision=MODEL_REVISION, local_files_only=not args.fetch
            )
        )
        size = _bounded_size(name, path.stat().st_size)
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