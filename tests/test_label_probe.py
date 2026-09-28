"""Offline tests of exact final-position label tokenization."""

import pytest

from embedded_jev.label_probe import (
    LabelProbeError,
    NON_THINKING_SUFFIX,
    _bounded_size,
    probe_label_boundary,
)


class FakeTokenizer:
    chat_template = "pinned template"
    all_special_ids = [0]

    def apply_chat_template(self, messages, **options):
        assert messages == [{"role": "user", "content": "Choice?"}]
        assert options["add_generation_prompt"] is True
        assert options["enable_thinking"] is False
        assert options["tokenize"] in (False, True)
        if options["tokenize"]:
            return {"input_ids": [3, 4], "attention_mask": [1, 1]}
        return "<|im_start|>user\nChoice?<|im_end|>" + NON_THINKING_SUFFIX

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        prefix = self.apply_chat_template(
            [{"role": "user", "content": "Choice?"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if text == prefix:
            return [3, 4]
        return [3, 4, ord(text.removeprefix(prefix))]


MESSAGES = [{"role": "user", "content": "Choice?"}]


def test_labels_extend_the_non_thinking_prefix_without_generation():
    report = probe_label_boundary(FakeTokenizer(), MESSAGES)
    assert report["label_token_ids"] == {label: ord(label) for label in "ABCDEFGHIJKLMNOP"}
    assert report["prompt_token_count"] == 2
    assert report["generated_tokens"] == 0
    assert report == probe_label_boundary(FakeTokenizer(), MESSAGES)


def test_rejects_ambiguous_or_unsupported_label_boundaries():
    class RetokenizingTokenizer(FakeTokenizer):
        def encode(self, text, *, add_special_tokens):
            ids = super().encode(text, add_special_tokens=add_special_tokens)
            return ids if len(ids) == 2 else [9, *ids[1:]]

    with pytest.raises(LabelProbeError, match="exact prompt"):
        probe_label_boundary(RetokenizingTokenizer(), MESSAGES)

    class DuplicateLabelTokenizer(FakeTokenizer):
        def encode(self, text, *, add_special_tokens):
            ids = super().encode(text, add_special_tokens=add_special_tokens)
            return ids if len(ids) == 2 else [*ids[:2], 100]

    with pytest.raises(LabelProbeError, match="shares a token ID"):
        probe_label_boundary(DuplicateLabelTokenizer(), MESSAGES)

    class DifferentTemplateEncoding(FakeTokenizer):
        def apply_chat_template(self, messages, **options):
            value = super().apply_chat_template(messages, **options)
            return {"input_ids": [4, 3]} if options["tokenize"] else value

    with pytest.raises(LabelProbeError, match="template tokenization"):
        probe_label_boundary(DifferentTemplateEncoding(), MESSAGES)

    class WrongTemplateTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, **options):
            return "<|im_start|>assistant\n<think>"

    with pytest.raises(LabelProbeError, match="assistant prefix"):
        probe_label_boundary(WrongTemplateTokenizer(), MESSAGES)
    with pytest.raises(LabelProbeError, match="prefix of A-P"):
        probe_label_boundary(FakeTokenizer(), MESSAGES, ("A", "C"))


def test_tokenizer_file_allowlist_enforces_per_file_caps():
    assert _bounded_size("tokenizer.json", 19_989_325) == 19_989_325
    for name, size in (
        ("tokenizer.json", 26 << 20),
        ("model.safetensors", 8),
        ("config.json", None),
        ("chat_template.jinja", 0),
    ):
        with pytest.raises(LabelProbeError, match="missing or oversized"):
            _bounded_size(name, size)