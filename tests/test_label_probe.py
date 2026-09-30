"""Offline tests of exact final-position label tokenization."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from embedded_jev.label_probe import (
    LabelProbeError,
    NON_THINKING_SUFFIX,
    _bounded_size,
    load_decision_fixture,
    probe_decision_cases,
    probe_label_boundary,
    probe_text_processor,
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
    assert _bounded_size("processor_config.json", 1191, processor=True) == 1191
    for name, size in (
        ("tokenizer.json", 26 << 20),
        ("model.safetensors", 8),
        ("config.json", None),
        ("chat_template.jinja", 0),
        ("processor_config.json", 1191),
    ):
        with pytest.raises(LabelProbeError, match="missing or oversized"):
            _bounded_size(name, size)
    with pytest.raises(LabelProbeError, match="missing or oversized"):
        _bounded_size("video_preprocessor_config.json", 2 << 20, processor=True)


def test_text_processor_must_match_prompt_ids_and_contain_no_media():
    class TextProcessor:
        def __call__(self, *, text, return_tensors):
            assert text.endswith(NON_THINKING_SUFFIX)
            assert return_tensors is None
            return {"input_ids": [[3, 4]], "attention_mask": [[1, 1]]}

    report = probe_text_processor(TextProcessor(), FakeTokenizer(), MESSAGES)
    assert report["processor_class"] == "TextProcessor"
    assert report["processor_text_only"] is True
    assert report["processor_mm_token_type_ids"] == "absent"

    class TypedTextProcessor(TextProcessor):
        def __call__(self, **kwargs):
            return {**super().__call__(**kwargs), "mm_token_type_ids": [[0, 0]]}

    assert probe_text_processor(TypedTextProcessor(), FakeTokenizer(), MESSAGES)[
        "processor_mm_token_type_ids"
    ] == "all_text"

    class MediaProcessor(TextProcessor):
        def __call__(self, **kwargs):
            return {**super().__call__(**kwargs), "pixel_values": [[0.0]]}

    with pytest.raises(LabelProbeError, match="contain media"):
        probe_text_processor(MediaProcessor(), FakeTokenizer(), MESSAGES)

    class MediaTokenProcessor(TypedTextProcessor):
        def __call__(self, **kwargs):
            return {**super().__call__(**kwargs), "mm_token_type_ids": [[0, 1]]}

    with pytest.raises(LabelProbeError, match="contain media"):
        probe_text_processor(MediaTokenProcessor(), FakeTokenizer(), MESSAGES)

    class RetokenizingProcessor(TextProcessor):
        def __call__(self, **kwargs):
            return {"input_ids": [[3, 9]], "attention_mask": [[1, 1]]}

    with pytest.raises(LabelProbeError, match="disagree with tokenizer"):
        probe_text_processor(RetokenizingProcessor(), FakeTokenizer(), MESSAGES)


class FixtureTokenizer:
    chat_template = "fixture template"
    all_special_ids = [0]

    def apply_chat_template(self, messages, **options):
        assert len(messages) == 1
        assert options["add_generation_prompt"] is True
        assert options["enable_thinking"] is False
        prompt = (
            f"<|im_start|>user\n{messages[0]['content']}<|im_end|>"
            + NON_THINKING_SUFFIX
        )
        if options["tokenize"]:
            return {"input_ids": self.encode(prompt, add_special_tokens=False)}
        return prompt

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return list(text.encode("utf-8"))


def test_synthetic_agent_tool_fixture_maps_stable_ids_across_option_orders():
    fixture, digest = load_decision_fixture(
        Path(__file__).parent / "fixtures" / "agent_tool_smoke.json"
    )
    assert len(digest) == 64
    assert fixture["schema_version"] == 1
    assert fixture["purpose"] == "synthetic_engineering_smoke_not_calibration_or_benchmark"
    reports = probe_decision_cases(FixtureTokenizer(), fixture["cases"])
    assert [report["expected_label"] for report in reports] == ["A", "A", "B", "C", "C"]
    assert all(set(report["label_token_ids"]) == {"A", "B", "C"} for report in reports)
    assert len({report["prompt_sha256"] for report in reports}) == len(reports)

    invalid = deepcopy(fixture["cases"])
    invalid[1]["id"] = invalid[0]["id"]
    with pytest.raises(LabelProbeError, match="duplicate or invalid"):
        probe_decision_cases(FixtureTokenizer(), invalid)
    invalid = deepcopy(fixture["cases"])
    invalid[0]["expected_option_id"] = "unknown"
    with pytest.raises(LabelProbeError, match="unknown expected choice"):
        probe_decision_cases(FixtureTokenizer(), invalid)


def test_fixture_loader_rejects_unsupported_schema_and_oversized_file(tmp_path):
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps({"schema_version": 2, "purpose": "synthetic", "cases": []}))
    with pytest.raises(LabelProbeError, match="schema or purpose"):
        load_decision_fixture(path)
    path.write_text("x" * (64 * 1024 + 1))
    with pytest.raises(LabelProbeError, match="exceeds 64 KiB"):
        load_decision_fixture(path)


def split_dataset_smoke():
    fixture, _ = load_decision_fixture(Path(__file__).parent / "fixtures" / "agent_tool_smoke.json")
    return {
        "schema_version": 1,
        "purpose": "synthetic_split_contract_smoke",
        "provenance": {"source": "repository synthetic engineering fixture", "license": "MIT"},
        "splits": {
            "calibration": [deepcopy(fixture["cases"][0])],
            "validation": [deepcopy(fixture["cases"][2])],
            "held_out": [deepcopy(fixture["cases"][4])],
        },
    }


def test_split_dataset_keeps_provenance_and_requires_explicit_case_split(tmp_path):
    from embedded_jev.decision_dataset import (
        DecisionDatasetError, load_decision_dataset, select_dataset_case,
    )

    dataset = split_dataset_smoke()
    path = tmp_path / "decisions.json"
    path.write_text(json.dumps(dataset))
    loaded, digest = load_decision_dataset(path)
    assert loaded == dataset and len(digest) == 64
    assert loaded["purpose"] == "synthetic_split_contract_smoke"
    case = select_dataset_case(loaded, split="held_out", case_id="publish-without-authorization")
    assert case["expected_option_id"] == "ask"
    with pytest.raises(DecisionDatasetError, match="requested split"):
        select_dataset_case(loaded, split="calibration", case_id=case["id"])
    with pytest.raises(DecisionDatasetError, match="unknown decision dataset split"):
        select_dataset_case(loaded, split="test", case_id=case["id"])


@pytest.mark.parametrize("leak", ["id", "group", "prompt"])
def test_split_dataset_rejects_cross_split_leakage(tmp_path, leak):
    from embedded_jev.decision_dataset import DecisionDatasetError, load_decision_dataset

    dataset = split_dataset_smoke()
    if leak == "id":
        dataset["splits"]["held_out"][0]["id"] = dataset["splits"]["calibration"][0]["id"]
    elif leak == "group":
        dataset["splits"]["held_out"][0]["group"] = dataset["splits"]["calibration"][0]["group"]
    else:
        copy = deepcopy(dataset["splits"]["calibration"][0])
        copy["id"] = "renamed-held-out-case"
        copy["group"] = "different-group"
        copy["state"] = "  " + copy["state"].replace(" ", "  ")
        copy["options"].reverse()
        dataset["splits"]["held_out"] = [copy]
    path = tmp_path / "decisions.json"
    path.write_text(json.dumps(dataset))
    with pytest.raises(DecisionDatasetError, match="splits"):
        load_decision_dataset(path)


def test_streamed_dataset_requires_explicit_split_and_exclusive_input(tmp_path):
    from embedded_jev.inventory import InventoryError
    from embedded_jev.streamed_text import run_streamed_text

    dataset_path = tmp_path / "decisions.json"
    with pytest.raises(InventoryError, match="explicit split"):
        run_streamed_text(None, prompt="unused", dataset_path=dataset_path, case_id="example")
    with pytest.raises(InventoryError, match="valid split and case id"):
        run_streamed_text(None, prompt="unused", dataset_path=dataset_path, split="validation")
    with pytest.raises(InventoryError, match="either a decision dataset"):
        run_streamed_text(
            None, prompt="unused", dataset_path=dataset_path, split="held_out",
            case_id="example", fixture_path=tmp_path / "fixture.json",
        )


@pytest.mark.parametrize("defect", ["version", "purpose", "provenance", "split", "cases", "duplicate_key", "oversized"])
def test_split_dataset_rejects_invalid_or_oversized_schema(tmp_path, defect):
    from embedded_jev.decision_dataset import DecisionDatasetError, load_decision_dataset

    dataset = split_dataset_smoke()
    if defect == "version":
        dataset["schema_version"] = True
    elif defect == "purpose":
        dataset["purpose"] = "calibrated_and_certified"
    elif defect == "provenance":
        dataset["provenance"]["license"] = " "
    elif defect == "split":
        del dataset["splits"]["held_out"]
    elif defect == "cases":
        dataset["splits"]["held_out"] = []
    payload = json.dumps(dataset)
    if defect == "duplicate_key":
        payload = '{"schema_version": 2, ' + payload[1:]
    elif defect == "oversized":
        payload = " " * ((1 << 20) + 1)
    path = tmp_path / "decisions.json"
    path.write_text(payload)
    with pytest.raises(DecisionDatasetError):
        load_decision_dataset(path)