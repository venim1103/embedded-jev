"""Offline tests of exact final-position label tokenization."""

import hashlib
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
from embedded_jev.model_profile import MIMO_PROFILE_PATH, mimo_profile, parse_model_profile


class FakeTokenizer:
    chat_template = "pinned template"
    all_special_ids = [0]

    def convert_tokens_to_ids(self, token):
        return 0 if token == "special" else None

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
    assert report["model"] is None and report["revision"] is None
    assert report["identity_verification"] == "unbound_tokenizer_inputs"
    assert report == probe_label_boundary(FakeTokenizer(), MESSAGES)


def _label_profile_record():
    record = json.loads(MIMO_PROFILE_PATH.read_bytes())
    record["profile_id"] = "synthetic-labels"
    record["source"].update(model="example/Labels", revision="9" * 40)
    template = FakeTokenizer.chat_template.encode()
    record["files"]["chat_template.jinja"] = {
        "bytes": len(template), "sha256": hashlib.sha256(template).hexdigest(),
    }
    record["tokenization"]["special_token_ids"] = {"special": 0}
    record["tokenization"]["label_token_ids"] = {label: ord(label) for label in "ABCDEFGHIJKLMNOP"}
    return record


def test_profile_bound_labels_derive_identity_and_refuse_template_or_token_drift():
    record = _label_profile_record()
    template = FakeTokenizer.chat_template.encode()
    profile = parse_model_profile(json.dumps(record).encode())
    report = probe_label_boundary(FakeTokenizer(), MESSAGES, profile=profile)
    assert report["model"] == "example/Labels" and report["revision"] == "9" * 40
    assert report["profile_id"] == "synthetic-labels" and report["profile_sha256"] == profile.sha256
    record["files"]["chat_template.jinja"]["sha256"] = "0" * 64
    with pytest.raises(LabelProbeError, match="template does not match"):
        probe_label_boundary(FakeTokenizer(), MESSAGES, profile=parse_model_profile(json.dumps(record).encode()))
    record["files"]["chat_template.jinja"]["sha256"] = hashlib.sha256(template).hexdigest()
    record["tokenization"]["label_token_ids"]["B"] = 300
    with pytest.raises(LabelProbeError, match="token ID does not match"):
        probe_label_boundary(FakeTokenizer(), MESSAGES, profile=parse_model_profile(json.dumps(record).encode()))


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


def test_calibration_capture_roundtrip_and_rejects_other_splits(tmp_path):
    import numpy as np

    from embedded_jev.decision_dataset import (
        DecisionDatasetError, load_calibration_capture, load_decision_dataset, save_calibration_capture,
    )

    dataset_path = tmp_path / "decisions.json"
    dataset_path.write_text(json.dumps(split_dataset_smoke()))
    _, digest = load_decision_dataset(dataset_path)
    values = np.random.default_rng(12).normal(size=(2, 12288)).astype(np.float32)
    directory = tmp_path / "capture"
    manifest = save_calibration_capture(
        directory, values, dataset_path=dataset_path, dataset_sha256=digest, case_id="inspect-before-answer",
    )
    loaded, reloaded = load_calibration_capture(directory)
    np.testing.assert_array_equal(loaded, values)
    assert reloaded == manifest and loaded.flags.writeable is False
    assert manifest["dataset"]["purpose"] == "synthetic_split_contract_smoke"
    assert manifest["dataset"]["split"] == "calibration"
    assert manifest["schema_version"] == 2 and manifest["source"]["model"] is None
    assert manifest["source"]["identity_verification"] == "unbound_activation_inputs"
    for case_id in ("edit-authorized", "publish-without-authorization"):
        with pytest.raises(DecisionDatasetError, match="requested split"):
            save_calibration_capture(
                tmp_path / case_id, values, dataset_path=dataset_path, dataset_sha256=digest, case_id=case_id,
            )
        assert not (tmp_path / case_id).exists()
    with pytest.raises(DecisionDatasetError, match="already exists"):
        save_calibration_capture(
            directory, values, dataset_path=dataset_path, dataset_sha256=digest, case_id="inspect-before-answer",
        )
    with (directory / "activations.npy").open("r+b") as destination:
        destination.seek(-1, 2)
        original = destination.read(1)
        destination.seek(-1, 2)
        destination.write(bytes([original[0] ^ 1]))
    with pytest.raises(DecisionDatasetError, match="hash mismatch"):
        load_calibration_capture(directory)
    changed = split_dataset_smoke()
    changed["provenance"]["source"] = "changed corpus"
    dataset_path.write_text(json.dumps(changed))
    with pytest.raises(DecisionDatasetError, match="dataset changed"):
        save_calibration_capture(
            tmp_path / "changed", values, dataset_path=dataset_path,
            dataset_sha256=digest, case_id="inspect-before-answer",
        )
    assert not (tmp_path / "changed").exists()


def test_calibration_capture_binds_its_profile_and_refuses_transplant(tmp_path):
    import numpy as np

    from embedded_jev.decision_dataset import (
        DecisionDatasetError, load_calibration_capture, load_decision_dataset, save_calibration_capture,
    )

    dataset_path = tmp_path / "decisions.json"
    dataset_path.write_text(json.dumps(split_dataset_smoke()))
    _, digest = load_decision_dataset(dataset_path)
    profile = parse_model_profile(json.dumps(_label_profile_record()).encode())
    values = np.zeros((2, 12288), dtype=np.float32)
    directory = tmp_path / "capture"
    manifest = save_calibration_capture(
        directory, values, dataset_path=dataset_path, dataset_sha256=digest,
        case_id="inspect-before-answer", profile=profile,
    )
    loaded, reloaded = load_calibration_capture(directory, profile=profile)
    np.testing.assert_array_equal(loaded, values)
    assert manifest == reloaded and manifest["source"]["model"] == "example/Labels"
    assert manifest["source"]["profile_sha256"] == profile.sha256
    with pytest.raises(DecisionDatasetError, match="does not match a verified model profile"):
        load_calibration_capture(directory)
    with pytest.raises(DecisionDatasetError, match="does not match a verified model profile"):
        load_calibration_capture(directory, profile=mimo_profile())


def test_calibration_legacy_readability_and_mixed_profile_refusal(tmp_path):
    import numpy as np

    from embedded_jev.decision_dataset import (
        DecisionDatasetError, load_balanced_calibration_captures, load_calibration_capture,
        load_decision_dataset, save_calibration_capture,
    )

    dataset_path = tmp_path / "decisions.json"
    dataset_path.write_text(json.dumps(split_dataset_smoke()))
    _, digest = load_decision_dataset(dataset_path)
    values = np.zeros((2, 12288), dtype=np.float32)
    profile = mimo_profile()
    bound, unbound = tmp_path / "bound", tmp_path / "unbound"
    manifest = save_calibration_capture(
        bound, values, dataset_path=dataset_path, dataset_sha256=digest,
        case_id="inspect-before-answer", profile=profile,
    )
    save_calibration_capture(
        unbound, values, dataset_path=dataset_path, dataset_sha256=digest, case_id="inspect-before-answer",
    )
    with pytest.raises(DecisionDatasetError, match="different model profiles"):
        load_balanced_calibration_captures([bound, unbound])
    with pytest.raises(DecisionDatasetError, match="unbound calibration capture"):
        load_calibration_capture(unbound, profile=profile)
    legacy = {
        **{key: manifest[key] for key in ("tensor", "transform", "shape", "dataset", "array")},
        "schema_version": 1, "format": "single_case_mimo_calibration_activations",
        "model": profile.model, "revision": profile.revision,
    }
    (bound / "manifest.json").write_text(json.dumps(legacy))
    loaded, reloaded = load_calibration_capture(bound, profile=profile)
    np.testing.assert_array_equal(loaded, values)
    assert reloaded == legacy
    legacy["revision"] = "main"
    (bound / "manifest.json").write_text(json.dumps(legacy))
    with pytest.raises(DecisionDatasetError, match="unsupported legacy"):
        load_calibration_capture(bound)


def test_streamed_capture_refuses_validation_and_held_out_before_model_import(tmp_path):
    from embedded_jev.inventory import InventoryError
    from embedded_jev.streamed_text import run_streamed_text

    for split in ("validation", "held_out"):
        with pytest.raises(InventoryError, match="BF16 calibration split"):
            run_streamed_text(
                None, layers=4, prompt="unused", dataset_path=tmp_path / "decisions.json",
                split=split, case_id="example", calibration_output=tmp_path / "capture",
            )
    with pytest.raises(InventoryError, match="activation observer requires"):
        run_streamed_text(
            None, layers=4, prompt="unused", dataset_path=tmp_path / "decisions.json",
            split="held_out", case_id="example", activation_observer=lambda values: None,
        )


@pytest.mark.parametrize("dtype", ["fp32", "ggml_bf16_rhs", "ggml_bf16_rhs_qk"])
def test_streamed_precision_diagnostics_refuse_unsafe_scope_before_model_import(tmp_path, dtype):
    from embedded_jev.inventory import InventoryError
    from embedded_jev.streamed_text import make_native_ffn_down, run_streamed_text

    for wrapper_dtype in ("fp32", "fp16"):
        with pytest.raises(InventoryError, match="existing frozen projection"):
            make_native_ffn_down(None, tmp_path / "native.so", compute_dtype=wrapper_dtype)
    with pytest.raises(InventoryError, match="unsupported streamed computation dtype"):
        run_streamed_text(None, prompt="unused", compute_dtype="fp16")
    for split in ("calibration", "validation", "held_out"):
        with pytest.raises(InventoryError, match="synthetic fixture"):
            run_streamed_text(None, prompt="unused", dataset_path=tmp_path / "decisions.json",
                              split=split, case_id="example", compute_dtype=dtype,
                              precision_trace_directory=tmp_path / "trace")
    for options in ({"layers": 3}, {"calibration_output": tmp_path / "capture"},
                    {"activation_observer": lambda values: None}):
        with pytest.raises(InventoryError, match="synthetic fixture"):
            run_streamed_text(None, prompt="unused", fixture_path=tmp_path / "fixture.json", case_id="example",
                              compute_dtype=dtype, precision_trace_directory=tmp_path / "trace", **options)
    with pytest.raises(InventoryError, match="existing frozen native projection"):
        run_streamed_text(None, prompt="unused", fixture_path=tmp_path / "fixture.json", case_id="example",
                          compute_dtype=dtype)
    with pytest.raises(InventoryError, match="destination already exists"):
        run_streamed_text(None, prompt="unused", fixture_path=tmp_path / "fixture.json", case_id="example",
                          precision_trace_directory=tmp_path)
    assert not (tmp_path / "trace").exists()


def test_precision_trace_comparison_is_bounded_and_reports_observed_errors(tmp_path):
    import numpy as np

    from embedded_jev.inventory import InventoryError
    from embedded_jev.streamed_text import compare_precision_traces

    records = [{"name": name, "shape": shape} for name, shape in {
        "model.input_embed": [3, 32], "l_out-0": [3, 32], "l_out-1": [3, 32], "l_out-2": [3, 32],
        "ffn_input": [3, 256], "ffn_output": [3, 32], "l_out-3": [3, 32], "final_norm": [1, 32],
    }.items()]
    for name in ("left", "right"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "manifest.json").write_text(json.dumps({
            "format": "jev-prefill-f32-trace-v1", "dtype": "float32", "byte_order": "little", "tensors": records,
        }))
        for record in records:
            np.ones(record["shape"], dtype="<f4").tofile(directory / (record["name"] + ".f32"))
    left, right = tmp_path / "left", tmp_path / "right"
    assert compare_precision_traces(left, right)["first_nonidentical_stage"] is None
    values = np.ones((3, 32), dtype="<f4")
    values[-1, 0] = 3
    values.tofile(left / "l_out-0.f32")
    report = compare_precision_traces(left, right)
    assert report["first_nonidentical_stage"] == "l_out-0"
    stage = next(stage for stage in report["stages"] if stage["name"] == "l_out-0")
    assert stage["max_abs_error"] == stage["last_token_max_abs_error"] == 2
    assert stage["rmse"] == pytest.approx(2 / np.sqrt(96))
    assert stage["relative_rmse"] == stage["rmse"]
    values[0, 0] = np.nan
    values.tofile(left / "l_out-0.f32")
    with pytest.raises(InventoryError, match="nonfinite"):
        compare_precision_traces(left, right)
    (left / "l_out-0.f32").write_bytes(b"short")
    with pytest.raises(InventoryError, match="size"):
        compare_precision_traces(left, right)


@pytest.mark.parametrize("defect", ["version", "split", "shape", "size", "pickle"])
def test_calibration_capture_rejects_unsafe_manifest_or_arrays(tmp_path, defect):
    import hashlib

    import numpy as np

    from embedded_jev.decision_dataset import (
        DecisionDatasetError, load_calibration_capture, load_decision_dataset, save_calibration_capture,
    )

    dataset_path = tmp_path / "decisions.json"
    dataset_path.write_text(json.dumps(split_dataset_smoke()))
    _, digest = load_decision_dataset(dataset_path)
    directory = tmp_path / "capture"
    manifest = save_calibration_capture(
        directory, np.ones((2, 12288), dtype=np.float32), dataset_path=dataset_path,
        dataset_sha256=digest, case_id="inspect-before-answer",
    )
    if defect == "version":
        manifest["schema_version"] = True
    elif defect == "split":
        manifest["dataset"]["split"] = "held_out"
    elif defect == "shape":
        manifest["shape"] = [129, 12288]
    elif defect == "size":
        with (directory / "activations.npy").open("r+b") as destination:
            destination.truncate(manifest["array"]["bytes"] + 1)
    else:
        np.save(directory / "activations.npy", np.array([object()], dtype=object), allow_pickle=True)
        array_bytes = (directory / "activations.npy").read_bytes()
        manifest["array"] = {"bytes": len(array_bytes), "sha256": hashlib.sha256(array_bytes).hexdigest()}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(DecisionDatasetError):
        load_calibration_capture(directory)


def test_pairwise_evaluator_reports_regressions_without_fitting(tmp_path, monkeypatch):
    import math
    from dataclasses import replace

    from embedded_jev import evaluation
    from embedded_jev.decision_dataset import DecisionDatasetError, load_decision_dataset
    from embedded_jev.model_profile import profile_source

    path = tmp_path / "decisions.json"
    path.write_text(json.dumps(split_dataset_smoke()))
    dataset, digest = load_decision_dataset(path)
    case = dataset["splits"]["held_out"][0]
    library = tmp_path / "kernel.so"
    library.write_bytes(b"mock library")
    identity = {"manifest_sha256": "0" * 64}
    monkeypatch.setattr(evaluation, "_candidate_identity", lambda directory: dict(identity))
    source_identity = evaluation._source_identity()
    original_sources = dict(source_identity)
    monkeypatch.setattr(evaluation, "_source_identity", lambda: dict(source_identity))
    runtime_versions = evaluation._runtime_versions()
    original_versions = dict(runtime_versions)
    monkeypatch.setattr(evaluation, "_runtime_versions", lambda: dict(runtime_versions))
    dependencies = {"test_dependency": {"sha256": "6" * 64}}
    original_dependencies = deepcopy(dependencies)
    monkeypatch.setattr(evaluation, "native_backend_dependencies", lambda library, backend: deepcopy(dependencies))
    calls = []
    defect = None
    profile = mimo_profile()

    def scorer(snapshot, **kwargs):
        calls.append(kwargs)
        native = "native_ffn_library" in kwargs
        probabilities = (0.6, 0.2, 0.2) if native else (0.1, 0.2, 0.7)
        options = [
            {"id": option["id"], "label": label, "token_id": 32 + index,
             "row_sha256": "0" * 64, "conditional_probability": probability, "logit": math.log(probability)}
            for index, (option, label, probability) in enumerate(zip(case["options"], "ABC", probabilities, strict=True))
        ]
        report = {
            **profile_source(profile), "generated_tokens": 0, "layers": 32,
            "prompt_sha256": "1" * 64, "ffn_down_input_sha256": "3" * 64, "tokens": 82,
            "dataset": {"sha256": digest, "split": "held_out", "case_id": case["id"]},
            "native_ffn_down": {
                "candidate_origin": "saved_hash_checked_native_fixture", "backend": kwargs.get("native_ffn_backend"),
                "backend_dependencies": deepcopy(dependencies),
            } if native else {},
            "decision": {"options": options, "chosen_option_id": "edit" if native else "ask"},
        }
        if native and defect == "generation":
            report["generated_tokens"] = 1
        elif native and defect == "prompt":
            report["prompt_sha256"] = "2" * 64
        elif native and defect == "mapping":
            options[0]["token_id"] = 99
        elif native and defect == "normalization":
            options[0]["conditional_probability"] = 0.1
        elif native and defect == "logits":
            options[0]["logit"] = 10.0
        elif native and defect == "tokens":
            report["tokens"] = 83
        elif native and defect == "activation":
            report["ffn_down_input_sha256"] = "4" * 64
        elif native and defect == "case":
            report["dataset"]["case_id"] = "different-case"
        elif native and defect == "profile":
            report["profile_sha256"] = "0" * 64
        elif not native and defect == "profile_record":
            monkeypatch.setattr(evaluation, "mimo_profile", lambda: replace(profile, sha256="f" * 64))
        elif not native and defect == "dataset":
            changed = deepcopy(dataset)
            changed["provenance"]["source"] = "changed after reference run"
            path.write_text(json.dumps(changed))
        elif not native and defect == "candidate":
            identity["manifest_sha256"] = "2" * 64
        elif not native and defect == "kernel":
            library.write_bytes(b"changed library")
        elif not native and defect == "source":
            source_identity["streamed_text.py"] = "5" * 64
        elif not native and defect == "runtime":
            runtime_versions["numpy"] = "changed"
        elif not native and defect == "dependency":
            dependencies["test_dependency"]["sha256"] = "changed"
        elif native and defect == "backend_report":
            report["native_ffn_down"]["backend"] = "incorrect"
        return report

    monkeypatch.setattr(evaluation, "run_streamed_text", scorer)
    report = evaluation.compare_dataset_cases(
        tmp_path, dataset_path=path, split="held_out", case_ids=[case["id"]],
        native_library=library, candidate=tmp_path / "candidate",
    )
    assert report["quantizer_fitting"] is False and report["generated_tokens"] == 0
    assert report["profile_id"] == profile.profile_id and report["profile_sha256"] == profile.sha256
    assert report["dataset"]["purpose"] == "synthetic_split_contract_smoke"
    assert report["runtime_versions"]["python"]
    assert len(report["scoring_source_sha256"]["streamed_text.py"]) == 64
    assert report["summary"] == {"cases": 1, "bf16_expected_matches": 1, "native_expected_matches": 0, "changed_choices": 1}
    assert report["cases"][0]["conditional_total_variation"] == pytest.approx(0.5)
    assert len(calls) == 2 and all(call["split"] == "held_out" for call in calls)
    assert calls[1]["native_ffn_backend"] == report["native_backend"] == "direct"
    assert "calibration_output" not in calls[0] and "calibration_output" not in calls[1]
    for defect in ("generation", "prompt", "mapping", "normalization", "logits", "tokens", "activation", "case", "profile", "profile_record", "dataset", "candidate", "kernel", "source", "runtime", "dependency", "backend_report"):
        path.write_text(json.dumps(dataset))
        identity["manifest_sha256"] = "0" * 64
        source_identity.update(original_sources)
        runtime_versions.update(original_versions)
        dependencies.clear()
        dependencies.update(deepcopy(original_dependencies))
        monkeypatch.setattr(evaluation, "mimo_profile", lambda: profile)
        library.write_bytes(b"mock library")
        with pytest.raises(DecisionDatasetError):
            evaluation.compare_dataset_cases(
                tmp_path, dataset_path=path, split="held_out", case_ids=[case["id"]],
                native_library=library, candidate=tmp_path / "candidate",
            )


@pytest.mark.parametrize("case_ids", [[], ["repeat", "repeat"], [str(index) for index in range(5)]])
def test_pairwise_evaluation_refuses_unbounded_or_duplicate_case_requests(tmp_path, case_ids):
    from embedded_jev.decision_dataset import DecisionDatasetError
    from embedded_jev.evaluation import compare_dataset_cases

    library = tmp_path / "kernel.so"
    library.write_bytes(b"mock library")
    with pytest.raises(DecisionDatasetError, match="1-4 unique case IDs"):
        compare_dataset_cases(
            tmp_path, dataset_path=tmp_path / "not-read.json", split="held_out", case_ids=case_ids,
            native_library=library, candidate=tmp_path / "not-read-candidate",
        )


def test_public_intent_proxy_preserves_official_split_roles_and_attribution(tmp_path):
    from embedded_jev.decision_dataset import load_decision_dataset
    from embedded_jev.public_data import _build_clinc_proxy

    source = {
        split: [[f"{split} request {index}", f"intent_{index}"] for index in range(4)]
        for split in ("train", "val", "test")
    }
    proxy = _build_clinc_proxy(source, per_split=2)
    assert proxy == _build_clinc_proxy(source, per_split=2)
    assert proxy["purpose"] == "public_intent_proxy" and proxy["provenance"]["license"] == "CC-BY-3.0"
    for origin, target in (("train", "calibration"), ("val", "validation"), ("test", "held_out")):
        assert len(proxy["splits"][target]) == 2
        assert all(case["id"].startswith(f"clinc150_{origin}_") for case in proxy["splits"][target])
        assert all(len(case["options"]) == 4 for case in proxy["splits"][target])
        assert all(case["expected_option_id"] in {option["id"] for option in case["options"]} for case in proxy["splits"][target])
    path = tmp_path / "decisions.json"
    path.write_text(json.dumps(proxy))
    loaded, _ = load_decision_dataset(path)
    assert loaded == proxy


@pytest.mark.parametrize("headers", [
    {"Content-Length": "99999999"}, {"Content-Length": "invalid"}, {"Content-Encoding": "gzip"},
])
def test_public_source_reader_rejects_bad_headers_before_body(monkeypatch, headers):
    from embedded_jev import public_data

    class Response:
        status = 200
        read_calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def geturl(self):
            return public_data.CLINC_URL + "README.md"

        def read(self, length):
            self.read_calls += 1
            pytest.fail("read an unbounded or encoded source response")

    response = Response()
    response.headers = headers
    monkeypatch.setattr(public_data, "urlopen", lambda request, timeout: response)
    with pytest.raises(public_data.PublicDataError):
        public_data._fetch_source_file("README.md")
    assert response.read_calls == 0


@pytest.mark.parametrize("count,seed", [(0, 902), (33, 902), (True, 902), (4, True)])
def test_public_proxy_selection_rejects_unbounded_requests(count, seed):
    from embedded_jev.public_data import PublicDataError, _build_clinc_proxy

    with pytest.raises(PublicDataError, match="1-32 cases"):
        _build_clinc_proxy({}, per_split=count, seed=seed)


def test_public_bundle_rejects_changed_source_without_creating_output(tmp_path, monkeypatch):
    from embedded_jev import public_data

    directory = tmp_path / "bundle"
    monkeypatch.setattr(public_data, "_fetch_source_file", lambda name: b"changed source")
    with pytest.raises(public_data.PublicDataError, match="SHA-256 mismatch"):
        public_data.prepare_clinc_proxy(directory)
    assert not directory.exists()
    directory.mkdir()
    monkeypatch.setattr(public_data, "_fetch_source_file", lambda name: pytest.fail("overwriting or refetching existing bundle"))
    with pytest.raises(public_data.PublicDataError, match="already exists"):
        public_data.prepare_clinc_proxy(directory)


def test_balanced_calibration_caps_tokens_preserves_provenance_and_rejects_mixing(tmp_path):
    import numpy as np

    from embedded_jev.decision_dataset import (
        DecisionDatasetError, load_balanced_calibration_captures, load_decision_dataset,
        save_calibration_capture,
    )

    dataset = split_dataset_smoke()
    extra = deepcopy(dataset["splits"]["calibration"][0])
    extra.update(id="second-training-case", group="second-training-group", state="Another calibration request.")
    dataset["splits"]["calibration"].append(extra)
    path = tmp_path / "decisions.json"
    path.write_text(json.dumps(dataset))
    _, digest = load_decision_dataset(path)
    directories = [tmp_path / "first", tmp_path / "second"]
    for index, directory in enumerate(directories):
        values = np.full((4 + index * 2, 12288), index + 1, dtype=np.float32)
        save_calibration_capture(
            directory, values, dataset_path=path, dataset_sha256=digest,
            case_id=dataset["splits"]["calibration"][index]["id"],
        )
    combined, manifest = load_balanced_calibration_captures(directories, max_tokens=6)
    assert combined.shape == (6, 12288) and not combined.flags.writeable
    assert manifest["dataset"]["purpose"] == "synthetic_split_contract_smoke"
    assert [record["selected_rows"] for record in manifest["captures"]] == [3, 3]
    assert manifest["captures"][0]["token_indices"] == [0, 1, 3]
    assert manifest["captures"][1]["token_indices"] == [0, 2, 5]
    np.testing.assert_array_equal(combined[:3], 1)
    np.testing.assert_array_equal(combined[3:], 2)
    with pytest.raises(DecisionDatasetError, match="repeated calibration"):
        load_balanced_calibration_captures([directories[0], directories[0]])
    for count in (1, 129, True):
        with pytest.raises(DecisionDatasetError, match="at most 128 tokens"):
            load_balanced_calibration_captures(directories, max_tokens=count)
    changed = json.loads((directories[1] / "manifest.json").read_text())
    changed["dataset"]["sha256"] = "f" * 64
    (directories[1] / "manifest.json").write_text(json.dumps(changed))
    with pytest.raises(DecisionDatasetError, match="different datasets"):
        load_balanced_calibration_captures(directories)