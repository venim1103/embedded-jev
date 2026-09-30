"""Pinned small public intent proxy, not an official benchmark or agent-tool dataset."""

import argparse
import hashlib
import json
import random
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from embedded_jev.decision_dataset import load_decision_dataset
from embedded_jev.inventory import _json_object


CLINC_REVISION = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
CLINC_URL = f"https://raw.githubusercontent.com/clinc/oos-eval/{CLINC_REVISION}/"
CLINC_DATA_SHA256 = "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0"
MAX_SOURCE_JSON_BYTES = 3 << 20
SOURCE_FILES = {
    "data/data_full.json": (2495390, "7a7b26c5f2dfbbf213f3e67d2dd0727e1af545aa"),
    "LICENSE": (19467, "1a16e05564d2aaa880bbe9e506a0a0226d8742cc"),
    "README.md": (2668, "ba0060730a2cdd7c05ac8635252b92f7db14b8de"),
}
SOURCE_SPLITS = {"train": "calibration", "val": "validation", "test": "held_out"}


class PublicDataError(ValueError):
    """A public source or proxy request does not match the pinned contract."""


def _fetch_source_file(name: str) -> bytes:
    expected_bytes, expected_blob = SOURCE_FILES[name]
    request = Request(CLINC_URL + name, headers={"Accept-Encoding": "identity"})
    try:
        with urlopen(request, timeout=30) as response:
            if response.status != 200 or urlsplit(response.geturl()).scheme != "https":
                raise PublicDataError("unexpected public dataset response")
            if response.headers.get("Content-Encoding", "identity") != "identity":
                raise PublicDataError("encoded public dataset response is unsupported")
            length = response.headers.get("Content-Length")
            if length is not None and (not length.isdecimal() or int(length) != expected_bytes):
                raise PublicDataError("public source length disagrees with pinned metadata")
            data = response.read(expected_bytes + 1)
    except (HTTPError, URLError) as exc:
        raise PublicDataError(f"unable to fetch pinned public source: {exc}") from exc
    blob = hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()
    if len(data) != expected_bytes or blob != expected_blob:
        raise PublicDataError("public source byte count or Git blob hash mismatch")
    return data


def _build_clinc_proxy(source: dict, *, per_split: int = 4, seed: int = 902) -> dict:
    if type(per_split) is not int or not 1 <= per_split <= 32 or type(seed) is not int:
        raise PublicDataError("proxy requires 1-32 cases per split and an integer seed")
    if not isinstance(source, dict) or not set(SOURCE_SPLITS) <= set(source):
        raise PublicDataError("missing CLINC source splits")
    for name in SOURCE_SPLITS:
        rows = source[name]
        if (
            not isinstance(rows, list) or not per_split <= len(rows) <= 30000
            or any(not isinstance(row, list) or len(row) != 2
                   or any(not isinstance(value, str) or not value.strip() or len(value) > 2000
                          for value in row) for row in rows)
        ):
            raise PublicDataError("unsupported CLINC intent rows")
    intents = sorted({row[1] for row in source["train"]})
    if len(intents) < 4 or any({row[1] for row in source[name]} != set(intents) for name in SOURCE_SPLITS):
        raise PublicDataError("CLINC intent sets disagree across source splits")
    generator = random.Random(seed)
    seen_utterances = set()
    splits = {}
    for source_split, split in SOURCE_SPLITS.items():
        indices = list(range(len(source[source_split])))
        generator.shuffle(indices)
        cases = []
        for index in indices:
            text, expected = source[source_split][index]
            normalized = " ".join(text.split())
            group = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if group in seen_utterances:
                continue
            seen_utterances.add(group)
            choices = [expected] + generator.sample([intent for intent in intents if intent != expected], 3)
            generator.shuffle(choices)
            cases.append({
                "id": f"clinc150_{source_split}_{index}", "group": f"utterance-{group}",
                "state": f"User request: {text}",
                "question": "Which listed intent best matches the user request?",
                "options": [{"id": intent, "description": f"Intent: {intent.replace('_', ' ')}"} for intent in choices],
                "expected_option_id": expected,
            })
            if len(cases) == per_split:
                break
        if len(cases) != per_split:
            raise PublicDataError("not enough distinct utterances for the requested proxy")
        splits[split] = cases
    return {
        "schema_version": 1, "purpose": "public_intent_proxy",
        "provenance": {
            "source": (
                f"CLINC150, Larson et al. (EMNLP 2019), https://aclanthology.org/D19-1131/; "
                f"{CLINC_URL}data/data_full.json; four-choice in-scope proxy; seed={seed}; "
                f"cases_per_split={per_split}; train->calibration, val->validation, test->held_out; OOS excluded"
            ),
            "license": "CC-BY-3.0",
        },
        "splits": splits,
    }


def prepare_clinc_proxy(directory: Path, *, per_split: int = 4, seed: int = 902) -> dict:
    """Explicitly fetch at most 2.6 MB of pinned data/attribution into one bundle."""
    if directory.exists():
        raise PublicDataError("public proxy bundle already exists")
    if type(per_split) is not int or not 1 <= per_split <= 32 or type(seed) is not int:
        raise PublicDataError("invalid public proxy selection bounds")
    files = {name: _fetch_source_file(name) for name in SOURCE_FILES}
    if hashlib.sha256(files["data/data_full.json"]).hexdigest() != CLINC_DATA_SHA256:
        raise PublicDataError("pinned CLINC content SHA-256 mismatch")
    source = _json_object(files["data/data_full.json"], "CLINC data", max_bytes=MAX_SOURCE_JSON_BYTES)
    if {name: len(source.get(name, [])) for name in SOURCE_SPLITS} != {"train": 15000, "val": 3000, "test": 4500}:
        raise PublicDataError("pinned CLINC split counts changed")
    proxy = _build_clinc_proxy(source, per_split=per_split, seed=seed)
    directory.mkdir(parents=True)
    records = {}
    for name, data in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        records[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    path = directory / "decisions.json"
    path.write_text(json.dumps(proxy, sort_keys=True), encoding="utf-8")
    _, digest = load_decision_dataset(path)
    manifest = {
        "revision": CLINC_REVISION, "files": records, "dataset_sha256": digest,
        "purpose": "four_choice_public_intent_proxy_not_official_clinc150_or_agent_quality",
        "dataset": str(path), "transfer_body_bytes": sum(len(data) for data in files.values()),
    }
    (directory / "source_manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a pinned attributed four-choice public intent proxy")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases-per-split", type=int, default=4)
    parser.add_argument("--seed", type=int, default=902)
    args = parser.parse_args()
    print(json.dumps(prepare_clinc_proxy(args.output, per_split=args.cases_per_split, seed=args.seed), sort_keys=True))


if __name__ == "__main__":
    main()