"""Synthetic tokenization checks; no model, source news or hub access."""

import json
from pathlib import Path

import pytest

from finsight.data.target_tokenization import audit_pairs
from finsight.sentiment.experiments import target_format_preparation as preparation
from finsight.sentiment.experiments.target_span_audit import audit_partition

ROOT = Path(__file__).resolve().parents[2]


class CharacterTokenizer:
    """One token per character makes exact budgets and lost markers testable."""

    padding_side = "right"
    truncation_side = "right"
    unk_token_id = 3

    def __len__(self):
        return 10000

    def num_special_tokens_to_add(self, pair):
        return 4 if pair else 2

    def build_inputs_with_special_tokens(self, first, second):
        return [1, *first, 2, 2, *second, 2]

    def decode(self, ids, **kwargs):
        return "".join(chr(i - 10) for i in ids)

    def __call__(self, texts, text_pair=None, **kwargs):
        ids = [[ord(c) + 10 for c in text] for text in texts]
        if text_pair is None:
            return {"input_ids": ids}
        sequences = []
        for first, second in zip(ids, text_pair, strict=True):
            budget = kwargs["max_length"] - len(first) - 4
            sequences.append(
                self.build_inputs_with_special_tokens(
                    first, [ord(c) + 10 for c in second[:budget]]
                )
            )
        width = max(map(len, sequences))
        return {
            "input_ids": [row + [0] * (width - len(row)) for row in sequences],
            "attention_mask": [
                [1] * len(row) + [0] * (width - len(row)) for row in sequences
            ],
        }


def example(partition, identifier, text, target):
    return {
        "record_id": identifier,
        "example_id": f"{identifier}-{target}",
        "group_id": identifier,
        "partition": partition,
        "headline": text,
        "target": target,
        "label": "neutral",
        "annotation_index": 0,
        "multi_target": False,
        "mixed_labels": False,
        "development_only": False,
        "review_reasons": [],
    }


def test_full_preservation_and_exact_budget_boundary():
    tokenizer = CharacterTokenizer()
    rows = audit_pairs(tokenizer, tokenizer, [("A", "B"), ("A", "BC")], max_length=6)
    assert rows[0]["eligible"]
    assert rows[0]["full_pair_tokens"] == 6
    assert rows[1]["reasons"] == ["full_pair_exceeds_max_length"]


def test_target_overflow_not_silently_truncated():
    tokenizer = CharacterTokenizer()
    result = audit_pairs(tokenizer, tokenizer, [("AB", "C")], max_length=6)[0]
    assert "target_leaves_no_headline_budget" in result["reasons"]
    assert not result["eligible"]


def test_slow_fast_mismatch_aborts_instead_of_changing_membership():
    class DifferentTokenizer(CharacterTokenizer):
        def __call__(self, *args, **kwargs):
            output = super().__call__(*args, **kwargs)
            output["input_ids"][0][0] += 1
            return output

    with pytest.raises(ValueError, match="slow/fast content"):
        audit_pairs(
            CharacterTokenizer(), DifferentTokenizer(), [("A", "B")], max_length=16
        )


def test_roundtrip_changes_rejected():
    class BrokenDecode(CharacterTokenizer):
        def decode(self, ids, **kwargs):
            return "different text"

    result = audit_pairs(
        BrokenDecode(), CharacterTokenizer(), [("A", "B")], max_length=16
    )[0]
    assert result["reasons"] == ["decoded_text_differs"]


def test_unknown_token_rejected_even_when_pair_budget_fits():
    class UnknownTokenizer(CharacterTokenizer):
        def __call__(self, *args, **kwargs):
            output = super().__call__(*args, **kwargs)
            output["input_ids"] = [
                [
                    self.unk_token_id if token == ord("A") + 10 else token
                    for token in row
                ]
                for row in output["input_ids"]
            ]
            return output

        def decode(self, ids, **kwargs):
            return "".join(
                "?" if token == self.unk_token_id else chr(token - 10) for token in ids
            )

    tokenizer = UnknownTokenizer()
    result = audit_pairs(tokenizer, tokenizer, [("A", "B")], max_length=16)[0]
    assert not result["eligible"]
    assert "unknown_token" in result["reasons"]


def test_invalid_paired_attention_mask_rejected():
    class BrokenMask(CharacterTokenizer):
        def __call__(self, texts, text_pair=None, **kwargs):
            output = super().__call__(texts, text_pair, **kwargs)
            if text_pair is not None:
                output["attention_mask"][0][0] = 0
            return output

    with pytest.raises(ValueError, match="preserve target"):
        audit_pairs(CharacterTokenizer(), BrokenMask(), [("A", "B")], max_length=16)


@pytest.mark.parametrize(
    "maximum,pairs", [(3, [("A", "B")]), (16, []), (16, [("", "B")])]
)
def test_invalid_requests_rejected(maximum, pairs):
    with pytest.raises(ValueError):
        audit_pairs(
            CharacterTokenizer(), CharacterTokenizer(), pairs, max_length=maximum
        )


def test_padding_policy_checked():
    tokenizer = CharacterTokenizer()
    tokenizer.padding_side = "left"
    with pytest.raises(ValueError, match="right"):
        audit_pairs(tokenizer, tokenizer, [("A", "B")], max_length=16)


def test_one_format_overflows_and_excludes_whole_headline_for_every_arm():
    tokenizer = CharacterTokenizer()
    rows = [
        example("train", "one", "A", "A"),
        example("train", "two", "ABCDEFGHIJKL Beta", "ABCDEFGHIJKL"),
        example("train", "two", "ABCDEFGHIJKL Beta", "Beta"),
    ]
    report, retained, rejected = preparation.audit_partition_views(
        rows, "train", tokenizer, tokenizer, 34
    )
    assert [r["example"]["record_id"] for r in retained] == ["one"]
    assert len(rejected) == 2
    assert report["additional_excluded_headlines"] == 1
    assert report["additional_excluded_annotations"] == 2
    assert (
        report["variants"]["target_pair"]["individually_token_eligible_annotations"]
        == 3
    )
    assert (
        report["variants"]["target_markers"]["individually_token_eligible_annotations"]
        == 1
    )
    assert retained[0]["example"]["target"] == "A"
    assert retained[0]["inputs"]["target_replacement"]["target"] == "[TARGET]"


def test_declared_protocol_has_nine_fresh_runs_and_no_test_access():
    plan, files = preparation.load_plan(ROOT)
    assert len(plan["trials"]) == 9
    assert not plan["frozen_test_access"]
    assert not plan["train_validation_combined"]
    assert "model.safetensors" not in files
    assert set(plan["seeds"]) == {42, 43, 44}


def test_frozen_plan_resolves_moved_audit_and_still_rejects_corruption(tmp_path):
    """Moving evidence must preserve both the protocol bytes and checksum guard."""
    plan_body = (ROOT / preparation.PLAN).read_bytes()
    plan = json.loads(plan_body)
    for name in ("backbone_plan_path", "training_config_path"):
        path = tmp_path / plan[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / plan[name]).read_bytes())
    (tmp_path / preparation.PLAN).write_bytes(plan_body)
    moved = tmp_path / preparation.REPORT.parent / "target-span-audit.json"
    moved.parent.mkdir(parents=True, exist_ok=True)
    moved.write_bytes(
        preparation.plan_reference(ROOT, plan["span_audit_path"]).read_bytes()
    )
    preparation.load_plan(tmp_path)
    assert (tmp_path / preparation.PLAN).read_bytes() == plan_body
    moved.write_text("corrupted")
    with pytest.raises(ValueError, match="reference differs"):
        preparation.load_plan(tmp_path)


def test_original_plan_reference_takes_precedence_over_migrated_copy(tmp_path):
    old = "reports/phase-04-target-sentiment/target-span-audit.json"
    original = tmp_path / old
    original.parent.mkdir(parents=True)
    original.write_text("original")
    moved = original.parent / "evaluation" / original.name
    moved.parent.mkdir()
    moved.write_text("moved")
    assert preparation.plan_reference(tmp_path, old) == original
    # Unrelated missing paths must not be redirected by the compatibility rule.
    other = "configs/missing.json"
    assert preparation.plan_reference(tmp_path, other) == tmp_path / other


def test_preparation_lock_private_bytes_and_corruption_guards(tmp_path, monkeypatch):
    from finsight.sentiment.artifacts import file_sha256

    train = [example("train", "one", "Acme rose", "Acme")]
    validation = [example("validation", "two", "Beta fell", "Beta")]
    source = tmp_path / "src/finsight"
    for name in preparation.SOURCES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic source\n")
    (tmp_path / "configs").mkdir()
    (tmp_path / preparation.PLAN).write_text("{}")
    (tmp_path / "uv.lock").write_text("synthetic lock")
    report_path = "reports/phase-04-target-sentiment/target-span-audit.json"
    (tmp_path / report_path).parent.mkdir(parents=True)
    span = {
        "inputs": {},
        "source_sha256": {},
        "partitions": {
            p: audit_partition(rows, p)[0]
            for p, rows in (("train", train), ("validation", validation))
        },
    }
    (tmp_path / report_path).write_text(json.dumps(span))
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "config.json").write_text("{}")
    files = {
        "config.json": {
            "size_bytes": 2,
            "sha256": file_sha256(tokenizer_dir / "config.json"),
        }
    }
    plan = {
        "span_audit_path": report_path,
        "max_length": 128,
        "experiment": "synthetic-study",
        "asset": {"model_id": "synthetic-model", "revision": "revision"},
    }
    monkeypatch.setattr(preparation, "load_plan", lambda root: (plan, files))
    monkeypatch.setattr(
        preparation,
        "load_tokenizers",
        lambda *args: (CharacterTokenizer(), CharacterTokenizer()),
    )
    monkeypatch.setattr(
        preparation, "development_inputs", lambda *args: (train, validation, {})
    )
    monkeypatch.setattr(preparation, "version", lambda name: "synthetic-version")
    monkeypatch.setattr(
        preparation.subprocess, "check_output", lambda *args, **kwargs: "revision\n"
    )
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if "benchmarks" in path.parts or path.name in {
            "model.safetensors",
            "test-records.jsonl",
        }:
            pytest.fail("preparation must not access frozen records or weights")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    report = preparation.prepare(tmp_path, tmp_path / "data", tokenizer_dir)
    assert not report["training_performed"]
    assert "Acme" not in json.dumps(report)
    assert (
        preparation.check_preparation(tmp_path, tmp_path / "data", tokenizer_dir)
        == report
    )
    directory = tmp_path / "data/interim/target-format" / report["data_manifest_sha256"]
    row = json.loads((directory / "train.jsonl").read_text())
    assert row["example"]["target"] == "Acme"
    assert row["inputs"]["target_replacement"]["target"] == "[TARGET]"
    with pytest.raises(FileExistsError):
        preparation.prepare(tmp_path, tmp_path / "data", tokenizer_dir)
    (directory / "train.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        preparation.check_preparation(tmp_path, tmp_path / "data", tokenizer_dir)
