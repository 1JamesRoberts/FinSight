"""A fair comparison excludes a whole headline when one target cannot align."""

import json
from pathlib import Path

import pytest

from finsight.sentiment.experiments import target_span_audit
from finsight.sentiment.experiments.target_span_audit import audit_partition


def row(identifier, index, text, target, label="neutral", partition="train"):
    return {
        "record_id": identifier,
        "example_id": f"{identifier}-{index}",
        "partition": partition,
        "headline": text,
        "target": target,
        "label": label,
    }


def test_complete_headline_selection_not_individual_target_selection():
    rows = [
        row("1", 0, "Acme rose", "Acme"),
        row("1", 1, "Acme rose", "Beta"),
        row("2", 0, "Gamma rose; Gamma hired", "Gamma"),
    ]
    result, selected, review = audit_partition(rows, "train")
    assert selected == ["2-0"]
    assert result["counts"]["individually_eligible_annotations"] == 2
    assert result["counts"]["shared_eligible_annotations"] == 1
    assert result["counts"]["excluded_annotations"] == 2
    assert result["counts"]["repeated_target_annotations"] == 1
    assert result["annotation_exclusion_reasons"] == {"no_literal_boundary_match": 1}
    assert len(review) == 2
    assert not review[0]["headline_eligible"]
    assert review[1]["headline_eligible"]
    assert len(set(result["variant_text_sha256"].values())) == 3


def test_eligibility_and_transformed_text_do_not_depend_on_gold_polarity():
    rows = [row("1", 0, "Acme rose", "Acme")]
    first, ids, _ = audit_partition(rows, "train")
    second, other, _ = audit_partition([{**rows[0], "label": "negative"}], "train")
    assert first == second
    assert ids == other


@pytest.mark.parametrize("partition", ["test", "test_candidate", "validation"])
def test_reserved_or_mismatched_partition_rejected(partition):
    with pytest.raises(ValueError, match="partition"):
        audit_partition([row("1", 0, "Acme rose", "Acme")], partition)


def test_duplicate_ids_and_inconsistent_headline_rejected():
    example = row("1", 0, "Acme rose", "Acme")
    with pytest.raises(ValueError, match="duplicate"):
        audit_partition([example, example], "train")
    with pytest.raises(ValueError, match="inconsistent"):
        audit_partition([example, row("1", 1, "Beta fell", "Beta")], "train")


def test_run_keeps_private_text_and_never_opens_frozen_records(tmp_path, monkeypatch):
    def example(partition):
        return {
            **row(partition, 0, "Acme profits rose", "Acme", partition=partition),
            "group_id": partition,
            "annotation_index": 0,
            "multi_target": False,
            "mixed_labels": False,
            "development_only": False,
            "review_reasons": [],
        }

    monkeypatch.setattr(
        target_span_audit,
        "development_inputs",
        lambda root, data: ([example("train")], [example("validation")], {}),
    )
    monkeypatch.setattr(
        target_span_audit.subprocess, "check_output", lambda *a, **kw: "revision\n"
    )
    for name in (
        "data/target_spans.py",
        "sentiment/experiments/target_span_audit.py",
        "sentiment/evaluation/target_data.py",
    ):
        source = tmp_path / "src/finsight" / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("# synthetic source\n")
    (tmp_path / target_span_audit.REPORT.parent).mkdir(parents=True)
    read_bytes = Path.read_bytes

    def guarded_read(path):
        if "benchmarks" in path.parts or path.name == "test-records.jsonl":
            pytest.fail("frozen records must never be opened")
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    result = target_span_audit.run(tmp_path, tmp_path / "data")
    public = json.loads((tmp_path / target_span_audit.REPORT).read_text())
    assert public == result
    assert "Acme" not in json.dumps(public)
    assert result["training_performed"] is False
    assert result["tokenizer_eligibility_verified"] is False
    directory = (
        tmp_path / "data/interim/target-span-audit" / result["private_manifest_sha256"]
    )
    assert json.loads((directory / "membership.json").read_text()) == {
        "train": ["train-0"],
        "validation": ["validation-0"],
    }
    assert target_span_audit.run(tmp_path, tmp_path / "data") == result
    (directory / "membership.json").write_text("changed")
    with pytest.raises(ValueError, match="private audit differs"):
        target_span_audit.run(tmp_path, tmp_path / "data")
