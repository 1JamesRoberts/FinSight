"""Test benchmark membership, immutable freezing, and development isolation."""

import hashlib
import json
from pathlib import Path

import pytest

from finsight.data.benchmark import (
    curate_benchmark,
    freeze_benchmark,
    load_development_records,
    render_benchmark_report,
)
from finsight.data.splits import SourceExample, make_split_plan

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def benchmark_inputs() -> tuple[dict, list[dict], dict]:
    split_policy = json.loads((ROOT / "configs/split-policy.json").read_text())
    policy = json.loads((ROOT / "configs/benchmark-policy.json").read_text())
    policy["curation_exclusions"] = {}
    review = {
        "summary": {
            "near_method": split_policy["near_duplicates"],
            "manifest_sha256": "synthetic",
            "revisions": {},
        },
        "exact_groups": [],
        "near_pairs": [],
    }
    examples = []
    for i, labels in enumerate(
        [
            ["negative"],
            ["neutral"],
            ["positive"],
            ["positive", "negative"],
            ["positive"],
            ["neutral"],
        ]
    ):
        text = f"Invented headline {i}"
        annotations = tuple((f"Entity {j}", label) for j, label in enumerate(labels))
        examples.append(
            SourceExample(
                "sentfin",
                str(i),
                text,
                annotations,
                {
                    "title": text,
                    "sentiments": [
                        {"target": target, "label": label}
                        for target, label in annotations
                    ],
                },
            )
        )
    plan, rows = make_split_plan(examples, review, split_policy)
    for decision, row in zip(plan["sources"], rows, strict=True):
        partition = "train" if row["record_id"] == "sentfin/4" else "test_candidate"
        row["partition"] = decision["partition"] = partition
    return plan, rows, policy


def test_curated_membership_preserves_targets_and_excludes_whole_groups(
    benchmark_inputs: tuple[dict, list[dict], dict],
) -> None:
    plan, rows, policy = benchmark_inputs
    # Both members of an excluded duplicate group must be withheld.
    rows[5]["group_id"] = plan["sources"][5]["group_id"] = rows[2]["group_id"]
    policy["curation_exclusions"] = {"sentfin/2": "malformed_target_name"}
    selected, summary = curate_benchmark(plan, rows, policy)
    assert {row["record_id"] for row in selected} == {
        "sentfin/0",
        "sentfin/1",
        "sentfin/3",
    }
    assert summary["curation_excluded_candidate_records"] == 2
    assert summary["curation_excluded_groups"] == 1


def test_cohorts_keep_mixed_target_gold_without_inventing_sentence_labels(
    benchmark_inputs: tuple[dict, list[dict], dict],
) -> None:
    plan, rows, policy = benchmark_inputs
    selected, summary = curate_benchmark(plan, rows, policy)
    assert len(selected) == 5
    assert summary["target_annotations"] == 6
    assert summary["cohorts"]["single_target"]["headlines"] == 4
    assert summary["cohorts"]["mixed_target_sentiment"]["target_annotations"] == 2
    assert (
        selected[3]["native_record"]["sentiments"]
        == rows[3]["native_record"]["sentiments"]
    )
    assert selected == curate_benchmark(plan, rows[::-1], policy)[0]


@pytest.mark.parametrize("issue", ["flag", "gold", "selection", "unknown_exclusion"])
def test_curation_fails_on_bad_inputs(
    benchmark_inputs: tuple[dict, list[dict], dict], issue: str
) -> None:
    plan, rows, policy = benchmark_inputs
    if issue == "flag":
        rows[0]["review_reasons"] = ["cross_corpus_overlap"]
    elif issue == "gold":
        rows[0]["native_record"]["sentiments"][0]["label"] = "invented"
    elif issue == "selection":
        policy["selection"] = "choose after model evaluation"
    else:
        policy["curation_exclusions"] = {"unknown": "bad"}
    with pytest.raises(ValueError):
        curate_benchmark(plan, rows, policy)


def test_freeze_reproduces_rejects_changes_and_isolates_development(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    benchmark_inputs: tuple[dict, list[dict], dict],
) -> None:
    plan, rows, policy = benchmark_inputs
    data = tmp_path / "data"
    records_body = "".join(json.dumps(row) + "\n" for row in rows)
    plan["prepared_records_sha256"] = hashlib.sha256(records_body.encode()).hexdigest()
    plan_body = json.dumps(plan) + "\n"
    digest = hashlib.sha256(plan_body.encode()).hexdigest()
    split_dir = data / "processed/splits" / digest
    split_dir.mkdir(parents=True)
    (split_dir / "split-manifest.json").write_text(plan_body)
    (split_dir / "prepared-records.jsonl").write_text(records_body)
    monkeypatch.setattr(
        "finsight.data.benchmark.prepare_splits",
        lambda *args, **kwargs: {
            "directory": str(split_dir),
            "split_manifest_sha256": digest,
        },
    )
    assets = tmp_path / "assets.json"
    assets.write_text(
        json.dumps(
            {
                "assets": {
                    "prosusai_finbert": {
                        "repository": "synthetic",
                        "revision": "synthetic",
                    }
                }
            }
        )
    )
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    receipt_path = tmp_path / "freeze.json"
    receipt = freeze_benchmark(assets, policy_path, policy_path, data, receipt_path)
    assert (
        freeze_benchmark(assets, policy_path, policy_path, data, receipt_path)
        == receipt
    )
    assert [
        row["record_id"] for row in load_development_records(data, receipt_path)
    ] == ["sentfin/4"]
    report = render_benchmark_report(receipt)
    assert "Invented headline" not in report
    assert "sentfin/0" not in report
    original = receipt_path.read_bytes()
    policy_path.write_text(json.dumps(policy | {"benchmark_id": "changed"}))
    with pytest.raises(ValueError, match="never overwrite"):
        freeze_benchmark(assets, policy_path, policy_path, data, receipt_path)
    assert receipt_path.read_bytes() == original
    frozen_dir = data / "processed/benchmarks" / receipt["benchmark_manifest_sha256"]
    (frozen_dir / "test-records.jsonl").write_text("corruption")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_development_records(data, receipt_path)
    policy_path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="never overwrite"):
        freeze_benchmark(assets, policy_path, policy_path, data, receipt_path)
