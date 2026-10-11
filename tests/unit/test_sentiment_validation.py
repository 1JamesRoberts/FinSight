"""Check metrics, development cohorts, and validation-only experiment logging."""

import json
from pathlib import Path

import pytest

from finsight.sentiment.__main__ import main
from finsight.sentiment.validation import classification_metrics, sentence_cohorts


def test_metrics_have_fixed_axes_and_zero_division() -> None:
    result = classification_metrics(
        ["negative", "neutral", "positive"], ["negative", "positive", "positive"]
    )
    assert result["confusion_matrix"] == [[1, 0, 0], [0, 0, 1], [0, 0, 1]]
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx(5 / 9)
    assert result["per_class"]["neutral"]["precision"] == 0
    absent = classification_metrics(["neutral"], ["neutral"])
    assert absent["macro_f1"] == pytest.approx(1 / 3)
    assert absent["per_class"]["negative"]["support"] == 0


@pytest.mark.parametrize(
    "gold,predicted",
    [
        ([], []),
        (["neutral"], []),
        (["unknown"], ["positive"]),
        (["positive"], ["unknown"]),
    ],
)
def test_invalid_metrics_fail(gold: list[str], predicted: list[str]) -> None:
    with pytest.raises(ValueError):
        classification_metrics(gold, predicted)


def row(identifier: str, corpus: str, partition: str, label: str, text: str) -> dict:
    native = (
        {"sentence": text, "label": label}
        if corpus == "phrasebank"
        else {
            "title": text,
            "sentiments": [{"target": "Example", "label": label}],
        }
    )
    return {
        "record_id": identifier,
        "corpus": corpus,
        "partition": partition,
        "native_record": native,
    }


def test_cohort_selection_excludes_fiqa_and_multiple_targets() -> None:
    records = [
        row("a", "phrasebank", "validation", "positive", "good"),
        row("b", "sentfin", "validation", "positive", "good"),
        row("c", "sentfin", "train", "negative", "bad"),
    ]
    records[2]["native_record"]["sentiments"].append(
        {"target": "Other", "label": "positive"}
    )
    records.append({"corpus": "fiqa", "partition": "validation"})
    cohorts = sentence_cohorts(records)
    assert [r["record_id"] for r in cohorts["phrasebank"]] == ["a"]
    assert [r["record_id"] for r in cohorts["sentfin_single_target"]] == ["b"]
    records.append({"corpus": "sentfin", "partition": "test_candidate"})
    with pytest.raises(ValueError, match="development records only"):
        sentence_cohorts(records)


def test_validation_command_uses_training_majority_and_logs_no_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = []
    for corpus in ("phrasebank", "sentfin"):
        for i, (partition, label, text) in enumerate(
            [
                ("train", "negative", "Training canary never exported"),
                ("train", "negative", "Other training canary"),
                ("validation", "positive", "The outlook is good."),
                ("validation", "neutral", "The meeting is on Tuesday."),
            ]
        ):
            records.append(row(f"{corpus}/{i}", corpus, partition, label, text))
    monkeypatch.setattr(
        "finsight.sentiment.validation.load_development_records", lambda *args: records
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "benchmark_manifest_sha256": "synthetic-benchmark",
                "parent_split_manifest_sha256": "synthetic-parent",
            }
        )
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "negative_threshold": -0.05,
                "positive_threshold": 0.05,
            }
        )
    )
    report = tmp_path / "report.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "finsight.sentiment",
            "validate-vader",
            "--data-dir",
            str(tmp_path / "data"),
            "--freeze-receipt",
            str(receipt),
            "--config",
            str(config),
            "--report-path",
            str(report),
        ],
    )
    main()
    run_path = next((tmp_path / "data/interim/vader-validation").glob("*/run.json"))
    run = json.loads(run_path.read_text())
    assert run["final_test_evaluated"] is False
    assert run["thresholds_tuned"] is False
    assert all(r["majority_label"] == "negative" for r in run["metrics"].values())
    assert all(r["vader"]["samples"] == 2 for r in run["metrics"].values())
    predicted = [
        json.loads(line)
        for line in run_path.with_name("predictions.jsonl").read_text().splitlines()
    ]
    assert {r["record_id"] for r in predicted} == {
        "phrasebank/2",
        "phrasebank/3",
        "sentfin/2",
        "sentfin/3",
    }
    assert "Training canary" not in run_path.read_text()
    assert "The outlook" not in run_path.with_name("predictions.jsonl").read_text()
    assert "phrasebank/2" not in report.read_text()
    assert json.loads(report.with_suffix(".json").read_text()) == run
    assert "Training canary" not in report.with_suffix(".json").read_text()
    assert run["model"]["license_notice_sha256"]
