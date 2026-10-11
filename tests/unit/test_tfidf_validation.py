"""The command trains on training rows and exports only aggregate reports."""

import hashlib
import json
from pathlib import Path

import pytest

from finsight.sentiment.__main__ import main
from finsight.sentiment.tfidf import TfidfSentiment
from finsight.sentiment.tfidf_validation import run_tfidf_validation


def records() -> list[dict]:
    rows = []
    for corpus in ("phrasebank", "sentfin"):
        for i, (partition, label, text) in enumerate(
            [
                ("train", "negative", "losses increase"),
                ("train", "negative", "profits decline"),
                ("train", "neutral", "meeting scheduled"),
                ("train", "positive", "profits increase"),
                ("validation", "positive", "validationcanary profits increase"),
                ("validation", "neutral", "validationcanary meeting scheduled"),
            ]
        ):
            native = (
                {"sentence": text, "label": label}
                if corpus == "phrasebank"
                else {
                    "title": text,
                    "sentiments": [{"target": "Example", "label": label}],
                }
            )
            rows.append(
                {
                    "record_id": f"{corpus}/{i}",
                    "corpus": corpus,
                    "partition": partition,
                    "native_record": native,
                }
            )
    return rows


def test_command_trains_only_training_and_persists_private_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "finsight.sentiment.tfidf_validation.load_development_records",
        lambda *args: records(),
    )
    calls = []
    original_fit = TfidfSentiment.fit

    def observed_fit(self: TfidfSentiment, texts: list[str], labels: list[str]) -> None:
        calls.append((texts, labels))
        original_fit(self, texts, labels)

    monkeypatch.setattr(TfidfSentiment, "fit", observed_fit)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "benchmark_manifest_sha256": "synthetic-freeze",
                "parent_split_manifest_sha256": "synthetic-parent",
            }
        )
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"schema_version": 1, "min_df": 1}))
    report = tmp_path / "report.md"
    data = tmp_path / "data"
    monkeypatch.setattr(
        "sys.argv",
        [
            "finsight.sentiment",
            "validate-tfidf-logreg",
            "--data-dir",
            str(data),
            "--freeze-receipt",
            str(receipt),
            "--config",
            str(config),
            "--report-path",
            str(report),
        ],
    )
    main()
    assert len(calls) == 2
    assert all(
        labels == ["negative", "negative", "neutral", "positive"] for _, labels in calls
    )
    assert all("validationcanary" not in text for texts, _ in calls for text in texts)
    run_path = next((data / "interim/tfidf-logreg-validation").glob("*/run.json"))
    run = json.loads(run_path.read_text())
    assert hashlib.sha256(run_path.read_bytes()).hexdigest() == run_path.parent.name
    assert report.with_suffix(".json").read_bytes() == run_path.read_bytes()
    assert (
        hashlib.sha256(run_path.with_name("predictions.jsonl").read_bytes()).hexdigest()
        == run["predictions_sha256"]
    )
    assert run["final_test_evaluated"] is False
    assert run["hyperparameters_tuned"] is False
    assert run["seed"] == 42
    for result in run["metrics"].values():
        assert result["majority_label"] == "negative"
        assert result["training_samples"] == 4
        assert result["logistic_regression"]["samples"] == 2
        model = TfidfSentiment.load(
            data / result["artifact_relative_path"],
            expected_sha256=result["artifact_sha256"],
        )
        assert "validationcanary" not in model.pipeline.named_steps["tfidf"].vocabulary_
    predictions = [
        json.loads(line)
        for line in run_path.with_name("predictions.jsonl").read_text().splitlines()
    ]
    assert {row["record_id"] for row in predictions} == {
        "phrasebank/4",
        "phrasebank/5",
        "sentfin/4",
        "sentfin/5",
    }
    assert json.loads(report.with_suffix(".json").read_text()) == run
    for path in (
        report,
        report.with_suffix(".json"),
        run_path.with_name("predictions.jsonl"),
    ):
        assert "validationcanary" not in path.read_text()
    assert "phrasebank/4" not in report.read_text()
    assert "phrasebank/4" not in report.with_suffix(".json").read_text()

    # Two equivalent fitted models can have different serialized bytes. A new
    # run must preserve the artifact pinned by the earlier run's checksum.
    original = run["metrics"]["phrasebank"]
    old_path = data / original["artifact_relative_path"]
    old_bytes = old_path.read_bytes()
    original_save = TfidfSentiment.save

    def changed_bytes(self: TfidfSentiment, path: Path) -> str:
        original_save(self, path)
        path.write_bytes(path.read_bytes() + b"second-local-serialization")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(TfidfSentiment, "save", changed_bytes)
    later = run_tfidf_validation(data, receipt, config, tmp_path / "later.md")
    assert old_path.read_bytes() == old_bytes
    assert later["metrics"]["phrasebank"]["artifact_relative_path"] != str(
        old_path.relative_to(data)
    )
    assert later["predictions_sha256"] == run["predictions_sha256"]


def test_test_candidate_is_rejected_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = records()
    rows[0]["partition"] = "test_candidate"
    monkeypatch.setattr(
        "finsight.sentiment.tfidf_validation.load_development_records",
        lambda *args: rows,
    )

    def forbidden_fit(*args: object) -> None:
        pytest.fail("training must not start with holdout records")

    monkeypatch.setattr(TfidfSentiment, "fit", forbidden_fit)
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}")
    with pytest.raises(ValueError, match="development records only"):
        run_tfidf_validation(
            tmp_path,
            receipt,
            Path("configs/tfidf-logreg.json"),
            tmp_path / "report.md",
        )


def test_phrasebank_cli_scope_and_balanced_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "finsight.sentiment.tfidf_validation.load_development_records",
        lambda *args: records(),
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "benchmark_manifest_sha256": "synthetic-freeze",
                "parent_split_manifest_sha256": "synthetic-parent",
            }
        )
    )
    config = tmp_path / "balanced.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "min_df": 1,
                "class_weight": "balanced",
            }
        )
    )
    report = tmp_path / "balanced.md"
    data = tmp_path / "data"
    monkeypatch.setattr(
        "sys.argv",
        [
            "finsight.sentiment",
            "validate-tfidf-logreg",
            "--data-dir",
            str(data),
            "--freeze-receipt",
            str(receipt),
            "--config",
            str(config),
            "--cohort",
            "phrasebank",
            "--report-path",
            str(report),
        ],
    )
    main()
    run = json.loads(report.with_suffix(".json").read_text())
    assert set(run["metrics"]) == {"phrasebank"}
    assert run["cohort_selection"] == "phrasebank"
    result = run["metrics"]["phrasebank"]
    assert result["effective_class_weights"] == pytest.approx(
        {
            "negative": 4 / 6,
            "neutral": 4 / 3,
            "positive": 4 / 3,
        }
    )
    assert result["model"]["classifier_parameters"]["class_weight"] == "balanced"
    assert result["training_samples"] == 4
    assert result["logistic_regression"]["samples"] == 2
    assert run["final_test_evaluated"] is False
    assert "balanced" in report.read_text()
    run_path = next((data / "interim/tfidf-logreg-validation").glob("*/run.json"))
    predicted = run_path.with_name("predictions.jsonl").read_text()
    assert "sentfin/" not in predicted
    assert "validationcanary" not in predicted


def test_unknown_cohort_fails_before_reading_data(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cohort"):
        run_tfidf_validation(
            tmp_path,
            tmp_path / "missing",
            tmp_path / "missing",
            tmp_path / "report.md",
            cohort="unknown",
        )


def test_vader_rejects_tfidf_cohort_option(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["finsight.sentiment", "validate-vader", "--cohort", "phrasebank"]
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
