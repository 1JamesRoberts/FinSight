"""Four SVM fits use matching training/validation rows without exporting text."""

import json
from pathlib import Path

import pytest

from finsight.sentiment.__main__ import main
from finsight.sentiment.svm import TfidfSVM
from finsight.sentiment.tfidf_validation import run_tfidf_validation


def examples() -> list[dict]:
    rows = []
    for corpus in ("phrasebank", "sentfin"):
        for i, (partition, label, text) in enumerate(
            [
                ("train", "negative", "losses rise"),
                ("train", "negative", "profits fall"),
                ("train", "neutral", "meeting scheduled"),
                ("train", "positive", "profits rise"),
                ("validation", "positive", "validationcanary profits rise"),
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
    rows.append({"corpus": "fiqa", "partition": "train"})
    rows.append(
        {
            "corpus": "sentfin",
            "partition": "train",
            "native_record": {
                "sentiments": [{"label": "positive"}, {"label": "negative"}],
            },
        }
    )
    return rows


@pytest.mark.parametrize("weight", [None, "balanced"])
def test_svm_cli_trains_both_cohorts_and_preserves_score_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    weight: str | None,
) -> None:
    monkeypatch.setattr(
        "finsight.sentiment.tfidf_validation.load_development_records",
        lambda *args: examples(),
    )
    seen = []
    original_fit = TfidfSVM.fit

    def observed(self: TfidfSVM, texts: list[str], labels: list[str]) -> None:
        seen.append((texts, labels))
        original_fit(self, texts, labels)

    monkeypatch.setattr(TfidfSVM, "fit", observed)
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
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "min_df": 1,
                "max_iter": 10000,
                "class_weight": weight,
            }
        )
    )
    report = tmp_path / "report.md"
    data = tmp_path / "data"
    monkeypatch.setattr(
        "sys.argv",
        [
            "finsight.sentiment",
            "validate-tfidf-svm",
            "--config",
            str(config),
            "--data-dir",
            str(data),
            "--freeze-receipt",
            str(receipt),
            "--report-path",
            str(report),
        ],
    )
    main()
    assert len(seen) == 2
    assert all(
        labels == ["negative", "negative", "neutral", "positive"] for _, labels in seen
    )
    assert all("validationcanary" not in text for texts, _ in seen for text in texts)
    run = json.loads(report.with_suffix(".json").read_text())
    assert set(run["metrics"]) == {"phrasebank", "sentfin_single_target"}
    assert run["model_kind"] == "linear_svm"
    assert run["final_test_evaluated"] is False
    for result in run["metrics"].values():
        assert result["training_samples"] == 4
        assert result["linear_svm"]["samples"] == 2
        assert result["majority_label"] == "negative"
        assert result["model"]["classifier_parameters"]["class_weight"] == weight
        expected = (
            {"negative": 2 / 3, "neutral": 4 / 3, "positive": 4 / 3}
            if weight
            else dict.fromkeys(("negative", "neutral", "positive"), 1.0)
        )
        assert result["effective_class_weights"] == pytest.approx(expected)
        model = TfidfSVM.load(
            data / result["artifact_relative_path"],
            expected_sha256=result["artifact_sha256"],
        )
        assert "validationcanary" not in model.pipeline.named_steps["tfidf"].vocabulary_
    local = next((data / "interim/tfidf-svm-validation").glob("*/predictions.jsonl"))
    predictions = [json.loads(line) for line in local.read_text().splitlines()]
    assert {row["record_id"] for row in predictions} == {
        "phrasebank/4",
        "phrasebank/5",
        "sentfin/4",
        "sentfin/5",
    }
    assert all("decision_scores" in row["prediction"] for row in predictions)
    assert all("probabilities" not in row["prediction"] for row in predictions)
    for path in (local, report, report.with_suffix(".json")):
        assert "validationcanary" not in path.read_text()
    assert "phrasebank/4" not in report.read_text()
    assert "phrasebank/4" not in report.with_suffix(".json").read_text()


def test_holdout_is_rejected_before_svm_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = examples()
    rows[0]["partition"] = "test_candidate"
    monkeypatch.setattr(
        "finsight.sentiment.tfidf_validation.load_development_records",
        lambda *args: rows,
    )

    def forbidden(*args: object) -> None:
        pytest.fail("SVM fitting must not start with holdout records")

    monkeypatch.setattr(TfidfSVM, "fit", forbidden)
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}")
    with pytest.raises(ValueError, match="development records only"):
        run_tfidf_validation(
            tmp_path,
            receipt,
            Path("configs/tfidf-svm.json"),
            tmp_path / "report.md",
            model_kind="linear_svm",
        )


def test_unknown_model_fails_before_loading(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="classifier"):
        run_tfidf_validation(
            tmp_path,
            tmp_path / "missing",
            tmp_path / "missing",
            tmp_path / "report.md",
            model_kind="unknown",
        )
