"""FinBERT runner cannot train, score reserved rows, or leak text into reports."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from finsight.sentiment.__main__ import main
from finsight.sentiment.finbert import FinBertConfig, probability_prediction
from finsight.sentiment.finbert_validation import run_finbert_validation, saved_controls
from finsight.sentiment.validation import classification_metrics


def baseline_fixture(rows: list[dict], receipt: dict) -> dict:
    candidates, selected = [], {}
    for cohort, corpus in (
        ("phrasebank", "phrasebank"),
        ("sentfin_single_target", "sentfin"),
    ):
        validation = [
            row
            for row in rows
            if row["corpus"] == corpus
            and row["partition"] == "validation"
            and (corpus == "phrasebank" or len(row["native_record"]["sentiments"]) == 1)
        ]
        membership = hashlib.sha256(
            json.dumps([row["record_id"] for row in validation]).encode()
        ).hexdigest()
        score = classification_metrics(
            ["positive", "neutral"], ["positive", "positive"]
        )
        selected[cohort] = {}
        for kind in ("logistic_regression", "linear_svm"):
            run = {
                **receipt,
                "final_test_evaluated": False,
                "metrics": {
                    cohort: {
                        kind: score,
                        "vader": score,
                        "training_majority": score,
                        "validation_membership_sha256": membership,
                    }
                },
            }
            identifier = hashlib.sha256(
                (json.dumps(run, indent=2, sort_keys=True) + "\n").encode()
            ).hexdigest()
            candidates.append({"run_id": identifier, "model_kind": kind, "run": run})
            selected[cohort][kind] = identifier
    return {"candidates": candidates, "selected_run_ids": selected}


@pytest.fixture
def experiment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    rows = []
    for corpus in ("phrasebank", "sentfin"):
        for index, (partition, text, label) in enumerate(
            [
                ("train", "TRAININGCANARY", "negative"),
                ("validation", "VALIDATIONCANARY first", "positive"),
                ("validation", "VALIDATIONCANARY second", "neutral"),
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
                    "corpus": corpus,
                    "record_id": f"{corpus}/{index}",
                    "partition": partition,
                    "native_record": native,
                }
            )
    rows.extend(
        [
            {"corpus": "fiqa", "record_id": "fiqa", "partition": "validation"},
            {
                "corpus": "sentfin",
                "record_id": "multi",
                "partition": "validation",
                "native_record": {"title": "MULTICANARY", "sentiments": [{}, {}]},
            },
        ]
    )
    receipt = {
        "benchmark_manifest_sha256": "synthetic-freeze",
        "parent_split_manifest_sha256": "synthetic-parent",
    }
    paths = {
        name: tmp_path / (name + ".json") for name in ("receipt", "baseline", "config")
    }
    paths["receipt"].write_text(json.dumps(receipt))
    paths["baseline"].write_text(json.dumps(baseline_fixture(rows, receipt)))
    paths["config"].write_text(
        json.dumps({"schema_version": 1, **asdict(FinBertConfig(device="cpu"))})
    )
    monkeypatch.setattr(
        "finsight.sentiment.finbert_validation.load_development_records",
        lambda *args: rows,
    )
    monkeypatch.setattr(
        "finsight.sentiment.finbert_validation.version", lambda name: "test"
    )
    return {
        **paths,
        "rows": rows,
        "data": tmp_path / "data",
        "report": tmp_path / "report.md",
    }


def test_cli_scores_only_validation_and_preserves_hashed_private_outputs(
    experiment: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = []

    class Model:
        def __init__(self, *args: object) -> None:
            pass

        def synchronize(self) -> None:
            pass

        def reset_peak_memory(self) -> None:
            pass

        def predict_batch(self, texts: list[str]) -> list:
            seen.extend(texts)
            return [probability_prediction([1.0, 0.0, 0.0], 7, 128) for _ in texts]

        def metadata(self) -> dict:
            return {
                "checkpoint_files": {
                    "pytorch_model.bin": {"sha256": "test", "size_bytes": 4}
                },
                "checkpoint_bytes": 4,
                "use_decision": "synthetic",
                "architecture": {
                    "class": "Synthetic",
                    "layers": 1,
                    "hidden_size": 2,
                    "attention_heads": 1,
                    "intermediate_size": 4,
                    "vocabulary_size": 5,
                    "total_parameters": 10,
                    "gradient_enabled_parameters": 0,
                    "parameter_bytes": 40,
                },
                "device": "cpu",
                "gpu": None,
                "cuda_runtime": None,
                "peak_gpu_allocated_bytes": None,
                "peak_gpu_reserved_bytes": None,
            }

    monkeypatch.setattr("finsight.sentiment.finbert_validation.FinBertSentiment", Model)
    monkeypatch.setattr(
        "sys.argv",
        [
            "sentiment",
            "validate-finbert",
            "--data-dir",
            str(experiment["data"]),
            "--freeze-receipt",
            str(experiment["receipt"]),
            "--config",
            str(experiment["config"]),
            "--baseline-report",
            str(experiment["baseline"]),
            "--report-path",
            str(experiment["report"]),
        ],
    )
    main()
    assert len(seen) == 5  # One invented warm-up, then four validation rows.
    assert all(
        "TRAININGCANARY" not in text and "MULTICANARY" not in text for text in seen
    )
    path = next((experiment["data"] / "interim/finbert-validation").glob("*/run.json"))
    run = json.loads(path.read_bytes())
    assert run["final_test_evaluated"] is False
    assert run["fine_tuned_in_project"] is False
    assert run["training_records_scored"] is False
    assert run["hyperparameters_tuned"] is False
    assert hashlib.sha256(path.read_bytes()).hexdigest() == path.parent.name
    assert experiment["report"].with_suffix(".json").read_bytes() == path.read_bytes()
    predictions = path.with_name("predictions.jsonl").read_bytes()
    assert hashlib.sha256(predictions).hexdigest() == run["predictions_sha256"]
    assert {json.loads(line)["record_id"] for line in predictions.splitlines()} == {
        "phrasebank/1",
        "phrasebank/2",
        "sentfin/1",
        "sentfin/2",
    }
    for output in (experiment["report"], experiment["report"].with_suffix(".json")):
        assert "VALIDATIONCANARY" not in output.read_text()
        assert "phrasebank/1" not in output.read_text()
    assert "supervision overlap" in experiment["report"].read_text()
    assert path.with_name("acquisition-receipt.json").exists()


def test_baseline_tampering_fails_before_model_load(
    experiment: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = json.loads(experiment["baseline"].read_text())
    baseline["candidates"][0]["run"]["final_test_evaluated"] = True
    experiment["baseline"].write_text(json.dumps(baseline))
    monkeypatch.setattr(
        "finsight.sentiment.finbert_validation.FinBertSentiment",
        lambda *args: pytest.fail("must reject lineage before acquisition"),
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        run_finbert_validation(
            experiment["data"],
            experiment["receipt"],
            experiment["config"],
            experiment["report"],
            experiment["baseline"],
        )


def test_reserved_rows_fail_before_model_load(
    experiment: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment["rows"].append({"partition": "test_candidate"})
    monkeypatch.setattr(
        "finsight.sentiment.finbert_validation.FinBertSentiment",
        lambda *args: pytest.fail("must reject reserved rows first"),
    )
    with pytest.raises(ValueError, match="development records only"):
        run_finbert_validation(
            experiment["data"],
            experiment["receipt"],
            experiment["config"],
            experiment["report"],
            experiment["baseline"],
        )


def test_saved_baselines_reject_different_membership(experiment: dict) -> None:
    baseline = json.loads(experiment["baseline"].read_text())
    receipt = json.loads(experiment["receipt"].read_text())
    with pytest.raises(ValueError, match="membership differs"):
        saved_controls(baseline, receipt, "phrasebank", "different-membership")
    receipt["parent_split_manifest_sha256"] = "different-parent"
    with pytest.raises(ValueError, match="lineage differs"):
        saved_controls(baseline, receipt, "phrasebank", "different-membership")
