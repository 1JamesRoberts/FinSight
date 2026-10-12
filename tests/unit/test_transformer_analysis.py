"""Probability diagnostics and private review boundaries, using synthetic inputs."""

import json

import pytest

from finsight.sentiment.transformer_analysis import (
    calibration_metrics,
    review_sample,
    summarize_annotations,
)


def row(gold, probabilities):
    return {
        "gold": gold,
        "prediction": max(probabilities, key=probabilities.get),
        "probabilities": probabilities,
    }


def test_perfect_predictions_and_empty_bins():
    rows = [row("positive", dict(negative=0.0, neutral=0.0, positive=1.0))]
    result = calibration_metrics(rows)
    assert result["ece"] == result["brier"] == result["nll"] == 0
    assert result["bins"][-1]["count"] == 1
    assert result["bins"][0]["accuracy"] is None


def test_brier_nll_and_inclusive_bin_boundary():
    result = calibration_metrics(
        [
            row("negative", dict(negative=0.2, neutral=0.3, positive=0.5)),
            row("positive", dict(negative=0.0, neutral=0.0, positive=1.0)),
        ]
    )
    assert result["brier"] == pytest.approx(0.49)
    assert result["ece"] == pytest.approx(0.25)
    assert result["nll"] == pytest.approx(-__import__("math").log(0.2) / 2)
    assert result["bins"][5]["count"] == 1  # [0.5, 0.6)
    assert sum(b["count"] for b in result["bins"]) == 2


@pytest.mark.parametrize(
    "bad",
    [
        {},
        dict(negative=-0.1, neutral=0.1, positive=1.0),
        dict(negative=0.2, neutral=0.2, positive=0.2),
        dict(negative=float("nan"), neutral=0.0, positive=1.0),
    ],
)
def test_invalid_probabilities_rejected(bad):
    with pytest.raises(ValueError):
        calibration_metrics(
            [dict(gold="positive", prediction="positive", probabilities=bad)]
        )


def test_empty_unknown_gold_and_wrong_argmax_rejected():
    valid = row("negative", dict(negative=0.8, neutral=0.1, positive=0.1))
    for rows in (
        [],
        [{**valid, "gold": "unknown"}],
        [{**valid, "prediction": "neutral"}],
    ):
        with pytest.raises(ValueError):
            calibration_metrics(rows)
    for bins in (0, True, 1.5):
        with pytest.raises(ValueError):
            calibration_metrics([valid], bins)


def test_sampling_is_deterministic_and_includes_recurring_and_variable():
    rows = [
        dict(
            record_id=str(i),
            gold="negative",
            predictions={
                "roberta_base": ["neutral"] * 3
                if i < 10
                else ["negative", "neutral", "negative"],
                "finbert": ["negative"] * 3,
            },
        )
        for i in range(20)
    ]
    chosen = review_sample(rows)
    assert chosen == review_sample(rows[::-1])
    assert len(chosen) == 4
    assert {r["review_strata"][0]["persistence"] for r in chosen} == {
        "recurring",
        "variable",
    }


def test_review_requires_exact_coverage_and_keeps_text_private():
    packet = {
        "examples": [dict(record_id="private-id", text="private headline")],
        "protocol": "fixed",
    }
    annotations = [
        dict(
            record_id="private-id",
            category="neutral_boundary",
            gold_ambiguous=True,
            note="private note",
        )
    ]
    result = summarize_annotations(packet, annotations)
    assert result["primary_categories"] == {"neutral_boundary": 1}
    assert result["suspected_gold_ambiguity"] == 1
    assert all(
        s not in json.dumps(result)
        for s in ("private-id", "private headline", "private note")
    )
    for bad in (
        [],
        annotations * 2,
        [{**annotations[0], "category": "other"}],
        [{**annotations[0], "note": ""}],
    ):
        with pytest.raises(ValueError):
            summarize_annotations(packet, bad)


@pytest.mark.parametrize(
    "corruption", [None, "bytes", "order", "gold", "metrics", "scope", "test_record"]
)
def test_prediction_audit_rejects_corruption(tmp_path, corruption):
    from finsight.sentiment.error_analysis import sha
    from finsight.sentiment.transformer_analysis import checked_run, json_body
    from finsight.sentiment.validation import classification_metrics

    validation = [
        dict(
            record_id=str(i),
            partition="validation",
            corpus="phrasebank",
            native_record=dict(sentence="synthetic", label=label),
        )
        for i, label in enumerate(("negative", "positive"))
    ]
    predictions = [
        {
            "record_id": str(i),
            **row(
                label,
                dict(
                    negative=1.0 if i == 0 else 0.0,
                    neutral=0.0,
                    positive=0.0 if i == 0 else 1.0,
                ),
            ),
        }
        for i, label in enumerate(("negative", "positive"))
    ]
    metrics = classification_metrics(["negative", "positive"], ["negative", "positive"])
    if corruption == "order":
        predictions.reverse()
    if corruption == "gold":
        predictions[0]["gold"] = "neutral"
    if corruption == "metrics":
        metrics = {**metrics, "accuracy": 0.0}
    if corruption == "test_record":
        validation[0]["partition"] = "test"
    receipt = dict(
        parent_split_manifest_sha256="parent", benchmark_manifest_sha256="benchmark"
    )
    prediction_body = b"".join(
        json_body(p).replace(b"\n", b"") + b"\n" for p in predictions
    )
    run = {
        **receipt,
        "configuration": dict(seed=42, learning_rate=2e-5),
        "final_test_evaluated": corruption == "scope",
        "predictions_sha256": sha(prediction_body),
        "validation_membership_sha256": sha(json.dumps(["0", "1"]).encode()),
        "metrics": metrics,
    }
    report = tmp_path / "report.json"
    report.write_bytes(json_body(run))
    entry = dict(
        model="roberta_base",
        seed=42,
        learning_rate=2e-5,
        report_path=str(report),
        report_sha256=sha(report.read_bytes()),
        validation_membership_sha256=run["validation_membership_sha256"],
        metrics=metrics,
    )
    directory = tmp_path / "interim/transformer-comparison" / entry["report_sha256"]
    directory.mkdir(parents=True)
    (directory / "run.json").write_bytes(report.read_bytes())
    (directory / "predictions.jsonl").write_bytes(
        prediction_body + (b" " if corruption == "bytes" else b"")
    )
    if corruption:
        with pytest.raises(ValueError):
            checked_run(entry, tmp_path, validation, receipt)
    else:
        actual, provenance = checked_run(entry, tmp_path, validation, receipt)
        assert actual == predictions
        assert provenance["predictions_sha256"] == sha(prediction_body)


def test_deberta_analysis_keeps_three_seeds_separate_and_private(tmp_path, monkeypatch):
    from finsight.sentiment import transformer_analysis as module
    from finsight.sentiment.validation import classification_metrics

    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}")
    validation = [
        dict(
            record_id="private-validation",
            partition="validation",
            corpus="sentfin",
            native_record=dict(
                title="synthetic private headline",
                sentiments=[dict(target="target", label="negative")],
            ),
        )
    ]
    training = [{**validation[0], "record_id": "train", "partition": "train"}]
    predictions = [
        dict(
            record_id="private-validation",
            **row("negative", dict(negative=0.1, neutral=0.8, positive=0.1)),
        )
    ]
    metrics = classification_metrics(["negative"], ["neutral"])
    study = dict(
        complete=True,
        final_test_evaluated=False,
        plan=dict(freeze_receipt_sha256=module.sha(receipt.read_bytes())),
        selected=dict(deberta_base=dict(learning_rate=2e-5)),
        seed_runs=dict(
            deberta_base=[
                dict(
                    model="deberta_base", seed=seed, learning_rate=2e-5, metrics=metrics
                )
                for seed in (42, 43, 44)
            ]
        ),
    )
    study_path = tmp_path / "study.json"
    study_path.write_bytes(module.json_body(study))
    monkeypatch.setattr(
        module, "load_development_records", lambda *_: training + validation
    )
    monkeypatch.setattr(
        module, "checked_run", lambda *_: (predictions, {"synthetic": True})
    )
    summary, packet = module.prepare_analysis(
        tmp_path, study_path, receipt, models=("deberta_base",)
    )
    assert summary["paired_counts"] == []
    assert summary["all_three_runs_wrong"] == 1
    assert list(summary["models"]) == ["deberta_base"]
    assert (
        summary["models"]["deberta_base"]["persistence"]["errors_in_k_of_three_seeds"][
            "3"
        ]
        == 1
    )
    assert len(packet["examples"]) == 1
    assert "private-validation" not in json.dumps(summary)
    study["seed_runs"]["deberta_base"][2]["seed"] = 43
    study_path.write_bytes(module.json_body(study))
    with pytest.raises(ValueError, match="three seeds"):
        module.prepare_analysis(tmp_path, study_path, receipt, models=("deberta_base",))
