"""Offline saved-error review integrity, sampling, and public-output boundaries."""

import json
from pathlib import Path

import pytest

from finsight.sentiment.error_analysis import (
    checked_predictions,
    disagreement_counts,
    prepare_packet,
    sample_errors,
    sha,
    summarize_review,
)
from finsight.sentiment.validation import classification_metrics


def test_sampling_is_order_independent_and_limits_each_error_cell() -> None:
    rows = [dict(record_id=str(i), gold="negative", lr="neutral") for i in range(20)]
    rows += [
        dict(record_id="rare", gold="positive", lr="negative"),
        dict(record_id="correct", gold="neutral", lr="neutral"),
    ]
    chosen = sample_errors(rows)
    assert chosen == sample_errors(list(reversed(rows)))
    assert len(chosen) == 6
    assert "rare" in {r["record_id"] for r in chosen}
    assert "correct" not in {r["record_id"] for r in chosen}
    for value in (0, True, 1.5):
        with pytest.raises(ValueError):
            sample_errors(rows, value)


def test_paired_counts_distinguish_same_wrong_from_different_wrong() -> None:
    rows = [
        dict(gold="positive", lr="positive", svm="positive", vader="neutral"),
        dict(gold="positive", lr="positive", svm="negative", vader="positive"),
        dict(gold="positive", lr="negative", svm="positive", vader="negative"),
        dict(gold="positive", lr="neutral", svm="negative", vader="negative"),
    ]
    assert disagreement_counts(rows) == dict(
        both_correct=1,
        lr_only_correct=1,
        svm_only_correct=1,
        both_wrong=1,
        lr_svm_disagree=3,
        lr_errors=2,
        svm_errors=2,
        vader_errors=3,
        all_three_wrong=1,
    )


def test_public_summary_never_exposes_notes_text_or_source_ids() -> None:
    packet = {
        "examples": [
            dict(record_id="private-id", text="private-text", cohort="phrasebank")
        ],
        "cohorts": {"phrasebank": {"reviewed_samples": 1}},
    }
    annotations = [
        dict(
            record_id="private-id",
            category="mixed_cues",
            gold_ambiguous=True,
            note="private-note",
        )
    ]
    summary = summarize_review(packet, annotations)
    result = summary["cohorts"]["phrasebank"]
    assert result["primary_categories"] == {"mixed_cues": 1}
    assert result["suspected_gold_ambiguity"] == 1
    assert summary["gold_labels_changed"] is False
    assert "primary_categories" not in packet["cohorts"]["phrasebank"]
    assert all(
        value not in json.dumps(summary)
        for value in ("private-id", "private-text", "private-note")
    )
    for bad in (
        [],
        annotations * 2,
        [{**annotations[0], "record_id": "unexpected"}],
        [{**annotations[0], "category": "unknown"}],
        [{**annotations[0], "note": " "}],
        [{**annotations[0], "gold_ambiguous": "yes"}],
    ):
        with pytest.raises(ValueError):
            summarize_review(packet, bad)


@pytest.mark.parametrize(
    "corruption", [None, "predictions", "artifact", "gold", "metrics", "membership"]
)
def test_saved_predictions_require_verified_original_artifacts(
    tmp_path: Path,
    corruption: str | None,
) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text("synthetic receipt")
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"synthetic artifact; never unpickled")
    validation = [
        dict(
            record_id="pb/1",
            corpus="phrasebank",
            partition="validation",
            native_record=dict(sentence="synthetic text", label="negative"),
        )
    ]
    row = dict(
        record_id="pb/1",
        cohort="phrasebank",
        gold="negative",
        prediction=dict(label="neutral"),
        vader=dict(label="positive"),
    )
    if corruption == "membership":
        row["record_id"] = "unexpected"
    if corruption == "gold":
        row["gold"] = "positive"
    pred_body = (json.dumps(row) + "\n").encode()
    result = dict(
        artifact_relative_path="model.pkl",
        artifact_sha256=sha(artifact.read_bytes()),
        validation_membership_sha256=sha(json.dumps(["pb/1"]).encode()),
        logistic_regression=classification_metrics(["negative"], ["neutral"]),
        vader=classification_metrics(["negative"], ["positive"]),
    )
    if corruption == "metrics":
        result["logistic_regression"]["accuracy"] = 1
    run = dict(
        final_test_evaluated=False,
        freeze_receipt_sha256=sha(receipt.read_bytes()),
        predictions_sha256=sha(pred_body),
        metrics=dict(phrasebank=result),
    )
    body = (json.dumps(run) + "\n").encode()
    run_id = sha(body)
    directory = tmp_path / "interim/tfidf-logreg-validation" / run_id
    directory.mkdir(parents=True)
    (directory / "run.json").write_bytes(body)
    (directory / "predictions.jsonl").write_bytes(pred_body)
    candidate = dict(model_kind="logistic_regression", run_id=run_id, run=run)
    if corruption == "predictions":
        (directory / "predictions.jsonl").write_bytes(pred_body + b" ")
    if corruption == "artifact":
        artifact.write_bytes(b"changed")
    if corruption:
        with pytest.raises(ValueError):
            checked_predictions(tmp_path, candidate, "phrasebank", validation, receipt)
    else:
        assert checked_predictions(
            tmp_path, candidate, "phrasebank", validation, receipt
        ) == {"pb/1": row}
        (directory / "run.json").write_bytes(body + b" ")
        with pytest.raises(ValueError, match="original run"):
            checked_predictions(tmp_path, candidate, "phrasebank", validation, receipt)


def test_review_rejects_holdout_before_accessing_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "finsight.sentiment.error_analysis.load_development_records",
        lambda *args: [dict(partition="test_candidate")],
    )
    tuning = tmp_path / "tuning.json"
    tuning.write_text(json.dumps(dict(final_test_evaluated=False, candidates=[])))
    with pytest.raises(ValueError, match="development records only"):
        prepare_packet(tmp_path, tmp_path / "missing-receipt", tuning)
