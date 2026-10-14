"""Headline predictions must be joined without using target gold to choose them."""

from dataclasses import replace

import pytest

from finsight.data.target_pairs import TargetExample
from finsight.sentiment.target_control import evaluate_cohorts, join_predictions


def example(identity: str, target: str, label: str) -> TargetExample:
    return TargetExample(
        identity,
        "headline",
        "group",
        "validation",
        0,
        target,
        "Alpha rises while Beta falls.",
        label,
        True,
        True,
        False,
        (),
    )


def prediction() -> dict:
    # Deliberately use a different insertion order from the report class order.
    return {
        "prediction": "positive",
        "probabilities": {
            "positive": 0.8,
            "negative": 0.1,
            "neutral": 0.1,
        },
    }


def test_same_prediction_for_conflicting_targets() -> None:
    examples = (example("a", "Alpha", "positive"), example("b", "Beta", "negative"))
    rows = join_predictions(examples, {"headline": prediction()})
    assert [r["gold"] for r in rows] == ["positive", "negative"]
    assert [r["prediction"] for r in rows] == ["positive", "positive"]
    assert rows[0]["probabilities"] == rows[1]["probabilities"]
    result = evaluate_cohorts(rows, {"Alpha"})
    assert result["all_targets"]["classification"]["accuracy"] == 0.5
    assert result["mixed_label_headlines"]["examples"] == 2
    assert result["seen_native_target"]["examples"] == 1
    assert result["unseen_native_target"]["examples"] == 1
    assert result["single_target"]["classification"] is None
    assert result["single_target"]["calibration"] is None


@pytest.mark.parametrize(
    "mode", ["missing", "extra", "duplicate", "reserved", "text", "probability"]
)
def test_invalid_join_rejected(mode: str) -> None:
    row = example("a", "Alpha", "positive")
    examples = (row,)
    predictions = {"headline": prediction()}
    if mode == "missing":
        predictions = {}
    elif mode == "extra":
        predictions["extra"] = prediction()
    elif mode == "duplicate":
        examples = (row, row)
    elif mode == "reserved":
        examples = (replace(row, partition="test"),)
    elif mode == "text":
        examples = (row, replace(row, example_id="b", headline="Other text"))
    else:
        predictions["headline"]["probabilities"]["positive"] = 0.9
    with pytest.raises(ValueError):
        join_predictions(examples, predictions)
