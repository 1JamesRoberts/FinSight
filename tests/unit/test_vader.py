"""Verify score boundaries and meaningful upstream rule behavior."""

import json
from pathlib import Path

import pytest

from finsight.sentiment.vader import VaderConfig, VaderSentiment, classify_compound


@pytest.mark.parametrize(
    "score,label",
    [
        (-1, "negative"),
        (-0.05, "negative"),
        (-0.04999, "neutral"),
        (0, "neutral"),
        (0.04999, "neutral"),
        (0.05, "positive"),
        (1, "positive"),
    ],
)
def test_class_boundaries(score: float, label: str) -> None:
    assert classify_compound(score).value == label


@pytest.mark.parametrize(
    "score", [float("nan"), float("inf"), 1.01, -1.01, True, "0.1"]
)
def test_invalid_scores_fail(score: float) -> None:
    with pytest.raises(ValueError):
        classify_compound(score)


@pytest.mark.parametrize(
    "negative,positive",
    [
        (0, 0.05),
        (-0.05, 0),
        (-1.1, 0.05),
        (-0.05, 1.1),
        (float("nan"), 0.05),
        (True, 0.05),
        (-0.05, "0.05"),
    ],
)
def test_invalid_thresholds_fail(negative: float, positive: float) -> None:
    with pytest.raises(ValueError):
        VaderConfig(negative, positive)


def test_config_and_custom_thresholds(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {"schema_version": 1, "negative_threshold": -0.2, "positive_threshold": 0.2}
        )
    )
    config = VaderConfig.load(path)
    assert classify_compound(0.1, config).value == "neutral"
    assert classify_compound(-0.2, config).value == "negative"
    path.write_text(json.dumps({"schema_version": 2}))
    with pytest.raises(ValueError):
        VaderConfig.load(path)


def test_rule_behavior_and_deterministic_inference() -> None:
    model = VaderSentiment()
    good = model.predict("The outlook is good.")
    assert good.label.value == "positive"
    assert model.predict("The outlook is not good.").label.value == "negative"
    assert model.predict("The outlook is very good!").compound > good.compound
    assert model.predict("The outlook is GOOD!!!").compound > good.compound
    assert model.predict("The meeting is on Tuesday.").label.value == "neutral"
    assert model.predict("The outlook is good.") == good
    assert good.negative + good.neutral + good.positive == pytest.approx(1, abs=0.002)
    metadata = model.metadata()
    assert metadata["version"] == "3.3.2"
    assert all(len(value) == 64 for value in metadata["resources_sha256"].values())


@pytest.mark.parametrize("text", ["", "  \n", None, 12])
def test_blank_and_nonstring_text_fails(text: str) -> None:
    with pytest.raises(ValueError):
        VaderSentiment().predict(text)
