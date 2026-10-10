"""Test descriptive statistics, denominators, and score interval boundaries."""

import pytest

from finsight.data.labels import label_distribution, score_distribution
from finsight.data.loaders import SentimentLabel


def test_label_percentages_use_annotation_count_and_fixed_classes() -> None:
    result = label_distribution(
        [SentimentLabel.POSITIVE] * 3
        + [SentimentLabel.NEUTRAL] * 2
        + [SentimentLabel.NEGATIVE]
    )
    assert result["total"] == 6
    assert result["counts"] == {"negative": 1, "neutral": 2, "positive": 3}
    assert result["percentages"]["positive"] == 50
    assert sum(result["percentages"].values()) == pytest.approx(100)
    assert result["largest_to_smallest_class_ratio"] == 3
    assert result["majority_share_percent"] == 50


def test_absent_classes_and_empty_labels_are_explicit() -> None:
    result = label_distribution([SentimentLabel.NEUTRAL])
    assert result["counts"]["negative"] == 0
    assert result["largest_to_smallest_class_ratio"] is None
    empty = label_distribution([])
    assert empty["total"] == 0
    assert empty["majority_share_percent"] is None
    assert all(value == 0 for value in empty["percentages"].values())


def test_score_summary_uses_inclusive_quartiles_and_exclusive_upper_bins() -> None:
    result = score_distribution([-1, -0.5, 0, 0.5, 1])
    assert result["mean"] == 0
    assert result["median"] == 0
    assert result["q25"] == -0.5
    assert result["q75"] == 0.5
    assert result["population_stddev"] == pytest.approx(0.5**0.5)
    assert result["histogram"] == {
        "[-1, -0.5)": 1,
        "[-0.5, 0)": 1,
        "[0, 0.5)": 1,
        "[0.5, 1]": 2,
    }
    assert sum(result["histogram"].values()) == result["total"]
    assert result["numeric_sign_counts"] == {"negative": 2, "zero": 1, "positive": 2}


def test_single_score_and_near_zero_scores_keep_numeric_meaning() -> None:
    result = score_distribution([0.1])
    assert result["q25"] == result["q75"] == result["median"] == 0.1
    assert result["population_stddev"] == 0
    assert score_distribution([-0.0001, 0.0001])["numeric_sign_counts"] == {
        "negative": 1,
        "zero": 0,
        "positive": 1,
    }


@pytest.mark.parametrize(
    "values", [[], [float("nan")], [float("inf")], [-1.01], [1.01]]
)
def test_score_summary_rejects_empty_or_invalid_inputs(values: list[float]) -> None:
    with pytest.raises(ValueError):
        score_distribution(values)
