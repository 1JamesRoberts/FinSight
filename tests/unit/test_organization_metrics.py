"""Exact boundaries count repeated occurrences independently."""

import pytest

from finsight.sentiment.evaluation.organization_metrics import exact_span_metrics


def test_partial_boundary_is_both_false_positive_and_false_negative():
    gold = {"a": [(0, 5), (10, 15)], "b": []}
    predicted = {"a": [(0, 5), (10, 13)], "b": [(0, 2)]}
    result = exact_span_metrics(gold, predicted)
    assert result["true_positives"] == 1
    assert result["false_positives"] == 2
    assert result["false_negatives"] == 1
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["recall"] == 0.5
    assert result["f1"] == 0.4


def test_no_entity_headlines_are_kept_and_zero_division_explicit():
    result = exact_span_metrics({"a": []}, {"a": []})
    assert result["headlines"] == 1
    assert result["precision"] is None
    assert result["recall"] is None
    assert result["f1"] is None
    assert exact_span_metrics({"a": [(0, 1)]}, {"a": []})["f1"] == 0


def test_membership_duplicates_and_bad_spans_rejected():
    with pytest.raises(ValueError, match="membership"):
        exact_span_metrics({"a": []}, {})
    with pytest.raises(ValueError, match="duplicate"):
        exact_span_metrics({"a": []}, {"a": [(0, 1), (0, 1)]})
    with pytest.raises(ValueError, match="span"):
        exact_span_metrics({"a": [(1, 1)]}, {"a": []})
