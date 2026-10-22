"""Synthetic renaming preserves context; output drift needs no synthetic gold."""

import pytest

from finsight.sentiment.experiments.target_name_stress import renamed_views, sensitivity


def prediction(negative, neutral, positive):
    p = dict(negative=negative, neutral=neutral, positive=positive)
    return {"probabilities": p, "prediction": max(p, key=p.__getitem__)}


def test_selected_name_changed_in_both_sequences_and_all_mentions():
    views = renamed_views(
        "Acme rose; Beta fell while Acme hired.", "Acme", ["Beta"], "Cedar Group"
    )
    assert views["target_pair"] == {
        "target": "Cedar Group",
        "headline": "Cedar Group rose; Beta fell while Cedar Group hired.",
    }
    assert (
        views["target_markers"]["headline"]
        == "[TARGET] Cedar Group [/TARGET] rose; Beta fell while "
        "[TARGET] Cedar Group [/TARGET] hired."
    )
    assert views["target_replacement"] == {
        "target": "[TARGET]",
        "headline": "[TARGET] rose; Beta fell while [TARGET] hired.",
    }


@pytest.mark.parametrize("name", ["Acme", "", "Beta", "Beta Group"])
def test_invalid_substitute_or_other_target_collision_rejected(name):
    with pytest.raises(ValueError):
        renamed_views("Acme rose; Beta fell.", "Acme", ["Beta"], name)


def test_missing_or_overlapping_original_target_rejected():
    with pytest.raises(ValueError, match="eligible"):
        renamed_views("Acme India rose.", "Acme", ["Acme India"], "Cedar Group")
    with pytest.raises(ValueError, match="eligible"):
        renamed_views("Beta rose.", "Acme", [], "Cedar Group")


def test_flip_and_probability_drift_have_explicit_units():
    original = [prediction(0.1, 0.2, 0.7), prediction(0.1, 0.2, 0.7)]
    changed = [prediction(0.6, 0.2, 0.2), prediction(0.2, 0.2, 0.6)]
    result = sensitivity(original, changed)
    assert result["comparisons"] == 2 and result["label_flips"] == 1
    assert result["label_flip_rate"] == 0.5
    assert result["mean_total_variation"] == pytest.approx(0.3)
    assert result["maximum_absolute_class_probability_change"] == pytest.approx(0.5)


def test_identical_output_is_a_zero_change_control():
    rows = [prediction(0.1, 0.2, 0.7)]
    result = sensitivity(rows, rows)
    assert result["label_flips"] == 0
    assert result["mean_total_variation"] == 0


@pytest.mark.parametrize("failure", ["empty", "length", "sum", "nan", "mapping"])
def test_invalid_predictions_rejected(failure):
    a, b = [prediction(0.1, 0.2, 0.7)], [prediction(0.1, 0.2, 0.7)]
    if failure == "empty":
        a = b = []
    elif failure == "length":
        b = []
    elif failure == "sum":
        b[0]["probabilities"]["positive"] = 0.8
    elif failure == "nan":
        b[0]["probabilities"]["positive"] = float("nan")
    else:
        b[0]["prediction"] = "negative"
    with pytest.raises(ValueError):
        sensitivity(a, b)
