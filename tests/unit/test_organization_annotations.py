"""Prediction-free sampling and review provenance guards."""

import copy

import pytest

from finsight.data.organization_annotations import (
    select_headlines,
    validate_annotations,
)


def source(i, text="Tesla meets UNICEF", partition="validation"):
    return dict(record_id=str(i), headline=text, partition=partition, target="ignored")


def annotation():
    return dict(
        record_id="1",
        headline="Tesla meets UNICEF",
        partition="validation",
        entities=[
            dict(text="Tesla", start=0, end=5, type="company", uncertain=False),
            dict(
                text="UNICEF",
                start=12,
                end=18,
                type="other_organization",
                uncertain=False,
            ),
        ],
        annotation_author="assistant",
        exhaustive=True,
        review=dict(status="pending", reviewer=None),
        notes="",
    )


def test_sampling_ignores_targets_predictions_and_input_order():
    rows = [source(i) for i in range(10)]
    selected = select_headlines(rows, size=4)
    changed = [{**r, "target": "Other", "prediction": "anything"} for r in rows]
    assert selected == select_headlines(list(reversed(changed)), size=4)
    assert all(set(r) == {"record_id", "headline", "partition"} for r in selected)
    assert len(selected) == 4
    assert select_headlines(rows, size=4, seed=43) != selected


def test_sampler_rejects_reserved_conflicting_or_insufficient_records():
    with pytest.raises(ValueError, match="validation"):
        select_headlines([source(1, partition="test")], size=1)
    with pytest.raises(ValueError, match="conflicting"):
        select_headlines([source(1), source(1, "Other")], size=1)
    with pytest.raises(ValueError, match="size"):
        select_headlines([source(1)], size=2)


def test_annotation_membership_offsets_and_review_gate():
    sources = [source(1)]
    rows = [annotation()]
    validate_annotations(sources, rows, provisional=True)
    with pytest.raises(ValueError, match="human review"):
        validate_annotations(sources, rows)
    rows[0]["review"] = dict(status="approved", reviewer="human reviewer")
    validate_annotations(sources, rows)
    rows[0]["entities"][0]["uncertain"] = True
    with pytest.raises(ValueError, match="uncertain"):
        validate_annotations(sources, rows)


def test_default_evaluation_rejects_drafts_before_loading_model(monkeypatch, tmp_path):
    import json

    from finsight.sentiment.experiments import organization_benchmark as benchmark

    monkeypatch.setattr(benchmark, "sample_rows", lambda *args: [source(1)])

    def forbidden(*args, **kwargs):
        raise AssertionError("pending annotations must not load models")

    monkeypatch.setattr(benchmark.OrganizationDetector, "load", forbidden)
    path = tmp_path / "annotations.jsonl"
    path.write_text(json.dumps(annotation()) + "\n")
    with pytest.raises(ValueError, match="human review"):
        benchmark.evaluate(tmp_path, tmp_path, tmp_path, path)


@pytest.mark.parametrize(
    "change",
    ["missing", "duplicate", "text", "offset", "type", "overlap", "incomplete"],
)
def test_invalid_annotations_fail_closed(change):
    row = annotation()
    rows = [row]
    if change == "missing":
        rows = []
    if change == "duplicate":
        rows.append(copy.deepcopy(row))
    if change == "text":
        row["headline"] += "!"
    if change == "offset":
        row["entities"][0]["end"] = 6
    if change == "type":
        row["entities"][0]["type"] = "person"
    if change == "overlap":
        row["entities"].append(copy.deepcopy(row["entities"][0]))
    if change == "incomplete":
        row["exhaustive"] = False
    with pytest.raises(ValueError):
        validate_annotations([source(1)], rows, provisional=True)
