"""Native-target coverage is a positive-only diagnostic, not NER precision."""

from pathlib import Path

import pytest

from finsight.sentiment.companies import load_registry
from finsight.sentiment.experiments.organization_detection import (
    compare_detections,
    validation_headlines,
)
from finsight.sentiment.organizations import OrganizationMention

ROOT = Path(__file__).resolve().parents[2]


def row(identifier, headline, target, partition="validation"):
    return dict(
        record_id=identifier, headline=headline, target=target, partition=partition
    )


def test_unique_headlines_and_reject_frozen_or_conflicting_rows():
    a = row("a", "Tesla met Ford", "Tesla")
    assert len(validation_headlines([a, row("a", a["headline"], "Ford")])) == 1
    with pytest.raises(ValueError, match="validation"):
        validation_headlines([row("a", "Tesla", "Tesla", "test_candidate")])
    with pytest.raises(ValueError, match="inconsistent"):
        validation_headlines([a, row("a", "Other", "Ford")])


def test_exact_overlap_nonliteral_and_unannotated_are_separate():
    rows = [
        row("a", "Tesla Inc. met UNICEF", "Tesla"),
        row("b", "Microsoft gained", "Microsoft"),
        row("c", "The car maker rose", "Tesla"),
        row("d", "Apple profits rose", "Apple"),
    ]
    ner = {
        "a": (
            OrganizationMention("Tesla Inc.", 0, 10, 0.9),
            OrganizationMention("UNICEF", 15, 21, 0.8),
        ),
        "b": (OrganizationMention("Microsoft", 0, 9, 0.9),),
        "c": (),
        "d": (),
    }
    summary, cases = compare_detections(
        rows, ner, load_registry(ROOT / "configs/companies.json"), {"Tesla", "Apple"}
    )
    native = summary["coverage"]["all_native_financial_targets"]
    assert native["annotations"] == 4
    assert native["literal_annotations"] == 3
    assert native["nonliteral_annotations"] == 1
    assert native["ner_org"]["exact"] == 1
    assert native["ner_org"]["overlap"] == 2
    assert native["registry_resolved"]["exact"] == 1
    assert native["registry_all_candidates"]["exact"] == 2
    assert summary["ner_span_annotation_relation"] == {
        "exact": 1,
        "overlap_only": 1,
        "no_literal_annotation_overlap": 1,
    }
    assert len(cases) == 4
    assert summary["coverage"]["fixed_company_allowlist"]["literal_annotations"] == 2


def test_missing_ner_membership_cannot_silently_reduce_denominator():
    with pytest.raises(ValueError, match="membership"):
        compare_detections(
            [row("a", "Tesla", "Tesla")],
            {},
            load_registry(ROOT / "configs/companies.json"),
            {"Tesla"},
        )
