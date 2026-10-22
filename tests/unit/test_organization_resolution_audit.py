"""Unknown companies count as abstentions, not correct non-company decisions."""

from pathlib import Path

import pytest

from finsight.sentiment.companies import load_registry
from finsight.sentiment.experiments.organization_resolution import evaluate_resolutions

ROOT = Path(__file__).resolve().parents[2]


def case(identifier, text, kind):
    return {
        "record_id": identifier,
        "headline": text,
        "entities": []
        if kind is None
        else [{"start": 0, "end": len(text), "text": text, "type": kind}],
    }


def test_types_abstentions_and_typed_extraction_denominators():
    rows = [
        case("a", "Microsoft", "company"),
        case("b", "Tesla", "company"),
        case("c", "Ministry of Finance", "other_organization"),
        case("d", "Gold", None),
    ]
    pred = {
        r["record_id"]: [dict(text=r["headline"], start=0, end=len(r["headline"]))]
        for r in rows
    }
    result, decisions, errors = evaluate_resolutions(
        rows, pred, load_registry(ROOT / "configs/companies.json")
    )
    assert result["decision_coverage"] == 0.5
    assert result["selective_typed_exact_precision"] == 1
    assert result["decisions_by_reference"]["unresolved"] == {
        "company": 1,
        "other_organization": 0,
        "unmatched_span": 1,
    }
    assert result["verified_company_extraction"]["precision"] == 1
    assert result["verified_company_extraction"]["recall"] == 0.5
    assert len(decisions) == 4 and len(errors) == 1


def test_no_decisions_yields_undefined_precision_not_perfect_score():
    rows = [case("a", "Tesla", "company")]
    result, _, _ = evaluate_resolutions(
        rows,
        {"a": [dict(text="Tesla", start=0, end=5)]},
        load_registry(ROOT / "configs/companies.json"),
    )
    assert result["decision_coverage"] == 0
    assert result["selective_typed_exact_precision"] is None
    assert result["verified_company_extraction"]["precision"] is None
    assert result["verified_company_extraction"]["recall"] == 0


def test_missing_headlines_rejected():
    with pytest.raises(ValueError, match="membership"):
        evaluate_resolutions(
            [case("a", "Tesla", "company")],
            {},
            load_registry(ROOT / "configs/companies.json"),
        )


def test_merged_names_cannot_be_hidden_by_resolution():
    text = "Microsoft & Tesla"
    rows = [
        {
            "record_id": "a",
            "headline": text,
            "entities": [
                dict(text="Microsoft", start=0, end=9, type="company"),
                dict(text="Tesla", start=12, end=17, type="company"),
            ],
        }
    ]
    result, _, errors = evaluate_resolutions(
        rows,
        {"a": [dict(text=text, start=0, end=len(text))]},
        load_registry(ROOT / "configs/companies.json"),
    )
    assert result["extraction_error_categories"]["merged_names"] == 1
    assert result["extraction_error_categories"]["reference_lost_to_boundary"] == 2
    assert len(errors) == 3
