"""Invented targets only: span alignment, identity removal and ambiguity guards."""

from dataclasses import replace

import pytest

from finsight.data.target_spans import (
    TargetSpan,
    align_targets,
    literal_spans,
    transform_input,
)


def test_literal_boundaries_case_punctuation_and_original_offsets():
    text = "ACME's profits rose; Acme fell. Acmeish stayed flat."
    spans = literal_spans(text, "Acme")
    assert [text[s.start : s.end] for s in spans] == ["ACME", "Acme"]
    assert literal_spans("A+B rose; AAB fell", "A+B") == (TargetSpan(0, 3),)
    assert not literal_spans("pineapple", "apple")


def test_no_normalization_or_alias_guessing():
    assert not literal_spans("Acme Corp gained", "Acme Corporation")
    assert not literal_spans("Acme  Corp gained", "Acme Corp")
    alignment = align_targets("Acme rose", ["Beta"])[0]
    assert alignment.reasons == ("no_literal_boundary_match",)


def test_repeated_spans_preserve_context_and_remove_both_identity_inputs():
    text = "Acme rose while Beta fell; Acme added jobs."
    alignment = align_targets(text, ["Acme", "Beta"])[0]
    assert transform_input(text, alignment, "target_pair") == ("Acme", text)
    assert transform_input(text, alignment, "target_markers") == (
        "Acme",
        "[TARGET] Acme [/TARGET] rose while Beta fell; "
        "[TARGET] Acme [/TARGET] added jobs.",
    )
    assert transform_input(text, alignment, "target_replacement") == (
        "[TARGET]",
        "[TARGET] rose while Beta fell; [TARGET] added jobs.",
    )
    beta = align_targets(text, ["Acme", "Beta"])[1]
    assert transform_input(text, beta, "target_replacement")[1] == (
        "Acme rose while [TARGET] fell; Acme added jobs."
    )


def test_nested_native_targets_rejected_for_both_annotations():
    result = align_targets("Acme India gained", ["Acme", "Acme India"])
    assert all(r.reasons == ("overlapping_native_targets",) for r in result)
    with pytest.raises(ValueError, match="eligible"):
        transform_input("Acme India gained", result[0], "target_markers")


def test_existing_markers_rejected():
    result = align_targets("Acme [target] rose", ["Acme"])[0]
    assert result.reasons == ("reserved_marker_in_original_input",)


def test_stale_offsets_and_unknown_variant_rejected():
    alignment = align_targets("Acme rose", ["Acme"])[0]
    with pytest.raises(ValueError, match="original headline"):
        transform_input("Acme fell; Acme rose", alignment, "target_markers")
    with pytest.raises(ValueError, match="unsupported"):
        transform_input("Acme rose", alignment, "other")
    with pytest.raises(ValueError, match="original headline"):
        transform_input(
            "Acme rose", replace(alignment, spans=(TargetSpan(1, 4),)), "target_pair"
        )


@pytest.mark.parametrize("text,target", [("", "Acme"), ("Acme", ""), (None, "Acme")])
def test_empty_invalid_inputs_rejected(text, target):
    with pytest.raises(ValueError, match="nonempty"):
        literal_spans(text, target)


def test_native_targets_must_be_nonempty_and_unique():
    for targets in ([], ["Acme", "Acme"]):
        with pytest.raises(ValueError, match="unique"):
            align_targets("Acme rose", targets)
