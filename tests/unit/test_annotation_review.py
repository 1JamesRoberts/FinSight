"""Protect duplicate identity and distinguish gold conflicts from target context."""

from itertools import combinations

import pytest

from finsight.data.audit import text_identity
from finsight.data.review import (
    ReviewRecord,
    conflicting_contexts,
    exact_duplicate_groups,
    near_duplicate_pairs,
    snippet_mismatches,
    text_hash,
)


def test_exact_group_preserves_rows_and_detects_same_sentence_conflict() -> None:
    rows = [
        ReviewRecord("phrasebank", "1", "ACME improves", (("sentence", "positive"),)),
        ReviewRecord("phrasebank", "2", " acme  improves ", (("sentence", "neutral"),)),
        ReviewRecord("phrasebank", "3", "ACME improves", (("sentence", "positive"),)),
    ]
    groups = exact_duplicate_groups(rows)
    assert len(groups) == 1
    assert len(groups[0]["records"]) == 3
    assert groups[0]["annotation_sets_differ"] is True
    assert len(groups[0]["conflicting_contexts"]) == 1
    assert groups[0]["records"][1]["text"] == " acme  improves "


def test_different_targets_and_missing_gold_are_not_conflicts() -> None:
    rows = [
        ReviewRecord(
            "sentfin", "1", "A rises; B falls", (("a", "positive"), ("b", "negative"))
        ),
        ReviewRecord("sentfin", "2", "A rises; B falls", (("a", "positive"),)),
        ReviewRecord("fiqa", "3", "A rises; B falls", (("a", "-0.4"),)),
        ReviewRecord("fiqa", "4", "A rises; B falls", (("a", None),)),
    ]
    assert conflicting_contexts(rows) == {}
    groups = exact_duplicate_groups(rows)
    assert len(groups) == 2
    assert all(group["annotation_sets_differ"] for group in groups)
    assert all(not group["conflicting_contexts"] for group in groups)


def test_gold_conflicts_require_same_text_and_aspect_snippet_context() -> None:
    rows = [
        ReviewRecord("fiqa", "1", "A gains", (("a/aspect1/span1", "0.2"),)),
        ReviewRecord("fiqa", "2", "A gains", (("a/aspect2/span1", "-0.2"),)),
        ReviewRecord("fiqa", "3", "A loses", (("a/aspect1/span1", "-0.2"),)),
    ]
    assert conflicting_contexts(rows) == {}
    rows.append(ReviewRecord("fiqa", "4", "A gains", (("a/aspect1/span1", "-0.3"),)))
    assert len(conflicting_contexts(rows)) == 1


def test_hash_preserves_numbers_punctuation_and_negation() -> None:
    assert text_hash(" ＡＣＭＥ   gains ") == text_hash("acme gains")
    assert text_hash("profit 10") != text_hash("profit 20")
    assert text_hash("profit rises") != text_hash("profit does not rise")
    assert text_hash("profit rises!") != text_hash("profit rises")


def test_near_search_matches_brute_force_and_is_deterministic() -> None:
    texts = [
        "one two three four five six",
        "ONE TWO THREE FOUR FIVE SIX",
        "one two three four five six seven",
        "one two three four five six seven eight",
        "entirely different sentence with no common content",
        "one two three",
    ]
    normalized = sorted(
        {text_identity(text) for text in texts if len(text.split()) >= 6}
    )
    expected = set()
    for left, right in combinations(normalized, 2):
        left_words, right_words = left.split(), right.split()
        a = {tuple(left_words[i : i + 3]) for i in range(len(left_words) - 2)}
        b = {tuple(right_words[i : i + 3]) for i in range(len(right_words) - 2)}
        if len(a & b) / len(a | b) >= 0.8:
            expected.add((text_hash(left), text_hash(right)))
    actual = near_duplicate_pairs(texts)
    assert {(p["left_sha256"], p["right_sha256"]) for p in actual} == expected
    assert len(actual) == 2
    assert near_duplicate_pairs(reversed(texts)) == actual


def test_numeric_changes_are_flagged_without_collapsing_distinct_text() -> None:
    stem = (
        "company reported solid earnings and steady margins during the quarter ending "
    )
    pairs = near_duplicate_pairs([stem + "2023", stem + "2024"])
    assert len(pairs) == 1
    assert pairs[0]["numeric_tokens_differ"] is True
    assert pairs[0]["left_sha256"] != pairs[0]["right_sha256"]
    assert near_duplicate_pairs(["short text", "short text again"]) == []


@pytest.mark.parametrize("threshold,min_words", [(0, 6), (1.1, 6), (0.8, 2)])
def test_invalid_screening_parameters_are_rejected(
    threshold: float, min_words: int
) -> None:
    with pytest.raises(ValueError):
        near_duplicate_pairs([], threshold=threshold, min_words=min_words)


def test_snippet_flags_preserve_missing_values_and_allow_normalized_spans() -> None:
    assert snippet_mismatches(
        "ACME  gains\n10%", [None, "acme gains", "10%", "Other declines"]
    ) == ("Other declines",)
