"""Coverage units, conservative mentions and development-only safeguards."""

from pathlib import Path

import pytest

from finsight.data.company_coverage import literal_match, summarize_coverage
from finsight.sentiment.companies import load_registry


def row(identifier, partition, title, annotations, group=None):
    return {
        "corpus": "sentfin",
        "partition": partition,
        "record_id": identifier,
        "group_id": group or identifier,
        "native_record": {
            "title": title,
            "sentiments": [
                {"target": target, "label": label} for target, label in annotations
            ],
        },
    }


@pytest.fixture
def registry():
    return load_registry(Path("configs/companies.json"))


def test_headlines_annotations_classes_and_multitarget_are_distinct(registry):
    rows = [
        row(
            "1",
            "train",
            "Apple rises while Microsoft falls",
            [("Apple", "positive"), ("Microsoft", "negative")],
        ),
        row("2", "train", "Apple holds steady", [("Apple Inc.", "neutral")]),
        row("3", "validation", "Microsoft rises", [("Microsoft", "positive")]),
    ]
    summary, review = summarize_coverage(rows, registry)
    train = summary["partitions"]["train"]
    assert train["overall"]["headlines"] == 2
    assert train["overall"]["target_annotations"] == 3
    apple = train["views"]["strict"]["AAPL"]
    assert apple["annotated_headlines"] == 2
    assert apple["target_annotations"] == 2
    assert apple["class_counts"] == {"negative": 0, "neutral": 1, "positive": 1}
    assert apple["single_target_headlines"] == apple["multi_target_headlines"] == 1
    assert apple["mixed_label_headlines"] == 1
    assert apple["literal_target_annotations"] == 1
    assert apple["registered_alias_only_annotations"] == 1
    assert any(r["reason"] == "target_not_literal" for r in review)
    assert (
        summary["partitions"]["validation"]["views"]["strict"]["AAPL"][
            "annotated_headlines"
        ]
        == 0
    )


def test_parent_rollup_is_separate_and_conflicts_are_not_silently_dropped(registry):
    rows = [
        row(
            "1",
            "train",
            "Google and Alphabet diverge",
            [("Google", "positive"), ("Alphabet", "negative")],
        )
    ]
    summary, review = summarize_coverage(rows, registry)
    result = summary["partitions"]["train"]
    strict = result["views"]["strict"]["GOOGL"]
    grouped = result["views"]["with_parent_rollup"]["GOOGL"]
    assert strict["target_annotations"] == 1
    assert grouped["target_annotations"] == 2
    assert grouped["annotated_headlines"] == 1
    assert grouped["same_company_conflict_headlines"] == 1
    assert any(r["reason"] == "parent_rollup_required" for r in review)
    assert result["target_resolution"]["ambiguous"] == 1


def test_unannotated_literal_candidates_never_become_labeled_examples(registry):
    rows = [
        row("1", "train", "Apple sells alongside Tata", [("Tata", "neutral")]),
        row("2", "train", "Pineapple harvest", [("Farms", "positive")]),
    ]
    summary, _ = summarize_coverage(rows, registry)
    counts = summary["partitions"]["train"]["views"]["strict"]["AAPL"]
    assert counts["literal_candidate_headlines"] == 1
    assert counts["unannotated_literal_candidate_headlines"] == 1
    assert counts["target_annotations"] == counts["annotated_headlines"] == 0


def test_nonliteral_annotation_and_unknown_name_cues_are_private_flags(registry):
    rows = [
        row("1", "train", "Device maker reports earnings", [("Apple", "positive")]),
        row("2", "train", "Apple Holdings rises", [("Apple Holdings", "positive")]),
    ]
    summary, review = summarize_coverage(rows, registry)
    part = summary["partitions"]["train"]
    assert part["views"]["strict"]["AAPL"]["no_registered_alias_annotations"] == 1
    assert part["unsupported_with_registry_name_cue_annotations"] == 1
    assert any(r["reason"] == "unsupported_with_registry_name_cue" for r in review)
    assert "Apple Holdings" not in str(summary)
    assert all("title" not in r for r in review)


@pytest.mark.parametrize("partition", ["test", "test_candidate", "final_test"])
def test_reserved_records_are_rejected_before_corpus_selection(registry, partition):
    with pytest.raises(ValueError, match="development"):
        summarize_coverage([{"corpus": "phrasebank", "partition": partition}], registry)


def test_duplicate_ids_or_cross_partition_groups_fail(registry):
    sample = row("1", "train", "Apple rises", [("Apple", "positive")])
    with pytest.raises(ValueError, match="unique"):
        summarize_coverage([sample, sample], registry)
    other = row(
        "2", "validation", "Microsoft falls", [("Microsoft", "negative")], group="1"
    )
    with pytest.raises(ValueError, match="group"):
        summarize_coverage([sample, other], registry)


@pytest.mark.parametrize("annotations", [[], [("Apple", "unknown")]])
def test_malformed_target_annotations_fail(registry, annotations):
    with pytest.raises(ValueError, match="annotation"):
        summarize_coverage([row("1", "train", "Apple", annotations)], registry)


def test_literal_screen_uses_boundaries_and_preserves_internal_punctuation():
    assert literal_match("($MSFT) rises", "MSFT")
    assert literal_match("Ａｐｐｌｅ,  Inc. rises", "Apple Inc.")
    assert not literal_match("pineapple and benefit", "Apple")
    assert not literal_match("profits", "IT")
    assert not literal_match("amazoncom", "Amazon.com")
