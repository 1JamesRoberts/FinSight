"""Check filtering, transitive leakage protection, and source lineage."""

import copy
import json
from pathlib import Path

import pytest

from finsight.data.review import text_hash
from finsight.data.splits import (
    SourceExample,
    assign_partition,
    duplicate_components,
    make_split_plan,
    validate_policy,
    validate_split_plan,
)

POLICY_PATH = Path(__file__).resolve().parents[2] / "configs/split-policy.json"


@pytest.fixture
def policy() -> dict:
    return json.loads(POLICY_PATH.read_text())


@pytest.fixture
def review(policy: dict) -> dict:
    return {
        "summary": {
            "near_method": policy["near_duplicates"],
            "manifest_sha256": "synthetic",
            "revisions": {"sentfin": "synthetic"},
        },
        "exact_groups": [],
        "near_pairs": [],
    }


def example(
    source_id: str,
    text: str = "Example profit rises",
    annotations: tuple = (("Example", "positive"),),
    *,
    corpus: str = "sentfin",
    snippet_issues: tuple[int, ...] = (),
    unscored: bool = False,
) -> SourceExample:
    return SourceExample(
        corpus,
        source_id,
        text,
        annotations,
        {"title": text, "revision": "synthetic", "original": "preserved"},
        snippet_issues,
        unscored,
    )


def near(left: SourceExample, right: SourceExample, *, conflict: bool = False) -> dict:
    return {
        "left_sha256": text_hash(left.text),
        "right_sha256": text_hash(right.text),
        "shared_context_gold_differs": conflict,
    }


def exact(row: SourceExample, *, conflict: bool = False, scope: bool = False) -> dict:
    return {
        "corpus": row.corpus,
        "text_sha256": text_hash(row.text),
        "conflicting_contexts": {"Example": ["positive", "neutral"]}
        if conflict
        else {},
        "annotation_sets_differ": scope,
    }


def test_components_are_transitive_and_order_independent() -> None:
    edges = [("a", "b"), ("b", "c")]
    groups = duplicate_components({"a", "b", "c", "d"}, edges)
    assert groups["a"] == groups["b"] == groups["c"]
    assert groups["d"] != groups["a"]
    assert groups == duplicate_components({"d", "c", "b", "a"}, edges[::-1])
    with pytest.raises(ValueError, match="unknown text"):
        duplicate_components({"a"}, edges)


def test_seeded_assignment_never_tests_development_groups(policy: dict) -> None:
    names = {
        assign_partition(str(i), policy, development_only=False) for i in range(500)
    }
    assert names == {"train", "validation", "test_candidate"}
    assert all(
        assign_partition(str(i), policy, development_only=True) != "test_candidate"
        for i in range(500)
    )
    changed = policy | {"seed": 7}
    assert any(
        assign_partition(str(i), policy, development_only=False)
        != assign_partition(str(i), changed, development_only=False)
        for i in range(50)
    )


@pytest.mark.parametrize(
    "change",
    [
        {"seed": True},
        {"unknown": 1},
        {"final_test_frozen": True},
        {"partition_weights": {"train": 80, "validation": 10, "test_candidate": 11}},
        {"quarantine": []},
        {"near_duplicates": {"jaccard_threshold": 0.5}},
    ],
)
def test_policy_rejects_unsupported_or_ignored_changes(
    policy: dict, review: dict, change: dict
) -> None:
    with pytest.raises(ValueError):
        validate_policy(policy | change, review)


def test_equivalent_duplicates_collapse_with_complete_lineage(
    policy: dict, review: dict
) -> None:
    a = example("a")
    b = example("b", "EXAMPLE  profit rises")
    c = example("c", "Example profit rises today")
    review["near_pairs"] = [near(a, c)]
    plan, rows = make_split_plan([c, b, a], review, policy)
    assert len(rows) == 2
    assert rows[0]["source_references"] == ["sentfin/a", "sentfin/b"]
    assert rows[0]["native_record"] == a.native_record
    assert len({row["partition"] for row in rows}) == 1
    assert len({row["group_id"] for row in rows}) == 1
    assert make_split_plan([a, b, c], review, policy) == (plan, rows)
    assert plan["sources"][1]["status"] == "redundant_exact_duplicate"


@pytest.mark.parametrize("conflict_type", ["exact", "near"])
def test_unresolved_conflicts_quarantine_entire_connected_group(
    policy: dict, review: dict, conflict_type: str
) -> None:
    a = example("a")
    b = example("b", annotations=(("Example", "neutral"),))
    c = example("c", "Example profit rises today")
    review["near_pairs"] = [near(a, c, conflict=conflict_type == "near")]
    if conflict_type == "exact":
        review["exact_groups"] = [exact(a, conflict=True)]
    plan, rows = make_split_plan([a, b, c], review, policy)
    assert not rows
    assert {row["status"] for row in plan["sources"]} == {"quarantined"}
    assert all(row["partition"] is None for row in plan["sources"])


def test_target_coverage_differences_preserve_mixed_targets(
    policy: dict, review: dict
) -> None:
    a = example("a", annotations=(("Example", "positive"), ("Other", "negative")))
    b = example("b")
    review["exact_groups"] = [exact(a, scope=True)]
    _, rows = make_split_plan([a, b], review, policy)
    assert len(rows) == 2
    assert all(row["development_only"] for row in rows)
    assert all(row["partition"] != "test_candidate" for row in rows)


def test_cross_corpus_overlap_is_development_only(policy: dict, review: dict) -> None:
    a = example("a")
    b = example("b", corpus="fiqa", annotations=(("target/aspect", "0.25"),))
    _, rows = make_split_plan([a, b], review, policy)
    assert len(rows) == 2
    assert all(row["development_only"] for row in rows)
    assert all(row["review_reasons"] == ["cross_corpus_overlap"] for row in rows)


def test_unscored_original_test_is_excluded_and_linked_gold_quarantined(
    policy: dict, review: dict
) -> None:
    a = example("train", corpus="fiqa", annotations=(("target/aspect", "0.25"),))
    b = example(
        "test", corpus="fiqa", annotations=(("target/aspect", None),), unscored=True
    )
    c = example("other", "Unrelated unscored text", corpus="fiqa", unscored=True)
    plan, rows = make_split_plan([a, b, c], review, policy)
    assert not rows
    statuses = {s["source_reference"]: s["status"] for s in plan["sources"]}
    assert statuses == {
        "fiqa/train": "quarantined",
        "fiqa/test": "unscored",
        "fiqa/other": "unscored",
    }


def test_snippet_flags_retain_continuous_gold_but_disable_snippet_supervision(
    policy: dict, review: dict
) -> None:
    a = example(
        "a",
        corpus="fiqa",
        annotations=(("first", "0.25"), ("second", "-0.4")),
        snippet_issues=(0,),
    )
    _, rows = make_split_plan([a], review, policy)
    row = rows[0]
    assert row["development_only"]
    assert row["partition"] != "test_candidate"
    assert row["native_record"] == a.native_record
    assert row["annotation_eligibility"][0] == {
        "annotation_index": 0,
        "target_score_supervision": True,
        "snippet_supervision": False,
        "review_required": True,
    }
    assert row["annotation_eligibility"][1]["snippet_supervision"] is True


@pytest.mark.parametrize("corruption", ["group", "hash", "edge", "lineage", "excluded"])
def test_validation_rejects_leakage_or_lost_lineage(
    policy: dict, review: dict, corruption: str
) -> None:
    a = example("a")
    b = example("b", "Different sentence")
    plan, rows = make_split_plan([a, b], review, policy)
    plan, rows = copy.deepcopy((plan, rows))
    for source, row, partition in zip(
        plan["sources"], rows, ["train", "validation"], strict=True
    ):
        source["partition"] = row["partition"] = partition
    if corruption == "group":
        plan["sources"][1]["group_id"] = rows[1]["group_id"] = rows[0]["group_id"]
    elif corruption == "hash":
        plan["sources"][1]["text_sha256"] = rows[1]["text_sha256"] = rows[0][
            "text_sha256"
        ]
    elif corruption == "edge":
        plan["known_near_edges"] = [[rows[0]["text_sha256"], rows[1]["text_sha256"]]]
    elif corruption == "lineage":
        rows[0]["source_references"] = []
    else:
        plan["sources"][0]["status"] = "quarantined"
    with pytest.raises(ValueError):
        validate_split_plan(plan, rows)
