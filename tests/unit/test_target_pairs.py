"""Native targets, fixed grouping, matched inputs and protected pair truncation."""

from copy import deepcopy

import pytest

from finsight.data.target_pairs import (
    audit_tokenization,
    build_examples,
    encode_batch,
    summarize_examples,
)


def source(
    identifier="1",
    partition="train",
    group="g1",
    target="Alpha",
    title="Alpha gains while Beta falls",
):
    return {
        "corpus": "sentfin",
        "partition": partition,
        "record_id": identifier,
        "group_id": group,
        "development_only": False,
        "review_reasons": [],
        "native_record": {
            "title": title,
            "sentiments": [
                {"target": target, "label": "positive"},
                {"target": "Beta", "label": "negative"},
            ],
        },
    }


def test_native_annotations_expand_without_mutation_or_splitting():
    row = source(target="ALPHA, Inc.")
    row["development_only"] = True
    row["review_reasons"] = ["target_coverage_difference"]
    original = deepcopy(row)
    examples = build_examples([row])
    assert len(examples) == 2
    assert examples[0].target == "ALPHA, Inc."
    assert examples[0].headline == examples[1].headline == row["native_record"]["title"]
    assert examples[0].label != examples[1].label
    assert examples[0].record_id == examples[1].record_id
    assert examples[0].group_id == examples[1].group_id
    assert examples[0].partition == examples[1].partition == "train"
    assert examples[0].example_id != examples[1].example_id
    assert examples[0].review_reasons == ("target_coverage_difference",)
    assert examples[0].development_only
    assert row == original


def test_nonliteral_targets_are_kept_and_summary_counts_examples():
    examples = build_examples(
        [
            source(target="Unmentioned"),
            source("2", "validation", "g2", target="Unmentioned", title="Other news"),
        ]
    )
    summary = summarize_examples(examples)
    assert len(examples) == 4
    assert summary["train"]["examples"] == 2
    assert summary["train"]["headlines"] == 1
    assert summary["train"]["mixed_label_examples"] == 2
    assert summary["validation"]["seen_native_target_examples"] == 2


@pytest.mark.parametrize("partition", ["test", "test_candidate", "reserved"])
def test_reserved_rows_fail_before_corpus_filtering(partition):
    with pytest.raises(ValueError, match="development"):
        build_examples([{"corpus": "phrasebank", "partition": partition}])


def test_duplicate_ids_group_and_text_overlap_fail():
    row = source()
    with pytest.raises(ValueError, match="unique"):
        build_examples([row, row])
    with pytest.raises(ValueError, match="group"):
        build_examples([row, source("2", "validation", "g1", title="different")])
    with pytest.raises(ValueError, match="headline"):
        build_examples([row, source("2", "validation", "g2")])


@pytest.mark.parametrize(
    "change",
    [
        "empty_target",
        "empty_text",
        "label",
        "duplicate_target",
        "development_only_validation",
    ],
)
def test_invalid_annotations_or_eligibility_fail(change):
    row = source()
    if change == "empty_target":
        row["native_record"]["sentiments"][0]["target"] = " "
    elif change == "empty_text":
        row["native_record"]["title"] = ""
    elif change == "label":
        row["native_record"]["sentiments"][0]["label"] = "unknown"
    elif change == "duplicate_target":
        row["native_record"]["sentiments"][1]["target"] = "Alpha"
    else:
        row.update(partition="validation", development_only=True)
    with pytest.raises(ValueError):
        build_examples([row])


@pytest.fixture
def tokenizer(tmp_path):
    t = pytest.importorskip("transformers")
    vocab = tmp_path / "vocab.txt"
    vocab.write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nalpha\nbeta\ngains\nwhile\nfalls\n"
    )
    return t.BertTokenizerFast(vocab_file=str(vocab))


def test_pair_keeps_entire_target_and_truncates_only_headline(tokenizer):
    examples = build_examples([source(target="Alpha Beta", title="gains " * 20)])
    encoded = encode_batch(tokenizer, examples, "target_pair", max_length=8)
    first = encoded["input_ids"][0]
    assert first == tokenizer.build_inputs_with_special_tokens([5, 6], [7, 7, 7])
    assert len(first) == 8
    assert examples[0].headline == "gains " * 20


def test_headline_only_omits_target_but_uses_same_example_rows(tokenizer):
    examples = build_examples([source()])
    encoded = encode_batch(tokenizer, examples, "matched_headline_only", max_length=16)
    assert encoded["input_ids"][0] == encoded["input_ids"][1]
    pairs = encode_batch(tokenizer, examples, "target_pair", max_length=16)
    assert pairs["input_ids"][0] != pairs["input_ids"][1]


def test_target_overflow_and_unknown_mode_fail(tokenizer):
    examples = build_examples([source(target="Alpha " * 8)])
    with pytest.raises(ValueError, match="target"):
        encode_batch(tokenizer, examples, "target_pair", max_length=8)
    with pytest.raises(ValueError, match="mode"):
        encode_batch(tokenizer, examples, "unknown", max_length=128)
    tokenizer.truncation_side = "left"
    with pytest.raises(ValueError, match="right"):
        encode_batch(tokenizer, examples, "target_pair", max_length=16)


def test_token_audit_exposes_headline_only_label_conflicts(tokenizer):
    examples = build_examples([source()])
    audit = audit_tokenization(tokenizer, examples, max_length=16)["train"]
    assert (
        audit["matched_headline_only"]["identical_inputs_with_conflicting_labels"] == 1
    )
    assert audit["target_pair"]["identical_inputs_with_conflicting_labels"] == 0
    assert audit["target_pair"]["headline_truncated_examples"] == 0
