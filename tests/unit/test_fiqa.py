"""Validate FiQA source preservation and reject malformed supervision offline."""

import json
from pathlib import Path

import pytest

from finsight.data.fiqa import load_fiqa
from finsight.data.loaders import DatasetValidationError


def training_row() -> dict:
    return {
        "sentence": "Example improves while Other declines.",
        "info": [
            {
                "target": "Example",
                "snippets": "[nan, 'improves']",
                "aspects": "['Stock/Price']",
                "sentiment_score": "0.250",
            },
            {
                "target": "Other",
                "snippets": "['declines']",
                "aspects": "['Stock/Price']",
                "sentiment_score": "-1",
            },
        ],
    }


def test_preserves_grouped_annotations_and_original_fields(tmp_path: Path) -> None:
    path = tmp_path / "train.json"
    row = training_row()
    path.write_text(json.dumps({"07": row}))
    records = load_fiqa(
        path, revision="pinned", source_type="headline", source_split="train"
    )
    assert len(records) == 1
    record = records[0]
    assert record.source_id == "07"
    assert record.sentence == row["sentence"]
    assert (record.revision, record.source_file, record.source_split) == (
        "pinned",
        "train.json",
        "train",
    )
    assert len(record.annotations) == 2
    first, second = record.annotations
    assert first.original_snippets == "[nan, 'improves']"
    assert first.snippets == (None, "improves")
    assert first.original_score == "0.250"
    assert first.score == 0.25
    assert first.aspects == ("Stock/Price",)
    assert first.original_aspects == "['Stock/Price']"
    assert second.target == "Other"
    assert second.score == -1


def test_test_inputs_keep_opaque_ids_and_missing_gold(tmp_path: Path) -> None:
    path = tmp_path / "test.json"
    path.write_text(
        json.dumps(
            {
                "123$EX": {
                    "sentence": "Example.",
                    "info": [{"target": "EX", "snippets": "['Example']"}],
                }
            }
        )
    )
    record = load_fiqa(
        path, revision="pinned", source_type="post", source_split="test"
    )[0]
    assert record.source_id == "123$EX"
    assert record.annotations[0].score is None
    assert record.annotations[0].original_score is None
    assert record.annotations[0].aspects is None


@pytest.mark.parametrize("score", ["nan", "inf", "-inf", "1.01", "-1.01", "bad", ""])
def test_rejects_invalid_scores(tmp_path: Path, score: str) -> None:
    row = training_row()
    row["info"][0]["sentiment_score"] = score
    path = tmp_path / "train.json"
    path.write_text(json.dumps({"1": row}))
    with pytest.raises(DatasetValidationError):
        load_fiqa(path, revision="r", source_type="headline", source_split="train")


@pytest.mark.parametrize(
    "field,value",
    [
        ("snippets", "[__import__('os').getcwd()]"),
        ("snippets", "[None]"),
        ("snippets", "[]"),
        ("aspects", "[nan]"),
        ("aspects", "{'x': 'y'}"),
        ("target", ""),
    ],
)
def test_rejects_invalid_annotation_fields(
    tmp_path: Path, field: str, value: str
) -> None:
    row = training_row()
    row["info"][0][field] = value
    path = tmp_path / "train.json"
    path.write_text(json.dumps({"1": row}))
    with pytest.raises(DatasetValidationError):
        load_fiqa(path, revision="r", source_type="headline", source_split="train")


@pytest.mark.parametrize(
    "body",
    [
        "{}",
        "[]",
        "{bad",
        '{"1": {}, "1": {}}',
        '{"1": {"sentence": "Example", "info": []}}',
        '{"1": {"sentence": "Example", "info": [{}]}}',
    ],
)
def test_rejects_invalid_json_or_schema(tmp_path: Path, body: str) -> None:
    path = tmp_path / "train.json"
    path.write_text(body)
    with pytest.raises(DatasetValidationError):
        load_fiqa(path, revision="r", source_type="headline", source_split="train")


def test_rejects_missing_train_gold_and_added_test_gold(tmp_path: Path) -> None:
    path = tmp_path / "source.json"
    row = training_row()
    path.write_text(json.dumps({"1": row}))
    with pytest.raises(DatasetValidationError):
        load_fiqa(path, revision="r", source_type="headline", source_split="test")
    del row["info"][0]["sentiment_score"]
    path.write_text(json.dumps({"1": row}))
    with pytest.raises(DatasetValidationError):
        load_fiqa(path, revision="r", source_type="headline", source_split="train")
