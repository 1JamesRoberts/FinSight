"""Synthetic examples only: no third-party text or downloads in tests."""

import csv
from pathlib import Path
from zipfile import ZipFile

import pytest

from finsight.data.loaders import (
    DatasetValidationError,
    SentimentLabel,
    load_entity_aliases,
    load_phrasebank,
    load_sentfin,
    parse_entity_sentiments,
)


def write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as source:
        writer = csv.writer(source)
        writer.writerow(headers)
        writer.writerows(rows)


def test_phrasebank_preserves_text_encoding_and_last_delimiter(tmp_path: Path) -> None:
    path = tmp_path / "sample.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "FinancialPhraseBank-v1.0/Sentences_50Agree.txt",
            "Café @ Example gains. @positive\n".encode("iso-8859-1"),
        )
    (record,) = load_phrasebank(path, revision="source-revision")
    assert record.sentence == "Café @ Example gains. "
    assert record.original_label == "positive"
    assert record.label is SentimentLabel.POSITIVE
    assert record.source_id == "50:1"
    assert record.revision == "source-revision"


@pytest.mark.parametrize(
    "text", ["missing separator", "@neutral", "Example@unknown", ""]
)
def test_phrasebank_rejects_bad_rows(tmp_path: Path, text: str) -> None:
    path = tmp_path / "sample.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("FinancialPhraseBank-v1.0/Sentences_50Agree.txt", text)
    with pytest.raises(DatasetValidationError):
        load_phrasebank(path, revision="test")


def test_phrasebank_rejects_missing_member_and_bad_agreement(tmp_path: Path) -> None:
    path = tmp_path / "sample.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("other.txt", "Example@neutral")
    with pytest.raises(DatasetValidationError):
        load_phrasebank(path, revision="test")
    with pytest.raises(ValueError, match="agreement"):
        load_phrasebank(path, revision="test", agreement="../other")


@pytest.mark.parametrize(
    "raw",
    [
        "{'Example': 'positive', 'Other': 'negative'}",
        '{"Example": "positive", "Other": "negative"}',
    ],
)
def test_decisions_preserve_targets_and_conflicting_labels(raw: str) -> None:
    result = parse_entity_sentiments(raw, "test")
    assert [(r.target, r.label) for r in result] == [
        ("Example", SentimentLabel.POSITIVE),
        ("Other", SentimentLabel.NEGATIVE),
    ]


def test_decisions_support_release_apostrophe_defect() -> None:
    result = parse_entity_sentiments("{'Example's Fund': 'neutral'}", "test")
    assert result[0].target == "Example's Fund"
    assert result[0].original_label == "neutral"


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        "[]",
        "{'Example': 'other'}",
        "{'': 'positive'}",
        "{'Example': 'positive', 'Example': 'negative'}",
        "{'Example': 'positive', ' example ': 'negative'}",
        "{'Example': 1}",
        "{1: 'positive'}",
        "{'Example': str('positive')}",
        "__import__('os').system('echo unsafe')",
        "{'Example's Fund': 'neutral'}; invalid",
        "{'Example's Fund': 'neutral', garbage}",
    ],
)
def test_decisions_reject_invalid_or_executable_input(raw: str) -> None:
    with pytest.raises(DatasetValidationError):
        parse_entity_sentiments(raw, "test")


def test_sentfin_preserves_raw_fields_and_headline_group(tmp_path: Path) -> None:
    path = tmp_path / "sentfin.csv"
    raw = "{'Example's Fund': 'neutral', 'Other': 'positive'}"
    write_csv(
        path,
        ["S No.", "Title", "Decisions", "Words"],
        [["0", "Example headline", raw, "2"]],
    )
    (record,) = load_sentfin(path, revision="test")
    assert record.source_id == "0"
    assert record.title == "Example headline"
    assert record.original_decisions == raw
    assert record.word_count == 2
    assert record.original_words == "2"
    assert len(record.sentiments) == 2


@pytest.mark.parametrize(
    "rows",
    [
        [["0", "Headline", "{'Example': 'neutral'}", "-1"]],
        [["0", "Headline", "{'Example': 'neutral'}", "0"]],
        [["0", "Headline", "{'Example': 'neutral'}", "2.5"]],
        [["0", "", "{'Example': 'neutral'}", "2"]],
        [["not-id", "Headline", "{'Example': 'neutral'}", "2"]],
        [["0", "Headline", "{'Example': 'neutral'}"]],
        [["0", "Headline", "{'Example': 'neutral'}", "2", "extra"]],
        [
            ["0", "Headline", "{'Example': 'neutral'}", "2"],
            ["00", "Other", "{'Other': 'positive'}", "1"],
        ],
    ],
)
def test_sentfin_rejects_invalid_schema_rows(
    tmp_path: Path, rows: list[list[str]]
) -> None:
    path = tmp_path / "sentfin.csv"
    write_csv(path, ["S No.", "Title", "Decisions", "Words"], rows)
    with pytest.raises(DatasetValidationError):
        load_sentfin(path, revision="test")


def test_csv_rejects_unexpected_headers_and_empty_source(tmp_path: Path) -> None:
    path = tmp_path / "sentfin.csv"
    write_csv(path, ["text", "label"], [])
    with pytest.raises(DatasetValidationError, match="headers"):
        load_sentfin(path, revision="test")
    write_csv(path, ["S No.", "Title", "Decisions", "Words"], [])
    with pytest.raises(DatasetValidationError, match="empty"):
        load_sentfin(path, revision="test")


def test_entity_database_preserves_alias_text_and_rejects_duplicate_symbol(
    tmp_path: Path,
) -> None:
    path = tmp_path / "entities.csv"
    write_csv(path, ["", "Symbol", "Entity"], [["0", "EX", "Example; Example Corp"]])
    (record,) = load_entity_aliases(path, revision="test")
    assert record.entity == "Example; Example Corp"
    assert record.symbol == "EX"
    write_csv(
        path, ["", "Symbol", "Entity"], [["0", "EX", "Example"], ["1", "EX", "Other"]]
    )
    with pytest.raises(DatasetValidationError, match="duplicate symbols"):
        load_entity_aliases(path, revision="test")
