"""Read reviewed source formats without changing text or assigning splits."""

import ast
import csv
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from zipfile import ZipFile


class DatasetValidationError(ValueError):
    """A source record does not satisfy its dataset schema."""


class SentimentLabel(StrEnum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


@dataclass(frozen=True)
class PhraseBankRecord:
    source_id: str
    sentence: str
    original_label: str
    label: SentimentLabel
    agreement: str
    revision: str
    source_member: str


@dataclass(frozen=True)
class EntitySentiment:
    target: str
    original_label: str
    label: SentimentLabel


@dataclass(frozen=True)
class SEntFiNRecord:
    source_id: str
    title: str
    original_decisions: str
    sentiments: tuple[EntitySentiment, ...]
    original_words: str
    word_count: int
    revision: str


@dataclass(frozen=True)
class EntityAliasRecord:
    source_id: str
    symbol: str
    entity: str
    revision: str


def _label(raw: str, location: str) -> SentimentLabel:
    try:
        return SentimentLabel(raw.strip().casefold())
    except ValueError:
        raise DatasetValidationError(f"{location}: unknown sentiment label") from None


def load_phrasebank(
    path: Path, *, revision: str, agreement: str = "50"
) -> tuple[PhraseBankRecord, ...]:
    """Read one ISO-8859-1 archive member; agreement subsets remain separate."""
    if agreement not in {"50", "66", "75", "all"}:
        raise ValueError("agreement must be 50, 66, 75, or all")
    member = f"FinancialPhraseBank-v1.0/Sentences_{agreement.title()}Agree.txt"
    with ZipFile(path) as archive:
        if archive.namelist().count(member) != 1:
            raise DatasetValidationError("archive must contain one requested member")
        lines = archive.read(member).decode("iso-8859-1").splitlines()
    records = []
    for number, line in enumerate(lines, start=1):
        location = f"PhraseBank {agreement}, line {number}"
        if "@" not in line:
            raise DatasetValidationError(f"{location}: missing label delimiter")
        sentence, raw_label = line.rsplit("@", 1)
        if not sentence.strip():
            raise DatasetValidationError(f"{location}: empty sentence")
        records.append(
            PhraseBankRecord(
                f"{agreement}:{number}",
                sentence,
                raw_label,
                _label(raw_label, location),
                agreement,
                revision,
                member,
            )
        )
    if not records:
        raise DatasetValidationError("PhraseBank source is empty")
    return tuple(records)


# A narrow fallback for the release's unescaped apostrophes in entity names.
# Only quoted target/label pairs and comma separators are accepted; no evaluation.
_DECISION_PAIR = re.compile(r"'(?P<target>.*?)'\s*:\s*'(?P<label>[^']*)'", re.DOTALL)


def parse_entity_sentiments(raw: str, location: str) -> tuple[EntitySentiment, ...]:
    """Parse a string dictionary, including known unescaped-apostrophe records.

    Valid Python dictionary syntax is inspected as an AST of string constants,
    never executed. On syntax failure, a restricted source-format parser is used.
    Duplicate keys, empty entities, expressions, and unknown labels are rejected.
    """
    pairs: list[tuple[str, str]] = []
    try:
        node = ast.parse(raw.strip(), mode="eval").body
    except SyntaxError:
        value = raw.strip()
        if not (value.startswith("{") and value.endswith("}")):
            raise DatasetValidationError(f"{location}: invalid decisions") from None
        inner = value[1:-1]
        end = 0
        for match in _DECISION_PAIR.finditer(inner):
            expected_gap = "" if not pairs else ","
            if inner[end : match.start()].strip() != expected_gap:
                raise DatasetValidationError(f"{location}: invalid decisions") from None
            pairs.append((match["target"], match["label"]))
            end = match.end()
        if inner[end:].strip():
            raise DatasetValidationError(f"{location}: invalid decisions") from None
    else:
        if not isinstance(node, ast.Dict):
            raise DatasetValidationError(f"{location}: decisions must be a dictionary")
        for key, value_node in zip(node.keys, node.values, strict=True):
            if not (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and isinstance(value_node, ast.Constant)
                and isinstance(value_node.value, str)
            ):
                raise DatasetValidationError(f"{location}: decisions require strings")
            pairs.append((key.value, value_node.value))
    if not pairs:
        raise DatasetValidationError(f"{location}: empty decisions")
    seen: set[str] = set()
    result = []
    for target, raw_label in pairs:
        identity = target.strip().casefold()
        if not identity or identity in seen:
            raise DatasetValidationError(f"{location}: empty or duplicate target")
        seen.add(identity)
        result.append(EntitySentiment(target, raw_label, _label(raw_label, location)))
    return tuple(result)


def _csv_rows(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    """Require the pinned schema, no missing cells, and unique numeric source IDs."""
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, strict=True)
        if tuple(reader.fieldnames or ()) != columns:
            raise DatasetValidationError("CSV headers do not match the source schema")
        rows = []
        seen: set[str] = set()
        try:
            for number, row in enumerate(reader, start=2):
                if None in row or any(
                    value is None or not value.strip() for value in row.values()
                ):
                    raise DatasetValidationError(
                        f"CSV row {number}: missing/extra cell"
                    )
                source_id = row[columns[0]]
                if not source_id.isdecimal() or str(int(source_id)) in seen:
                    raise DatasetValidationError(
                        f"CSV row {number}: invalid/duplicate ID"
                    )
                seen.add(str(int(source_id)))
                rows.append(row)
        except csv.Error:
            raise DatasetValidationError("invalid CSV quoting") from None
    if not rows:
        raise DatasetValidationError("CSV source is empty")
    return rows


def load_sentfin(path: Path, *, revision: str) -> tuple[SEntFiNRecord, ...]:
    """Preserve headline groups and raw dictionary/word-count fields."""
    records = []
    for row in _csv_rows(path, ("S No.", "Title", "Decisions", "Words")):
        location = f"SEntFiN ID {row['S No.']}"
        if not row["Words"].isdecimal() or int(row["Words"]) <= 0:
            raise DatasetValidationError(f"{location}: word count must be positive")
        records.append(
            SEntFiNRecord(
                row["S No."],
                row["Title"],
                row["Decisions"],
                parse_entity_sentiments(row["Decisions"], location),
                row["Words"],
                int(row["Words"]),
                revision,
            )
        )
    return tuple(records)


def load_entity_aliases(path: Path, *, revision: str) -> tuple[EntityAliasRecord, ...]:
    """Preserve released alias strings; do not infer company/ticker mappings."""
    rows = _csv_rows(path, ("", "Symbol", "Entity"))
    if len({row["Symbol"] for row in rows}) != len(rows):
        raise DatasetValidationError("entity database contains duplicate symbols")
    return tuple(
        EntityAliasRecord(row[""], row["Symbol"], row["Entity"], revision)
        for row in rows
    )
