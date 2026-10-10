"""Load original-format FiQA Task 1 groups, including unscored test inputs."""

import ast
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from finsight.data.loaders import DatasetValidationError


@dataclass(frozen=True)
class FiQAAnnotation:
    target: str
    original_snippets: str
    snippets: tuple[str | None, ...]
    original_aspects: str | None
    aspects: tuple[str, ...] | None
    original_score: str | None
    score: float | None


@dataclass(frozen=True)
class FiQARecord:
    source_id: str
    sentence: str
    annotations: tuple[FiQAAnnotation, ...]
    source_type: Literal["headline", "post"]
    source_split: Literal["train", "test"]
    revision: str
    source_file: str


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DatasetValidationError("FiQA JSON contains a duplicate key")
        result[key] = value
    return result


def _text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError("FiQA requires nonempty string fields")
    return value


def _strings(raw: str, *, allow_missing: bool = False) -> tuple[str | None, ...]:
    """Inspect list constants; the known bare nan snippet means missing, not text."""
    try:
        node = ast.parse(raw, mode="eval").body
    except (SyntaxError, ValueError):
        raise DatasetValidationError(
            "FiQA contains an invalid annotation list"
        ) from None
    if not isinstance(node, ast.List) or not node.elts:
        raise DatasetValidationError("FiQA annotations require nonempty lists")
    result: list[str | None] = []
    for item in node.elts:
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            result.append(_text(item.value))
        elif allow_missing and isinstance(item, ast.Name) and item.id == "nan":
            result.append(None)
        else:
            raise DatasetValidationError("FiQA annotation list contains invalid values")
    return tuple(result)


def load_fiqa(
    path: Path,
    *,
    revision: str,
    source_type: Literal["headline", "post"],
    source_split: Literal["train", "test"],
) -> tuple[FiQARecord, ...]:
    """Preserve source groups and splits; never invent gold scores or class labels."""
    if source_type not in {"headline", "post"} or source_split not in {"train", "test"}:
        raise ValueError("unsupported FiQA source type or split")
    try:
        data = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
    except (json.JSONDecodeError, UnicodeError):
        raise DatasetValidationError("FiQA source is not valid UTF-8 JSON") from None
    if not isinstance(data, dict) or not data:
        raise DatasetValidationError("FiQA requires a nonempty record dictionary")
    records = []
    for source_id, row in data.items():
        _text(source_id)
        if not isinstance(row, dict) or set(row) != {"sentence", "info"}:
            raise DatasetValidationError(f"FiQA ID {source_id}: invalid record schema")
        sentence = _text(row["sentence"])
        info = row["info"]
        if not isinstance(info, list) or not info:
            raise DatasetValidationError(f"FiQA ID {source_id}: missing annotations")
        annotations = []
        for annotation in info:
            expected = {"target", "snippets"}
            if source_split == "train":
                expected |= {"aspects", "sentiment_score"}
            if not isinstance(annotation, dict) or set(annotation) != expected:
                raise DatasetValidationError(
                    f"FiQA ID {source_id}: invalid annotation schema"
                )
            target = _text(annotation["target"])
            raw_snippets = _text(annotation["snippets"])
            snippets = _strings(raw_snippets, allow_missing=True)
            raw_aspects = raw_score = None
            aspects = None
            score = None
            if source_split == "train":
                raw_aspects = _text(annotation["aspects"])
                parsed_aspects = _strings(raw_aspects)
                aspects = tuple(value for value in parsed_aspects if value is not None)
                raw_score = _text(annotation["sentiment_score"])
                try:
                    score = float(raw_score)
                except ValueError:
                    raise DatasetValidationError(
                        f"FiQA ID {source_id}: invalid score"
                    ) from None
                if not math.isfinite(score) or not -1 <= score <= 1:
                    raise DatasetValidationError(
                        f"FiQA ID {source_id}: score outside [-1, 1]"
                    )
            annotations.append(
                FiQAAnnotation(
                    target,
                    raw_snippets,
                    snippets,
                    raw_aspects,
                    aspects,
                    raw_score,
                    score,
                )
            )
        records.append(
            FiQARecord(
                source_id,
                sentence,
                tuple(annotations),
                source_type,
                source_split,
                revision,
                path.name,
            )
        )
    return tuple(records)
