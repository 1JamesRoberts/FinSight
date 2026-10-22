"""Prediction-free development sampling and explicit annotation review states."""

from finsight.sentiment.artifacts import sha

EXPERIMENT = "organization-mentions-dev-v1"
ENTITY_TYPES = {"company", "other_organization"}


def select_headlines(rows: list[dict], *, size: int = 50, seed: int = 42) -> list[dict]:
    """Rank unique validation IDs, ignoring targets and all detector outputs."""
    unique = {}
    for row in rows:
        if row["partition"] != "validation":
            raise ValueError("only validation sources are allowed")
        identifier, text = row["record_id"], row["headline"]
        if (
            not isinstance(identifier, str)
            or not isinstance(text, str)
            or not text.strip()
        ):
            raise ValueError("nonempty headline identity and text required")
        if identifier in unique and unique[identifier]["headline"] != text:
            raise ValueError("conflicting headline identity")
        unique[identifier] = dict(
            record_id=identifier, headline=text, partition="validation"
        )
    if type(size) is not int or not 0 < size <= len(unique):
        raise ValueError("sample size must fit available headlines")
    identifiers = sorted(
        unique, key=lambda i: (sha(f"{EXPERIMENT}|{seed}|{i}".encode()), i)
    )
    return [unique[i] for i in identifiers[:size]]


def validate_annotations(
    sources: list[dict], annotations: list[dict], *, provisional: bool = False
) -> None:
    """Validate all mentions, including empty rows; never auto-approve drafts.

    Human review is a recorded reviewer assertion, not something software can
    independently establish. The default gate rejects pending assistant drafts.
    """
    expected = {r["record_id"]: r for r in sources}
    observed = {r["record_id"]: r for r in annotations}
    if (
        not expected
        or len(expected) != len(sources)
        or len(observed) != len(annotations)
        or set(observed) != set(expected)
    ):
        raise ValueError("annotation membership differs or contains duplicates")
    for identifier, row in observed.items():
        source = expected[identifier]
        text = source["headline"]
        if (
            source["partition"] != "validation"
            or row["partition"] != "validation"
            or row["headline"] != text
            or row.get("exhaustive") is not True
        ):
            raise ValueError("annotation source differs or is incomplete")
        spans = []
        for entity in row["entities"]:
            start, end = entity["start"], entity["end"]
            if (
                type(start) is not int
                or type(end) is not int
                or not 0 <= start < end <= len(text)
                or text[start:end] != entity["text"]
                or not entity["text"].strip()
                or entity["type"] not in ENTITY_TYPES
                or type(entity["uncertain"]) is not bool
            ):
                raise ValueError("invalid annotation span, text or type")
            spans.append((start, end))
            if entity["uncertain"] and not provisional:
                raise ValueError("uncertain entities require adjudication")
        spans.sort()
        if any(a[1] > b[0] for a, b in zip(spans, spans[1:], strict=False)):
            raise ValueError("overlapping or duplicate annotation spans")
        review = row.get("review", {})
        reviewer = review.get("reviewer")
        if not provisional and (
            review.get("status") != "approved"
            or not isinstance(reviewer, str)
            or not reviewer.strip()
            or reviewer.strip().casefold() in {"assistant", "codex", "ai"}
        ):
            raise ValueError("human review is required for reference evaluation")
