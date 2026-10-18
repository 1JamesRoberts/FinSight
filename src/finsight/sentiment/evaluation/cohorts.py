"""Select development sentence cohorts without admitting reserved test records."""

from typing import Any


def sentence_cohorts(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Never admit test candidates, FiQA scores, or mixed-target sentence gold."""
    if any(row["partition"] not in {"train", "validation"} for row in records):
        raise ValueError("sentence checks require development records only")
    return {
        "phrasebank": [row for row in records if row["corpus"] == "phrasebank"],
        "sentfin_single_target": [
            row
            for row in records
            if row["corpus"] == "sentfin"
            and len(row["native_record"]["sentiments"]) == 1
        ],
    }


def _gold(row: dict[str, Any]) -> str:
    native = row["native_record"]
    return (
        native["label"]
        if row["corpus"] == "phrasebank"
        else native["sentiments"][0]["label"]
    )


def sentence_and_label(row: dict[str, Any]) -> tuple[str, str]:
    native = row["native_record"]
    if row["corpus"] == "phrasebank":
        return native["sentence"], native["label"]
    return native["title"], native["sentiments"][0]["label"]
