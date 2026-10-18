"""Equal-weight voting rules over aligned model predictions."""

from collections import Counter
from statistics import mean
from typing import Any

from finsight.sentiment.evaluation.calibration import calibration_metrics
from finsight.sentiment.evaluation.metrics import LABELS


def combine(members: list[dict[str, Any]]) -> dict[str, Any]:
    """Three equal members; soft ties use class order, hard three-way ties neutral."""
    if len(members) != 3 or any(
        r["record_id"] != members[0]["record_id"] or r["gold"] != members[0]["gold"]
        for r in members
    ):
        raise ValueError("three aligned members required")
    calibration_metrics(members)
    probabilities = {c: mean(r["probabilities"][c] for r in members) for c in LABELS}
    counts = Counter(r["prediction"] for r in members)
    tie = len(counts) == 3
    return {
        "record_id": members[0]["record_id"],
        "gold": members[0]["gold"],
        "soft": {
            "prediction": max(LABELS, key=probabilities.get),
            "probabilities": probabilities,
        },
        "hard": {
            "prediction": "neutral" if tie else max(LABELS, key=lambda c: counts[c]),
            "votes": {c: counts[c] for c in LABELS},
            "tie": tie,
        },
    }
