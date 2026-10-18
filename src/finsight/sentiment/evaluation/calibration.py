"""Probability diagnostics and scalar temperature fitting; no data loading."""

import math
from statistics import mean
from typing import Any

from finsight.sentiment.evaluation.metrics import LABELS

BOUNDS = (0.05, 20.0)
FLOOR = 1e-15


def calibration_metrics(rows: list[dict[str, Any]], bins: int = 10) -> dict[str, Any]:
    """Top-label equal-width ECE; unnormalized multiclass Brier (range 0–2)."""
    if not rows or type(bins) is not int or bins < 1:
        raise ValueError("nonempty rows and positive integer bins required")
    cells: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    brier, losses, confidences, correct = [], [], [], []
    for row in rows:
        p = row["probabilities"]
        if row["gold"] not in LABELS or set(p) != set(LABELS):
            raise ValueError("unknown gold or probability classes")
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
            for v in p.values()
        ):
            raise ValueError("invalid probability values")
        if not math.isclose(sum(p.values()), 1, abs_tol=1e-6):
            raise ValueError("probabilities must sum to one")
        confidence = max(p.values())
        predicted = row["prediction"]
        if predicted not in LABELS or p[predicted] != confidence:
            raise ValueError("prediction must be a probability argmax")
        ok = predicted == row["gold"]
        cells[min(int(confidence * bins), bins - 1)].append((confidence, ok))
        confidences.append(confidence)
        correct.append(ok)
        brier.append(sum((p[label] - (label == row["gold"])) ** 2 for label in LABELS))
        losses.append(-math.log(max(p[row["gold"]], 1e-15)))
    reliability = []
    for i, cell in enumerate(cells):
        reliability.append(
            dict(
                lower=i / bins,
                upper=(i + 1) / bins,
                count=len(cell),
                confidence=mean(c for c, _ in cell) if cell else None,
                accuracy=mean(ok for _, ok in cell) if cell else None,
            )
        )
    return {
        "samples": len(rows),
        "bin_count": bins,
        "bins": reliability,
        "ece": sum(
            b["count"] / len(rows) * abs(b["accuracy"] - b["confidence"])
            for b in reliability
            if b["count"]
        ),
        "brier": mean(brier),
        "nll": mean(losses),
        "nll_clip": 1e-15,
        "accuracy": mean(correct),
        "mean_confidence": mean(confidences),
        "confidence_minus_accuracy": mean(confidences) - mean(correct),
        "high_confidence_errors": sum(
            not ok and c >= 0.9 for c, ok in zip(confidences, correct, strict=True)
        ),
        "errors": sum(not ok for ok in correct),
    }


def _logs(rows: list[dict[str, Any]], floor: float) -> list[list[float]]:
    if (
        type(floor) not in (int, float)
        or not math.isfinite(floor)
        or not 0 < floor < 1 / 3
    ):
        raise ValueError("probability floor must be finite and between zero and 1/3")
    calibration_metrics(rows)
    return [[math.log(max(r["probabilities"][c], floor)) for c in LABELS] for r in rows]


def _softmax(values: list[float], beta: float) -> list[float]:
    maximum = max(values)
    weights = [math.exp((v - maximum) * beta) for v in values]
    total = sum(weights)
    return [w / total for w in weights]


def apply_temperature(
    rows: list[dict[str, Any]], temperature: float, *, floor: float = FLOOR
) -> list[dict[str, Any]]:
    """Return new rows with p^(1/T), normalized; preserve saved argmax tie choice."""
    if (
        type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("temperature must be positive and finite")
    # Scaling log-probabilities by 1/T changes confidence, preserving class order.
    logs = _logs(rows, floor)
    adjusted = []
    for row, values in zip(rows, logs, strict=True):
        probabilities = dict(
            zip(LABELS, _softmax(values, 1 / temperature), strict=True)
        )
        if temperature == 1 and sum(row["probabilities"].values()) == 1:
            probabilities = dict(row["probabilities"])
        if probabilities[row["prediction"]] != max(probabilities.values()):
            raise ValueError("temperature changed the predicted class")
        adjusted.append({**row, "probabilities": probabilities})
    calibration_metrics(adjusted)
    return adjusted


def fit_temperature(
    rows: list[dict[str, Any]], *, floor: float = FLOOR
) -> dict[str, Any]:
    """Minimize convex NLL in beta=1/T on fixed [0.05,20] T bounds."""
    logs = _logs(rows, floor)
    gold = [LABELS.index(r["gold"]) for r in rows]

    def derivative(beta: float) -> float:
        return mean(
            sum(p * v for p, v in zip(_softmax(vs, beta), vs, strict=True)) - vs[g]
            for vs, g in zip(logs, gold, strict=True)
        )

    low, high = 1 / BOUNDS[1], 1 / BOUNDS[0]
    boundary = None
    if all(max(vs) == min(vs) for vs in logs):
        temperature = 1.0  # NLL is constant; retain identity.
    elif derivative(low) >= 0:
        temperature, boundary = BOUNDS[1], "upper_temperature"
    elif derivative(high) <= 0:
        temperature, boundary = BOUNDS[0], "lower_temperature"
    else:
        for _ in range(80):
            middle = (low + high) / 2
            if derivative(middle) > 0:
                high = middle
            else:
                low = middle
        temperature = 1 / ((low + high) / 2)
    return {
        "temperature": temperature,
        "boundary": boundary,
        "fit_samples": len(rows),
        "floored_probability_values": sum(
            p < floor for r in rows for p in r["probabilities"].values()
        ),
    }
