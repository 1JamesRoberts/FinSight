"""Fixed-label classification metrics, independent of model and experiment code."""

from typing import Any

LABELS = ("negative", "neutral", "positive")


def classification_metrics(gold: list[str], predictions: list[str]) -> dict[str, Any]:
    """Fixed class order; undefined precision/recall/F1 are zero, never omitted."""
    if not gold or len(gold) != len(predictions):
        raise ValueError("metrics require equal nonempty gold/prediction lists")
    if any(label not in LABELS for label in gold + predictions):
        raise ValueError("unknown classification label")
    # Fixed axes keep absent classes visible and macro F1 comparable across cohorts.
    matrix = [[0] * 3 for _ in LABELS]
    for actual, predicted in zip(gold, predictions, strict=True):
        matrix[LABELS.index(actual)][LABELS.index(predicted)] += 1
    per_class = {}
    for i, label in enumerate(LABELS):
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        correct = matrix[i][i]
        precision = correct / predicted if predicted else 0.0
        recall = correct / support if support else 0.0
        per_class[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": 2 * correct / (support + predicted) if support + predicted else 0.0,
        }
    return {
        "samples": len(gold),
        "class_order": list(LABELS),
        "accuracy": sum(matrix[i][i] for i in range(3)) / len(gold),
        "macro_f1": sum(row["f1"] for row in per_class.values()) / 3,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "confusion_matrix_axes": "rows=gold, columns=predicted",
    }
