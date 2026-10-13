"""Fixed-label metrics and paired duplicate-group bootstrap for locked evaluation."""

from collections import Counter
from statistics import mean, stdev
from typing import Any

import numpy as np

from finsight.sentiment.error_analysis import sha
from finsight.sentiment.transformer_analysis import calibration_metrics
from finsight.sentiment.validation import LABELS, classification_metrics


def validate_rows(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    """Require ordered membership/gold, valid labels and complete probabilities."""
    if not records or len(rows) != len(records):
        raise ValueError("complete nonempty prediction membership required")
    ids = [r["record_id"] for r in records]
    if len(set(ids)) != len(ids) or [r["record_id"] for r in rows] != ids:
        raise ValueError("prediction membership/order mismatch")
    if [r["gold"] for r in rows] != [r["gold"] for r in records]:
        raise ValueError("prediction gold mismatch")
    classification_metrics([r["gold"] for r in rows], [r["prediction"] for r in rows])
    probabilities = ["probabilities" in r for r in rows]
    if any(probabilities) and not all(probabilities):
        raise ValueError("partial probability outputs")
    if all(probabilities):
        calibration_metrics(rows)


def family_keys() -> dict[str, list[str]]:
    families = {
        name: [name]
        for name in (
            "training_prior",
            "vader",
            "tfidf_logreg",
            "tfidf_svm",
            "finbert_pretrained",
        )
    }
    for model in ("finbert", "roberta_base", "deberta_base"):
        for stage in ("raw", "calibrated"):
            families[f"{model}.{stage}"] = [
                f"{model}.{s}.{stage}" for s in (42, 43, 44)
            ]
    for method in ("soft", "hard"):
        families[method] = [f"{method}.{s}" for s in (42, 43, 44)]
    return families


def evaluate_outputs(
    records: list[dict[str, Any]],
    outputs: dict[str, list[dict[str, Any]]],
    *,
    families: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    families = families or family_keys()
    if set(outputs) != {key for keys in families.values() for key in keys}:
        raise ValueError("exact declared comparison variants required")
    runs = {}
    for key, rows in outputs.items():
        validate_rows(records, rows)
        runs[key] = {
            "classification": classification_metrics(
                [r["gold"] for r in rows], [r["prediction"] for r in rows]
            ),
            "probability_metrics": calibration_metrics(rows)
            if "probabilities" in rows[0]
            else None,
            "ece_sensitivity": {
                str(b): calibration_metrics(rows, b)["ece"] for b in (5, 15)
            }
            if "probabilities" in rows[0]
            else None,
        }
    statistics = {}
    for family, keys in families.items():
        metrics = {
            m: [runs[k]["classification"][m] for k in keys]
            for m in ("macro_f1", "accuracy")
        }
        if runs[keys[0]]["probability_metrics"] is not None:
            metrics.update(
                {
                    m: [runs[k]["probability_metrics"][m] for k in keys]
                    for m in (
                        "nll",
                        "brier",
                        "ece",
                        "mean_confidence",
                        "confidence_minus_accuracy",
                    )
                }
            )
        statistics[family] = {
            m: {"mean": mean(v), "sample_std": stdev(v) if len(v) > 1 else None}
            for m, v in metrics.items()
        }
    return {"runs": runs, "statistics": statistics}


def bootstrap(
    records: list[dict[str, Any]],
    outputs: dict[str, list[dict[str, Any]]],
    *,
    families: dict[str, list[str]] | None = None,
    replicates: int = 1000,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Same complete-group draws for every variant; mean metrics across model seeds."""
    families = families or family_keys()
    if type(replicates) is not int or replicates < 1:
        raise ValueError("positive bootstrap replicates required")
    if set(outputs) != {k for keys in families.values() for k in keys}:
        raise ValueError("bootstrap comparison variants differ")
    groups = sorted({r["group_id"] for r in records})
    group_index = {g: i for i, g in enumerate(groups)}
    ids = np.array([group_index[r["group_id"]] for r in records])
    draws = np.random.default_rng(seed).integers(
        0, len(groups), size=(replicates, len(groups))
    )
    weights = np.stack([np.bincount(d, minlength=len(groups)) for d in draws]).astype(
        np.int64
    )
    group_sizes = np.bincount(ids, minlength=len(groups))
    denominators = weights @ group_sizes
    gold = np.array([LABELS.index(r["gold"]) for r in records])
    samples = {}
    for key, rows in outputs.items():
        validate_rows(records, rows)
        pred = np.array([LABELS.index(r["prediction"]) for r in rows])
        cells = np.zeros((len(groups), 9))
        np.add.at(cells, (ids, gold * 3 + pred), 1)
        matrices = (weights @ cells).reshape(replicates, 3, 3)
        tp = matrices.diagonal(axis1=1, axis2=2)
        divisor = matrices.sum(axis=1) + matrices.sum(axis=2)
        f1 = np.divide(2 * tp, divisor, out=np.zeros_like(tp), where=divisor != 0).mean(
            axis=1
        )
        values = {"macro_f1": f1, "accuracy": tp.sum(axis=1) / denominators}
        if "probabilities" in rows[0]:
            p = np.array([[r["probabilities"][c] for c in LABELS] for r in rows])
            nll = -np.log(np.maximum(p[np.arange(len(rows)), gold], 1e-15))
            brier = ((p - np.eye(3)[gold]) ** 2).sum(axis=1)
            for metric, losses in (("nll", nll), ("brier", brier)):
                total = np.bincount(ids, weights=losses, minlength=len(groups))
                values[metric] = (weights @ total) / denominators
            confidence = p.max(axis=1)
            bins = np.minimum((confidence * 10).astype(int), 9)
            correct, confidences = [np.zeros((len(groups), 10)) for _ in range(2)]
            np.add.at(correct, (ids, bins), pred == gold)
            np.add.at(confidences, (ids, bins), confidence)
            # count*|accuracy-confidence| == |correct sum-confidence sum|.
            values["ece"] = (
                np.abs(weights @ correct - weights @ confidences).sum(axis=1)
                / denominators
            )
        samples[key] = values
    family_samples = {
        f: {m: np.mean([samples[k][m] for k in keys], axis=0) for m in samples[keys[0]]}
        for f, keys in families.items()
    }

    def interval(values: Any) -> list[float]:
        return np.quantile(values, [0.025, 0.975], method="linear").tolist()

    intervals = {
        f: {m: interval(v) for m, v in values.items()}
        for f, values in family_samples.items()
    }
    comparisons = {}
    declared = [
        ("deberta_base.raw", "roberta_base.raw", "macro_f1"),
        ("deberta_base.raw", "finbert.raw", "macro_f1"),
        *(
            (f"{m}.raw", "tfidf_logreg", "macro_f1")
            for m in ("finbert", "roberta_base", "deberta_base")
        ),
        *(
            (f"{m}.calibrated", f"{m}.raw", "nll")
            for m in ("finbert", "roberta_base", "deberta_base")
        ),
        ("soft", "deberta_base.raw", "macro_f1"),
        ("hard", "deberta_base.raw", "macro_f1"),
        ("soft", "deberta_base.calibrated", "nll"),
    ]
    for a, b, metric in declared:
        if (
            a in family_samples
            and b in family_samples
            and metric in family_samples[a]
            and metric in family_samples[b]
        ):
            comparisons[f"{a} minus {b}: {metric}"] = interval(
                family_samples[a][metric] - family_samples[b][metric]
            )
    private = {"groups": groups, "draws": draws.tolist()}
    return {
        "replicates": replicates,
        "seed": seed,
        "unit": "duplicate_group",
        "percentile_method": "linear",
        "confidence_level": 0.95,
        "groups": len(groups),
        "group_sizes": dict(Counter(group_sizes.tolist())),
        "draws_sha256": sha(draws.astype("<i8").tobytes()),
        "intervals": intervals,
        "paired_differences": comparisons,
    }, private
