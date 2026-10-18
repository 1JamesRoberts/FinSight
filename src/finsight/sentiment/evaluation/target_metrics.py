"""Locked native-target cohorts, mean seed metrics and paired group uncertainty."""

from collections import Counter
from statistics import mean

from finsight.sentiment.evaluation.final_metrics import (
    bootstrap,
    evaluate_outputs,
    validate_rows,
)
from finsight.sentiment.evaluation.metrics import LABELS, classification_metrics
from finsight.sentiment.evaluation.target_protocol import ARMS, MODELS, SEEDS


def families() -> dict[str, list[str]]:
    result = {
        f"{model}.{arm}": [f"{model}.{arm}.{seed}" for seed in SEEDS]
        for model in MODELS
        for arm in ARMS
    }
    result.update(
        {
            k: [k]
            for k in ("tfidf.headline_only", "tfidf.headline_target", "training_prior")
        }
    )
    return result


def cohort_indices(
    records: list[dict], training_targets: set[str]
) -> dict[str, list[int]]:
    return {
        "multi_target": list(range(len(records))),
        "mixed_label_headlines": [
            i for i, r in enumerate(records) if r["mixed_labels"]
        ],
        "same_label_headlines": [
            i for i, r in enumerate(records) if not r["mixed_labels"]
        ],
        "seen_native_target": [
            i for i, r in enumerate(records) if r["target"] in training_targets
        ],
        "unseen_native_target": [
            i for i, r in enumerate(records) if r["target"] not in training_targets
        ],
    }


def summarize(
    records: list[dict],
    outputs: dict[str, list[dict]],
    training_targets: set[str],
    plan: dict,
) -> tuple[dict, dict]:
    """All labels stay in evaluation; inference has already returned its predictions."""
    expected = families()
    if not records or any(not r["multi_target"] for r in records):
        raise ValueError("nonempty multi-target evaluation examples required")
    if set(outputs) != {k for keys in expected.values() for k in keys}:
        raise ValueError("all 21 locked variants required")
    for rows in outputs.values():
        validate_rows(records, rows)
    cohorts, private_draws = {}, {}
    for name, indices in cohort_indices(records, training_targets).items():
        selected = [records[i] for i in indices]
        counts = {
            "examples": len(selected),
            "headlines": len({r["headline_id"] for r in selected}),
            "duplicate_groups": len({r["group_id"] for r in selected}),
            "class_counts": dict(Counter(r["gold"] for r in selected)),
        }
        if not selected:
            cohorts[name] = {
                **counts,
                "metrics": None,
                "uncertainty": None,
                "paired_differences": None,
            }
            continue
        predictions = {k: [rows[i] for i in indices] for k, rows in outputs.items()}
        metrics = evaluate_outputs(selected, predictions, families=expected)
        for run in metrics["runs"].values():
            run.pop("ece_sensitivity", None)  # Only the locked ten-bin ECE is reported.
        differences = [
            (d["left"], d["right"], "macro_f1") for d in plan["paired_comparisons"]
        ]
        intervals, draws = bootstrap(
            selected,
            predictions,
            families=expected,
            replicates=plan["uncertainty"]["replicates"],
            seed=plan["uncertainty"]["seed"],
            paired_comparisons=differences,
        )
        paired = {}
        for difference in plan["paired_comparisons"]:
            left, right = difference["left"], difference["right"]
            paired[difference["id"]] = {
                **difference,
                "macro_f1_difference": metrics["statistics"][left]["macro_f1"]["mean"]
                - metrics["statistics"][right]["macro_f1"]["mean"],
                "interval_95": intervals["paired_differences"][
                    f"{left} minus {right}: macro_f1"
                ],
            }
        cohorts[name] = {
            **counts,
            "metrics": metrics,
            "uncertainty": intervals,
            "paired_differences": paired,
        }
        private_draws[name] = draws
    # Native names and sparse target-level statistics stay in private results only.
    per_target = {}
    for target in sorted({r["target"] for r in records}):
        indices = [i for i, r in enumerate(records) if r["target"] == target]
        per_target[target] = {
            "support": len(indices),
            "sparse_below_20": len(indices) < 20,
            "class_counts": {
                c: sum(records[i]["gold"] == c for i in indices) for c in LABELS
            },
            "families": {},
        }
        for name, keys in expected.items():
            values = [
                classification_metrics(
                    [records[i]["gold"] for i in indices],
                    [outputs[k][i]["prediction"] for i in indices],
                )
                for k in keys
            ]
            per_target[target]["families"][name] = {
                "runs": dict(zip(keys, values, strict=True)),
                "mean_macro_f1": mean(v["macro_f1"] for v in values),
            }
    return {
        "cohorts": cohorts,
        "native_target_count": len(per_target),
        "sparse_native_target_count": sum(
            v["sparse_below_20"] for v in per_target.values()
        ),
    }, {"bootstrap_draws": private_draws, "per_native_target": per_target}
