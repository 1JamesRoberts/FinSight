"""Cross-fit scalar temperatures on checked validation predictions only."""

import argparse
import json
import platform
import subprocess
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from sklearn.model_selection import StratifiedGroupKFold

from finsight.data.benchmark import load_development_records
from finsight.sentiment.artifacts import sha
from finsight.sentiment.evaluation.calibration import _logs as _logs
from finsight.sentiment.evaluation.calibration import _softmax as _softmax
from finsight.sentiment.evaluation.calibration import (
    apply_temperature,
    calibration_metrics,
    fit_temperature,
)
from finsight.sentiment.evaluation.cohorts import sentence_and_label, sentence_cohorts
from finsight.sentiment.evaluation.metrics import LABELS, classification_metrics
from finsight.sentiment.experiments.sentence_error_analysis import (
    SEEDS,
    checked_run,
    json_body,
)

BOUNDS = (0.05, 20.0)
FLOOR = 1e-15
METRICS = ("ece", "brier", "nll", "accuracy", "mean_confidence")


def validation_folds(
    records: list[dict[str, Any]],
) -> list[tuple[list[int], list[int]]]:
    """Five shared stratified group folds; reject test, train and duplicate IDs."""
    if not records or any(r["partition"] != "validation" for r in records):
        raise ValueError("validation records only")
    ids = [r["record_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("unique validation IDs required")
    gold = [sentence_and_label(r)[1] for r in records]
    groups = [r["group_id"] for r in records]
    if set(gold) != set(LABELS) or any(
        len({groups[i] for i, g in enumerate(gold) if g == c}) < 5 for c in LABELS
    ):
        raise ValueError("five groups per class required")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = []
    for fit, assess in splitter.split(ids, gold, groups):
        fit, assess = fit.tolist(), assess.tolist()
        if {groups[i] for i in fit} & {groups[i] for i in assess}:
            raise ValueError("duplicate group crossed calibration folds")
        if set(gold[i] for i in fit) != set(LABELS):
            raise ValueError("calibration fit lacks a class")
        folds.append((fit, assess))
    return folds


def _membership(rows: list[dict[str, Any]], indices: list[int]) -> str:
    return sha(json_body([rows[i]["record_id"] for i in indices]))


def cross_fit(
    rows: list[dict[str, Any]], folds: list[tuple[list[int], list[int]]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit each temperature on complement only; restore original assessment order."""
    calibration_metrics(rows)
    universe = set(range(len(rows)))
    if len(folds) != 5 or sorted(i for _, assess in folds for i in assess) != list(
        range(len(rows))
    ):
        raise ValueError("five assessment folds must cover each row exactly once")
    adjusted: dict[int, dict[str, Any]] = {}
    details = []
    for number, (fit, assess) in enumerate(folds):
        if (
            not fit
            or not assess
            or len(set(fit)) != len(fit)
            or len(set(assess)) != len(assess)
            or set(fit) & set(assess)
            or set(fit) | set(assess) != universe
        ):
            raise ValueError("fit/assessment must be disjoint complete complements")
        fitted = fit_temperature([rows[i] for i in fit])
        scaled = apply_temperature([rows[i] for i in assess], fitted["temperature"])
        adjusted.update(zip(assess, scaled, strict=True))
        details.append(
            {
                "fold": number,
                "fit": fitted,
                "assessment_samples": len(assess),
                "fit_membership_sha256": _membership(rows, fit),
                "assessment_membership_sha256": _membership(rows, assess),
            }
        )
    pooled = [adjusted[i] for i in range(len(rows))]
    if [r["prediction"] for r in pooled] != [r["prediction"] for r in rows]:
        raise ValueError("calibration changed class predictions")
    gold = [r["gold"] for r in rows]
    return {
        "folds": details,
        "before": calibration_metrics(rows),
        "after": calibration_metrics(pooled),
        "ece_sensitivity": {
            str(b): {
                "before": calibration_metrics(rows, b)["ece"],
                "after": calibration_metrics(pooled, b)["ece"],
            }
            for b in (5, 15)
        },
        "classification_before": classification_metrics(
            gold, [r["prediction"] for r in rows]
        ),
        "classification_after": classification_metrics(
            gold, [r["prediction"] for r in pooled]
        ),
    }, pooled


def run_experiment(
    data_dir: Path, plan_path: Path, report_path: Path
) -> dict[str, Any]:
    """Audit nine saved runs; keep memberships and adjusted outputs private."""
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    if (
        plan["schema_version"] != 1
        or plan["cohort"] != "sentfin_single_target_validation"
        or plan["folds"] != 5
        or plan["fold_seed"] != 42
        or plan["seeds"] != list(SEEDS)
        or plan["temperature_bounds"] != list(BOUNDS)
        or plan["probability_floor"] != FLOOR
        or plan["objective"] != "negative_log_likelihood"
        or plan["optimizer"]
        != "convex inverse-temperature derivative bisection, 80 iterations"
        or plan["final_test_evaluated"] is not False
        or [s["models"] for s in plan["studies"]]
        != [["roberta_base", "finbert"], ["deberta_base"]]
    ):
        raise ValueError("unsupported calibration protocol")
    receipt_path = Path("configs/benchmark-freeze.json")
    receipt_bytes = receipt_path.read_bytes()
    if sha(receipt_bytes) != plan["freeze_receipt_sha256"]:
        raise ValueError("freeze receipt checksum mismatch")
    receipt = json.loads(receipt_bytes)
    records = sentence_cohorts(load_development_records(data_dir, receipt_path))[
        "sentfin_single_target"
    ]
    validation = [r for r in records if r["partition"] == "validation"]
    folds = validation_folds(validation)
    results, inputs, private_runs = {}, [], {}
    for source in plan["studies"]:
        body = Path(source["path"]).read_bytes()
        if sha(body) != source["sha256"]:
            raise ValueError("study checksum mismatch")
        study = json.loads(body)
        if (
            not study["complete"]
            or study["final_test_evaluated"]
            or set(study["seed_runs"]) != set(source["models"])
            or study["plan"]["freeze_receipt_sha256"] != sha(receipt_bytes)
        ):
            raise ValueError("completed validation-only study required")
        for model in source["models"]:
            entries = sorted(study["seed_runs"][model], key=lambda e: e["seed"])
            if [e["seed"] for e in entries] != list(SEEDS) or any(
                e["model"] != model
                or e["learning_rate"] != study["selected"][model]["learning_rate"]
                for e in entries
            ):
                raise ValueError("three seeds at the selected rate required")
            runs = []
            for entry in entries:
                rows, provenance = checked_run(entry, data_dir, validation, receipt)
                inputs.append({"model": model, **provenance})
                result, adjusted = cross_fit(rows, folds)
                full_fit = fit_temperature(rows)
                result.update(seed=entry["seed"], full_validation_fit=full_fit)
                runs.append(result)
                private_runs[f"{model}-{entry['seed']}"] = adjusted
            results[model] = {
                "runs": runs,
                "statistics": {
                    metric: {
                        stage: {
                            "mean": mean(r[stage][metric] for r in runs),
                            "sample_std": stdev(r[stage][metric] for r in runs),
                        }
                        for stage in ("before", "after")
                    }
                    for metric in METRICS
                },
            }
    packet = {
        "plan_sha256": sha(plan_bytes),
        "records": [{k: r[k] for k in ("record_id", "group_id")} for r in validation],
        "folds": [{"fit": f, "assess": a} for f, a in folds],
        "out_of_fold_predictions": private_runs,
    }
    packet_bytes = json_body(packet)
    calibrator_bytes = json_body(
        {
            "plan_sha256": sha(plan_bytes),
            "class_order": list(LABELS),
            "probability_floor": FLOOR,
            "temperature_bounds": list(BOUNDS),
            "inputs": inputs,
            "models": {
                m: {str(r["seed"]): r["full_validation_fit"] for r in v["runs"]}
                for m, v in results.items()
            },
            "final_test_evaluated": False,
        }
    )
    directory = data_dir / "interim/temperature-scaling" / sha(packet_bytes)
    summary = {
        "schema_version": 1,
        "experiment": "validation-cross-fitted-temperature-scaling",
        "plan": plan,
        "plan_sha256": sha(plan_bytes),
        "samples": len(validation),
        "groups": len({r["group_id"] for r in validation}),
        "folds": [
            {
                "fold": i,
                "fit_samples": len(f),
                "assessment_samples": len(a),
                "assessment_class_counts": dict(
                    Counter(sentence_and_label(validation[j])[1] for j in a)
                ),
            }
            for i, (f, a) in enumerate(folds)
        ],
        "models": results,
        "inputs": inputs,
        "private_packet_sha256": sha(packet_bytes),
        "calibrators_sha256": sha(calibrator_bytes),
        "implementation_sha256": sha(Path(__file__).read_bytes()),
        "audit_implementation_sha256": sha(
            Path(__file__).with_name("sentence_error_analysis.py").read_bytes()
        ),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "environment": {
            "python": platform.python_version(),
            **{p: version(p) for p in ("scikit-learn", "numpy", "scipy")},
        },
        "dependency_lock_sha256": sha(Path("uv.lock").read_bytes()),
        "final_test_evaluated": False,
        "limitation": (
            "Cross-fitting isolates calibration fitting, not earlier epoch/rate "
            "selection. This reused validation cohort is not an independent test. "
            "Saved FP32 probabilities replace logits; flooring loses information "
            "below 1e-15."
        ),
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "packet.json").write_bytes(packet_bytes)
    (directory / "calibrators.json").write_bytes(calibrator_bytes)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(json_body(summary))
    report_path.with_suffix(".md").write_bytes(render_report(summary).encode())
    return summary


def render_report(report: dict[str, Any]) -> str:
    """Generate aggregate-only results; keep row IDs/probabilities out of Git."""
    lines = [
        "# Three-model temperature-scaling experiment",
        "",
        f"SEntFiN single-target validation: {report['samples']} records. "
        "Shared five-fold grouped cross-fitting, seeds 42/43/44; "
        "no new model inference or training.",
        "",
        "Each assessed row uses a temperature fitted on the other four folds. "
        "Pool out-of-fold probabilities per seed, then report mean/sample SD. "
        "Earlier epoch/rate selection used these records: "
        "these remain exploratory development results.",
        "",
        "| Model | ECE before → after | Brier before → after | "
        "NLL before → after | Confidence before → after |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, result in report["models"].items():
        cells = []
        for metric in ("ece", "brier", "nll", "mean_confidence"):
            stat = result["statistics"][metric]
            cells.append(
                " → ".join(
                    f"{stat[s]['mean']:.4f} ± {stat[s]['sample_std']:.4f}"
                    for s in ("before", "after")
                )
            )
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Accuracy, macro F1, per-class metrics and confusion matrices "
        "remain unchanged. Temperature scaling adjusts confidence, not "
        "the chosen class. Lower ECE/Brier/NLL is better; "
        "ECE alone does not measure discrimination.",
        "",
        "| Model | Seed | Full-validation T (future use) | Fold T range | "
        "NLL before → OOF after |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, result in report["models"].items():
        for r in result["runs"]:
            ts = [f["fit"]["temperature"] for f in r["folds"]]
            lines.append(
                f"| {model} | {r['seed']} | "
                f"{r['full_validation_fit']['temperature']:.4f} | "
                f"{min(ts):.4f}–{max(ts):.4f} | "
                f"{r['before']['nll']:.4f} → {r['after']['nll']:.4f} |"
            )
    lines += [
        "",
        "Full-validation temperatures are separately fitted saved candidates "
        "for a future locked evaluation. Their in-sample improvement is not "
        "reported as evidence. No best seed or ensemble is selected.",
        "",
        "Saved probabilities are converted to log probabilities with floor "
        "1e-15; this is equivalent to logit temperature scaling up to an "
        "additive constant for positive, unfloored probabilities. "
        "FP32 rounding/flooring limits recovery of the original logits. "
        "All floor counts, optimizer boundaries, per-seed reliability bins, "
        "5/15-bin sensitivity and unchanged classification metrics are in JSON.",
        "",
        "Bounds T=[0.05,20], objective NLL, fixed 80-iteration "
        "inverse-temperature bisection. "
        "Method: [Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html). "
        "The class mapping remains negative/neutral/positive; "
        "no labels or parent partitions change.",
        "",
        "Final tests remain frozen and unscored. Source data, per-record "
        "adjusted outputs, fold IDs and calibrator artifacts stay ignored; "
        "public files contain aggregates and hashes only.",
        "",
        f"Protocol SHA-256: `{report['plan_sha256']}`. "
        f"Private packet SHA-256: `{report['private_packet_sha256']}`.",
        "",
        "See [reproduction and limitations](../../docs/temperature-scaling.md).",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--plan", type=Path, default=Path("configs/temperature-scaling.json")
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/experiments/temperature-scaling.json"),
    )
    args = parser.parse_args()
    result = run_experiment(args.data_dir, args.plan, args.report)
    print(
        f"Aggregate report: {args.report}; "
        f"private packet: {result['private_packet_sha256']}"
    )


if __name__ == "__main__":
    main()
