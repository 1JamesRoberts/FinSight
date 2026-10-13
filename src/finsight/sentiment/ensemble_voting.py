"""Equal-weight cross-model voting on audited saved development predictions."""

import argparse
import json
import math
import platform
import subprocess
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from finsight.data.benchmark import load_development_records
from finsight.sentiment.error_analysis import sha
from finsight.sentiment.evaluation_protocol import safe_path, verify_locked_plan
from finsight.sentiment.finbert import file_sha256
from finsight.sentiment.temperature_scaling import apply_temperature, validation_folds
from finsight.sentiment.transformer_analysis import (
    calibration_metrics,
    checked_run,
    json_body,
)
from finsight.sentiment.validation import (
    LABELS,
    classification_metrics,
    sentence_cohorts,
)

MODELS = ("finbert", "roberta_base", "deberta_base")
SEEDS = (42, 43, 44)


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


def evaluate_seed(
    members: dict[str, list[dict[str, Any]]], *, seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare both votes with their individual members on the same ordered rows."""
    if set(members) != set(MODELS) or seed not in SEEDS:
        raise ValueError("declared models and seed required")
    lengths = {len(rows) for rows in members.values()}
    if len(lengths) != 1 or not next(iter(lengths)):
        raise ValueError("equal nonempty member rows required")
    ids = [r["record_id"] for r in members[MODELS[0]]]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate prediction IDs")
    joined = [combine([members[m][i] for m in MODELS]) for i in range(len(ids))]
    gold = [r["gold"] for r in joined]
    result: dict[str, Any] = {"seed": seed, "members": {}}
    for model, rows in members.items():
        result["members"][model] = {
            "classification": classification_metrics(
                gold, [r["prediction"] for r in rows]
            ),
            "probability_metrics": calibration_metrics(rows),
        }
    for method in ("soft", "hard"):
        rows = [{"gold": r["gold"], **r[method]} for r in joined]
        paired = Counter()
        for row, reference in zip(rows, members["deberta_base"], strict=True):
            a, b = (
                row["prediction"] == row["gold"],
                reference["prediction"] == row["gold"],
            )
            paired[
                "both_correct"
                if a and b
                else "ensemble_only_correct"
                if a
                else "deberta_only_correct"
                if b
                else "both_wrong"
            ] += 1
            paired["prediction_disagreements"] += (
                row["prediction"] != reference["prediction"]
            )
        result[method] = {
            "classification": classification_metrics(
                gold, [r["prediction"] for r in rows]
            ),
            "probability_metrics": calibration_metrics(rows)
            if method == "soft"
            else None,
            "versus_deberta": {
                k: paired[k]
                for k in (
                    "both_correct",
                    "ensemble_only_correct",
                    "deberta_only_correct",
                    "both_wrong",
                    "prediction_disagreements",
                )
            },
        }
        if method == "soft":
            result[method]["ece_sensitivity"] = {
                str(b): calibration_metrics(rows, b)["ece"] for b in (5, 15)
            }
    result["hard"]["three_way_ties"] = sum(r["hard"]["tie"] for r in joined)
    result["member_agreement"] = dict(
        Counter(
            str(len(set(members[m][i]["prediction"] for m in MODELS)))
            for i in range(len(ids))
        )
    )
    return result, joined


def verify_extension(root: Path, data_dir: Path) -> dict[str, Any]:
    """Verify additive v2 protocol while retaining the exact original primary lock."""
    path = root / "configs/final-evaluation-ensembles-v2.json"
    lock = json.loads(
        (root / "configs/final-evaluation-ensembles-v2-lock.json").read_bytes()
    )
    if file_sha256(path) != lock["protocol_sha256"]:
        raise ValueError("ensemble extension checksum mismatch")
    extension = json.loads(path.read_bytes())
    if (
        extension["protocol_id"] != "phase3-sentfin-single-target-v2"
        or extension["methods"] != ["soft", "hard"]
        or extension["test_evaluated"] is not False
    ):
        raise ValueError("unsupported ensemble extension")
    for field in ("parent_plan", "ensemble_plan"):
        if (
            file_sha256(safe_path(root, extension[field + "_path"]))
            != extension[field + "_sha256"]
        ):
            raise ValueError("ensemble extension input checksum mismatch")
    verify_locked_plan(root, data_dir)
    return extension


def run_validation(data_dir: Path, report_path: Path) -> dict[str, Any]:
    """Audit raw/OOF members; publish aggregates and save private votes."""
    root = Path.cwd()
    extension = verify_extension(root, data_dir)
    recipe_path = Path("configs/ensemble-voting.json")
    recipe = json.loads(recipe_path.read_bytes())
    if (
        recipe["models"] != list(MODELS)
        or recipe["seeds"] != list(SEEDS)
        or recipe["member_weights"] != [1, 1, 1]
        or recipe["cohort"] != "sentfin_single_target_validation"
        or any(
            recipe[k] is not False
            for k in (
                "weight_search",
                "best_seed_selection",
                "additional_calibration_fit",
                "final_test_evaluated",
            )
        )
        or file_sha256(Path(recipe["calibration_report_path"]))
        != recipe["calibration_report_sha256"]
    ):
        raise ValueError("unsupported or changed voting recipe")
    calibration = json.loads(Path(recipe["calibration_report_path"]).read_bytes())
    packet_path = (
        data_dir
        / "interim/temperature-scaling"
        / calibration["private_packet_sha256"]
        / "packet.json"
    )
    if file_sha256(packet_path) != calibration["private_packet_sha256"]:
        raise ValueError("OOF packet checksum mismatch")
    packet = json.loads(packet_path.read_bytes())
    receipt_path = Path("configs/benchmark-freeze.json")
    receipt = json.loads(receipt_path.read_bytes())
    records = sentence_cohorts(load_development_records(data_dir, receipt_path))[
        "sentfin_single_target"
    ]
    validation = [r for r in records if r["partition"] == "validation"]
    if packet["records"] != [
        {k: r[k] for k in ("record_id", "group_id")} for r in validation
    ]:
        raise ValueError("OOF membership differs")
    folds = validation_folds(validation)
    if packet["folds"] != [{"fit": f, "assess": a} for f, a in folds]:
        raise ValueError("OOF folds differ")
    primary = json.loads(Path(recipe["parent_final_plan_path"]).read_bytes())
    results, outputs, inputs = [], {}, []
    for seed in SEEDS:
        members = {}
        for model in MODELS:
            entry = next(r for r in primary["transformers"][model] if r["seed"] == seed)
            original = json.loads(Path(entry["report_path"]).read_bytes())
            checked_entry = {
                **entry,
                "model": model,
                "learning_rate": entry["configuration"]["learning_rate"],
                "metrics": original["metrics"],
            }
            raw, provenance = checked_run(checked_entry, data_dir, validation, receipt)
            inputs.append({"model": model, **provenance})
            rows = packet["out_of_fold_predictions"][f"{model}-{seed}"]
            analysis = next(
                r for r in calibration["models"][model]["runs"] if r["seed"] == seed
            )
            # Reapply each saved fold temperature to its own audited raw rows.
            expected = {}
            for (_, assess), fold in zip(folds, analysis["folds"], strict=True):
                expected.update(
                    zip(
                        assess,
                        apply_temperature(
                            [raw[i] for i in assess], fold["fit"]["temperature"]
                        ),
                        strict=True,
                    )
                )
            calibration_metrics(rows)
            if len(rows) != len(raw) or any(
                {k: v for k, v in row.items() if k != "probabilities"}
                != {k: v for k, v in expected[i].items() if k != "probabilities"}
                or any(
                    not math.isclose(
                        row["probabilities"][c],
                        expected[i]["probabilities"][c],
                        abs_tol=1e-12,
                        rel_tol=0,
                    )
                    for c in LABELS
                )
                for i, row in enumerate(rows)
            ):
                raise ValueError(
                    "OOF probabilities/labels differ from checked fold fits"
                )
            members[model] = rows
        result, private = evaluate_seed(members, seed=seed)
        results.append(result)
        outputs[str(seed)] = private
    names = (*MODELS, "soft", "hard")
    statistics = {}
    for name in names:
        variants = [r["members"][name] if name in MODELS else r[name] for r in results]
        statistics[name] = {
            m: {
                "mean": mean(v["classification"][m] for v in variants),
                "sample_std": stdev(v["classification"][m] for v in variants),
            }
            for m in ("accuracy", "macro_f1")
        }
        if name != "hard":
            statistics[name].update(
                {
                    m: {
                        "mean": mean(v["probability_metrics"][m] for v in variants),
                        "sample_std": stdev(
                            v["probability_metrics"][m] for v in variants
                        ),
                    }
                    for m in ("ece", "brier", "nll", "mean_confidence")
                }
            )
    private_bytes = json_body(
        {"recipe_sha256": file_sha256(recipe_path), "predictions": outputs}
    )
    directory = data_dir / "interim/ensemble-voting" / sha(private_bytes)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "predictions.json").write_bytes(private_bytes)
    summary = {
        "schema_version": 1,
        "experiment": recipe["experiment"],
        "protocol_id": extension["protocol_id"],
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "python_version": platform.python_version(),
        "dependency_lock_sha256": file_sha256(Path("uv.lock")),
        "cohort": recipe["cohort"],
        "samples": len(validation),
        "recipe": recipe,
        "recipe_sha256": file_sha256(recipe_path),
        "extension_sha256": file_sha256(
            Path("configs/final-evaluation-ensembles-v2.json")
        ),
        "runs": results,
        "statistics": statistics,
        "inputs": inputs,
        "private_predictions_sha256": sha(private_bytes),
        "implementation_sha256": file_sha256(Path(__file__)),
        "ensemble_profile": {
            "members": 3,
            "total_member_parameters": 418556937,
            "learned_voting_parameters": 0,
            "new_weights_saved": False,
            "runtime": (
                "Three member inferences per seed; sequential loading can limit "
                "peak memory. Voting adds a small CPU reduction; no measured "
                "new model-inference timings because saved outputs are reused."
            ),
        },
        "limitations": (
            "Reused epoch/rate-selected validation; exploratory, not an "
            "independent test. Cross-fitting isolates member calibration only; "
            "no ensemble calibrator is fitted. Hard vote fractions are not "
            "probability estimates. Three matching-seed ensembles are not "
            "independent datasets."
        ),
        "final_test_evaluated": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(json_body(summary))
    report_path.with_suffix(".md").write_bytes(render_report(summary).encode())
    return summary


def render_report(report: dict[str, Any]) -> str:
    """Render aggregate metrics and correction counts without source texts."""
    lines = [
        "# Soft and democratic voting on SEntFiN validation",
        "",
        "Three matching-seed ensembles (42/43/44), each combining calibrated FinBERT, "
        "RoBERTa Base and DeBERTa Base with equal weights. 781 validation records; "
        "cross-fitted member temperatures, no new training or calibration. "
        "Results remain exploratory because this validation "
        "already selected rates/epochs.",
        "",
        "| Model/method | Macro F1 ± SD | Accuracy ± SD | "
        "NLL ± SD | Brier ± SD | ECE ± SD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in report["statistics"].items():
        cells = [
            f"{metrics[m]['mean']:.4f} ± {metrics[m]['sample_std']:.4f}"
            if m in metrics
            else "N/A"
            for m in ("macro_f1", "accuracy", "nll", "brier", "ece")
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Soft voting averages calibrated probabilities then takes argmax; "
        "exact ties use negative/neutral/positive order. Democratic/hard voting "
        "takes the majority label; three-way ties resolve to neutral. "
        "Vote counts are not calibrated confidence, so "
        "hard voting has no probability metrics.",
        "",
        "| Method | Seed | Macro F1 | Corrected DeBERTa errors | "
        "Lost DeBERTa correct cases | Hard three-way ties |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in report["runs"]:
        for method in ("soft", "hard"):
            v = r[method]
            lines.append(
                f"| {method} | {r['seed']} | {v['classification']['macro_f1']:.4f} | "
                f"{v['versus_deberta']['ensemble_only_correct']} | "
                f"{v['versus_deberta']['deberta_only_correct']} | "
                f"{r['hard']['three_way_ties'] if method == 'hard' else '—'} |"
            )
    lines += [
        "",
        "No weight search, stacking, best-seed selection "
        "or fitted ensemble calibration. "
        "The soft average itself is not guaranteed calibrated. Per-seed class metrics, "
        "confusion matrices, reliability bins, confidence, ECE sensitivity and member "
        "agreement are in JSON; individual votes/probabilities remain private.",
        "",
        "Both voting methods were declared as supplementary final comparisons "
        "before this analysis. The additive v2 protocol retains the original "
        "primary individual-model "
        "comparison. Final test predictions must use the locked full-validation member "
        "temperatures, not these assessment-fold temperatures. "
        "Frozen tests remain unscored.",
        "",
        "Architecture: three existing classifiers, 418,556,937 "
        "total member parameters, zero learned voting parameters and "
        "no new weight files. Serving requires three "
        "member inferences per seed, rather than one; measured validation reuse here "
        "does not establish serving latency or throughput.",
        "",
        "See [reproduction and interpretation](../../docs/ensemble-voting.md).",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--report", type=Path, default=Path("reports/experiments/ensemble-voting.json")
    )
    parser.add_argument("--verify-final-extension", action="store_true")
    args = parser.parse_args()
    if args.verify_final_extension:
        extension = verify_extension(Path.cwd(), args.data_dir)
        print(f"Verified {extension['protocol_id']}; no test inference")
    else:
        run_validation(args.data_dir, args.report)
        print(f"Aggregate report: {args.report}")


if __name__ == "__main__":
    main()
