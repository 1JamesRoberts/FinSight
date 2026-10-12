"""Analyze six saved validation runs; no model fitting or new inference."""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from finsight.data.benchmark import load_development_records
from finsight.sentiment.error_analysis import CATEGORIES, sha
from finsight.sentiment.tfidf_validation import sentence_and_label
from finsight.sentiment.validation import (
    LABELS,
    classification_metrics,
    sentence_cohorts,
)

MODELS = ("roberta_base", "finbert")
SEEDS = (42, 43, 44)


def json_body(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


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


def checked_run(
    entry: dict[str, Any],
    data_dir: Path,
    validation: list[dict[str, Any]],
    receipt: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Check report/private bytes, split lineage, ordered membership, gold, metrics."""
    if not validation or any(r["partition"] != "validation" for r in validation):
        raise ValueError("validation records only")
    body = Path(entry["report_path"]).read_bytes()
    if sha(body) != entry["report_sha256"]:
        raise ValueError("report checksum mismatch")
    run = json.loads(body)
    if (
        run["final_test_evaluated"]
        or run["configuration"]["seed"] != entry["seed"]
        or run["configuration"]["learning_rate"] != entry["learning_rate"]
    ):
        raise ValueError("run scope/configuration mismatch")
    for field in ("parent_split_manifest_sha256", "benchmark_manifest_sha256"):
        if run[field] != receipt[field]:
            raise ValueError("split lineage mismatch")
    namespace = (
        "finbert-finetuning"
        if entry["model"] == "finbert"
        else "transformer-comparison"
    )
    directory = data_dir / "interim" / namespace / entry["report_sha256"]
    if (directory / "run.json").read_bytes() != body:
        raise ValueError("private run mismatch")
    predictions_body = (directory / "predictions.jsonl").read_bytes()
    if sha(predictions_body) != run["predictions_sha256"]:
        raise ValueError("prediction checksum mismatch")
    rows = [json.loads(line) for line in predictions_body.splitlines()]
    ids = [r["record_id"] for r in validation]
    if [r["record_id"] for r in rows] != ids or len(set(ids)) != len(ids):
        raise ValueError("validation membership/order mismatch")
    if (
        sha(json.dumps(ids).encode()) != run["validation_membership_sha256"]
        or run["validation_membership_sha256"] != entry["validation_membership_sha256"]
    ):
        raise ValueError("membership digest mismatch")
    gold = [sentence_and_label(r)[1] for r in validation]
    if [r["gold"] for r in rows] != gold:
        raise ValueError("saved gold mismatch")
    if (
        classification_metrics(gold, [r["prediction"] for r in rows]) != run["metrics"]
        or run["metrics"] != entry["metrics"]
    ):
        raise ValueError("saved metrics mismatch")
    calibration_metrics(rows)
    return rows, {
        "report_path": entry["report_path"],
        "report_sha256": sha(body),
        "predictions_sha256": sha(predictions_body),
        "seed": entry["seed"],
        "learning_rate": entry["learning_rate"],
        "validation_membership_sha256": run["validation_membership_sha256"],
    }


def persistence_counts(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    errors = Counter(sum(p != r["gold"] for p in r["predictions"][model]) for r in rows)
    return {
        "errors_in_k_of_three_seeds": {str(k): errors[k] for k in range(4)},
        "prediction_varies_across_seeds": sum(
            len(set(r["predictions"][model])) > 1 for r in rows
        ),
        "recurring_error_gold_counts": dict(
            sorted(
                Counter(
                    r["gold"]
                    for r in rows
                    if all(p != r["gold"] for p in r["predictions"][model])
                ).items()
            )
        ),
    }


def review_sample(
    rows: list[dict[str, Any]], per_cell: int = 2
) -> list[dict[str, Any]]:
    """Two per model/persistence/gold/wrong-label cell; deduplicate source IDs."""
    if type(per_cell) is not int or per_cell < 1:
        raise ValueError("per_cell must be a positive integer")
    selected: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        for persistence in ("recurring", "variable"):
            for gold in LABELS:
                for wrong in LABELS:
                    if gold == wrong:
                        continue
                    cell = []
                    for r in rows:
                        predictions = r["predictions"][model]
                        n = sum(p != r["gold"] for p in predictions)
                        if (
                            r["gold"] == gold
                            and wrong in predictions
                            and (
                                (n == 3) if persistence == "recurring" else (0 < n < 3)
                            )
                        ):
                            cell.append(r)
                    for r in sorted(
                        cell, key=lambda r: sha(f"phase3:42:{r['record_id']}".encode())
                    )[:per_cell]:
                        if r["record_id"] not in selected:
                            selected[r["record_id"]] = {**r, "review_strata": []}
                        selected[r["record_id"]]["review_strata"].append(
                            dict(
                                model=model,
                                persistence=persistence,
                                gold=gold,
                                wrong=wrong,
                            )
                        )
    negation = [
        r
        for r in rows
        if re.search(r"\b(?:no|not|never|without)\b|n't\b", r.get("text", ""), re.I)
        and any(p != r["gold"] for m in MODELS for p in r["predictions"][m])
    ]
    for r in sorted(
        negation, key=lambda r: sha(f"phase3:42:{r['record_id']}".encode())
    )[:3]:
        if r["record_id"] not in selected:
            selected[r["record_id"]] = {**r, "review_strata": []}
        selected[r["record_id"]]["review_strata"].append(
            {"supplement": "explicit_negation"}
        )
    return sorted(selected.values(), key=lambda r: r["record_id"])


def prepare_analysis(
    data_dir: Path, study_path: Path, receipt_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    study_bytes, receipt_bytes = study_path.read_bytes(), receipt_path.read_bytes()
    study, receipt = json.loads(study_bytes), json.loads(receipt_bytes)
    if (
        not study["complete"]
        or study["final_test_evaluated"]
        or set(study["seed_runs"]) != set(MODELS)
    ):
        raise ValueError("complete validation-only two-model study required")
    if study["plan"]["freeze_receipt_sha256"] != sha(receipt_bytes):
        raise ValueError("freeze receipt differs")
    records = sentence_cohorts(load_development_records(data_dir, receipt_path))[
        "sentfin_single_target"
    ]
    validation = [r for r in records if r["partition"] == "validation"]
    training_counts = Counter(
        sentence_and_label(r)[1] for r in records if r["partition"] == "train"
    )
    prior = {
        label: training_counts[label] / sum(training_counts.values())
        for label in LABELS
    }
    prior_predictions = [
        {
            "gold": sentence_and_label(r)[1],
            "prediction": max(prior, key=prior.get),
            "probabilities": prior,
        }
        for r in validation
    ]
    predictions, results, inputs = {}, {}, []
    for model in MODELS:
        entries = sorted(study["seed_runs"][model], key=lambda r: r["seed"])
        if [r["seed"] for r in entries] != list(SEEDS) or any(
            r["model"] != model
            or r["learning_rate"] != study["selected"][model]["learning_rate"]
            for r in entries
        ):
            raise ValueError("expected three seeds at selected rate")
        predictions[model], results[model] = [], {"runs": []}
        for entry in entries:
            rows, provenance = checked_run(entry, data_dir, validation, receipt)
            inputs.append(provenance)
            predictions[model].append(rows)
            results[model]["runs"].append(
                {
                    "seed": entry["seed"],
                    "classification": entry["metrics"],
                    **calibration_metrics(rows),
                    "ece_sensitivity": {
                        str(b): calibration_metrics(rows, b)["ece"] for b in (5, 15)
                    },
                }
            )
        results[model]["statistics"] = {
            metric: {
                "mean": mean(r[metric] for r in results[model]["runs"]),
                "sample_std": stdev(r[metric] for r in results[model]["runs"]),
            }
            for metric in ("ece", "brier", "nll", "accuracy", "mean_confidence")
        }
    joined = []
    for i, source in enumerate(validation):
        text, gold = sentence_and_label(source)
        joined.append(
            {
                "record_id": source["record_id"],
                "text": text,
                "gold": gold,
                "target": source["native_record"]["sentiments"][0]["target"],
                "predictions": {
                    m: [rows[i]["prediction"] for rows in predictions[m]]
                    for m in MODELS
                },
                "confidences": {
                    m: [
                        max(rows[i]["probabilities"].values())
                        for rows in predictions[m]
                    ]
                    for m in MODELS
                },
            }
        )
    for m in MODELS:
        results[m]["persistence"] = persistence_counts(joined, m)
    paired = []
    for j, seed in enumerate(SEEDS):
        counts = Counter()
        for r in joined:
            a, b = (r["predictions"][m][j] == r["gold"] for m in MODELS)
            counts[
                "both_correct"
                if a and b
                else "roberta_only_correct"
                if a
                else "finbert_only_correct"
                if b
                else "both_wrong"
            ] += 1
            counts["prediction_disagreements"] += (
                r["predictions"][MODELS[0]][j] != r["predictions"][MODELS[1]][j]
            )
        paired.append({"seed": seed, **dict(sorted(counts.items()))})
    packet = {
        "protocol": (
            "Up to two per model/persistence/gold/wrong-label cell; "
            "SHA-256(phase3:42:record_id) order; deduplicate IDs. "
            "Recurring = wrong in all three seeds; variable = wrong in "
            "one or two. Supplement: up to three errors containing "
            "no/not/never/without/n't in the same hash order. Categories "
            "fixed from Phase 2; no inferred automatic annotations."
        ),
        "study_sha256": sha(study_bytes),
        "examples": review_sample(joined),
    }
    summary = {
        "schema_version": 1,
        "cohort": "sentfin_single_target_validation",
        "samples": len(validation),
        "final_test_evaluated": False,
        "probabilities_adjusted": False,
        "models": results,
        "training_prior_control": {
            "training_class_counts": dict(training_counts),
            "probabilities": prior,
            "metrics": calibration_metrics(prior_predictions),
        },
        "paired_counts": paired,
        "all_six_runs_wrong": sum(
            all(p != r["gold"] for m in MODELS for p in r["predictions"][m])
            for r in joined
        ),
        "inputs": inputs,
        "study_sha256": sha(study_bytes),
        "freeze_receipt_sha256": sha(receipt_bytes),
        "implementation_sha256": sha(Path(__file__).read_bytes()),
        "definitions": {
            "ece": (
                "Top-label ECE, 10 equal-width bins [lower,upper); last "
                "includes 1. Empty bins ignored. Sensitivity: 5 and 15 bins."
            ),
            "brier": (
                "Mean sum over three classes of (p - one-hot gold)^2; "
                "unnormalized range 0–2; lower is better."
            ),
            "nll": "Mean negative log gold probability; natural log, lower clip 1e-15.",
            "statistics": (
                "Mean/sample SD across three separately evaluated seeds, no "
                "probability averaging/ensemble. Same validation used for "
                "training selection; descriptive only."
            ),
        },
    }
    return summary, packet


def summarize_annotations(
    packet: dict[str, Any], annotations: list[dict[str, Any]]
) -> dict[str, Any]:
    ids = {r["record_id"] for r in packet["examples"]}
    if (
        len({r["record_id"] for r in annotations}) != len(annotations)
        or {r["record_id"] for r in annotations} != ids
    ):
        raise ValueError("annotations must cover exact sample once")
    for r in annotations:
        if (
            r["category"] not in CATEGORIES
            or type(r["gold_ambiguous"]) is not bool
            or not isinstance(r["note"], str)
            or not r["note"].strip()
        ):
            raise ValueError("invalid annotation")
    return {
        "reviewed_samples": len(annotations),
        "primary_categories": dict(
            sorted(Counter(r["category"] for r in annotations).items())
        ),
        "suspected_gold_ambiguity": sum(r["gold_ambiguous"] for r in annotations),
        "protocol": packet["protocol"],
        "packet_sha256": sha(json_body(packet)),
        "annotations_sha256": sha(json_body(annotations)),
        "reviewer": "Codex exploratory review; not independent human adjudication",
        "gold_labels_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--study",
        type=Path,
        default=Path("reports/experiments/transformer-tuning-runs/study.json"),
    )
    parser.add_argument("--annotations", type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/experiments/transformer-analysis.json"),
    )
    args = parser.parse_args()
    summary, packet = prepare_analysis(
        args.data_dir, args.study, Path("configs/benchmark-freeze.json")
    )
    directory = args.data_dir / "interim/phase3-error-review" / sha(json_body(packet))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "packet.json").write_bytes(json_body(packet))
    (directory / "calibration.json").write_bytes(json_body(summary))
    if args.annotations:
        annotations = json.loads(args.annotations.read_bytes())
        summary["qualitative_review"] = summarize_annotations(packet, annotations)
        (directory / "annotations.json").write_bytes(json_body(annotations))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes(json_body(summary))
        print(f"Aggregate report: {args.report}")
    print(f"Private review: {directory / 'packet.json'}")


if __name__ == "__main__":
    main()
