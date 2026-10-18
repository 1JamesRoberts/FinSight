"""Inspect saved validation errors; never fit or score a model."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from finsight.config import Settings
from finsight.data.benchmark import load_development_records
from finsight.sentiment.artifacts import sha as sha
from finsight.sentiment.evaluation.cohorts import sentence_and_label, sentence_cohorts
from finsight.sentiment.evaluation.metrics import LABELS, classification_metrics

CATEGORIES = (
    "financial_context",
    "numeric_comparison",
    "mixed_cues",
    "event_interpretation",
    "neutral_boundary",
    "negation_scope",
)


def sample_errors(
    rows: list[dict[str, Any]], per_cell: int = 5
) -> list[dict[str, Any]]:
    """Deterministic hash ordering within each off-diagonal confusion cell."""
    if type(per_cell) is not int or per_cell < 1:
        raise ValueError("per_cell must be a positive integer")
    selected = []
    for gold in LABELS:
        for predicted in LABELS:
            if gold == predicted:
                continue
            cell = [r for r in rows if r["gold"] == gold and r["lr"] == predicted]
            selected.extend(
                sorted(cell, key=lambda r: sha(f"42:{r['record_id']}".encode()))[
                    :per_cell
                ]
            )
    return selected


def disagreement_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Paired correctness and disagreements across the complete validation cohort."""
    counts: Counter[str] = Counter()
    for row in rows:
        a, b = row["lr"] == row["gold"], row["svm"] == row["gold"]
        counts[
            "both_correct"
            if a and b
            else "lr_only_correct"
            if a
            else "svm_only_correct"
            if b
            else "both_wrong"
        ] += 1
        counts["lr_svm_disagree"] += row["lr"] != row["svm"]
        counts["lr_errors"] += not a
        counts["svm_errors"] += not b
        counts["vader_errors"] += row["vader"] != row["gold"]
        counts["all_three_wrong"] += not a and not b and row["vader"] != row["gold"]
    return {
        key: counts[key]
        for key in (
            "both_correct",
            "lr_only_correct",
            "svm_only_correct",
            "both_wrong",
            "lr_svm_disagree",
            "lr_errors",
            "svm_errors",
            "vader_errors",
            "all_three_wrong",
        )
    }


def checked_predictions(
    data_dir: Path,
    candidate: dict[str, Any],
    cohort: str,
    validation: list[dict[str, Any]],
    receipt_path: Path,
) -> dict[str, dict[str, Any]]:
    """Verify original run, predictions, artifact, membership, gold, and metrics."""
    kind, run = candidate["model_kind"], candidate["run"]
    prefix = "tfidf-logreg" if kind == "logistic_regression" else "tfidf-svm"
    directory = data_dir / f"interim/{prefix}-validation" / candidate["run_id"]
    body = (directory / "run.json").read_bytes()
    if sha(body) != candidate["run_id"] or json.loads(body) != run:
        raise ValueError("original run mismatch")
    if run["final_test_evaluated"] or run["freeze_receipt_sha256"] != sha(
        receipt_path.read_bytes()
    ):
        raise ValueError("invalid evaluation scope or freeze")
    predictions_body = (directory / "predictions.jsonl").read_bytes()
    if sha(predictions_body) != run["predictions_sha256"]:
        raise ValueError("prediction checksum mismatch")
    result = run["metrics"][cohort]
    if (
        sha((data_dir / result["artifact_relative_path"]).read_bytes())
        != result["artifact_sha256"]
    ):
        raise ValueError("model checksum mismatch")
    rows = [json.loads(line) for line in predictions_body.splitlines()]
    rows = [row for row in rows if row["cohort"] == cohort]
    indexed = {row["record_id"]: row for row in rows}
    ids = [row["record_id"] for row in validation]
    if len(indexed) != len(rows) or set(indexed) != set(ids):
        raise ValueError("validation prediction membership mismatch")
    if sha(json.dumps(ids).encode()) != result["validation_membership_sha256"]:
        raise ValueError("validation membership digest mismatch")
    gold = [sentence_and_label(row)[1] for row in validation]
    if gold != [indexed[i]["gold"] for i in ids]:
        raise ValueError("validation gold mismatch")
    for metric, field in ((kind, "prediction"), ("vader", "vader")):
        if (
            classification_metrics(gold, [indexed[i][field]["label"] for i in ids])
            != result[metric]
        ):
            raise ValueError("saved metrics mismatch")
    return indexed


def prepare_packet(
    data_dir: Path,
    receipt_path: Path,
    tuning_path: Path,
) -> dict[str, Any]:
    """Build an ignored local review packet from verified development rows only."""
    records = load_development_records(data_dir, receipt_path)
    tuning_bytes = tuning_path.read_bytes()
    tuning = json.loads(tuning_bytes)
    if tuning["final_test_evaluated"]:
        raise ValueError("validation-only review required")
    candidates = {c["run_id"]: c for c in tuning["candidates"]}
    examples, cohorts = [], {}
    for cohort, development in sentence_cohorts(records).items():
        validation = [row for row in development if row["partition"] == "validation"]
        selected = tuning["selected_run_ids"][cohort]
        chosen = {
            kind: candidates[selected[kind]]
            for kind in ("logistic_regression", "linear_svm")
        }
        predictions = {
            kind: checked_predictions(
                data_dir, candidate, cohort, validation, receipt_path
            )
            for kind, candidate in chosen.items()
        }
        rows = []
        for source in validation:
            record_id = source["record_id"]
            lr = predictions["logistic_regression"][record_id]
            svm = predictions["linear_svm"][record_id]
            if lr["vader"] != svm["vader"]:
                raise ValueError("VADER controls differ")
            text, gold = sentence_and_label(source)
            rows.append(
                {
                    "record_id": record_id,
                    "cohort": cohort,
                    "text": text,
                    "gold": gold,
                    "lr": lr["prediction"]["label"],
                    "svm": svm["prediction"]["label"],
                    "vader": lr["vader"]["label"],
                    "target": source["native_record"]
                    .get("sentiments", [{}])[0]
                    .get("target"),
                }
            )
        sample = sample_errors(rows)
        examples.extend(sample)
        cohorts[cohort] = {
            "validation_samples": len(rows),
            "reviewed_samples": len(sample),
            "paired_counts": disagreement_counts(rows),
            "sample_confusion_cells": dict(
                sorted(Counter(f"{r['gold']}->{r['lr']}" for r in sample).items())
            ),
            "selected_run_ids": selected,
        }
    return {
        "schema_version": 1,
        "tuning_sha256": sha(tuning_bytes),
        "freeze_receipt_sha256": sha(receipt_path.read_bytes()),
        "method": "Up to five LR errors per off-diagonal cell; "
        "SHA-256(42:record_id) ordering.",
        "cohorts": cohorts,
        "examples": examples,
    }


def summarize_review(
    packet: dict[str, Any], annotations: list[dict[str, Any]]
) -> dict[str, Any]:
    """Publish category counts only; free-text notes and source IDs remain private."""
    examples = {row["record_id"]: row for row in packet["examples"]}
    indexed = {row["record_id"]: row for row in annotations}
    if len(indexed) != len(annotations) or set(indexed) != set(examples):
        raise ValueError("annotations must cover the exact review sample once")
    for row in annotations:
        if row["category"] not in CATEGORIES or type(row["gold_ambiguous"]) is not bool:
            raise ValueError("invalid review category or ambiguity flag")
        if not isinstance(row["note"], str) or not row["note"].strip():
            raise ValueError("each review requires a note")
    summary = {key: value for key, value in packet.items() if key != "examples"}
    # Copy nested aggregates; never mutate the private source packet.
    summary = json.loads(json.dumps(summary))
    for cohort, result in summary["cohorts"].items():
        reviewed = [
            indexed[i] for i, example in examples.items() if example["cohort"] == cohort
        ]
        result["primary_categories"] = dict(
            sorted(Counter(r["category"] for r in reviewed).items())
        )
        result["suspected_gold_ambiguity"] = sum(r["gold_ambiguous"] for r in reviewed)
    summary.update(
        {
            "reviewer": "Codex qualitative review; not independent human adjudication",
            "gold_labels_changed": False,
            "final_test_evaluated": False,
            "implementation_sha256": sha(Path(__file__).read_bytes()),
            "packet_sha256": sha(
                (json.dumps(packet, indent=2, sort_keys=True) + "\n").encode()
            ),
            "annotations_sha256": sha(
                (json.dumps(annotations, indent=2, sort_keys=True) + "\n").encode()
            ),
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Settings.from_env().data_dir)
    parser.add_argument(
        "--freeze-receipt", type=Path, default=Path("configs/benchmark-freeze.json")
    )
    parser.add_argument(
        "--tuning-report",
        type=Path,
        default=Path("reports/experiments/tfidf-tuning.json"),
    )
    parser.add_argument("--annotations", type=Path)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/experiments/phase2-error-analysis.json"),
    )
    args = parser.parse_args()
    packet = prepare_packet(args.data_dir, args.freeze_receipt, args.tuning_report)
    body = json.dumps(packet, indent=2, sort_keys=True) + "\n"
    directory = args.data_dir / "interim/phase2-error-review" / sha(body.encode())
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "packet.json"
    path.write_text(body)
    if args.annotations:
        annotations = json.loads(args.annotations.read_text())
        summary = summarize_review(packet, annotations)
        annotation_body = json.dumps(annotations, indent=2, sort_keys=True) + "\n"
        annotation_path = (
            directory / f"{sha(annotation_body.encode())}-annotations.json"
        )
        annotation_path.write_text(annotation_body)
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        print(f"Aggregate review saved to {args.report_path}")
    print(f"Local review packet saved to {path}")


if __name__ == "__main__":
    main()
