"""Bounded validation-only VADER check, with a training-majority control."""

import hashlib
import json
import platform
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from finsight.data.benchmark import load_development_records
from finsight.sentiment.artifacts import project_root, source_manifest
from finsight.sentiment.evaluation.cohorts import _gold, sentence_cohorts
from finsight.sentiment.evaluation.metrics import LABELS as LABELS
from finsight.sentiment.evaluation.metrics import classification_metrics
from finsight.sentiment.models.vader import VaderConfig, VaderSentiment


def run_validation(
    data_dir: Path,
    receipt_path: Path,
    config_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Log validation metrics, scoring resources, provenance, and local predictions."""
    records = load_development_records(data_dir, receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    model = VaderSentiment(VaderConfig.load(config_path))
    model.predict("The outlook is good.")  # One invented warm-up; no corpus labels.
    metrics = {}
    predictions = []
    for name, cohort in sentence_cohorts(records).items():
        training = [row for row in cohort if row["partition"] == "train"]
        validation = [row for row in cohort if row["partition"] == "validation"]
        if not training or not validation:
            raise ValueError(
                "each sentence cohort requires train and validation records"
            )
        training_labels = Counter(_gold(row) for row in training)
        majority = max(LABELS, key=lambda label: training_labels[label])
        gold = []
        predicted = []
        elapsed = 0.0
        for row in validation:
            native = row["native_record"]
            text = (
                native["sentence"] if row["corpus"] == "phrasebank" else native["title"]
            )
            started = perf_counter()
            result = model.predict(text)
            elapsed += perf_counter() - started
            gold.append(_gold(row))
            predicted.append(result.label.value)
            predictions.append(
                {
                    "cohort": name,
                    "record_id": row["record_id"],
                    "gold": gold[-1],
                    "prediction": asdict(result),
                }
            )
        metrics[name] = {
            "vader": classification_metrics(gold, predicted),
            "training_majority": classification_metrics(gold, [majority] * len(gold)),
            "majority_label": majority,
            "training_samples": len(training),
            "training_class_counts": dict(sorted(training_labels.items())),
            "inference_seconds": elapsed,
            "mean_inference_ms_per_text": elapsed * 1000 / len(validation),
        }
    root = project_root()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    snapshot = source_manifest()
    body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions)
    run = {
        "experiment": "vader-validation-v1",
        "schema_version": 1,
        "scope": "validation only; PhraseBank and single-target SEntFiN",
        "thresholds_tuned": False,
        "final_test_evaluated": False,
        "benchmark_manifest_sha256": receipt["benchmark_manifest_sha256"],
        "parent_split_manifest_sha256": receipt["parent_split_manifest_sha256"],
        "model": model.metadata(),
        "metrics": metrics,
        "code_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "working_tree_dirty": bool(dirty.stdout) if dirty.returncode == 0 else None,
        "implementation_sha256": snapshot,
        "predictions_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "timing": (
            "batch size 1; one synthetic warm-up; scoring only, "
            "excludes loading/serialization; single pass"
        ),
        "seed": None,
        "seed_reason": "deterministic rules; no random sampling or training",
    }
    run_body = json.dumps(run, indent=2, sort_keys=True) + "\n"
    run_id = hashlib.sha256(run_body.encode()).hexdigest()
    directory = data_dir / "interim/vader-validation" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    # Persist the exact hashed UTF-8 bytes, including LF on Windows.
    (directory / "run.json").write_bytes(run_body.encode("utf-8"))
    (directory / "predictions.jsonl").write_bytes(body.encode("utf-8"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_validation_report(run, run_id), encoding="utf-8")
    report_path.with_suffix(".json").write_bytes(run_body.encode("utf-8"))
    return run


def render_validation_report(run: dict[str, Any], run_id: str) -> str:
    """Publish aggregate validation scores only; predictions/IDs remain ignored."""
    lines = [
        "# VADER validation sanity check",
        "",
        run["scope"],
        "",
        f"Thresholds: negative <= {run['model']['config']['negative_threshold']}, "
        f"positive >= {run['model']['config']['positive_threshold']}, neutral between.",
        "No threshold tuning and no final-test evaluation. Majority control uses",
        "training labels from the corresponding sentence cohort only.",
        "",
        "| Cohort | Validation texts | VADER accuracy | VADER macro F1 "
        "| Majority accuracy | Majority macro F1 | Mean inference ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in run["metrics"].items():
        model, majority = result["vader"], result["training_majority"]
        lines.append(
            f"| {name} | {model['samples']} | {model['accuracy']:.4f} | "
            f"{model['macro_f1']:.4f} | {majority['accuracy']:.4f} | "
            f"{majority['macro_f1']:.4f} | {result['mean_inference_ms_per_text']:.4f} |"
        )
    for name, result in run["metrics"].items():
        model = result["vader"]
        lines += [
            "",
            f"### {name}: per-class metrics",
            "",
            "| Class | Support | Precision | Recall | F1 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for label, value in model["per_class"].items():
            lines.append(
                f"| {label} | {value['support']} | {value['precision']:.4f} | "
                f"{value['recall']:.4f} | {value['f1']:.4f} |"
            )
        lines += [
            "",
            "Confusion matrix: rows=gold, columns=predicted; "
            "order negative, neutral, positive.",
            "",
            "```text",
            *(str(row) for row in model["confusion_matrix"]),
            "```",
            "",
        ]
    lines += [
        f"Run ID: `{run_id}`.",
        f"Parent split: `{run['parent_split_manifest_sha256']}`.",
        f"Code revision at execution: `{run['code_revision']}`; "
        f"dirty: `{run['working_tree_dirty']}`.",
        "Implementation file digests distinguish uncommitted implementation changes.",
        "Aggregate settings, resources, environment, and timing accompany it as JSON.",
        "The complete local run is also preserved in ignored `run.json`.",
        "Local prediction IDs/gold/scores remain ignored; no source text is exported.",
        "Latency is a single-pass local measurement, not a deployment benchmark.",
        "See `docs/vader.md` for scoring, reproduction, and financial-domain limits.",
    ]
    return "\n".join(lines) + "\n"
