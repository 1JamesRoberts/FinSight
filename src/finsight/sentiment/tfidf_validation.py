"""Fit initial logistic baselines on training only; score validation only."""

import hashlib
import json
import platform
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

from finsight.data.benchmark import load_development_records
from finsight.sentiment.tfidf import TfidfConfig, TfidfSentiment
from finsight.sentiment.vader import VaderConfig, VaderSentiment
from finsight.sentiment.validation import (
    LABELS,
    classification_metrics,
    sentence_cohorts,
)


def sentence_and_label(row: dict[str, Any]) -> tuple[str, str]:
    native = row["native_record"]
    if row["corpus"] == "phrasebank":
        return native["sentence"], native["label"]
    return native["title"], native["sentiments"][0]["label"]


def _digest(body: str) -> str:
    return hashlib.sha256(body.encode()).hexdigest()


def run_tfidf_validation(
    data_dir: Path,
    receipt_path: Path,
    config_path: Path,
    report_path: Path,
    vader_config_path: Path = Path("configs/vader.json"),
    *,
    cohort: str = "all",
) -> dict[str, Any]:
    """Separate fitted models for matching VADER cohorts; no search or test scoring."""
    if cohort not in {"all", "phrasebank", "sentfin_single_target"}:
        raise ValueError("unknown sentence cohort")
    records = load_development_records(data_dir, receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    config = TfidfConfig.load(config_path)
    vader = VaderSentiment(VaderConfig.load(vader_config_path))
    vader.predict("The outlook is good.")
    root = Path(__file__).resolve().parents[3]
    snapshot = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("tfidf.py", "tfidf_validation.py", "validation.py", "vader.py")
    }
    metrics: dict[str, Any] = {}
    predictions = []
    for name, rows in sentence_cohorts(records).items():
        if cohort != "all" and name != cohort:
            continue
        training = [row for row in rows if row["partition"] == "train"]
        validation = [row for row in rows if row["partition"] == "validation"]
        if not training or not validation:
            raise ValueError("each cohort requires training and validation records")
        training_examples = [sentence_and_label(row) for row in training]
        train_texts = [text for text, _ in training_examples]
        train_labels = [label for _, label in training_examples]
        counts = Counter(train_labels)
        majority = max(LABELS, key=lambda label: counts[label])
        model = TfidfSentiment(config)
        started = perf_counter()
        model.fit(train_texts, train_labels)
        training_seconds = perf_counter() - started
        # Pin ordered training membership locally without exposing source IDs.
        training_digest = _digest(json.dumps([row["record_id"] for row in training]))
        identity = {
            "cohort": name,
            "training_membership_sha256": training_digest,
            "parent_split_manifest_sha256": receipt["parent_split_manifest_sha256"],
            "metadata": model.metadata(),
            "implementation_sha256": snapshot,
        }
        model_id = _digest(json.dumps(identity, sort_keys=True))
        artifact_root = data_dir / "interim/tfidf-logreg-models"
        artifact_root.mkdir(parents=True, exist_ok=True)
        # Address by serialized bytes: repeated fits must not overwrite an older
        # run's artifact even if identical configuration produces different bytes.
        with TemporaryDirectory(dir=artifact_root) as temporary:
            staging = Path(temporary) / "model.pkl"
            artifact_digest = model.save(staging)
            artifact = artifact_root / artifact_digest / "model.pkl"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            if artifact.exists():
                if artifact.read_bytes() != staging.read_bytes():
                    raise ValueError("existing model artifact is corrupted")
            else:
                staging.replace(artifact)
        # Score the persisted model to verify the actual artifact used for inference.
        model = TfidfSentiment.load(artifact, expected_sha256=artifact_digest)
        model.predict("The outlook is good.")
        gold, predicted, vader_predicted = [], [], []
        elapsed, vader_elapsed = 0.0, 0.0
        for row in validation:
            text, label = sentence_and_label(row)
            started = perf_counter()
            result = model.predict(text)
            elapsed += perf_counter() - started
            started = perf_counter()
            control = vader.predict(text)
            vader_elapsed += perf_counter() - started
            gold.append(label)
            predicted.append(result.label.value)
            vader_predicted.append(control.label.value)
            predictions.append(
                {
                    "cohort": name,
                    "record_id": row["record_id"],
                    "gold": label,
                    "prediction": asdict(result),
                    "vader": asdict(control),
                }
            )
        metrics[name] = {
            "logistic_regression": classification_metrics(gold, predicted),
            "vader": classification_metrics(gold, vader_predicted),
            "training_majority": classification_metrics(gold, [majority] * len(gold)),
            "majority_label": majority,
            "training_samples": len(training),
            "training_class_counts": dict(sorted(counts.items())),
            "effective_class_weights": {
                label: len(training) / (len(LABELS) * counts[label])
                if config.class_weight == "balanced"
                else 1.0
                for label in LABELS
            },
            "training_seconds": training_seconds,
            "inference_seconds": elapsed,
            "mean_inference_ms_per_text": elapsed * 1000 / len(validation),
            "vader_mean_inference_ms_per_text": vader_elapsed * 1000 / len(validation),
            "model": model.metadata(),
            "model_id": model_id,
            "artifact_relative_path": str(artifact.relative_to(data_dir)),
            "artifact_sha256": artifact_digest,
            "training_membership_sha256": training_digest,
        }
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
    body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions)
    run = {
        "schema_version": 1,
        "experiment": "tfidf-logreg-validation-v1",
        "scope": "Training-only fits; validation-only scores for " + ", ".join(metrics),
        "cohort_selection": cohort,
        "configuration": asdict(config),
        "hyperparameters_tuned": False,
        "final_test_evaluated": False,
        "seed": config.seed,
        "benchmark_manifest_sha256": receipt["benchmark_manifest_sha256"],
        "parent_split_manifest_sha256": receipt["parent_split_manifest_sha256"],
        "freeze_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "lockfile_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
        "vader": vader.metadata(),
        "metrics": metrics,
        "code_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "working_tree_dirty": bool(dirty.stdout) if dirty.returncode == 0 else None,
        "implementation_sha256": snapshot,
        "predictions_sha256": _digest(body),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "timing": "batch size 1; synthetic warm-up; inference only; one local pass",
    }
    run_body = json.dumps(run, indent=2, sort_keys=True) + "\n"
    run_id = _digest(run_body)
    directory = data_dir / "interim/tfidf-logreg-validation" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run.json").write_text(run_body, encoding="utf-8")
    (directory / "predictions.jsonl").write_text(body, encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(run, run_id), encoding="utf-8")
    report_path.with_suffix(".json").write_text(run_body, encoding="utf-8")
    return run


def render_report(run: dict[str, Any], run_id: str) -> str:
    lines = [
        "# TF-IDF + Logistic Regression validation",
        "",
        run["scope"],
        "",
        "One fixed configuration; no tuning or final-test scoring.",
        f"Class weighting: `{run['configuration']['class_weight']}`.",
        "Separate models fitted on each cohort's training texts/labels only.",
        "Vocabulary and IDF use training only. VADER scores the same validation rows.",
        "",
        "| Cohort | Train | Validation | Model | Accuracy | Macro F1 |",
        "| --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for name, result in run["metrics"].items():
        for model_name in ("logistic_regression", "vader", "training_majority"):
            score = result[model_name]
            lines.append(
                f"| {name} | {result['training_samples']} | {score['samples']} | "
                f"{model_name} | {score['accuracy']:.4f} | {score['macro_f1']:.4f} |"
            )
    for name, result in run["metrics"].items():
        score = result["logistic_regression"]
        lines += [
            "",
            f"## {name}",
            "",
            "| Class | Support | Precision | Recall | F1 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for label in LABELS:
            value = score["per_class"][label]
            lines.append(
                f"| {label} | {value['support']} | {value['precision']:.4f} | "
                f"{value['recall']:.4f} | {value['f1']:.4f} |"
            )
        lines += [
            "",
            "Confusion matrix: rows=gold, columns=predicted.",
            "Class order: negative/neutral/positive.",
            "",
            "```text",
            *(str(row) for row in score["confusion_matrix"]),
            "```",
            "",
            f"Vocabulary: {result['model']['vocabulary_size']} features. "
            f"Training: {result['training_seconds']:.3f} seconds. "
            f"Mean inference: {result['mean_inference_ms_per_text']:.3f} ms/text "
            f"(VADER {result['vader_mean_inference_ms_per_text']:.3f}).",
            f"Local model ID: `{result['model_id']}`.",
            f"Artifact SHA-256: `{result['artifact_sha256']}`.",
        ]
    lines += [
        "",
        f"Run ID: `{run_id}`.",
        f"Parent split: `{run['parent_split_manifest_sha256']}`.",
        f"Code revision: `{run['code_revision']}`; "
        f"dirty: `{run['working_tree_dirty']}`.",
        "Implementation digests identify code used before its commit.",
        "Settings, all estimator parameters, dependency versions, hashes, and metrics",
        "are in the accompanying aggregate JSON. Models and per-record outputs stay",
        "in ignored storage. Vocabulary, headlines, and record IDs stay local.",
        "These are development results, with separate corpus-specific training; they",
        "do not measure cross-corpus transfer, target-aware sentiment, calibration,",
        "or frozen-test generalization. Latency is descriptive on one local machine.",
        "See `docs/tfidf-logreg.md` for usage, preprocessing, and limitations.",
    ]
    return "\n".join(lines) + "\n"
