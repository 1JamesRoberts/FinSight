"""Coordinate FinBERT adaptation, verified reload and validation reporting."""

import hashlib
import json
import platform
import subprocess
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

from finsight.data.benchmark import load_development_records
from finsight.sentiment.artifacts import file_sha256, project_root, source_manifest
from finsight.sentiment.evaluation.cohorts import sentence_and_label
from finsight.sentiment.evaluation.metrics import classification_metrics
from finsight.sentiment.experiments.finbert_validation import saved_controls
from finsight.sentiment.models.finbert import MODEL_ID, REVISION, FinBertSentiment
from finsight.sentiment.training.config import FineTuneConfig
from finsight.sentiment.training.sentence import (
    fit_checkpoint,
    membership_digest,
    sentfin_partitions,
)


def run_finetuning(
    data_dir: Path,
    receipt_path: Path,
    config_path: Path,
    report_path: Path,
    baseline_path: Path = Path("reports/experiments/tfidf-tuning.json"),
    pretrained_path: Path = Path("reports/experiments/finbert-validation.json"),
) -> dict[str, Any]:
    """One fixed SEntFiN adaptation run; reload and score the saved best checkpoint."""
    config = FineTuneConfig.load(config_path)
    training, validation = sentfin_partitions(
        load_development_records(data_dir, receipt_path)
    )
    receipt = json.loads(receipt_path.read_bytes())
    baseline = json.loads(baseline_path.read_bytes())
    training_digest, validation_digest = (
        membership_digest(training),
        membership_digest(validation),
    )
    controls = saved_controls(
        baseline, receipt, "sentfin_single_target", validation_digest
    )
    selected_lr = baseline["selected_run_ids"]["sentfin_single_target"][
        "logistic_regression"
    ]
    reference = next(
        item["run"]["metrics"]["sentfin_single_target"]
        for item in baseline["candidates"]
        if item["run_id"] == selected_lr
    )
    counts = dict(
        sorted(Counter(sentence_and_label(row)[1] for row in training).items())
    )
    if (
        reference["training_membership_sha256"] != training_digest
        or reference["training_samples"] != len(training)
        or reference["training_class_counts"] != counts
    ):
        raise ValueError("baseline training membership differs")
    pretrained = json.loads(pretrained_path.read_bytes())
    for key in ("benchmark_manifest_sha256", "parent_split_manifest_sha256"):
        if pretrained[key] != receipt[key]:
            raise ValueError("pretrained baseline lineage differs")
    if (
        pretrained["final_test_evaluated"] is not False
        or pretrained["fine_tuned_in_project"] is not False
        or pretrained["configuration"]["revision"] != REVISION
        or pretrained["configuration"]["max_length"] != config.max_length
        or pretrained["metrics"]["sentfin_single_target"][
            "validation_membership_sha256"
        ]
        != validation_digest
    ):
        raise ValueError("pretrained baseline comparison differs")
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
    print(
        f"Adapting existing FinBERT sentiment checkpoint: {len(training)} train, "
        f"{len(validation)} validation texts; no final test.",
        flush=True,
    )
    fit = fit_checkpoint(config, training, validation, data_dir)
    directory = fit.pop("artifact_directory")
    model = FinBertSentiment.from_artifact(
        config.inference_config(), directory, fit["artifact_manifest_sha256"]
    )
    model.predict_batch(["The company reported higher earnings."])
    model.reset_peak_memory()
    texts = [sentence_and_label(row)[0] for row in validation]
    gold = [sentence_and_label(row)[1] for row in validation]
    model.synchronize()
    started = perf_counter()
    predictions = model.predict_batch(texts)
    model.synchronize()
    elapsed = perf_counter() - started
    scores = classification_metrics(
        gold, [prediction.label.value for prediction in predictions]
    )
    if scores != fit["history"][fit["best_epoch"] - 1]["validation"]:
        raise ValueError("reloaded artifact metrics differ from selected epoch")
    metadata = model.metadata()
    body = "".join(
        json.dumps(
            {
                "record_id": row["record_id"],
                "gold": label,
                "prediction": prediction.label.value,
                "probabilities": prediction.probabilities,
            },
            sort_keys=True,
        )
        + "\n"
        for row, label, prediction in zip(validation, gold, predictions, strict=True)
    ).encode()
    run = {
        "schema_version": 1,
        "experiment": "finbert-sentfin-finetuning-v1",
        "strategy": "adapt_existing_sentiment_checkpoint",
        "cohort": "sentfin_single_target",
        "configuration": asdict(config),
        "model_id": MODEL_ID,
        "revision": REVISION,
        "fine_tuned_in_project": True,
        "hyperparameter_grid_searched": False,
        "final_test_evaluated": False,
        "training_samples": len(training),
        "validation_samples": len(validation),
        "training_class_counts": counts,
        "training_membership_sha256": training_digest,
        "validation_membership_sha256": validation_digest,
        "benchmark_manifest_sha256": receipt["benchmark_manifest_sha256"],
        "parent_split_manifest_sha256": receipt["parent_split_manifest_sha256"],
        "freeze_receipt_sha256": file_sha256(receipt_path),
        "config_sha256": file_sha256(config_path),
        "lockfile_sha256": file_sha256(root / "uv.lock"),
        "baseline_report_sha256": file_sha256(baseline_path),
        "pretrained_report_sha256": file_sha256(pretrained_path),
        "code_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "working_tree_dirty": bool(dirty.stdout) if dirty.returncode == 0 else None,
        "implementation_sha256": source_manifest(),
        "selection": "maximum validation macro F1; strict improvement; earliest "
        "exact tie",
        "optimizer": "AdamW, betas=(0.9,0.999), eps=1e-8; bias/LayerNorm excluded "
        "from decay",
        "schedule": "linear warm-up then linear decay over maximum configured steps",
        "precision": "FP32; deterministic algorithms; TF32 disabled; no AMP",
        "fit": fit,
        "model": metadata,
        "artifact_relative_path": directory.relative_to(data_dir).as_posix(),
        "metrics": scores,
        "saved_baselines": controls,
        "pretrained_finbert": pretrained["metrics"]["sentfin_single_target"]["finbert"],
        "inference_seconds": elapsed,
        "amortized_ms_per_text": elapsed * 1000 / len(validation),
        "predictions_sha256": hashlib.sha256(body).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": version("torch"),
            "transformers": version("transformers"),
        },
        "scope": "SEntFiN development only; single configuration/seed; full "
        "adaptation of a "
        "PhraseBank-supervised checkpoint; weight license scope unresolved; local "
        "research decision",
    }
    run_body = (json.dumps(run, indent=2, sort_keys=True) + "\n").encode()
    identifier = hashlib.sha256(run_body).hexdigest()
    local = data_dir / "interim/finbert-finetuning" / identifier
    local.mkdir(parents=True, exist_ok=True)
    (local / "run.json").write_bytes(run_body)
    (local / "predictions.jsonl").write_bytes(body)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.with_suffix(".json").write_bytes(run_body)
    report_path.write_bytes(render_report(run, identifier).encode())
    print(
        f"Best epoch {fit['best_epoch']}: reloaded validation macro F1 "
        f"{scores['macro_f1']:.4f}",
        flush=True,
    )
    return run


def render_report(run: dict[str, Any], identifier: str) -> str:
    config, fit, model = run["configuration"], run["fit"], run["model"]
    architecture = model["architecture"]
    lines = [
        "# FinBERT adaptation on SEntFiN",
        "",
        run["scope"],
        "",
        "Continues training all parameters of the existing financial sentiment "
        "checkpoint. "
        "This is adaptation, not training a sentiment head on a base language model.",
        "",
        f"Train: {run['training_samples']}; validation: {run['validation_samples']}; "
        "PhraseBank, FiQA, multi-target headlines, and frozen test groups excluded "
        "from this run.",
        "",
        "| Model | Accuracy | Macro F1 |",
        "| --- | ---: | ---: |",
    ]
    comparisons = {
        name: item["metrics"] for name, item in run["saved_baselines"].items()
    }
    comparisons["finbert_pretrained"] = run["pretrained_finbert"]
    comparisons["finbert_adapted"] = run["metrics"]
    for name, score in comparisons.items():
        lines.append(f"| {name} | {score['accuracy']:.4f} | {score['macro_f1']:.4f} |")
    lines.extend(
        [
            "",
            "## Fixed configuration and checkpoint selection",
            "",
            f"Learning rate {config['learning_rate']}; train batch "
            f"{config['train_batch_size']}; "
            f"evaluation batch {config['evaluation_batch_size']}; max length "
            f"{config['max_length']}; "
            f"seed {config['seed']}; up to {config['max_epochs']} epochs; patience "
            f"{config['patience']}.",
            run["optimizer"] + ".",
            run["schedule"] + ".",
            run["precision"] + ".",
            f"Weight decay {config['weight_decay']}; warm-up {fit['warmup_steps']} "
            f"steps; "
            f"gradient clip norm {config['max_grad_norm']}. Training order "
            f"shuffled with fixed generator; "
            "no validation gradients or class weighting.",
            run["selection"] + ".",
            f"Selected epoch **{fit['best_epoch']}**; early stopped: "
            f"{fit['early_stopped']}. "
            f"Optimizer steps "
            f"{fit['optimizer_steps']}/{fit['maximum_optimizer_steps']}.",
            "",
            "| Epoch | Train loss | Validation loss | Accuracy | Macro F1 | "
            "Improved | Seconds |",
            "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for epoch in fit["history"]:
        score = epoch["validation"]
        lines.append(
            f"| {epoch['epoch']} | {epoch['training_loss']:.4f} | "
            f"{epoch['validation_loss']:.4f} | {score['accuracy']:.4f} | "
            f"{score['macro_f1']:.4f} | {epoch['improved']} | {epoch['seconds']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Model architecture, storage, and timing",
            "",
            f"{architecture['class']}: {architecture['layers']} layers, "
            f"hidden {architecture['hidden_size']}, "
            f"{architecture['attention_heads']} heads, "
            f"feed-forward {architecture['intermediate_size']}, vocabulary "
            f"{architecture['vocabulary_size']}.",
            f"Parameters trained: {fit['parameters_updated']:,}; "
            f"FP32 parameter bytes {architecture['parameter_bytes']:,}; "
            f"saved model/tokenizer files {model['checkpoint_bytes']:,} bytes.",
            f"GPU: {model['gpu']}; CUDA {model['cuda_runtime']}. "
            f"Training wall time (epochs/validation/saves): "
            f"{fit['training_seconds_including_validation_and_saving']:.2f}s.",
            f"Peak training allocated/reserved memory: "
            f"{fit['peak_training_gpu_allocated_bytes']} / "
            f"{fit['peak_training_gpu_reserved_bytes']} bytes.",
            f"Reloaded inference: {run['inference_seconds']:.3f}s, "
            f"{run['amortized_ms_per_text']:.3f} amortized ms/text at evaluation "
            f"batch size; "
            "synchronized, synthetic warm-up, includes tokenization/length audit "
            "and output transfer.",
            "Historical baseline timings use different hardware/batches; no direct "
            "speed claim.",
            f"Truncation audit: `{json.dumps(fit['truncation'], sort_keys=True)}`.",
            "",
            "## Class metrics",
            "",
            "| Class | Support | Precision | Recall | F1 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, score in run["metrics"]["per_class"].items():
        lines.append(
            f"| {label} | {score['support']} | {score['precision']:.4f} | "
            f"{score['recall']:.4f} | {score['f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Confusion matrix (rows=gold, columns=predicted; "
            "negative/neutral/positive):",
            "",
            "```text",
            *[str(row) for row in run["metrics"]["confusion_matrix"]],
            "```",
            "",
            "## Reproducibility and limitations",
            "",
            f"Run `{identifier}`; code revision `{run['code_revision']}`; "
            f"working tree dirty `{run['working_tree_dirty']}`.",
            f"Best artifact: ignored data/{run['artifact_relative_path']}; "
            f"manifest SHA-256 `{fit['artifact_manifest_sha256']}`.",
            "The best saved artifact was reloaded and its validation metrics "
            "matched the selected epoch. "
            "JSON logs source/model/artifact/configuration hashes, frozen lineage, "
            "ordered membership, "
            "per-epoch metrics, baselines, timing and environment. Predictions and "
            "weights stay ignored.",
            "Validation chooses the epoch and therefore is not an unbiased final "
            "performance estimate. "
            "One configuration and seed; no hyperparameter grid or statistical "
            "superiority claim. "
            "FinBERT previously used PhraseBank supervision; complete raw-news "
            "pretraining exposure is unknown. "
            "Probabilities remain uncalibrated. The frozen test remains unscored.",
            "Local research and derivative storage follow the documented user "
            "decision; "
            "hosted weight license scope remains unresolved and public "
            "redistribution is outside scope.",
            "",
        ]
    )
    return "\n".join(lines)
