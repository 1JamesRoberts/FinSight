"""Reviewed transformer validation, adaptation and separate corpus training."""

import argparse
import gc
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from finsight.data.benchmark import load_development_records
from finsight.sentiment.artifacts import file_sha256, save_model, source_manifest
from finsight.sentiment.artifacts import verify_files as verify_files
from finsight.sentiment.evaluation.cohorts import sentence_and_label
from finsight.sentiment.evaluation.metrics import classification_metrics
from finsight.sentiment.experiments.finbert_validation import saved_controls
from finsight.sentiment.models.finbert import INDEX_LABELS
from finsight.sentiment.models.transformer import (
    checked_labels,
    load_saved,
    load_source,
    score_model,
)
from finsight.sentiment.models.transformer import runtime as runtime
from finsight.sentiment.models.transformer import validate_loading as validate_loading
from finsight.sentiment.training.config import FineTuneConfig
from finsight.sentiment.training.sentence import (
    cohort_partitions,
    fit_checkpoint,
    membership_digest,
    sentfin_partitions,
)

ASSETS = Path("configs/transformer-assets.json")
CONFIG = Path("configs/transformer-training.json")
FOLLOWUP_EXPERIMENTS = {
    "financial_roberta_adapted": ("financial_roberta", "sentfin_single_target"),
    "financial_deberta_adapted": ("financial_deberta", "sentfin_single_target"),
    "roberta_phrasebank": ("roberta_base", "phrasebank"),
    "deberta_phrasebank": ("deberta_base", "phrasebank"),
}
INFERENCE_EXPERIMENTS = {
    "financial_roberta_phrasebank": ("financial_roberta", "phrasebank"),
    "financial_deberta_phrasebank": ("financial_deberta", "phrasebank"),
}
RECEIPT = Path("configs/benchmark-freeze.json")
REPORTS = Path("reports/experiments")


def predictions_bytes(rows: list[dict], predictions: list) -> bytes:
    return "".join(
        json.dumps(
            {
                "record_id": row["record_id"],
                "gold": sentence_and_label(row)[1],
                "prediction": p.label.value,
                "probabilities": p.probabilities,
            },
            sort_keys=True,
        )
        + "\n"
        for row, p in zip(rows, predictions, strict=True)
    ).encode()


def run_experiment(
    key: str,
    data_dir: Path,
    config_path: Path = CONFIG,
    *,
    report_dir: Path | None = None,
) -> dict:
    assets = json.loads(ASSETS.read_bytes())
    source_key, cohort = (FOLLOWUP_EXPERIMENTS | INFERENCE_EXPERIMENTS).get(
        key, (key, "sentfin_single_target")
    )
    asset = assets["models"][source_key]
    trains = asset["mode"] == "fine_tune_base" or key.endswith("_adapted")
    strategy = (
        "new_sentiment_head"
        if asset["mode"] == "fine_tune_base"
        else "adapt_existing_sentiment_head"
        if trains
        else "unchanged_inference"
    )
    config = FineTuneConfig.load(config_path)
    receipt = json.loads(RECEIPT.read_bytes())
    records = load_development_records(data_dir, RECEIPT)
    training, validation = cohort_partitions(records, cohort)
    baseline = json.loads((REPORTS / "tfidf-tuning.json").read_bytes())
    controls = saved_controls(baseline, receipt, cohort, membership_digest(validation))
    reference = next(
        item["run"]["metrics"][cohort]
        for item in baseline["candidates"]
        if item["run_id"] == controls["logistic_regression"]["source_run_id"]
    )
    if reference["training_membership_sha256"] != membership_digest(
        training
    ) or reference["training_samples"] != len(training):
        raise ValueError("training baseline membership differs")
    historical = {}
    for name in ("finbert-validation", "finbert-finetuning"):
        if cohort == "phrasebank" and name.endswith("finetuning"):
            continue
        report = REPORTS / f"{name}.json"
        run = json.loads(report.read_bytes())
        for field in ("benchmark_manifest_sha256", "parent_split_manifest_sha256"):
            if run[field] != receipt[field]:
                raise ValueError("FinBERT comparison lineage differs")
        membership = (
            run["validation_membership_sha256"]
            if name.endswith("finetuning")
            else run["metrics"][cohort]["validation_membership_sha256"]
        )
        if (
            membership != membership_digest(validation)
            or run["final_test_evaluated"] is not False
        ):
            raise ValueError("FinBERT comparison membership differs")
        if name.endswith("finetuning") and (
            run["training_membership_sha256"] != membership_digest(training)
            or run["training_samples"] != len(training)
        ):
            raise ValueError("FinBERT training membership differs")
        historical[name] = {
            "report_sha256": file_sha256(report),
            "metrics": run["metrics"]
            if name.endswith("finetuning")
            else run["metrics"][cohort]["finbert"],
        }
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    )
    fit, artifact = None, None
    print(
        f"Starting {key}: {strategy}; train {len(training) if trains else 0}, "
        f"validation {len(validation)}",
        flush=True,
    )
    if trains:
        # Every backbone uses the same sentence loop; only loading/saving differ.
        fit = fit_checkpoint(
            config,
            training,
            validation,
            data_dir,
            checkpoint_loader=lambda *_: load_source(asset, config, data_dir),
            artifact_saver=lambda model, tokenizer, _root, conf, epoch, membership: (
                save_model(model, tokenizer, data_dir, asset, conf, epoch, membership)
            ),
            checkpoint_files=tuple(asset["files"]),
            run_namespace=key,
            training_cohort=cohort,
            index_labels=(
                INDEX_LABELS
                if asset["mode"] == "fine_tune_base"
                else checked_labels(asset["id2label"])
            ),
        )
        directory = fit.pop("artifact_directory")
        # Score the checkpoint reloaded from disk to verify persistence fidelity.
        torch, tokenizer, model, manifest = load_saved(
            directory, fit["artifact_manifest_sha256"], asset, config
        )
        artifact = directory.relative_to(data_dir).as_posix()
        model_files = manifest["files"]
    else:
        torch, tokenizer, model, directory = load_source(asset, config, data_dir)
        model_files = asset["files"]
    predictions, metadata = score_model(
        torch,
        tokenizer,
        model,
        [sentence_and_label(row)[0] for row in validation],
        config,
    )
    scores = classification_metrics(
        [sentence_and_label(row)[1] for row in validation],
        [p.label.value for p in predictions],
    )
    if fit and scores != fit["history"][fit["best_epoch"] - 1]["validation"]:
        raise ValueError("saved checkpoint validation metrics differ")
    body = "".join(
        json.dumps(
            {
                "record_id": row["record_id"],
                "gold": sentence_and_label(row)[1],
                "prediction": p.label.value,
                "probabilities": p.probabilities,
            },
            sort_keys=True,
        )
        + "\n"
        for row, p in zip(validation, predictions, strict=True)
    ).encode()
    transfer, transfer_body = None, None
    if fit and cohort == "phrasebank":
        _, transfer_rows = sentfin_partitions(records)
        if {row["group_id"] for row in training} & {
            row["group_id"] for row in transfer_rows
        }:
            raise ValueError("training/transfer validation groups overlap")
        transfer_predictions, transfer_metadata = score_model(
            torch,
            tokenizer,
            model,
            [sentence_and_label(row)[0] for row in transfer_rows],
            config,
        )
        transfer_body = predictions_bytes(transfer_rows, transfer_predictions)
        transfer = {
            "cohort": "sentfin_single_target",
            "samples": len(transfer_rows),
            "used_for_checkpoint_selection": False,
            "validation_membership_sha256": membership_digest(transfer_rows),
            "metrics": classification_metrics(
                [sentence_and_label(row)[1] for row in transfer_rows],
                [p.label.value for p in transfer_predictions],
            ),
            "model": transfer_metadata,
            "predictions_sha256": hashlib.sha256(transfer_body).hexdigest(),
            "saved_baselines": saved_controls(
                baseline,
                receipt,
                "sentfin_single_target",
                membership_digest(transfer_rows),
            ),
        }
    run = {
        "schema_version": 1,
        "experiment": key,
        "training_cohort": cohort if fit else None,
        "evaluation_cohort": cohort,
        "training_strategy": strategy,
        "transfer_validation": transfer,
        "asset": asset,
        "fine_tuned_in_project": fit is not None,
        "final_test_evaluated": False,
        "training_samples": len(training) if fit else 0,
        "validation_samples": len(validation),
        "training_membership_sha256": membership_digest(training) if fit else None,
        "validation_membership_sha256": membership_digest(validation),
        "benchmark_manifest_sha256": receipt["benchmark_manifest_sha256"],
        "parent_split_manifest_sha256": receipt["parent_split_manifest_sha256"],
        "configuration": asdict(config),
        "code_revision": revision,
        "working_tree_dirty": dirty,
        "provenance": {
            "assets_sha256": file_sha256(ASSETS),
            "config_sha256": file_sha256(config_path),
            "freeze_receipt_sha256": file_sha256(RECEIPT),
            "lockfile_sha256": file_sha256(Path("uv.lock")),
            "implementation_sha256": source_manifest(),
        },
        "fit": fit,
        "artifact_relative_path": artifact,
        "model": metadata,
        "model_files": model_files,
        "model_file_bytes": sum(f["size_bytes"] for f in model_files.values()),
        "metrics": scores,
        "saved_baselines": controls,
        "finbert_comparisons": historical,
        "predictions_sha256": hashlib.sha256(body).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": version("torch"),
            "transformers": version("transformers"),
            "sentencepiece": version("sentencepiece"),
        },
        "selection": "validation macro F1; strict improvement; earliest exact tie"
        if fit
        else "none",
        "scope": (
            f"{cohort} selects trained epochs; "
            if fit
            else f"Unchanged checkpoint inference on {cohort}; no project training; "
        )
        + "development validation only; "
        "no final test; one seed/configuration. "
        "Models differ in size and prior training. "
        "Raw-text pretraining exposure unknown. Financial RoBERTa supervised "
        "provenance incomplete. "
        "Financial DeBERTa appears PhraseBank-supervised. Probabilities uncalibrated.",
    }
    content = (json.dumps(run, indent=2, sort_keys=True) + "\n").encode()
    identifier = hashlib.sha256(content).hexdigest()
    local = data_dir / "interim/transformer-comparison" / identifier
    local.mkdir(parents=True, exist_ok=True)
    (local / "run.json").write_bytes(content)
    (local / "predictions.jsonl").write_bytes(body)
    if transfer_body is not None:
        (local / "transfer-predictions.jsonl").write_bytes(transfer_body)
    destination = report_dir or REPORTS
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{key}.json").write_bytes(content)
    (destination / f"{key}.md").write_bytes(render_report(run, identifier).encode())
    print(
        f"{key}: accuracy {scores['accuracy']:.4f}, macro F1 "
        f"{scores['macro_f1']:.4f}; run {identifier}",
        flush=True,
    )
    del model, tokenizer
    gc.collect()
    if config.device == "cuda":
        torch.cuda.empty_cache()
    return run


def render_report(run: dict, identifier: str) -> str:
    lines = [
        f"# {run['experiment']}",
        "",
        run["scope"],
        "",
        f"Checkpoint `{run['asset']['model_id']}` at `{run['asset']['revision']}`.",
        f"Mode: {run['training_strategy']}; train {run['training_samples']}, "
        f"validation {run['validation_samples']}.",
        "",
        "| Model | Accuracy | Macro F1 |",
        "| --- | ---: | ---: |",
    ]
    scores = {name: item["metrics"] for name, item in run["saved_baselines"].items()}
    scores.update(
        {name: item["metrics"] for name, item in run["finbert_comparisons"].items()}
    )
    scores[run["experiment"]] = run["metrics"]
    lines.extend(
        f"| {name} | {s['accuracy']:.4f} | {s['macro_f1']:.4f} |"
        for name, s in scores.items()
    )
    lines.extend(
        [
            "",
            "## Configuration and model",
            "",
            f"Configuration: `{json.dumps(run['configuration'], sort_keys=True)}`.",
            f"Architecture: "
            f"`{json.dumps(run['model']['architecture'], sort_keys=True)}`.",
            f"Model/tokenizer/source-notice file bytes: {run['model_file_bytes']:,}.",
            f"GPU: {run['model']['gpu']}; CUDA {run['model']['cuda_runtime']}; "
            f"{run['model']['precision']}.",
            f"Inference: {run['model']['inference_seconds']:.3f}s; "
            f"{run['model']['amortized_ms_per_text']:.3f} ms/text at batch size 32.",
            "Timing includes tokenization, length audit and output transfer after "
            "synthetic warm-up; synchronized.",
            f"Truncated validation texts: {run['model']['truncated_texts']}; "
            f"maximum tokens: {run['model']['maximum_tokens']}.",
        ]
    )
    if run["fit"]:
        fit = run["fit"]
        lines.extend(
            [
                "",
                "## Training",
                "",
                f"Strategy: {run['training_strategy']}; all parameters trained "
                f"on {run['training_cohort']} only.",
                "AdamW (betas 0.9/0.999, eps 1e-8); bias/LayerNorm excluded from "
                "decay; linear warm-up/decay; gradient clipping.",
                f"Best epoch {fit['best_epoch']}; patience triggered "
                f"{fit['early_stopped']}; total steps {fit['optimizer_steps']}.",
                f"Training/validation/saving: "
                f"{fit['training_seconds_including_validation_and_saving']:.2f}s; "
                f"peak allocated/reserved bytes "
                f"{fit['peak_training_gpu_allocated_bytes']}/{fit['peak_training_gpu_reserved_bytes']}.",
                f"Training truncation: "
                f"`{json.dumps(fit['truncation'], sort_keys=True)}`.",
                f"Best artifact: `data/{run['artifact_relative_path']}`; manifest "
                f"`{fit['artifact_manifest_sha256']}`.",
                "Selected checkpoint reloaded and metrics reproduced. No optimizer "
                "state saved.",
                "",
                "| Epoch | Train loss | Validation loss | Accuracy | Macro F1 |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for epoch in fit["history"]:
            lines.append(
                f"| {epoch['epoch']} | {epoch['training_loss']:.4f} | "
                f"{epoch['validation_loss']:.4f} | "
                f"{epoch['validation']['accuracy']:.4f} | "
                f"{epoch['validation']['macro_f1']:.4f} |"
            )
    if run["transfer_validation"] is not None:
        transfer = run["transfer_validation"]
        lines.extend(
            [
                "",
                "## SEntFiN transfer validation",
                "",
                "Not used for checkpoint selection; no SEntFiN training.",
                f"Accuracy {transfer['metrics']['accuracy']:.4f}; "
                f"macro F1 {transfer['metrics']['macro_f1']:.4f} "
                f"on {transfer['samples']} texts.",
                "PhraseBank FinBERT comparison is diagnostic: "
                "known upstream supervision.",
            ]
        )
    lines.extend(
        [
            "",
            "## Class metrics",
            "",
            "| Class | Support | Precision | Recall | F1 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, s in run["metrics"]["per_class"].items():
        lines.append(
            f"| {label} | {s['support']} | {s['precision']:.4f} | "
            f"{s['recall']:.4f} | {s['f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Confusion matrix: rows gold, columns predicted; "
            "negative/neutral/positive.",
            "",
            "```text",
            *[str(row) for row in run["metrics"]["confusion_matrix"]],
            "```",
            "",
            f"Run `{identifier}`; code `{run['code_revision']}`; dirty "
            f"`{run['working_tree_dirty']}`.",
            "JSON pins source/model/configuration/code hashes, membership, "
            "baseline provenance and environment.",
            "Weights and individual predictions remain ignored. Validation selects "
            "trained checkpoints; no unbiased test claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        choices=(
            "financial_roberta",
            "financial_deberta",
            "roberta_base",
            "deberta_base",
            *FOLLOWUP_EXPERIMENTS,
            *INFERENCE_EXPERIMENTS,
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    run_experiment(args.experiment, args.data_dir, args.config)


if __name__ == "__main__":
    main()
