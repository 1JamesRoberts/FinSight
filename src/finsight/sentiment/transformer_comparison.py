"""Reviewed RoBERTa/DeBERTa validation and SEntFiN base-model fine-tuning."""

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

from finsight.data.benchmark import load_development_records
from finsight.sentiment.finbert import INDEX_LABELS, file_sha256, probability_prediction
from finsight.sentiment.finbert_training import (
    FineTuneConfig,
    cohort_partitions,
    fit_checkpoint,
    membership_digest,
    sentfin_partitions,
)
from finsight.sentiment.finbert_validation import saved_controls
from finsight.sentiment.tfidf_validation import sentence_and_label
from finsight.sentiment.validation import classification_metrics

ASSETS = Path("configs/transformer-assets.json")
CONFIG = Path("configs/transformer-training.json")
FOLLOWUP_EXPERIMENTS = {
    "financial_roberta_adapted": ("financial_roberta", "sentfin_single_target"),
    "financial_deberta_adapted": ("financial_deberta", "sentfin_single_target"),
    "roberta_phrasebank": ("roberta_base", "phrasebank"),
    "deberta_phrasebank": ("deberta_base", "phrasebank"),
}
RECEIPT = Path("configs/benchmark-freeze.json")
REPORTS = Path("reports/experiments")


def checked_labels(mapping: dict) -> tuple[str, ...]:
    labels = tuple(str(mapping[str(i)]).lower() for i in range(3))
    if set(labels) != set(INDEX_LABELS):
        raise ValueError("checkpoint must have explicit three-class sentiment labels")
    return labels


def validate_loading(info: dict, model_type: str, training: bool) -> None:
    """Only base training may initialize known classification-head parameters."""
    if info.get("mismatched_keys") or info.get("error_msgs"):
        raise ValueError("checkpoint loading errors")
    missing, unexpected = (
        set(info.get("missing_keys", [])),
        info.get("unexpected_keys", []),
    )
    if not training:
        if missing or unexpected:
            raise ValueError("inference checkpoint loading must be exact")
        return
    head = (
        {
            "classifier.dense.weight",
            "classifier.dense.bias",
            "classifier.out_proj.weight",
            "classifier.out_proj.bias",
        }
        if model_type == "roberta"
        else {
            "classifier.weight",
            "classifier.bias",
            "pooler.dense.weight",
            "pooler.dense.bias",
        }
    )
    prefixes = (
        ("lm_head.",)
        if model_type == "roberta"
        else ("lm_predictions.", "mask_predictions.")
    )
    if missing != head or any(not key.startswith(prefixes) for key in unexpected):
        raise ValueError(f"unexpected base checkpoint loading: {info}")


def runtime(config: FineTuneConfig) -> Any:
    import torch

    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def verify_files(directory: Path, files: dict) -> None:
    for name, expected in files.items():
        if Path(name).name != name or "\\" in name or name in {".", ".."}:
            raise ValueError("unsafe checkpoint filename")
        path = directory / name
        if (
            path.stat().st_size != expected["size_bytes"]
            or file_sha256(path) != expected["sha256"]
        ):
            raise ValueError(f"checkpoint checksum differs: {name}")


def load_source(asset: dict, config: FineTuneConfig, data_dir: Path) -> tuple:
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch = runtime(config)
    directory = Path(
        snapshot_download(
            repo_id=asset["model_id"],
            revision=asset["revision"],
            cache_dir=str(data_dir / "interim/transformer-checkpoint-cache"),
            allow_patterns=list(asset["files"]),
        )
    )
    verify_files(directory, asset["files"])
    base = asset["mode"] == "fine_tune_base"
    kwargs = {}
    if base:
        kwargs = {
            "num_labels": 3,
            "id2label": dict(enumerate(INDEX_LABELS)),
            "label2id": {label: i for i, label in enumerate(INDEX_LABELS)},
        }
    model, info = AutoModelForSequenceClassification.from_pretrained(
        directory,
        local_files_only=True,
        trust_remote_code=False,
        weights_only=True,
        use_safetensors="model.safetensors" in asset["files"],
        attn_implementation="eager",
        output_loading_info=True,
        **kwargs,
    )
    validate_loading(info, asset["model_type"], base)
    tokenizer = AutoTokenizer.from_pretrained(
        directory,
        local_files_only=True,
        trust_remote_code=False,
        use_fast="tokenizer.json" in asset["files"],
    )
    model.to(device=config.device, dtype=torch.float32)
    print(f"Loaded verified {asset['model_id']}; loading info: {info}", flush=True)
    return torch, tokenizer, model, directory


def save_model(
    model: Any,
    tokenizer: Any,
    data_dir: Path,
    asset: dict,
    config: FineTuneConfig,
    epoch: int,
    membership: str,
) -> tuple[Path, str]:
    root = data_dir / "interim/transformer-finetuned-models"
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=root) as temporary:
        stage = Path(temporary) / "checkpoint"
        model.save_pretrained(stage, safe_serialization=True)
        tokenizer.save_pretrained(stage)
        manifest = {
            "schema_version": 1,
            "source": asset,
            "configuration": asdict(config),
            "epoch": epoch,
            "training_membership_sha256": membership,
            "files": {
                p.name: {"sha256": file_sha256(p), "size_bytes": p.stat().st_size}
                for p in sorted(stage.iterdir())
                if p.is_file()
            },
        }
        body = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        identifier = hashlib.sha256(body).hexdigest()
        (stage / "artifact-manifest.json").write_bytes(body)
        destination = root / identifier
        if destination.exists():
            if (destination / "artifact-manifest.json").read_bytes() != body:
                raise ValueError("artifact manifest differs")
            verify_files(destination, manifest["files"])
        else:
            stage.replace(destination)
    return destination, identifier


def load_saved(
    directory: Path, identifier: str, asset: dict, config: FineTuneConfig
) -> tuple:
    if file_sha256(directory / "artifact-manifest.json") != identifier:
        raise ValueError("artifact manifest differs")
    manifest = json.loads((directory / "artifact-manifest.json").read_bytes())
    if manifest["source"] != asset or manifest["configuration"] != asdict(config):
        raise ValueError("saved artifact ancestry/configuration differs")
    verify_files(directory, manifest["files"])
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch = runtime(config)
    model, info = AutoModelForSequenceClassification.from_pretrained(
        directory,
        local_files_only=True,
        trust_remote_code=False,
        weights_only=True,
        use_safetensors=True,
        attn_implementation="eager",
        output_loading_info=True,
    )
    validate_loading(info, asset["model_type"], False)
    tokenizer = AutoTokenizer.from_pretrained(
        directory,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=(directory / "tokenizer.json").exists(),
    )
    model.to(device=config.device, dtype=torch.float32)
    return torch, tokenizer, model, manifest


def score_model(
    torch: Any, tokenizer: Any, model: Any, texts: list[str], config: FineTuneConfig
) -> tuple[list, dict]:
    mapping = {str(k): v for k, v in model.config.id2label.items()}
    labels = checked_labels(mapping)
    model.eval()
    model.requires_grad_(False)
    # Reorder probabilities to the shared sentiment mapping without altering weights.
    indices = [labels.index(label) for label in INDEX_LABELS]
    warm = tokenizer(["The company reported higher earnings."], return_tensors="pt")
    with torch.inference_mode():
        model(**warm.to(config.device))
    if config.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    predictions = []
    for start in range(0, len(texts), config.evaluation_batch_size):
        batch = texts[start : start + config.evaluation_batch_size]
        lengths = [
            len(ids)
            for ids in tokenizer(batch, padding=False, truncation=False)["input_ids"]
        ]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
        ).to(config.device)
        with torch.inference_mode():
            values = torch.softmax(model(**inputs).logits, dim=-1).cpu().tolist()
        predictions.extend(
            probability_prediction(
                [value[i] for i in indices], length, config.max_length
            )
            for value, length in zip(values, lengths, strict=True)
        )
    if config.device == "cuda":
        torch.cuda.synchronize()
    elapsed = perf_counter() - started
    c = model.config
    metadata = {
        "architecture": {
            "class": type(model).__name__,
            "layers": c.num_hidden_layers,
            "hidden_size": c.hidden_size,
            "attention_heads": c.num_attention_heads,
            "intermediate_size": c.intermediate_size,
            "vocabulary_size": c.vocab_size,
            "parameters": sum(p.numel() for p in model.parameters()),
            "parameter_bytes": sum(
                p.numel() * p.element_size() for p in model.parameters()
            ),
        },
        "id2label": mapping,
        "precision": "FP32; eager attention; deterministic; TF32 disabled",
        "inference_seconds": elapsed,
        "amortized_ms_per_text": elapsed * 1000 / len(texts),
        "gpu": torch.cuda.get_device_name(0) if config.device == "cuda" else None,
        "cuda_runtime": torch.version.cuda,
        "peak_inference_allocated_bytes": torch.cuda.max_memory_allocated()
        if config.device == "cuda"
        else None,
        "peak_inference_reserved_bytes": torch.cuda.max_memory_reserved()
        if config.device == "cuda"
        else None,
        "truncated_texts": sum(p.truncated for p in predictions),
        "maximum_tokens": max(p.token_count for p in predictions),
    }
    return predictions, metadata


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


def run_experiment(key: str, data_dir: Path, config_path: Path = CONFIG) -> dict:
    assets = json.loads(ASSETS.read_bytes())
    source_key, cohort = FOLLOWUP_EXPERIMENTS.get(key, (key, "sentfin_single_target"))
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
        f"Starting {key}: {asset['mode']}; train {len(training)}, validation "
        f"{len(validation)}",
        flush=True,
    )
    if trains:
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
    if cohort == "phrasebank":
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
        "training_cohort": cohort,
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
            "implementation_sha256": {
                name: file_sha256(Path("src/finsight/sentiment") / name)
                for name in ("transformer_comparison.py", "finbert_training.py")
            },
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
        "scope": f"{cohort} selects trained epochs; development validation only; "
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
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{key}.json").write_bytes(content)
    (REPORTS / f"{key}.md").write_bytes(render_report(run, identifier).encode())
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
    if run["training_cohort"] == "phrasebank":
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
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    run_experiment(args.experiment, args.data_dir, args.config)


if __name__ == "__main__":
    main()
