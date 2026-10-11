"""Adapt the reviewed sentiment checkpoint on SEntFiN training records only."""

import hashlib
import json
import math
import platform
import random
import subprocess
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

from finsight.data.benchmark import load_development_records
from finsight.sentiment.finbert import (
    CHECKPOINT_FILES,
    INDEX_LABELS,
    MODEL_ID,
    REVISION,
    FinBertConfig,
    FinBertSentiment,
    _load_checkpoint,
    file_sha256,
    probability_prediction,
    verify_finetuned_artifact,
)
from finsight.sentiment.finbert_validation import saved_controls
from finsight.sentiment.tfidf_validation import sentence_and_label
from finsight.sentiment.validation import classification_metrics, sentence_cohorts


@dataclass(frozen=True)
class FineTuneConfig:
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    train_batch_size: int = 16
    evaluation_batch_size: int = 32
    max_length: int = 128
    max_epochs: int = 5
    patience: int = 2
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = 42
    device: str = "cuda"

    def __post_init__(self) -> None:
        FinBertConfig(
            batch_size=self.evaluation_batch_size,
            max_length=self.max_length,
            seed=self.seed,
            device=self.device,
        )
        for name in ("train_batch_size", "max_epochs", "patience"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "weight_decay", "warmup_ratio", "max_grad_norm"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.max_grad_norm <= 0:
            raise ValueError("invalid optimizer configuration")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")

    @classmethod
    def load(cls, path: Path) -> "FineTuneConfig":
        body = json.loads(path.read_bytes())
        if (
            not isinstance(body, dict)
            or type(body.get("schema_version")) is not int
            or body["schema_version"] != 1
            or set(body) != {"schema_version", *cls.__dataclass_fields__}
        ):
            raise ValueError("unsupported fine-tuning configuration")
        return cls(**{k: v for k, v in body.items() if k != "schema_version"})

    def inference_config(self) -> FinBertConfig:
        return FinBertConfig(
            batch_size=self.evaluation_batch_size,
            max_length=self.max_length,
            seed=self.seed,
            device=self.device,
        )


@dataclass
class EarlyStopping:
    patience: int
    best_epoch: int = 0
    best_score: float = -1.0
    stale_epochs: int = 0

    def observe(self, epoch: int, score: float) -> bool:
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("validation macro F1 must be finite and in [0, 1]")
        if score > self.best_score:
            self.best_score, self.best_epoch, self.stale_epochs = score, epoch, 0
            return True
        self.stale_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.stale_epochs >= self.patience


def sentfin_partitions(records: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Select only single-target SEntFiN; reject reserved or overlapping groups."""
    rows = sentence_cohorts(records)["sentfin_single_target"]
    training = [row for row in rows if row["partition"] == "train"]
    validation = [row for row in rows if row["partition"] == "validation"]
    if not training or not validation:
        raise ValueError("SEntFiN training and validation must be nonempty")
    if {row["record_id"] for row in training} & {
        row["record_id"] for row in validation
    } or {row["group_id"] for row in training} & {
        row["group_id"] for row in validation
    }:
        raise ValueError("training/validation membership overlap")
    if set(sentence_and_label(row)[1] for row in training) != set(INDEX_LABELS):
        raise ValueError("training must contain the three sentiment classes")
    for row in validation:
        if sentence_and_label(row)[1] not in INDEX_LABELS:
            raise ValueError("invalid validation label")
    return training, validation


def membership_digest(rows: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps([row["record_id"] for row in rows]).encode()
    ).hexdigest()


def save_artifact(
    model: Any,
    tokenizer: Any,
    root: Path,
    config: FineTuneConfig,
    epoch: int,
    training_membership: str,
) -> tuple[Path, str]:
    """Save model/tokenizer as a complete, immutable content-addressed artifact."""
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=root) as temporary:
        stage = Path(temporary) / "checkpoint"
        model.save_pretrained(stage, safe_serialization=True)
        tokenizer.save_pretrained(stage)
        files = {
            path.name: {"size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in sorted(stage.iterdir())
            if path.is_file()
        }
        manifest = {
            "schema_version": 1,
            "model_id": MODEL_ID,
            "revision": REVISION,
            "strategy": "adapt_existing_sentiment_checkpoint",
            "cohort": "sentfin_single_target",
            "epoch": epoch,
            "training_membership_sha256": training_membership,
            "configuration": asdict(config),
            "files": files,
            "weight_license_spdx": None,
            "use_decision": "user_authorized_local_research_license_scope_unresolved",
        }
        body = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        identifier = hashlib.sha256(body).hexdigest()
        (stage / "artifact-manifest.json").write_bytes(body)
        destination = root / identifier
        if destination.exists():
            verify_finetuned_artifact(destination, identifier)
        else:
            stage.replace(destination)
    return destination, identifier


def fit_checkpoint(
    config: FineTuneConfig,
    training: list[dict],
    validation: list[dict],
    data_dir: Path,
) -> dict[str, Any]:
    """Adapt all parameters; select checkpoints using validation only."""
    checked_train, checked_validation = sentfin_partitions(training + validation)
    if checked_train != training or checked_validation != validation:
        raise ValueError("training and validation arguments have incorrect partitions")
    import numpy as np
    from transformers import get_linear_schedule_with_warmup

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch, tokenizer, model, snapshot = _load_checkpoint(
        config.inference_config(), data_dir / "interim/finbert-checkpoint-cache"
    )
    if {int(k): v for k, v in model.config.id2label.items()} != dict(
        enumerate(INDEX_LABELS)
    ):
        raise ValueError("unexpected checkpoint label mapping")
    model.to(device=config.device, dtype=torch.float32)
    model.requires_grad_(True)
    total_parameters = sum(p.numel() for p in model.parameters())
    source_files = {
        name: {
            "size_bytes": (snapshot / name).stat().st_size,
            "sha256": file_sha256(snapshot / name),
        }
        for name in CHECKPOINT_FILES
    }
    examples = [
        (sentence_and_label(row)[0], INDEX_LABELS.index(sentence_and_label(row)[1]))
        for row in training
    ]
    validation_examples = [sentence_and_label(row) for row in validation]
    truncation_counts = {}
    for name, texts in (
        ("training", [text for text, _ in examples]),
        ("validation", [text for text, _ in validation_examples]),
    ):
        lengths = [
            len(ids)
            for ids in tokenizer(texts, padding=False, truncation=False)["input_ids"]
        ]
        truncation_counts[name] = {
            "maximum_tokens": max(lengths),
            "truncated_texts": sum(n > config.max_length for n in lengths),
        }

    def collate(batch: list[tuple[str, int]]) -> dict:
        inputs = tokenizer(
            [text for text, _ in batch],
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
        )
        inputs["labels"] = torch.tensor([label for _, label in batch], dtype=torch.long)
        return inputs

    loader = torch.utils.data.DataLoader(
        examples,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(config.seed),
        collate_fn=collate,
    )
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (
            no_decay if name.endswith("bias") or "LayerNorm.weight" in name else decay
        ).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    maximum_steps = len(loader) * config.max_epochs
    warmup_steps = int(maximum_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, maximum_steps)
    stopping = EarlyStopping(config.patience)
    history, optimizer_steps = [], 0
    progress_dir = data_dir / "interim/finbert-training-progress" / uuid.uuid4().hex
    progress_dir.mkdir(parents=True)
    best_directory, best_identifier = None, None
    if config.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss, samples = 0.0, 0
        epoch_started = perf_counter()
        for batch in loader:
            inputs = {key: value.to(config.device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**inputs).loss
            if not torch.isfinite(loss).item():
                raise ValueError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            optimizer_steps += 1
            count = inputs["labels"].shape[0]
            train_loss += loss.item() * count
            samples += count
        model.eval()
        gold, predicted, validation_loss = [], [], 0.0
        with torch.inference_mode():
            for start in range(
                0, len(validation_examples), config.evaluation_batch_size
            ):
                batch_examples = validation_examples[
                    start : start + config.evaluation_batch_size
                ]
                batch = collate(
                    [
                        (text, INDEX_LABELS.index(label))
                        for text, label in batch_examples
                    ]
                )
                inputs = {key: value.to(config.device) for key, value in batch.items()}
                output = model(**inputs)
                validation_loss += output.loss.item() * len(batch_examples)
                probabilities = torch.softmax(output.logits, dim=-1).cpu().tolist()
                gold.extend(label for _, label in batch_examples)
                predicted.extend(
                    probability_prediction(values, 0, config.max_length).label.value
                    for values in probabilities
                )
        if config.device == "cuda":
            torch.cuda.synchronize()
        scores = classification_metrics(gold, predicted)
        improved = stopping.observe(epoch, scores["macro_f1"])
        if improved:
            best_directory, best_identifier = save_artifact(
                model,
                tokenizer,
                data_dir / "interim/finbert-finetuned-models",
                config,
                epoch,
                membership_digest(training),
            )
        entry = {
            "epoch": epoch,
            "training_loss": train_loss / samples,
            "validation_loss": validation_loss / len(validation),
            "validation": scores,
            "optimizer_steps": optimizer_steps,
            "learning_rate_end": scheduler.get_last_lr()[0],
            "seconds": perf_counter() - epoch_started,
            "improved": improved,
            "best_epoch": stopping.best_epoch,
            "stale_epochs": stopping.stale_epochs,
            "best_artifact_manifest_sha256": best_identifier,
        }
        history.append(entry)
        (progress_dir / "history.json").write_bytes(
            (json.dumps(history, indent=2, sort_keys=True) + "\n").encode()
        )
        print(
            f"Epoch {epoch}/{config.max_epochs}: train loss "
            f"{entry['training_loss']:.4f}; "
            f"validation accuracy {scores['accuracy']:.4f}, macro F1 "
            f"{scores['macro_f1']:.4f}; "
            f"best epoch {stopping.best_epoch}",
            flush=True,
        )
        if stopping.should_stop:
            print("Early stopping: validation macro F1 did not improve.", flush=True)
            break
    return {
        "history": history,
        "best_epoch": stopping.best_epoch,
        "best_validation_macro_f1": stopping.best_score,
        "early_stopped": stopping.should_stop,
        "optimizer_steps": optimizer_steps,
        "maximum_optimizer_steps": maximum_steps,
        "warmup_steps": warmup_steps,
        "training_seconds_including_validation_and_saving": perf_counter() - started,
        "peak_training_gpu_allocated_bytes": torch.cuda.max_memory_allocated()
        if config.device == "cuda"
        else None,
        "peak_training_gpu_reserved_bytes": torch.cuda.max_memory_reserved()
        if config.device == "cuda"
        else None,
        "parameters_updated": total_parameters,
        "source_checkpoint_files": source_files,
        "truncation": truncation_counts,
        "artifact_directory": best_directory,
        "artifact_manifest_sha256": best_identifier,
        "progress_relative_path": progress_dir.relative_to(data_dir).as_posix(),
    }


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
    root = Path(__file__).resolve().parents[3]
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
        "implementation_sha256": {
            name: file_sha256(root / "src/finsight" / name)
            for name in (
                "sentiment/finbert_training.py",
                "sentiment/finbert.py",
                "data/benchmark.py",
                "sentiment/validation.py",
            )
        },
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
