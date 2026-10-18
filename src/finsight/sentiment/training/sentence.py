"""Shared sentence-classifier training loop; callers supply the backbone and corpus."""

import hashlib
import json
import random
import uuid
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

from finsight.sentiment.artifacts import file_sha256, save_artifact
from finsight.sentiment.evaluation.cohorts import sentence_and_label, sentence_cohorts
from finsight.sentiment.evaluation.metrics import classification_metrics
from finsight.sentiment.models.finbert import (
    CHECKPOINT_FILES,
    INDEX_LABELS,
    _load_checkpoint,
    probability_prediction,
)
from finsight.sentiment.training.config import EarlyStopping
from finsight.sentiment.training.config import FineTuneConfig as FineTuneConfig


def cohort_partitions(
    records: list[dict[str, Any]], cohort: str
) -> tuple[list[dict], list[dict]]:
    """Select one development cohort; reject reserved or overlapping groups."""
    if cohort not in {"phrasebank", "sentfin_single_target"}:
        raise ValueError("unsupported training cohort")
    rows = sentence_cohorts(records)[cohort]
    training = [row for row in rows if row["partition"] == "train"]
    validation = [row for row in rows if row["partition"] == "validation"]
    if not training or not validation:
        raise ValueError(f"{cohort} training and validation must be nonempty")
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


def sentfin_partitions(records: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Compatibility entry point for single-target SEntFiN development only."""
    return cohort_partitions(records, "sentfin_single_target")


def membership_digest(rows: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps([row["record_id"] for row in rows]).encode()
    ).hexdigest()


def fit_checkpoint(
    config: FineTuneConfig,
    training: list[dict],
    validation: list[dict],
    data_dir: Path,
    *,
    checkpoint_loader: Callable | None = None,
    artifact_saver: Callable | None = None,
    checkpoint_files: tuple[str, ...] = CHECKPOINT_FILES,
    run_namespace: str = "finbert",
    training_cohort: str = "sentfin_single_target",
    index_labels: tuple[str, ...] = INDEX_LABELS,
    frozen_parameters: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Adapt all parameters; select checkpoints using validation only."""
    if len(index_labels) != 3 or set(index_labels) != set(INDEX_LABELS):
        raise ValueError("invalid training label mapping")
    # Reject reserved records and leakage before loading a model or fitting anything.
    checked_train, checked_validation = cohort_partitions(
        training + validation, training_cohort
    )
    if checked_train != training or checked_validation != validation:
        raise ValueError("training and validation arguments have incorrect partitions")
    import numpy as np
    from transformers import get_linear_schedule_with_warmup

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch, tokenizer, model, snapshot = (checkpoint_loader or _load_checkpoint)(
        config.inference_config(), data_dir / "interim/finbert-checkpoint-cache"
    )
    if {int(k): v for k, v in model.config.id2label.items()} != dict(
        enumerate(index_labels)
    ):
        raise ValueError("unexpected checkpoint label mapping")
    model.to(device=config.device, dtype=torch.float32)
    model.requires_grad_(True)
    parameters = dict(model.named_parameters())
    if set(frozen_parameters) - parameters.keys():
        raise ValueError("unknown frozen parameter")
    for name in frozen_parameters:
        parameters[name].requires_grad_(False)
    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not total_parameters:
        raise ValueError("training requires trainable parameters")
    source_files = {
        name: {
            "size_bytes": (snapshot / name).stat().st_size,
            "sha256": file_sha256(snapshot / name),
        }
        for name in checkpoint_files
    }
    # Inputs are raw training texts; labels become class indices for cross-entropy.
    examples = [
        (sentence_and_label(row)[0], index_labels.index(sentence_and_label(row)[1]))
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

    # Tokenize each minibatch and pad to its longest sequence, up to max_length.
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

    # This seeded generator controls training order independently of model RNG.
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
        if not parameter.requires_grad:
            continue
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
    progress_dir = (
        data_dir / "interim" / f"{run_namespace}-training-progress" / uuid.uuid4().hex
    )
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
            # Backpropagate training loss, clip large gradients, then update weights.
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
        # Validation measures the current checkpoint without building gradients.
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
                        (text, index_labels.index(label))
                        for text, label in batch_examples
                    ]
                )
                inputs = {key: value.to(config.device) for key, value in batch.items()}
                output = model(**inputs)
                validation_loss += output.loss.item() * len(batch_examples)
                probabilities = torch.softmax(output.logits, dim=-1).cpu().tolist()
                gold.extend(label for _, label in batch_examples)
                predicted.extend(
                    probability_prediction(
                        [values[index_labels.index(label)] for label in INDEX_LABELS],
                        0,
                        config.max_length,
                    ).label.value
                    for values in probabilities
                )
        if config.device == "cuda":
            torch.cuda.synchronize()
        scores = classification_metrics(gold, predicted)
        # Save only a strict validation macro-F1 improvement; exact ties keep
        # the earlier epoch. No test data participates in checkpoint selection.
        improved = stopping.observe(epoch, scores["macro_f1"])
        if improved:
            best_directory, best_identifier = (artifact_saver or save_artifact)(
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


# These names support historical callers; experiment orchestration lives separately.


def __getattr__(name: str) -> Any:
    if name in {"run_finetuning", "render_report"}:
        from finsight.sentiment.experiments import finbert_training

        return getattr(finbert_training, name)
    raise AttributeError(name)
