"""Shared target-aware optimization over native target/headline examples."""

import hashlib
import math
import uuid
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

from finsight.data.audit import text_identity
from finsight.data.target_pairs import LABELS, TargetExample, TargetText
from finsight.data.target_preparation import json_bytes
from finsight.sentiment.artifacts import save_model
from finsight.sentiment.evaluation.metrics import classification_metrics
from finsight.sentiment.models.target import MODES as MODES
from finsight.sentiment.models.target import collate_examples, score_examples
from finsight.sentiment.models.target import load_fresh as load_fresh
from finsight.sentiment.models.target import state_sha256 as state_sha256
from finsight.sentiment.training.config import EarlyStopping, FineTuneConfig


def validate_partitions(
    training: tuple[TargetExample, ...], validation: tuple[TargetExample, ...]
) -> None:
    """Reject reserved examples and cross-partition membership/text/group overlap."""
    if not training or not validation:
        raise ValueError("nonempty training and validation examples required")
    if any(e.partition != "train" for e in training) or any(
        e.partition != "validation" or e.development_only for e in validation
    ):
        raise ValueError("training/validation partitions differ")
    rows = training + validation
    if len({e.example_id for e in rows}) != len(rows):
        raise ValueError("duplicate example IDs")
    if set(e.label for e in training) != set(LABELS) or any(
        e.label not in LABELS for e in validation
    ):
        raise ValueError("training must cover all three known labels")
    for field in ("record_id", "group_id", "headline"):
        values = [
            {
                text_identity(getattr(e, field))
                if field == "headline"
                else getattr(e, field)
                for e in partition
            }
            for partition in (training, validation)
        ]
        if values[0] & values[1]:
            raise ValueError(f"training/validation {field} overlap")


def membership_sha256(examples: tuple[TargetExample, ...] | list[TargetExample]) -> str:
    return hashlib.sha256(json_bytes([e.example_id for e in examples])).hexdigest()


def fit_target(
    torch: Any,
    tokenizer: Any,
    model: Any,
    training: tuple[TargetExample, ...],
    validation: tuple[TargetExample, ...],
    *,
    config: FineTuneConfig,
    data_dir: Path,
    asset: dict,
    context: dict,
    input_views: Mapping[str, TargetText] | None = None,
) -> dict:
    """Update every parameter; select strictly by all-target validation macro F1."""
    from transformers import get_linear_schedule_with_warmup

    # Keep all native targets for a headline within its reviewed data partition.
    validate_partitions(training, validation)
    mode = context["input_mode"]
    if mode not in MODES or {
        int(k): v for k, v in model.config.id2label.items()
    } != dict(enumerate(LABELS)):
        raise ValueError("input mode or new-head label order differs")
    model.requires_grad_(True)
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
    # Collation supplies labels for training; the scorer below receives them
    # only as metadata for evaluation, never as model input.
    loader = torch.utils.data.DataLoader(
        training,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(config.seed),
        collate_fn=lambda rows: (
            rows,
            collate_examples(
                torch,
                tokenizer,
                rows,
                mode,
                config.max_length,
                labels=True,
                input_views=input_views,
            ),
        ),
    )
    maximum_steps = len(loader) * config.max_epochs
    warmup = int(maximum_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, maximum_steps)
    stopping = EarlyStopping(config.patience)
    progress = data_dir / "interim/target-deberta-progress" / uuid.uuid4().hex
    progress.mkdir(parents=True)
    history, steps, best_directory, best_identity = [], 0, None, None
    train_peak = 0
    if config.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        loss_sum, count, order = 0.0, 0, []
        epoch_started = perf_counter()
        for batch_number, (examples, batch) in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            output = model(**{k: v.to(config.device) for k, v in batch.items()})
            if not torch.isfinite(output.loss).item():
                raise ValueError("nonfinite training loss")
            # One optimizer update per training minibatch; all parameters train.
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            steps += 1
            loss_sum += output.loss.item() * len(examples)
            count += len(examples)
            order.extend(examples)
            if batch_number % 200 == 0:
                print(
                    f"{mode} seed {config.seed}, epoch {epoch}: "
                    f"batch {batch_number}/{len(loader)}",
                    flush=True,
                )
        if count != len(training):
            raise ValueError("training epoch membership incomplete")
        if config.device == "cuda":
            torch.cuda.synchronize()
            train_peak = max(train_peak, torch.cuda.max_memory_allocated())
        # The shared scorer switches to eval/inference mode and disables gradients.
        rows, _ = score_examples(
            torch,
            tokenizer,
            model,
            validation,
            mode,
            config,
            input_views=input_views,
        )
        scores = classification_metrics(
            [r["gold"] for r in rows], [r["prediction"] for r in rows]
        )
        nll = sum(
            -math.log(max(r["probabilities"][r["gold"]], 1e-15)) for r in rows
        ) / len(rows)
        # Both input arms use the same validation metric and early-stopping rule.
        improved = stopping.observe(epoch, scores["macro_f1"])
        if improved:
            best_directory, best_identity = save_model(
                model,
                tokenizer,
                data_dir,
                asset,
                config,
                epoch,
                membership_sha256(training),
                context=context,
            )
        entry = {
            "epoch": epoch,
            "training_loss": loss_sum / count,
            "validation": scores,
            "validation_nll": nll,
            "improved": improved,
            "best_epoch": stopping.best_epoch,
            "stale_epochs": stopping.stale_epochs,
            "optimizer_steps": steps,
            "learning_rate_end": scheduler.get_last_lr()[0],
            "training_order_sha256": membership_sha256(order),
            "training_examples": count,
            "seconds": perf_counter() - epoch_started,
            "best_artifact_manifest_sha256": best_identity,
        }
        history.append(entry)
        (progress / "history.json").write_bytes(json_bytes(history))
        print(
            f"{mode} seed {config.seed}: epoch {epoch}, "
            f"macro F1 {scores['macro_f1']:.6f}; best {stopping.best_epoch}",
            flush=True,
        )
        if stopping.should_stop:
            break
    return {
        "history": history,
        "best_epoch": stopping.best_epoch,
        "best_validation_macro_f1": stopping.best_score,
        "early_stopped": stopping.should_stop,
        "optimizer_steps": steps,
        "maximum_optimizer_steps": maximum_steps,
        "warmup_steps": warmup,
        "parameters_updated": sum(p.numel() for p in model.parameters()),
        "training_seconds": perf_counter() - started,
        "peak_training_allocated_bytes": train_peak
        if config.device == "cuda"
        else None,
        "artifact_directory": best_directory,
        "artifact_manifest_sha256": best_identity,
        "progress_relative_path": progress.relative_to(data_dir).as_posix(),
    }
