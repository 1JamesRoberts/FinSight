"""Fresh DeBERTa training and separate inference for native target examples."""

import hashlib
import json
import math
import random
import uuid
from pathlib import Path
from time import perf_counter
from typing import Any

from finsight.data.audit import text_identity
from finsight.data.target_pairs import LABELS, TargetExample, encode_batch
from finsight.data.target_preparation import json_bytes
from finsight.sentiment.finbert_training import EarlyStopping, FineTuneConfig
from finsight.sentiment.transformer_comparison import (
    checked_labels,
    runtime,
    save_model,
    validate_loading,
    verify_files,
)
from finsight.sentiment.validation import classification_metrics

MODES = ("matched_headline_only", "target_pair")


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


def state_sha256(model: Any) -> str:
    """Fingerprint the entire initial state, including the new pooler/head."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_fresh(asset: dict, config: FineTuneConfig, data_dir: Path) -> tuple:
    """Load only the original reviewed cache; initialize the same head per seed."""
    import numpy as np
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch = runtime(config)
    directory = (
        data_dir
        / "interim/transformer-checkpoint-cache"
        / ("models--" + asset["model_id"].replace("/", "--"))
        / "snapshots"
        / asset["revision"]
    )
    verify_files(directory, asset["files"])
    model, info = AutoModelForSequenceClassification.from_pretrained(
        directory,
        local_files_only=True,
        trust_remote_code=False,
        weights_only=True,
        use_safetensors="model.safetensors" in asset["files"],
        attn_implementation="eager",
        output_loading_info=True,
        num_labels=3,
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
    )
    validate_loading(info, asset["model_type"], True)
    model.float()
    initial_state = state_sha256(model)
    tokenizer = AutoTokenizer.from_pretrained(
        directory,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=False,
    )
    model.to(device=config.device, dtype=torch.float32)
    rng = hashlib.sha256(torch.get_rng_state().numpy().tobytes())
    if config.device == "cuda":
        rng.update(torch.cuda.get_rng_state().cpu().numpy().tobytes())
    return (
        torch,
        tokenizer,
        model,
        {
            "state_sha256": initial_state,
            "rng_sha256": rng.hexdigest(),
            "loading_info": info,
            "source_files": asset["files"],
        },
    )


def collate_examples(
    torch: Any,
    tokenizer: Any,
    examples: tuple[TargetExample, ...] | list[TargetExample],
    mode: str,
    max_length: int,
    *,
    labels: bool,
) -> dict:
    if any(e.partition not in ("train", "validation") for e in examples):
        raise ValueError("development examples only")
    encoded = encode_batch(tokenizer, examples, mode, max_length=max_length)
    inputs = {
        key: torch.tensor(value, dtype=torch.long) for key, value in encoded.items()
    }
    if labels:
        inputs["labels"] = torch.tensor(
            [LABELS.index(e.label) for e in examples], dtype=torch.long
        )
    return inputs


def score_examples(
    torch: Any,
    tokenizer: Any,
    model: Any,
    examples: tuple[TargetExample, ...],
    mode: str,
    config: FineTuneConfig,
) -> tuple[list[dict], dict]:
    """Inference receives no gold-label tensor and never enables gradients."""
    from dataclasses import asdict

    if not examples or any(e.partition != "validation" for e in examples):
        raise ValueError("nonempty validation examples only")
    mapping = {str(k): v for k, v in model.config.id2label.items()}
    labels = checked_labels(mapping)
    model.eval()
    with torch.inference_mode():
        warm = collate_examples(
            torch, tokenizer, examples[:1], mode, config.max_length, labels=False
        )
        model(**{k: v.to(config.device) for k, v in warm.items()})
    if config.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started, rows, max_tokens = perf_counter(), [], 0
    for start in range(0, len(examples), config.evaluation_batch_size):
        batch = examples[start : start + config.evaluation_batch_size]
        inputs = collate_examples(
            torch, tokenizer, batch, mode, config.max_length, labels=False
        )
        max_tokens = max(
            max_tokens, int(inputs["attention_mask"].sum(dim=1).max().item())
        )
        with torch.inference_mode():
            values = (
                torch.softmax(
                    model(**{k: v.to(config.device) for k, v in inputs.items()}).logits,
                    dim=-1,
                )
                .cpu()
                .tolist()
            )
        for example, value in zip(batch, values, strict=True):
            if any(not math.isfinite(v) for v in value):
                raise ValueError("nonfinite inference probabilities")
            p = {label: value[labels.index(label)] for label in LABELS}
            rows.append(
                {
                    **asdict(example),
                    "gold": example.label,
                    "prediction": max(LABELS, key=p.__getitem__),
                    "probabilities": p,
                }
            )
    if config.device == "cuda":
        torch.cuda.synchronize()
    elapsed = perf_counter() - started
    c = model.config
    return rows, {
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
        "precision": "FP32; eager; deterministic; TF32 disabled",
        "gpu": torch.cuda.get_device_name(0) if config.device == "cuda" else None,
        "inference_seconds": elapsed,
        "inference_ms_per_annotation": elapsed * 1000 / len(rows),
        "maximum_encoded_tokens": max_tokens,
        "peak_inference_allocated_bytes": torch.cuda.max_memory_allocated()
        if config.device == "cuda"
        else None,
    }


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
) -> dict:
    """Update every parameter; select strictly by all-target validation macro F1."""
    from transformers import get_linear_schedule_with_warmup

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
    loader = torch.utils.data.DataLoader(
        training,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(config.seed),
        collate_fn=lambda rows: (
            rows,
            collate_examples(
                torch, tokenizer, rows, mode, config.max_length, labels=True
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
        rows, _ = score_examples(torch, tokenizer, model, validation, mode, config)
        scores = classification_metrics(
            [r["gold"] for r in rows], [r["prediction"] for r in rows]
        )
        nll = sum(
            -math.log(max(r["probabilities"][r["gold"]], 1e-15)) for r in rows
        ) / len(rows)
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
