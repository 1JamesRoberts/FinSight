"""Fresh target backbones, explicit pair collation and inference without gradients."""

import hashlib
import json
import math
import random
from pathlib import Path
from time import perf_counter
from typing import Any

from finsight.data.target_pairs import LABELS, TargetExample, encode_batch
from finsight.sentiment.artifacts import verify_files
from finsight.sentiment.models.finbert import standardize_bert_head
from finsight.sentiment.models.transformer import (
    checked_labels,
    runtime,
    validate_loading,
)
from finsight.sentiment.training.config import FineTuneConfig

MODES = ("matched_headline_only", "target_pair")


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
    # Pair mode passes the target and headline as distinct tokenizer sequences;
    # the tokenizer inserts architecture-specific separators (no manual [SEP]).
    encoded = encode_batch(tokenizer, examples, mode, max_length=max_length)
    inputs = {
        key: torch.tensor(value, dtype=torch.long) for key, value in encoded.items()
    }
    # Gold labels are attached only for the training loss.
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
    # Disable dropout and gradient recording; validation cannot update weights.
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


def load_backbone(
    model_key: str, asset: dict, config: FineTuneConfig, data_dir: Path
) -> tuple:
    """Original local snapshots only, never a previously SEntFiN-adapted model."""
    if model_key == "roberta_base":
        return load_fresh(asset, config, data_dir)
    if model_key != "finbert":
        raise ValueError("backbone outside supplemental plan")
    import numpy as np
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch = runtime(config)
    directory = (
        data_dir
        / "interim/finbert-checkpoint-cache"
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
    )
    validate_loading(info, asset["model_type"], False)
    standardize_bert_head(torch, model)
    model.float()
    initial_state = state_sha256(model)
    tokenizer = AutoTokenizer.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False, use_fast=False
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
            "head_policy": "pretrained head retained; semantic output rows permuted",
        },
    )
