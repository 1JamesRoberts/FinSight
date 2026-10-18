"""Shared checked Hugging Face loading and sentence prediction."""

import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from typing import Any

from finsight.sentiment.artifacts import file_sha256, verify_files
from finsight.sentiment.models import deberta, roberta
from finsight.sentiment.models.finbert import INDEX_LABELS, probability_prediction
from finsight.sentiment.training.config import FineTuneConfig


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
    # Only the known head may be initialized; missing backbone tensors are errors.
    head, prefixes = (
        (roberta.BASE_HEAD_KEYS, roberta.LANGUAGE_HEAD_PREFIXES)
        if model_type == "roberta"
        else (deberta.BASE_HEAD_KEYS, deberta.LANGUAGE_HEAD_PREFIXES)
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
    # A base checkpoint gets a new sentiment head. A financial checkpoint must
    # retain its reviewed existing head and explicit label mapping.
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


def load_saved(
    directory: Path,
    identifier: str,
    asset: dict,
    config: FineTuneConfig,
    *,
    device: str | None = None,
) -> tuple:
    # Check ancestry, configuration and file bytes before constructing a model.
    if file_sha256(directory / "artifact-manifest.json") != identifier:
        raise ValueError("artifact manifest differs")
    manifest = json.loads((directory / "artifact-manifest.json").read_bytes())
    if manifest["source"] != asset or manifest["configuration"] != asdict(config):
        raise ValueError("saved artifact ancestry/configuration differs")
    verify_files(directory, manifest["files"])
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # Validate original training provenance above, then apply an explicit runtime
    # device override without rewriting the saved configuration or artifact.
    execution = replace(config, device=device) if device is not None else config
    torch = runtime(execution)
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
    model.to(device=execution.device, dtype=torch.float32)
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
