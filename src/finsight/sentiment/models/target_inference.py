"""Label-free target prediction from already-verified local checkpoints."""

import gc
from pathlib import Path
from time import perf_counter
from typing import Any

from finsight.data.target_pairs import TargetText as TargetText
from finsight.data.target_pairs import encode_batch
from finsight.sentiment.evaluation.metrics import LABELS
from finsight.sentiment.evaluation.protocol import safe_path, verify_artifact
from finsight.sentiment.models.target_tfidf import TargetTfidf
from finsight.sentiment.models.transformer import checked_labels, load_saved
from finsight.sentiment.training.config import FineTuneConfig


def unique_inputs(
    inputs: list[TargetText], arm: str
) -> tuple[list[TargetText], list[int]]:
    if not inputs or arm not in ("matched_headline_only", "target_pair"):
        raise ValueError("nonempty target inputs and supported arm required")
    unique, indices, seen, headlines = [], [], {}, {}
    for row in inputs:
        if not all(
            isinstance(v, str) and v.strip()
            for v in (row.record_id, row.headline, row.target)
        ):
            raise ValueError(
                "nonempty input identities, headlines and targets required"
            )
        if row.record_id in headlines and headlines[row.record_id] != row.headline:
            raise ValueError("one headline identity has conflicting text")
        headlines[row.record_id] = row.headline
        key = (
            (row.record_id,)
            if arm == "matched_headline_only"
            else (row.record_id, row.target)
        )
        if key not in seen:
            seen[key] = len(unique)
            unique.append(row)
        indices.append(seen[key])
    return unique, indices


def score_inputs(
    torch: Any,
    tokenizer: Any,
    model: Any,
    inputs: list[TargetText],
    arm: str,
    config: FineTuneConfig,
) -> tuple[list[dict], dict]:
    """Tokenize without gold, disable dropout/gradients and restore annotation order."""
    unique, indices = unique_inputs(inputs, arm)
    labels = checked_labels({str(k): v for k, v in model.config.id2label.items()})
    model.eval()
    model.requires_grad_(False)

    def batch(rows: list[TargetText]) -> dict:
        encoded = encode_batch(tokenizer, rows, arm, max_length=config.max_length)
        return {
            k: torch.tensor(v, dtype=torch.long, device=config.device)
            for k, v in encoded.items()
        }

    with torch.inference_mode():
        model(**batch(unique[:1]))
    if config.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started, predictions, maximum = perf_counter(), [], 0
    for start in range(0, len(unique), config.evaluation_batch_size):
        encoded = batch(unique[start : start + config.evaluation_batch_size])
        maximum = max(maximum, int(encoded["attention_mask"].sum(1).max().item()))
        with torch.inference_mode():
            values = torch.softmax(model(**encoded).logits, -1).cpu().tolist()
        for value in values:
            p = {label: value[labels.index(label)] for label in LABELS}
            predictions.append(
                {"prediction": max(LABELS, key=p.__getitem__), "probabilities": p}
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
        "batch_size": config.evaluation_batch_size,
        "device": config.device,
        "inference_seconds": elapsed,
        "maximum_encoded_tokens": maximum,
        "annotations": len(inputs),
        "unique_inputs_scored": len(unique),
        "broadcast": arm == "matched_headline_only",
        "probabilities_calibrated": False,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated()
        if config.device == "cuda"
        else None,
    }
    return [predictions[i].copy() for i in indices], metadata


def predict(
    spec: dict, inputs: list[TargetText], data_dir: Path
) -> tuple[list[dict], dict]:
    """Verify local artifacts before loading; never acquire or fit models."""
    started = perf_counter()
    arm = spec["arm"]
    unique, indices = unique_inputs(inputs, arm)
    if spec["kind"] == "prior":
        return [
            {
                "prediction": spec["prediction"],
                "probabilities": dict(spec["probabilities"]),
            }
            for _ in inputs
        ], {"kind": "training_prior", "loading_seconds": 0, "inference_seconds": 0}
    if spec["kind"] == "tfidf":
        model = TargetTfidf.load(
            safe_path(data_dir, spec["relative_path"]), expected_sha256=spec["sha256"]
        )
        expected = (
            "headline_only" if arm == "matched_headline_only" else "headline_target"
        )
        if model.mode != expected:
            raise ValueError("sparse model input arm differs")
        loading = perf_counter() - started
        model.predict_many([unique[0].headline], [unique[0].target])
        started = perf_counter()
        result = model.predict_many(
            [r.headline for r in unique], [r.target for r in unique]
        )
        predictions = [
            {"prediction": r.label.value, "probabilities": r.probabilities}
            for r in result
        ]
        return [predictions[i].copy() for i in indices], {
            "model": model.metadata(),
            "loading_seconds": loading,
            "inference_seconds": perf_counter() - started,
            "unique_inputs_scored": len(unique),
            "annotations": len(inputs),
            "broadcast": arm == "matched_headline_only",
        }
    if spec["kind"] != "transformer":
        raise ValueError("unsupported target model kind")
    import json

    verify_artifact(data_dir, spec["artifact"], full=True)
    directory = safe_path(data_dir, spec["artifact"]["relative_path"])
    manifest = json.loads((directory / "artifact-manifest.json").read_bytes())
    config = FineTuneConfig(**spec["configuration"])
    torch, tokenizer, model, _ = load_saved(
        directory, spec["artifact"]["manifest_sha256"], manifest["source"], config
    )
    try:
        torch.cuda.synchronize()
        loading = perf_counter() - started
        rows, metadata = score_inputs(torch, tokenizer, model, inputs, arm, config)
        return rows, {
            **metadata,
            "loading_seconds": loading,
            "artifact_manifest_sha256": spec["artifact"]["manifest_sha256"],
        }
    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
