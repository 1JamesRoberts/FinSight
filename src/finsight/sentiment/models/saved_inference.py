"""Inference from locked local artifacts; no fitting, acquisition or gold labels."""

import gc
import os
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from finsight.sentiment.evaluation.protocol import safe_path, verify_artifact
from finsight.sentiment.models.svm import TfidfSVM
from finsight.sentiment.models.tfidf import TfidfSentiment
from finsight.sentiment.models.vader import VaderConfig, VaderSentiment


def infer_candidate(
    name: str, spec: dict[str, Any], texts: list[str], data_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Produce ordered predictions without ever receiving gold labels."""
    if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("nonempty texts required")
    loading = perf_counter()
    if "artifact" in spec:
        verify_artifact(data_dir, spec["artifact"], full=True)
    if name == "training_prior":
        rows = [
            {"prediction": spec["prediction"], "probabilities": spec["probabilities"]}
            for _ in texts
        ]
        return rows, {
            "architecture": "training-frequency prior",
            "loading_seconds": 0,
            "inference_seconds": 0,
            "batch_size": 1,
        }
    if name == "vader":
        config = {
            k: v for k, v in spec["configuration"].items() if k != "schema_version"
        }
        model = VaderSentiment(VaderConfig(**config))
        if model.metadata() != spec["resource_identity"]:
            raise ValueError("VADER resource identity differs")
    elif name in ("tfidf_logreg", "tfidf_svm"):
        artifact = spec["artifact"]
        cls = TfidfSentiment if name == "tfidf_logreg" else TfidfSVM
        model = cls.load(
            safe_path(data_dir, artifact["relative_path"]),
            expected_sha256=artifact["files"]["model.pkl"]["sha256"],
        )
        if asdict(model.config) != spec["configuration"]:
            raise ValueError("saved classifier configuration differs")
    else:
        return _gpu_inference(name, spec, texts, data_dir, loading)
    load_seconds = perf_counter() - loading
    model.predict("The company reported higher earnings.")
    started = perf_counter()
    rows = []
    for text in texts:
        prediction = model.predict(text)
        row = {"prediction": prediction.label.value}
        if name == "tfidf_logreg":
            row["probabilities"] = prediction.probabilities
        elif name == "tfidf_svm":
            row["decision_scores"] = prediction.decision_scores
        else:
            row["compound"] = prediction.compound
        rows.append(row)
    seconds = perf_counter() - started
    metadata = {
        "loading_seconds": load_seconds,
        "inference_seconds": seconds,
        "amortized_ms_per_text": seconds * 1000 / len(texts),
        "batch_size": 1,
        "model": model.metadata(),
    }
    if "artifact" in spec:
        metadata["saved_artifact_bytes"] = sum(
            f["size_bytes"] for f in spec["artifact"]["files"].values()
        )
    return rows, metadata


def _gpu_inference(
    name: str, spec: dict[str, Any], texts: list[str], data_dir: Path, started: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Explicit local paths and offline mode: inference never acquires assets.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from finsight.sentiment.models.transformer import (
        runtime,
        score_model,
        validate_loading,
    )
    from finsight.sentiment.training.config import FineTuneConfig

    config = FineTuneConfig(
        seed=spec["configuration"]["seed"], max_length=128, evaluation_batch_size=32
    )
    torch = runtime(config)
    directory = safe_path(data_dir, spec["artifact"]["relative_path"])
    model, info = AutoModelForSequenceClassification.from_pretrained(
        directory,
        local_files_only=True,
        trust_remote_code=False,
        weights_only=True,
        use_safetensors=name != "finbert_pretrained",
        attn_implementation="eager",
        output_loading_info=True,
    )
    try:
        validate_loading(info, model.config.model_type, False)
        tokenizer = AutoTokenizer.from_pretrained(
            directory,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=name.startswith("finbert")
            or (directory / "tokenizer.json").exists(),
        )
        model.to(device="cuda", dtype=torch.float32)
        torch.cuda.synchronize()
        loading_seconds = perf_counter() - started
        predictions, metadata = score_model(torch, tokenizer, model, texts, config)
        metadata.update(
            loading_seconds=loading_seconds,
            batch_size=32,
            saved_artifact_bytes=sum(
                f["size_bytes"] for f in spec["artifact"]["files"].values()
            ),
        )
        rows = [
            {
                "prediction": p.label.value,
                "probabilities": p.probabilities,
                "token_count": p.token_count,
                "truncated": p.truncated,
            }
            for p in predictions
        ]
        return rows, metadata
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
