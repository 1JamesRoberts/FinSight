"""Pinned pretrained FinBERT inference; optional GPU dependencies load on demand."""

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from finsight.data.loaders import SentimentLabel

MODEL_ID = "ProsusAI/finbert"
REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"
INDEX_LABELS = ("positive", "negative", "neutral")
CHECKPOINT_FILES = (
    "README.md",
    "config.json",
    "pytorch_model.bin",
    "vocab.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
CHECKPOINT_HASHES = {
    "README.md": "23bf85400efe1e88cbe2bf79cec8c5ff81fe9806c599f3ca85e18288ab871008",
    "config.json": "f6449ddda85eb726207a40be59c0cd3bd4b142ccb27298d5e45f9ae3396b1abe",
    "pytorch_model.bin": (
        "e15a7b5738df7f17553399b6d94c6e2ff69c89245d066e8e5d183f5803a554e3"
    ),
}


def file_sha256(path: Path) -> str:
    """Hash checkpoint files without reading all weights into Python memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class FinBertConfig:
    model_id: str = MODEL_ID
    revision: str = REVISION
    batch_size: int = 32
    max_length: int = 128
    device: str = "cuda"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.model_id != MODEL_ID or self.revision != REVISION:
            raise ValueError("only the reviewed pinned FinBERT checkpoint is supported")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if type(self.max_length) is not int or not 2 <= self.max_length <= 512:
            raise ValueError("max_length must be an integer in [2, 512]")
        if self.device not in {"cuda", "cpu"}:
            raise ValueError("device must be cuda or cpu; no silent fallback")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a nonnegative 32-bit integer")

    @classmethod
    def load(cls, path: Path) -> "FinBertConfig":
        body = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(body, dict)
            or type(body.get("schema_version")) is not int
            or body["schema_version"] != 1
            or set(body) != {"schema_version", *cls.__dataclass_fields__}
        ):
            raise ValueError("unsupported FinBERT configuration")
        return cls(
            **{key: value for key, value in body.items() if key != "schema_version"}
        )


@dataclass(frozen=True)
class FinBertPrediction:
    label: SentimentLabel
    probabilities: dict[str, float]
    token_count: int
    truncated: bool


def probability_prediction(
    values: list[float], token_count: int, max_length: int
) -> FinBertPrediction:
    """Map checkpoint indices explicitly; softmax is not calibrated confidence."""
    if (
        len(values) != 3
        or any(not math.isfinite(value) or not 0 <= value <= 1 for value in values)
        or not math.isclose(sum(values), 1.0, abs_tol=1e-5)
    ):
        raise ValueError("probabilities must be three finite values summing to one")
    index = max(range(3), key=values.__getitem__)
    return FinBertPrediction(
        SentimentLabel(INDEX_LABELS[index]),
        dict(zip(INDEX_LABELS, values, strict=True)),
        token_count,
        token_count > max_length,
    )


def _load_checkpoint(
    config: FinBertConfig, cache_dir: Path
) -> tuple[Any, Any, Any, Path]:
    try:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "install the optional transformer extra before inference"
        ) from exc
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    snapshot = Path(
        snapshot_download(
            repo_id=config.model_id,
            revision=config.revision,
            cache_dir=str(cache_dir),
            allow_patterns=list(CHECKPOINT_FILES),
        )
    )
    for name, expected in CHECKPOINT_HASHES.items():
        if file_sha256(snapshot / name) != expected:
            raise ValueError("reviewed checkpoint checksum mismatch")
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    model, loading = AutoModelForSequenceClassification.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        weights_only=True,
        use_safetensors=False,
        attn_implementation="eager",
        output_loading_info=True,
    )
    if any(
        loading.get(key)
        for key in (
            "missing_keys",
            "unexpected_keys",
            "mismatched_keys",
            "error_msgs",
        )
    ):
        raise ValueError("checkpoint did not load exactly; refusing a random head")
    return torch, tokenizer, model, snapshot


class FinBertSentiment:
    """One unchanged checkpoint, dynamic padding, FP32, evaluation/inference mode."""

    def __init__(self, config: FinBertConfig, cache_dir: Path) -> None:
        self.config = config
        self._torch, self._tokenizer, self._model, self._snapshot = _load_checkpoint(
            config,
            cache_dir,
        )
        mapping = {
            int(key): value for key, value in self._model.config.id2label.items()
        }
        if mapping != dict(enumerate(INDEX_LABELS)):
            raise ValueError("unexpected checkpoint label mapping")
        self._model.to(device=config.device, dtype=self._torch.float32)
        self._model.eval()
        self._model.requires_grad_(False)

    def synchronize(self) -> None:
        if self.config.device == "cuda":
            self._torch.cuda.synchronize()

    def reset_peak_memory(self) -> None:
        if self.config.device == "cuda":
            self._torch.cuda.reset_peak_memory_stats()

    def predict_batch(self, texts: list[str]) -> list[FinBertPrediction]:
        """Preserve input order; reject empty/non-string records before scoring."""
        if not isinstance(texts, list) or any(
            not isinstance(text, str) or not text.strip() for text in texts
        ):
            raise ValueError("FinBERT inputs must be a list of nonempty strings")
        results = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = texts[start : start + self.config.batch_size]
            lengths = [
                len(ids)
                for ids in self._tokenizer(
                    batch,
                    truncation=False,
                    padding=False,
                )["input_ids"]
            ]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            ).to(self.config.device)
            with self._torch.inference_mode():
                logits = self._model(**inputs).logits
                probabilities = self._torch.softmax(logits, dim=-1).cpu().tolist()
            if len(probabilities) != len(batch):
                raise ValueError("checkpoint returned the wrong number of predictions")
            results.extend(
                probability_prediction(values, length, self.config.max_length)
                for values, length in zip(probabilities, lengths, strict=True)
            )
        self.synchronize()
        return results

    def metadata(self) -> dict[str, Any]:
        configuration = self._model.config
        files = {
            name: {
                "sha256": file_sha256(self._snapshot / name),
                "size_bytes": (self._snapshot / name).stat().st_size,
            }
            for name in CHECKPOINT_FILES
        }
        return {
            "configuration": asdict(self.config),
            "id2label": dict(enumerate(INDEX_LABELS)),
            "architecture": {
                "class": type(self._model).__name__,
                "model_type": configuration.model_type,
                "layers": configuration.num_hidden_layers,
                "hidden_size": configuration.hidden_size,
                "attention_heads": configuration.num_attention_heads,
                "intermediate_size": configuration.intermediate_size,
                "vocabulary_size": configuration.vocab_size,
                "maximum_positions": configuration.max_position_embeddings,
                "classes": configuration.num_labels,
                "total_parameters": sum(p.numel() for p in self._model.parameters()),
                "gradient_enabled_parameters": sum(
                    p.numel() for p in self._model.parameters() if p.requires_grad
                ),
                "parameter_bytes": sum(
                    p.numel() * p.element_size() for p in self._model.parameters()
                ),
                "dtype": "float32",
                "attention_implementation": "eager",
            },
            "checkpoint_files": files,
            "checkpoint_bytes": sum(item["size_bytes"] for item in files.values()),
            "device": self.config.device,
            "gpu": self._torch.cuda.get_device_name(0)
            if self.config.device == "cuda"
            else None,
            "cuda_runtime": self._torch.version.cuda,
            "peak_gpu_allocated_bytes": self._torch.cuda.max_memory_allocated()
            if self.config.device == "cuda"
            else None,
            "peak_gpu_reserved_bytes": self._torch.cuda.max_memory_reserved()
            if self.config.device == "cuda"
            else None,
            "fine_tuned_in_project": False,
            "probabilities_calibrated": False,
            "weight_license_spdx": None,
            "use_decision": "user_authorized_local_research_license_scope_unresolved",
        }
