"""Validated settings and validation-only early stopping shared by all backbones."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from finsight.sentiment.models.finbert import FinBertConfig


@dataclass(frozen=True)
class FineTuneConfig:
    """One optimizer/runtime configuration shared by sentence and target studies."""

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
    """Track strict improvements; stop after patience non-improving epochs."""

    patience: int
    best_epoch: int = 0
    best_score: float = -1.0
    stale_epochs: int = 0

    def observe(self, epoch: int, score: float) -> bool:
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("validation macro F1 must be finite and in [0, 1]")
        # Strict comparison deliberately preserves the first epoch on exact ties.
        if score > self.best_score:
            self.best_score, self.best_epoch, self.stale_epochs = score, epoch, 0
            return True
        self.stale_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.stale_epochs >= self.patience
