"""Reusable target-aware RoBERTa inference, with explicit optional company checks."""

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from finsight.data.target_pairs import LABELS, TargetText, encode_batch
from finsight.sentiment.artifacts import file_sha256, project_root
from finsight.sentiment.companies import (
    CompanyRegistry,
    CompanyResolution,
    resolve_company,
)
from finsight.sentiment.deployment import CONFIG, check_deployment
from finsight.sentiment.evaluation.protocol import safe_path
from finsight.sentiment.models.transformer import checked_labels, load_saved
from finsight.sentiment.training.config import FineTuneConfig


@dataclass(frozen=True)
class TargetPrediction:
    target: str
    label: str
    probabilities: dict[str, float]
    model_version: str
    deployment_sha256: str
    probabilities_calibrated: bool = False


@dataclass(frozen=True)
class CompanyPrediction:
    resolution: CompanyResolution
    prediction: TargetPrediction | None


class TargetSentimentPredictor:
    """Load once, then score explicit native targets without training or gold labels.

    Callers supply the intended target. Scoring does not identify entities, verify
    target relevance, or imply reliable coverage of the five-company registry.
    """

    def __init__(
        self,
        torch: Any,
        tokenizer: Any,
        model: Any,
        config: FineTuneConfig,
        *,
        model_version: str,
        deployment_sha256: str,
    ) -> None:
        if model.config.model_type != "roberta":
            raise ValueError("this interface requires a target-aware RoBERTa model")
        self.torch, self.tokenizer, self.model, self.config = (
            torch,
            tokenizer,
            model,
            config,
        )
        self.labels = checked_labels(
            {str(k): v for k, v in model.config.id2label.items()}
        )
        self.model_version = model_version
        self.deployment_sha256 = deployment_sha256
        model.eval()
        model.requires_grad_(False)

    @classmethod
    def load(
        cls,
        *,
        root: Path | None = None,
        data_dir: Path | None = None,
        config_path: Path | None = None,
        device: str = "cuda",
        batch_size: int = 32,
    ) -> "TargetSentimentPredictor":
        """Verify selection and artifact bytes, then load local weights exactly once.

        Device/batch overrides affect inference only. The original training config
        is still checked against the artifact manifest before model construction.
        """
        root = root or project_root()
        data_dir = data_dir or root / "data"
        config_path = config_path or root / CONFIG
        deployment = check_deployment(root, config_path)
        training = FineTuneConfig(**deployment["training_configuration"])
        execution = replace(training, device=device, evaluation_batch_size=batch_size)
        artifact = deployment["artifact"]
        torch, tokenizer, model, _ = load_saved(
            safe_path(data_dir, artifact["relative_path"]),
            artifact["manifest_sha256"],
            deployment["source"],
            training,
            device=device,
        )
        return cls(
            torch,
            tokenizer,
            model,
            execution,
            model_version=artifact["manifest_sha256"],
            deployment_sha256=file_sha256(config_path),
        )

    def predict(self, text: str, target_company: str) -> TargetPrediction:
        """Score one caller-supplied target; preserve its original spelling."""
        return self.predict_many([text], [target_company])[0]

    def predict_many(
        self, texts: list[str], targets: list[str]
    ) -> list[TargetPrediction]:
        """Paired lists, one result per input, with model/tokenizer reused in memory."""
        if (
            not texts
            or len(texts) != len(targets)
            or any(not isinstance(v, str) or not v.strip() for v in [*texts, *targets])
        ):
            raise ValueError(
                "equal nonempty lists of nonempty text and targets required"
            )
        inputs = [
            TargetText(str(i), text, target)
            for i, (text, target) in enumerate(zip(texts, targets, strict=True))
        ]
        results = []
        self.model.eval()
        for start in range(0, len(inputs), self.config.evaluation_batch_size):
            rows = inputs[start : start + self.config.evaluation_batch_size]
            # Target first; only the headline can be truncated. The existing
            # encoder rejects a target that leaves no room for headline tokens.
            encoded = encode_batch(
                self.tokenizer, rows, "target_pair", max_length=self.config.max_length
            )
            tensors = {
                k: self.torch.tensor(
                    v, dtype=self.torch.long, device=self.config.device
                )
                for k, v in encoded.items()
            }
            with self.torch.inference_mode():
                values = (
                    self.torch.softmax(self.model(**tensors).logits, -1).cpu().tolist()
                )
            if len(values) != len(rows):
                raise ValueError("incomplete model output")
            for row, value in zip(rows, values, strict=True):
                if (
                    len(value) != 3
                    or any(not math.isfinite(v) or not 0 <= v <= 1 for v in value)
                    or not math.isclose(sum(value), 1.0, abs_tol=1e-5)
                ):
                    raise ValueError("invalid model probabilities")
                probabilities = {
                    label: value[self.labels.index(label)] for label in LABELS
                }
                results.append(
                    TargetPrediction(
                        row.target,
                        max(LABELS, key=probabilities.__getitem__),
                        probabilities,
                        self.model_version,
                        self.deployment_sha256,
                    )
                )
        return results

    def predict_company(
        self,
        text: str,
        target_company: str,
        registry: CompanyRegistry,
        *,
        company_context: bool = False,
        allow_parent_rollup: bool = False,
    ) -> CompanyPrediction:
        """Optional registry guard: unresolved identities receive no model score.

        Resolution is metadata only: do not replace the supplied target with a
        ticker or canonical name, since training used native target strings.
        """
        if (
            not isinstance(text, str)
            or not text.strip()
            or not isinstance(target_company, str)
            or not target_company.strip()
        ):
            raise ValueError("nonempty text and target required")
        resolution = resolve_company(
            target_company,
            registry,
            company_context=company_context,
            allow_parent_rollup=allow_parent_rollup,
        )
        prediction = (
            self.predict(text, target_company)
            if resolution.status == "resolved"
            else None
        )
        return CompanyPrediction(resolution, prediction)
