"""Typed sentence-wide VADER inference with explicit classification thresholds."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from importlib.metadata import distribution, version
from importlib.resources import files
from pathlib import Path

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from finsight.data.loaders import SentimentLabel


@dataclass(frozen=True)
class VaderConfig:
    negative_threshold: float = -0.05
    positive_threshold: float = 0.05

    def __post_init__(self) -> None:
        values = (self.negative_threshold, self.positive_threshold)
        if any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in values
        ):
            raise ValueError("thresholds must be finite numbers")
        if not -1 <= self.negative_threshold < 0 < self.positive_threshold <= 1:
            raise ValueError(
                "thresholds must satisfy -1 <= negative < 0 < positive <= 1"
            )

    @classmethod
    def load(cls, path: Path) -> "VaderConfig":
        body = json.loads(path.read_text(encoding="utf-8"))
        if (
            set(body) != {"schema_version", "negative_threshold", "positive_threshold"}
            or body["schema_version"] != 1
        ):
            raise ValueError("unsupported VADER configuration")
        return cls(body["negative_threshold"], body["positive_threshold"])


DEFAULT_CONFIG = VaderConfig()


def classify_compound(
    score: float, config: VaderConfig = DEFAULT_CONFIG
) -> SentimentLabel:
    """Map finite compound scores; polar endpoints are inclusive."""
    if (
        type(score) not in (int, float)
        or not math.isfinite(score)
        or not -1 <= score <= 1
    ):
        raise ValueError("compound score must be finite and within [-1, 1]")
    if score <= config.negative_threshold:
        return SentimentLabel.NEGATIVE
    if score >= config.positive_threshold:
        return SentimentLabel.POSITIVE
    return SentimentLabel.NEUTRAL


@dataclass(frozen=True)
class VaderPrediction:
    label: SentimentLabel
    compound: float
    negative: float
    neutral: float
    positive: float


class VaderSentiment:
    """Reuse one bundled analyzer; no fitting, custom lexicon, or network access."""

    def __init__(self, config: VaderConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self._analyzer = SentimentIntensityAnalyzer()

    def predict(self, text: str) -> VaderPrediction:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("VADER input must be a nonempty string")
        scores = self._analyzer.polarity_scores(text)
        return VaderPrediction(
            classify_compound(scores["compound"], self.config),
            scores["compound"],
            scores["neg"],
            scores["neu"],
            scores["pos"],
        )

    def metadata(self) -> dict:
        """Bind predictions to the installed scoring code and bundled resources."""
        package = files("vaderSentiment")
        dist = distribution("vaderSentiment")
        return {
            "package": "vaderSentiment",
            "version": version("vaderSentiment"),
            "config": asdict(self.config),
            "resources_sha256": {
                name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                for name in (
                    "vaderSentiment.py",
                    "vader_lexicon.txt",
                    "emoji_utf8_lexicon.txt",
                )
            },
            "proportions_are_calibrated_probabilities": False,
            "license_notice_sha256": {
                str(path): hashlib.sha256(
                    dist.locate_file(path).read_bytes()
                ).hexdigest()
                for path in dist.files or []
                if "license" in str(path).lower()
            },
        }
