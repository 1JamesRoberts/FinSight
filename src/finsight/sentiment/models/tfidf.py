"""A trainable sentence classifier; inference never fits text preprocessing."""

import hashlib
import json
import math
import pickle
import warnings
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from finsight.data.loaders import SentimentLabel

LABELS = ("negative", "neutral", "positive")


@dataclass(frozen=True)
class TfidfConfig:
    """Explicit settings for a fixed run; weights are derived from training only."""

    c: float = 1.0
    min_df: int = 2
    max_features: int = 50_000
    max_iter: int = 1000
    seed: int = 42
    class_weight: Literal["balanced"] | None = None

    def __post_init__(self) -> None:
        if self.class_weight is not None and self.class_weight != "balanced":
            raise ValueError("class_weight must be null or 'balanced'")
        if (
            isinstance(self.c, bool)
            or not isinstance(self.c, (float, int))
            or not math.isfinite(self.c)
            or self.c <= 0
        ):
            raise ValueError("c must be finite and positive")
        for name in ("min_df", "max_features", "max_iter", "seed"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"invalid {name}")
        if self.seed >= 2**32:
            raise ValueError("seed must fit an unsigned 32-bit integer")

    @classmethod
    def load(cls, path: Path) -> "TfidfConfig":
        body = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(body, dict)
            or type(body.get("schema_version")) is not int
            or body["schema_version"] != 1
            or body.keys() - {"schema_version", *cls.__dataclass_fields__}
        ):
            raise ValueError("unsupported TF-IDF configuration")
        return cls(
            **{key: value for key, value in body.items() if key != "schema_version"}
        )


@dataclass(frozen=True)
class TfidfPrediction:
    label: SentimentLabel
    probabilities: dict[str, float]


def validate_texts(texts: list[str]) -> None:
    if not texts or any(
        not isinstance(text, str) or not text.strip() for text in texts
    ):
        raise ValueError("texts must be nonempty strings")


def dependency_versions() -> dict[str, str]:
    return {name: version(name) for name in ("scikit-learn", "numpy", "scipy")}


def build_vectorizer(config: TfidfConfig) -> TfidfVectorizer:
    """Identical sparse word features for both traditional learned classifiers."""
    return TfidfVectorizer(
        lowercase=True,
        stop_words=None,
        ngram_range=(1, 2),
        min_df=config.min_df,
        max_features=config.max_features,
        sublinear_tf=True,
        norm="l2",
        smooth_idf=True,
    )


class TfidfSentiment:
    """TF-IDF word unigrams/bigrams followed by multinomial L2 logistic regression."""

    def __init__(self, config: TfidfConfig | None = None) -> None:
        self.config = config or TfidfConfig()
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    build_vectorizer(self.config),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=self.config.c,
                        solver="lbfgs",
                        max_iter=self.config.max_iter,
                        random_state=self.config.seed,
                        class_weight=self.config.class_weight,
                        tol=1e-4,
                    ),
                ),
            ]
        )

    def fit(self, texts: list[str], labels: list[str]) -> None:
        """Caller supplies training only. Both vocabulary and IDF are fitted here."""
        validate_texts(texts)
        if len(texts) != len(labels) or set(labels) != set(LABELS):
            raise ValueError("training requires matching texts and all three labels")
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            # Fit vocabulary/IDF on training texts before fitting the classifier.
            self.pipeline.fit(texts, labels)

    def predict_many(self, texts: list[str]) -> list[TfidfPrediction]:
        validate_texts(texts)
        check_is_fitted(self.pipeline.named_steps["classifier"])
        classes = self.pipeline.classes_.tolist()
        results = self.pipeline.predict_proba(texts)
        return [
            TfidfPrediction(
                label=SentimentLabel(classes[int(scores.argmax())]),
                probabilities={
                    label: float(scores[classes.index(label)]) for label in LABELS
                },
            )
            for scores in results
        ]

    def predict(self, text: str) -> TfidfPrediction:
        return self.predict_many([text])[0]

    def metadata(self) -> dict[str, Any]:
        check_is_fitted(self.pipeline.named_steps["classifier"])
        vectorizer = self.pipeline.named_steps["tfidf"]
        classifier = self.pipeline.named_steps["classifier"]
        vectorizer_params = vectorizer.get_params(deep=False)
        vectorizer_params["dtype"] = str(vectorizer_params["dtype"])
        return {
            "config": asdict(self.config),
            "dependencies": dependency_versions(),
            "tfidf_parameters": vectorizer_params,
            "classifier_parameters": classifier.get_params(deep=False),
            "classes": classifier.classes_.tolist(),
            "vocabulary_size": len(vectorizer.vocabulary_),
            "iterations": classifier.n_iter_.tolist(),
            "probabilities_calibrated": False,
        }

    def save(self, path: Path) -> str:
        """Save locally; fitted vocabulary/coefficients are restricted artifacts."""
        check_is_fitted(self.pipeline.named_steps["classifier"])
        body = pickle.dumps(
            {
                "schema_version": 1,
                "dependencies": dependency_versions(),
                "model": self,
            },
            protocol=5,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return hashlib.sha256(body).hexdigest()

    @classmethod
    def load(cls, path: Path, *, expected_sha256: str) -> "TfidfSentiment":
        """Load only trusted local artifacts; a digest does not make pickle safe."""
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != expected_sha256:
            raise ValueError("model checksum mismatch")
        payload = pickle.loads(body)
        if (
            payload["schema_version"] != 1
            or payload["dependencies"] != dependency_versions()
            or not isinstance(payload["model"], cls)
        ):
            raise ValueError("unsupported artifact schema or dependency versions")
        model = payload["model"]
        check_is_fitted(model.pipeline.named_steps["classifier"])
        return model
