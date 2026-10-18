"""Sparse headline-only and additive headline/target logistic regression."""

import hashlib
import pickle
import warnings
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal

from scipy.sparse import csr_matrix, hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from finsight.data.loaders import SentimentLabel
from finsight.data.target_pairs import TargetExample
from finsight.sentiment.models.tfidf import (
    LABELS,
    TfidfConfig,
    TfidfPrediction,
    build_vectorizer,
    dependency_versions,
    validate_texts,
)

Mode = Literal["headline_only", "headline_target"]


class TargetTfidf:
    """Target features add class-specific offsets; no target/text interactions."""

    def __init__(
        self,
        mode: Mode,
        config: TfidfConfig | None = None,
        *,
        target_min_df: int = 1,
        target_max_features: int = 10_000,
    ) -> None:
        if mode not in ("headline_only", "headline_target"):
            raise ValueError("unsupported target TF-IDF mode")
        self.mode = mode
        self.config = config or TfidfConfig()
        self.target_config = replace(
            self.config, min_df=target_min_df, max_features=target_max_features
        )
        self.headline_vectorizer = build_vectorizer(self.config)
        self.target_vectorizer = build_vectorizer(self.target_config)
        self.classifier = LogisticRegression(
            C=self.config.c,
            solver="lbfgs",
            max_iter=self.config.max_iter,
            random_state=self.config.seed,
            class_weight=self.config.class_weight,
            tol=1e-4,
        )

    @staticmethod
    def _validate(headlines: list[str], targets: list[str]) -> None:
        validate_texts(headlines)
        validate_texts(targets)
        if len(headlines) != len(targets):
            raise ValueError("headline/target lengths differ")

    def fit(self, examples: tuple[TargetExample, ...]) -> None:
        """Fit vocabulary, IDF and coefficients exclusively on training examples."""
        if not examples or any(e.partition != "train" for e in examples):
            raise ValueError("nonempty training examples only")
        if len({e.example_id for e in examples}) != len(examples):
            raise ValueError("duplicate annotation IDs")
        headlines, targets = (
            [e.headline for e in examples],
            [e.target for e in examples],
        )
        self._validate(headlines, targets)
        labels = [e.label for e in examples]
        if set(labels) != set(LABELS):
            raise ValueError("training requires all three labels")
        x = self.headline_vectorizer.fit_transform(headlines)
        if self.mode == "headline_target":
            x = hstack([x, self.target_vectorizer.fit_transform(targets)], format="csr")
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            self.classifier.fit(x, labels)

    def transform(self, headlines: list[str], targets: list[str]) -> csr_matrix:
        """Transform with training vocabularies; each block has its own L2 norm."""
        self._validate(headlines, targets)
        check_is_fitted(self.classifier)
        x = self.headline_vectorizer.transform(headlines)
        if self.mode == "headline_target":
            return hstack([x, self.target_vectorizer.transform(targets)], format="csr")
        return csr_matrix(x)

    def predict_many(
        self, headlines: list[str], targets: list[str]
    ) -> list[TfidfPrediction]:
        x = self.transform(headlines, targets)
        classes = self.classifier.classes_.tolist()
        return [
            TfidfPrediction(
                SentimentLabel(classes[int(p.argmax())]),
                {label: float(p[classes.index(label)]) for label in LABELS},
            )
            for p in self.classifier.predict_proba(x)
        ]

    def diagnostics(self, headlines: list[str], targets: list[str]) -> dict:
        self._validate(headlines, targets)
        check_is_fitted(self.classifier)
        return {
            "examples": len(headlines),
            "zero_headline_rows": int(
                (
                    self.headline_vectorizer.transform(headlines).getnnz(axis=1) == 0
                ).sum()
            ),
            "zero_target_rows": int(
                (self.target_vectorizer.transform(targets).getnnz(axis=1) == 0).sum()
            )
            if self.mode == "headline_target"
            else None,
        }

    def metadata(self) -> dict:
        check_is_fitted(self.classifier)
        clf = self.classifier
        headline = len(self.headline_vectorizer.vocabulary_)
        target = (
            len(self.target_vectorizer.vocabulary_)
            if self.mode == "headline_target"
            else 0
        )
        return {
            "mode": self.mode,
            "configuration": asdict(self.config),
            "target_configuration": asdict(self.target_config),
            "dependencies": dependency_versions(),
            "architecture": (
                "sparse word unigram/bigram TF-IDF; "
                "multinomial L2 logistic regression; "
                "additive target block only"
            ),
            "headline_features": headline,
            "target_features": target,
            "features": headline + target,
            "coefficient_shape": list(clf.coef_.shape),
            "intercept_shape": list(clf.intercept_.shape),
            "parameters": int(clf.coef_.size + clf.intercept_.size),
            "coefficient_and_intercept_bytes": clf.coef_.nbytes + clf.intercept_.nbytes,
            "idf_values": headline + target,
            "classes": clf.classes_.tolist(),
            "iterations": clf.n_iter_.tolist(),
            "classifier_parameters": clf.get_params(deep=False),
            "tfidf_policy": {
                "lowercase": True,
                "ngram_range": [1, 2],
                "sublinear_tf": True,
                "smooth_idf": True,
                "norm": "l2 per block",
                "stop_words": None,
                "token_pattern": self.headline_vectorizer.token_pattern,
                "block_scale": 1.0,
                "joint_normalization": False,
            },
            "probabilities_calibrated": False,
        }

    def save(self, path: Path) -> str:
        """Persist fitted vocabulary/weights privately, following existing artifacts."""
        check_is_fitted(self.classifier)
        body = pickle.dumps(
            {"schema_version": 1, "dependencies": dependency_versions(), "model": self},
            protocol=5,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return hashlib.sha256(body).hexdigest()

    @classmethod
    def load(cls, path: Path, *, expected_sha256: str) -> "TargetTfidf":
        """Load trusted local artifacts only, with checksum and dependency checks."""
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != expected_sha256:
            raise ValueError("model checksum mismatch")
        payload = pickle.loads(body)
        if (
            payload["schema_version"] != 1
            or payload["dependencies"] != dependency_versions()
            or not isinstance(payload["model"], cls)
        ):
            raise ValueError("artifact schema or dependencies differ")
        model = payload["model"]
        check_is_fitted(model.classifier)
        return model
