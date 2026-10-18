"""Linear SVM sentence inference with signed decision scores, not probabilities."""

import hashlib
import pickle
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted

from finsight.data.loaders import SentimentLabel
from finsight.sentiment.models.tfidf import (
    LABELS,
    TfidfConfig,
    build_vectorizer,
    dependency_versions,
    validate_texts,
)


@dataclass(frozen=True)
class SVMPrediction:
    label: SentimentLabel
    decision_scores: dict[str, float]


class TfidfSVM:
    """Shared TF-IDF features followed by three one-versus-rest linear SVMs."""

    def __init__(self, config: TfidfConfig | None = None) -> None:
        self.config = config or TfidfConfig(max_iter=10000)
        self.pipeline = Pipeline(
            [
                ("tfidf", build_vectorizer(self.config)),
                (
                    "classifier",
                    LinearSVC(
                        C=self.config.c,
                        penalty="l2",
                        loss="squared_hinge",
                        dual=True,
                        tol=1e-4,
                        max_iter=self.config.max_iter,
                        random_state=self.config.seed,
                        class_weight=self.config.class_weight,
                        multi_class="ovr",
                        fit_intercept=True,
                        intercept_scaling=1.0,
                    ),
                ),
            ]
        )

    def fit(self, texts: list[str], labels: list[str]) -> None:
        """Fit vocabulary, IDF, and margins together on training examples only."""
        validate_texts(texts)
        if len(texts) != len(labels) or set(labels) != set(LABELS):
            raise ValueError("training requires matching texts and all three labels")
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            # Fit vocabulary/IDF on training texts before fitting the classifier.
            self.pipeline.fit(texts, labels)

    def predict_many(self, texts: list[str]) -> list[SVMPrediction]:
        validate_texts(texts)
        check_is_fitted(self.pipeline.named_steps["classifier"])
        classes = self.pipeline.classes_.tolist()
        scores = self.pipeline.decision_function(texts)
        return [
            SVMPrediction(
                label=SentimentLabel(classes[int(row.argmax())]),
                decision_scores={
                    label: float(row[classes.index(label)]) for label in LABELS
                },
            )
            for row in scores
        ]

    def predict(self, text: str) -> SVMPrediction:
        return self.predict_many([text])[0]

    def metadata(self) -> dict[str, Any]:
        check_is_fitted(self.pipeline.named_steps["classifier"])
        vectorizer = self.pipeline.named_steps["tfidf"]
        classifier = self.pipeline.named_steps["classifier"]
        parameters = vectorizer.get_params(deep=False)
        parameters["dtype"] = str(parameters["dtype"])
        return {
            "config": asdict(self.config),
            "dependencies": dependency_versions(),
            "tfidf_parameters": parameters,
            "classifier_parameters": classifier.get_params(deep=False),
            "classes": classifier.classes_.tolist(),
            "vocabulary_size": len(vectorizer.vocabulary_),
            "iterations": int(classifier.n_iter_),
            "decision_scores_are_probabilities": False,
        }

    def save(self, path: Path) -> str:
        """Persist to reviewed local storage; no public artifact rights inferred."""
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
    def load(cls, path: Path, *, expected_sha256: str) -> "TfidfSVM":
        """Load trusted local pickle files; a digest does not establish trust."""
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
