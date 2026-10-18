"""Aggregate architecture and exact local artifact size for fitted linear models."""

import hashlib
from pathlib import Path
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted


def profile_linear_pipeline(
    pipeline: Pipeline,
    artifact_path: Path,
    expected_sha256: str,
) -> dict[str, Any]:
    """Read fitted arrays only; never fit, predict, or expose vocabulary strings."""
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    check_is_fitted(classifier)
    check_is_fitted(vectorizer)
    if not isinstance(classifier, (LogisticRegression, LinearSVC)):
        raise ValueError("unsupported linear classifier profile")
    body = artifact_path.read_bytes()
    if hashlib.sha256(body).hexdigest() != expected_sha256:
        raise ValueError("model checksum mismatch")
    logistic = isinstance(classifier, LogisticRegression)
    classifier_bytes = int(classifier.coef_.nbytes + classifier.intercept_.nbytes)
    idf_bytes = int(vectorizer.idf_.nbytes)
    return {
        "schema_version": 1,
        "pipeline": "TF-IDF word unigrams/bigrams -> " + type(classifier).__name__,
        "classifier": type(classifier).__name__,
        "multiclass_strategy": "multinomial" if logistic else "one-versus-rest",
        "input_features": len(vectorizer.vocabulary_),
        "output_classes": classifier.classes_.tolist(),
        "output_kind": "probabilities" if logistic else "decision_scores",
        "coefficient_shape": list(classifier.coef_.shape),
        "intercept_shape": list(classifier.intercept_.shape),
        "coefficient_values": int(classifier.coef_.size),
        "intercept_values": int(classifier.intercept_.size),
        "classifier_parameters": int(
            classifier.coef_.size + classifier.intercept_.size
        ),
        "tfidf_idf_values": int(vectorizer.idf_.size),
        "coefficient_dtype": str(classifier.coef_.dtype),
        "idf_dtype": str(vectorizer.idf_.dtype),
        "classifier_array_bytes": classifier_bytes,
        "idf_array_bytes": idf_bytes,
        "numeric_array_bytes": classifier_bytes + idf_bytes,
        "artifact_bytes": len(body),
        "artifact_kib": len(body) / 1024,
        "size_definition": (
            "Classifier parameters count stored coefficients plus intercepts; "
            "IDF values are separate fitted preprocessing state. Numeric bytes cover "
            "these arrays only. Artifact bytes cover the entire serialized pipeline, "
            "including vocabulary/metadata/serialization overhead; "
            "neither is runtime RAM."
        ),
    }
