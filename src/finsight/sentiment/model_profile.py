"""Aggregate architecture and exact local artifact size for fitted linear models."""

import argparse
import hashlib
import json
import subprocess
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


def enrich_report(report_path: Path, data_dir: Path) -> dict[str, Any]:
    """Append profiles to an existing report; preserve the original local run."""
    from finsight.sentiment.svm import TfidfSVM
    from finsight.sentiment.tfidf import TfidfSentiment
    from finsight.sentiment.tfidf_validation import render_report

    aggregate_path = report_path.with_suffix(".json")
    published = json.loads(aggregate_path.read_text(encoding="utf-8"))
    original_id = published.get("report_enrichment", {}).get("original_run_id")
    if original_id is None:
        body = json.dumps(published, indent=2, sort_keys=True) + "\n"
        original_id = hashlib.sha256(body.encode()).hexdigest()
    prefix = published["experiment"].removesuffix("-validation-v1")
    original_path = data_dir / f"interim/{prefix}-validation" / original_id / "run.json"
    original_body = original_path.read_bytes()
    if hashlib.sha256(original_body).hexdigest() != original_id:
        raise ValueError("original run checksum mismatch")
    run = json.loads(original_body)
    kind = run.get("model_kind", "logistic_regression")
    model_class = TfidfSentiment if kind == "logistic_regression" else TfidfSVM
    for result in run["metrics"].values():
        artifact = data_dir / result["artifact_relative_path"]
        model = model_class.load(artifact, expected_sha256=result["artifact_sha256"])
        result["model_profile"] = profile_linear_pipeline(
            model.pipeline, artifact, result["artifact_sha256"]
        )
    root = Path(__file__).resolve().parents[3]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    run["report_enrichment"] = {
        "original_run_id": original_id,
        "code_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "working_tree_dirty": bool(dirty.stdout) if dirty.returncode == 0 else None,
        "implementation_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("model_profile.py", "tfidf_validation.py")
        },
        "scope": "read original fitted artifacts only; no fitting or prediction",
    }
    report_path.write_text(render_report(run, original_id), encoding="utf-8")
    aggregate_path.write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run


def main() -> None:
    from finsight.config import Settings

    parser = argparse.ArgumentParser(description="Add profiles without retraining.")
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--data-dir", type=Path, default=Settings.from_env().data_dir)
    args = parser.parse_args()
    for report in args.reports:
        enrich_report(report, args.data_dir)
        print(f"Model architecture/size added to {report}")


if __name__ == "__main__":
    main()
