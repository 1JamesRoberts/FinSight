"""Model size means stored fitted values and bytes, not runtime process memory."""

import hashlib
import json
from pathlib import Path

import pytest
from sklearn.exceptions import NotFittedError

from finsight.sentiment.model_profile import enrich_report, profile_linear_pipeline
from finsight.sentiment.svm import TfidfSVM
from finsight.sentiment.tfidf import TfidfConfig, TfidfSentiment


@pytest.mark.parametrize("kind", ["logreg", "svm"])
def test_profile_matches_fitted_arrays_and_exact_artifact(
    tmp_path: Path, kind: str
) -> None:
    cls = TfidfSentiment if kind == "logreg" else TfidfSVM
    model = cls(TfidfConfig(min_df=1))
    model.fit(["profits", "losses", "meeting"], ["positive", "negative", "neutral"])
    path = tmp_path / "model.pkl"
    digest = model.save(path)
    profile = profile_linear_pipeline(model.pipeline, path, digest)
    assert profile["input_features"] == 3
    assert profile["output_classes"] == ["negative", "neutral", "positive"]
    assert profile["coefficient_shape"] == [3, 3]
    assert profile["intercept_shape"] == [3]
    assert profile["classifier_parameters"] == 12
    assert profile["tfidf_idf_values"] == 3
    assert profile["classifier_array_bytes"] == 12 * 8
    assert profile["idf_array_bytes"] == 3 * 8
    assert profile["numeric_array_bytes"] == 15 * 8
    assert profile["artifact_bytes"] == len(path.read_bytes())
    assert profile["artifact_kib"] == profile["artifact_bytes"] / 1024
    assert profile["numeric_array_bytes"] < profile["artifact_bytes"]
    assert profile["classifier"] == (
        "LogisticRegression" if kind == "logreg" else "LinearSVC"
    )
    assert profile["output_kind"] == (
        "probabilities" if kind == "logreg" else "decision_scores"
    )
    assert "profits" not in str(profile)
    assert "runtime_memory_bytes" not in profile


def test_profile_rejects_unfitted_model_and_changed_artifact(tmp_path: Path) -> None:
    path = tmp_path / "model.pkl"
    with pytest.raises(NotFittedError):
        profile_linear_pipeline(TfidfSVM().pipeline, path, "unused")
    model = TfidfSVM(TfidfConfig(min_df=1))
    model.fit(["profits", "losses", "meeting"], ["positive", "negative", "neutral"])
    digest = model.save(path)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        profile_linear_pipeline(model.pipeline, path, digest)


def test_enrichment_preserves_run_and_never_trains_or_predicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from finsight.sentiment.tfidf_validation import run_tfidf_validation

    records = [
        {
            "corpus": "phrasebank",
            "record_id": str(i),
            "partition": partition,
            "native_record": {"sentence": text, "label": label},
        }
        for i, (partition, text, label) in enumerate(
            [
                ("train", "profits", "positive"),
                ("train", "losses", "negative"),
                ("train", "meeting", "neutral"),
                ("validation", "profits rise", "positive"),
            ]
        )
    ]
    monkeypatch.setattr(
        "finsight.sentiment.tfidf_validation.load_development_records",
        lambda *args: records,
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "benchmark_manifest_sha256": "synthetic-freeze",
                "parent_split_manifest_sha256": "synthetic-parent",
            }
        )
    )
    config = tmp_path / "config.json"
    config.write_text('{"schema_version": 1, "min_df": 1}')
    data, report = tmp_path / "data", tmp_path / "report.md"
    run = run_tfidf_validation(data, receipt, config, report, cohort="phrasebank")
    original_body = report.with_suffix(".json").read_bytes()
    original_id = hashlib.sha256(original_body).hexdigest()
    local = data / "interim/tfidf-logreg-validation" / original_id / "run.json"
    assert local.read_bytes() == original_body
    artifact = data / run["metrics"]["phrasebank"]["artifact_relative_path"]
    original_artifact = artifact.read_bytes()

    def forbidden(*args: object) -> None:
        pytest.fail("profile extraction must not load dataset, fit, or predict")

    monkeypatch.setattr(TfidfSentiment, "fit", forbidden)
    monkeypatch.setattr(TfidfSentiment, "predict", forbidden)
    monkeypatch.setattr(
        "finsight.sentiment.tfidf_validation.load_development_records",
        forbidden,
    )
    updated = enrich_report(report, data)
    repeated = enrich_report(report, data)
    # JSON normalizes parameter tuples to lists; compare the persisted run form.
    assert (
        updated["metrics"]
        == json.loads(original_body)["metrics"]
        == repeated["metrics"]
    )
    assert repeated["report_enrichment"]["original_run_id"] == original_id
    assert local.read_bytes() == original_body
    assert artifact.read_bytes() == original_artifact
    assert "Model architecture and size" in report.read_text()
    assert "Serialized pipeline bytes" in report.read_text()
