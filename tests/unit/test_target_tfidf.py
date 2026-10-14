"""Train-only vocabulary fitting and additive target-feature behavior."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.exceptions import NotFittedError

from finsight.data.target_pairs import TargetExample
from finsight.sentiment.target_tfidf import TargetTfidf
from finsight.sentiment.tfidf import TfidfConfig


def training() -> tuple[TargetExample, ...]:
    return tuple(
        TargetExample(
            str(i),
            str(i),
            str(i),
            "train",
            0,
            target,
            headline,
            label,
            False,
            False,
            False,
            (),
        )
        for i, (target, headline, label) in enumerate(
            [
                ("Alpha", "Alpha reports strong gains", "positive"),
                ("Beta", "Beta reports strong gains", "positive"),
                ("Alpha", "Alpha reports severe losses", "negative"),
                ("Beta", "Beta reports severe losses", "negative"),
                ("Alpha", "Alpha reports unchanged results", "neutral"),
                ("Beta", "Beta reports unchanged results", "neutral"),
            ]
        )
    )


def model(mode: str = "headline_target") -> TargetTfidf:
    return TargetTfidf(mode, TfidfConfig(min_df=1))


def test_inference_does_not_fit_vocabulary_or_idf() -> None:
    m = model()
    m.fit(training())
    headline_vocab = dict(m.headline_vectorizer.vocabulary_)
    target_vocab = dict(m.target_vectorizer.vocabulary_)
    idf = m.headline_vectorizer.idf_.copy()
    p = m.predict_many(["validationexclusive words"], ["unseentarget"])
    assert set(p[0].probabilities) == {"negative", "neutral", "positive"}
    assert m.headline_vectorizer.vocabulary_ == headline_vocab
    assert m.target_vectorizer.vocabulary_ == target_vocab
    np.testing.assert_array_equal(idf, m.headline_vectorizer.idf_)
    assert (
        m.diagnostics(["validationexclusive"], ["unseentarget"])["zero_target_rows"]
        == 1
    )


def test_target_changes_only_its_separate_sparse_feature_block() -> None:
    m = model()
    m.fit(training())
    x = m.transform(["Alpha gains while Beta falls"] * 2, ["Alpha", "Beta"])
    n = len(m.headline_vectorizer.vocabulary_)
    assert issparse(x)
    np.testing.assert_array_equal(x[0, :n].toarray(), x[1, :n].toarray())
    assert (x[0, n:] != x[1, n:]).nnz > 0
    np.testing.assert_allclose(np.asarray(x.multiply(x).sum(axis=1)).ravel(), [2, 2])


def test_headline_arm_ignores_target_and_matches_existing_classifier() -> None:
    from finsight.sentiment.tfidf import TfidfSentiment

    m = model("headline_only")
    rows = training()
    m.fit(rows)
    old = TfidfSentiment(m.config)
    old.fit([e.headline for e in rows], [e.label for e in rows])
    outputs = m.predict_many(["Alpha strong gains"] * 2, ["Alpha", "Beta"])
    assert outputs[0] == outputs[1] == old.predict("Alpha strong gains")
    assert m.metadata()["target_features"] == 0


@pytest.mark.parametrize("partition", ["validation", "test"])
def test_fit_rejects_nontraining_examples(partition: str) -> None:
    rows = training()
    with pytest.raises(ValueError, match="training"):
        model().fit((replace(rows[0], partition=partition), *rows[1:]))


def test_fit_rejects_duplicate_annotation_ids_and_missing_class() -> None:
    with pytest.raises(ValueError):
        model().fit((*training(), training()[0]))
    with pytest.raises(ValueError):
        model().fit(training()[:2])


def test_unfitted_and_invalid_input_rejected() -> None:
    with pytest.raises(NotFittedError):
        model().predict_many(["News"], ["Alpha"])
    with pytest.raises(ValueError):
        model("unsupported")
    m = model()
    m.fit(training())
    for texts, targets in [
        ([], []),
        (["News"], []),
        ([""], ["Alpha"]),
        (["News"], [""]),
    ]:
        with pytest.raises(ValueError):
            m.predict_many(texts, targets)


def test_save_reload_and_model_size(tmp_path) -> None:
    m = model()
    m.fit(training())
    path = tmp_path / "model.pkl"
    digest = m.save(path)
    loaded = TargetTfidf.load(path, expected_sha256=digest)
    assert loaded.predict_many(["Alpha gains"], ["Alpha"]) == m.predict_many(
        ["Alpha gains"], ["Alpha"]
    )
    info = loaded.metadata()
    assert info["coefficient_shape"] == [3, info["features"]]
    assert info["parameters"] == 3 * info["features"] + 3
    with pytest.raises(ValueError, match="checksum"):
        TargetTfidf.load(path, expected_sha256="0" * 64)


def test_experiment_saves_models_and_predictions_without_test_input(
    tmp_path, monkeypatch
) -> None:
    import json
    from pathlib import Path

    from finsight.sentiment import target_tfidf_experiment as experiment

    train = training()
    validation = (
        replace(
            train[0],
            example_id="va",
            record_id="multi",
            group_id="vg0",
            partition="validation",
            multi_target=True,
            mixed_labels=True,
        ),
        replace(
            train[2],
            example_id="vb",
            record_id="multi",
            group_id="vg0",
            partition="validation",
            headline=train[0].headline,
            target="Beta",
            multi_target=True,
            mixed_labels=True,
        ),
        replace(
            train[4],
            example_id="vc",
            record_id="single",
            group_id="vg1",
            partition="validation",
            target="UnknownEntity",
        ),
    )
    monkeypatch.setattr(
        experiment, "verified_inputs", lambda _: ({}, {}, train + validation, [])
    )
    monkeypatch.setattr(experiment, "REPORT", tmp_path / "public" / "comparison")
    result = experiment.run(tmp_path / "data")
    assert result["final_test_evaluated"] is False
    assert result["training_examples"] == 6
    assert result["matched_headline_features_identical"] is True
    for mode, arm in result["models"].items():
        assert arm["cohorts"]["all_targets"]["examples"] == 3
        artifact = tmp_path / "data" / arm["artifact_relative_path"]
        assert artifact.is_file()
        loaded = TargetTfidf.load(artifact, expected_sha256=arm["artifact_sha256"])
        assert loaded.mode == mode
    saved = json.loads((tmp_path / "public/comparison.json").read_bytes())
    assert (
        experiment.render_report(saved)
        == (tmp_path / "public/comparison.md").read_text()
    )
    with pytest.raises(FileExistsError):
        experiment.run(Path("nonexistent-data"))
