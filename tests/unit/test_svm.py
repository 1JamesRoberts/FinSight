"""Linear margins, train-only features, weighting, and local SVM artifacts."""

import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning, NotFittedError

from finsight.sentiment.svm import TfidfSVM
from finsight.sentiment.tfidf import TfidfConfig, TfidfSentiment


def train(weight: str | None = None) -> TfidfSVM:
    model = TfidfSVM(TfidfConfig(min_df=1, max_iter=10000, class_weight=weight))
    model.fit(
        ["profits rise", "losses rise", "not good"] + ["meeting scheduled"] * 6,
        ["positive", "negative", "negative"] + ["neutral"] * 6,
    )
    return model


def test_scores_are_fixed_class_margins_and_features_do_not_refit() -> None:
    model = train()
    vectorizer = model.pipeline.named_steps["tfidf"]
    vocab, idf = dict(vectorizer.vocabulary_), vectorizer.idf_.copy()
    assert {"not", "not good"} <= vocab.keys()
    result = model.predict("validationcanary profits rise")
    assert list(result.decision_scores) == ["negative", "neutral", "positive"]
    assert result.label.value == max(
        result.decision_scores, key=result.decision_scores.get
    )
    scores = model.pipeline.decision_function(["validationcanary profits rise"])[0]
    for label, score in zip(model.pipeline.classes_, scores, strict=True):
        assert result.decision_scores[label] == score
    assert not hasattr(result, "probabilities")
    assert vectorizer.vocabulary_ == vocab
    assert "validationcanary" not in vocab
    np.testing.assert_array_equal(vectorizer.idf_, idf)
    assert set(model.predict("unseenword").decision_scores) == {
        "negative",
        "neutral",
        "positive",
    }


def test_weighted_fit_changes_coefficients_and_matches_lr_features() -> None:
    plain, balanced = train(), train("balanced")
    assert (
        plain.pipeline.named_steps["tfidf"].vocabulary_
        == balanced.pipeline.named_steps["tfidf"].vocabulary_
    )
    np.testing.assert_array_equal(
        plain.pipeline.named_steps["tfidf"].idf_,
        balanced.pipeline.named_steps["tfidf"].idf_,
    )
    assert not np.allclose(
        plain.pipeline.named_steps["classifier"].coef_,
        balanced.pipeline.named_steps["classifier"].coef_,
    )
    lr = TfidfSentiment(TfidfConfig(min_df=1))
    lr.fit(
        ["profits rise", "losses rise", "not good"] + ["meeting scheduled"] * 6,
        ["positive", "negative", "negative"] + ["neutral"] * 6,
    )
    assert (
        lr.pipeline.named_steps["tfidf"].vocabulary_
        == plain.pipeline.named_steps["tfidf"].vocabulary_
    )
    np.testing.assert_array_equal(
        lr.pipeline.named_steps["tfidf"].idf_, plain.pipeline.named_steps["tfidf"].idf_
    )


@pytest.mark.parametrize("weight", [None, "balanced"])
def test_repeatability_and_artifact_roundtrip(
    tmp_path: Path, weight: str | None
) -> None:
    first, second = train(weight), train(weight)
    assert first.predict("profits rise") == second.predict("profits rise")
    path = tmp_path / "svm.pkl"
    digest = first.save(path)
    loaded = TfidfSVM.load(path, expected_sha256=digest)
    assert loaded.predict("profits rise") == first.predict("profits rise")
    assert loaded.metadata() == first.metadata()
    assert loaded.metadata()["classifier_parameters"]["class_weight"] == weight
    assert loaded.metadata()["decision_scores_are_probabilities"] is False
    with pytest.raises(ValueError, match="checksum"):
        TfidfSVM.load(path, expected_sha256="incorrect")


@pytest.mark.parametrize(
    "texts,labels",
    [
        ([], []),
        (["good"], ["positive"]),
        (["good", "bad", "meeting"], ["positive", "negative", "unknown"]),
        (["good", "bad", ""], ["positive", "negative", "neutral"]),
        (["good", "bad", "meeting"], ["positive", "negative"]),
    ],
)
def test_invalid_training(texts: list[str], labels: list[str]) -> None:
    with pytest.raises(ValueError):
        TfidfSVM().fit(texts, labels)


def test_unfitted_and_empty_prediction_fail() -> None:
    with pytest.raises(NotFittedError):
        TfidfSVM().predict("good")
    with pytest.raises(ValueError):
        train().predict(" ")


def test_nonconverged_fit_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    model = TfidfSVM()

    def unfinished(*args: object) -> None:
        warnings.warn("synthetic unfinished SVM", ConvergenceWarning, stacklevel=2)

    monkeypatch.setattr(model.pipeline, "fit", unfinished)
    with pytest.raises(ConvergenceWarning):
        model.fit(["profits", "losses", "meeting"], ["positive", "negative", "neutral"])


def test_wrong_model_artifact_is_rejected(tmp_path: Path) -> None:
    lr = TfidfSentiment(TfidfConfig(min_df=1))
    lr.fit(["profits", "losses", "meeting"], ["positive", "negative", "neutral"])
    path = tmp_path / "logreg.pkl"
    digest = lr.save(path)
    with pytest.raises(ValueError, match="artifact"):
        TfidfSVM.load(path, expected_sha256=digest)
