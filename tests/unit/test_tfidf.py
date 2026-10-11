"""Train-only features, fixed class probabilities, and local artifact behavior."""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning, NotFittedError

from finsight.sentiment.tfidf import TfidfConfig, TfidfSentiment


def fitted() -> TfidfSentiment:
    model = TfidfSentiment(TfidfConfig(min_df=1))
    model.fit(
        ["profits rise", "profits grow", "losses rise", "losses grow", "meeting today"],
        ["positive", "positive", "negative", "negative", "neutral"],
    )
    return model


def test_validation_does_not_change_vocabulary_or_idf() -> None:
    model = fitted()
    vectorizer = model.pipeline.named_steps["tfidf"]
    vocabulary = dict(vectorizer.vocabulary_)
    idf = vectorizer.idf_.copy()
    # Five training documents, two containing "profits"; smoothed training IDF.
    assert idf[vocabulary["profits"]] == pytest.approx(np.log(6 / 3) + 1)
    result = model.predict("validationcanary profits rise")
    assert "validationcanary" not in vectorizer.vocabulary_
    assert vectorizer.vocabulary_ == vocabulary
    np.testing.assert_array_equal(vectorizer.idf_, idf)
    assert list(result.probabilities) == ["negative", "neutral", "positive"]
    assert sum(result.probabilities.values()) == pytest.approx(1)
    assert result.label.value == max(result.probabilities, key=result.probabilities.get)
    # Unknown tokens are valid; an all-zero vector uses learned intercepts.
    assert sum(model.predict("unseenword").probabilities.values()) == pytest.approx(1)


def test_word_features_keep_negation_and_bigrams() -> None:
    model = TfidfSentiment(TfidfConfig(min_df=1))
    model.fit(
        ["NOT good!", "good", "scheduled meeting"],
        list(reversed(["neutral", "positive", "negative"])),
    )
    vocab = model.pipeline.named_steps["tfidf"].vocabulary_
    assert {"not", "not good", "good"} <= vocab.keys()
    assert "NOT" not in vocab


def test_deterministic_training_and_artifact_roundtrip(tmp_path: Path) -> None:
    first, second = fitted(), fitted()
    text = "profits grow"
    assert first.predict(text) == second.predict(text)
    path = tmp_path / "model.pkl"
    digest = first.save(path)
    loaded = TfidfSentiment.load(path, expected_sha256=digest)
    assert loaded.predict(text) == first.predict(text)
    assert loaded.metadata() == first.metadata()
    with pytest.raises(ValueError, match="checksum"):
        TfidfSentiment.load(path, expected_sha256="incorrect")


@pytest.mark.parametrize(
    "texts,labels",
    [
        ([], []),
        (["good"], []),
        (["good"], ["unknown"]),
        (["good"], ["positive"]),
        (["", "bad"], ["positive", "negative"]),
    ],
)
def test_invalid_training(texts: list[str], labels: list[str]) -> None:
    with pytest.raises(ValueError):
        TfidfSentiment().fit(texts, labels)


def test_unfitted_and_blank_prediction_fail() -> None:
    with pytest.raises(NotFittedError):
        TfidfSentiment().predict("good")
    with pytest.raises(ValueError):
        fitted().predict("  ")


def test_document_frequency_floor_uses_training_only() -> None:
    model = TfidfSentiment()
    model.fit(
        ["rarecanary profits", "profits grow", "meeting today", "meeting next"],
        ["negative", "positive", "neutral", "neutral"],
    )
    vocab = model.pipeline.named_steps["tfidf"].vocabulary_
    assert set(vocab) == {"profits", "meeting"}
    model.predict_many(["rarecanary"] * 3)
    assert set(vocab) == {"profits", "meeting"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"c": 0},
        {"c": float("nan")},
        {"c": True},
        {"min_df": 0},
        {"min_df": True},
        {"max_features": -1},
        {"max_iter": 0},
        {"seed": True},
    ],
)
def test_invalid_config(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        TfidfConfig(**kwargs)


def test_config_load_is_strict(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"schema_version": 1, "c": 2, "seed": 42}))
    assert TfidfConfig.load(path).c == 2
    for body in ({"schema_version": 2}, {"schema_version": 1, "unknown": 4}):
        path.write_text(json.dumps(body))
        with pytest.raises(ValueError):
            TfidfConfig.load(path)


def test_nonconverged_fit_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    model = TfidfSentiment()

    def unfinished_fit(*args: object) -> None:
        warnings.warn(
            "synthetic optimizer did not converge", ConvergenceWarning, stacklevel=2
        )

    monkeypatch.setattr(model.pipeline, "fit", unfinished_fit)
    with pytest.raises(ConvergenceWarning):
        model.fit(["losses", "meeting", "profits"], ["negative", "neutral", "positive"])
