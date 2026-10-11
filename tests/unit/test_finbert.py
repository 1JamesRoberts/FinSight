"""Check pinned configuration and class mapping without downloading weights."""

import json
from pathlib import Path

import pytest

from finsight.sentiment.finbert import FinBertConfig, probability_prediction


@pytest.mark.parametrize(
    "values,label",
    [
        ([0.8, 0.1, 0.1], "positive"),
        ([0.1, 0.8, 0.1], "negative"),
        ([0.1, 0.1, 0.8], "neutral"),
        ([0.5, 0.5, 0.0], "positive"),
    ],
)
def test_checkpoint_order_is_not_alphabetical(values: list[float], label: str) -> None:
    result = probability_prediction(values, 9, 128)
    assert result.label == label
    assert result.probabilities == dict(
        zip(("positive", "negative", "neutral"), values, strict=True)
    )
    assert not result.truncated
    assert probability_prediction(values, 129, 128).truncated


@pytest.mark.parametrize(
    "values", [[0.5, 0.5], [1, 1, 1], [-1, 1, 1], [float("nan"), 0, 1]]
)
def test_invalid_probabilities_fail(values: list[float]) -> None:
    with pytest.raises(ValueError, match="probabilities"):
        probability_prediction(values, 10, 128)


@pytest.mark.parametrize(
    "change",
    [
        {"revision": "main"},
        {"model_id": "different/model"},
        {"batch_size": 0},
        {"batch_size": True},
        {"max_length": 513},
        {"max_length": 1},
        {"device": "auto"},
        {"seed": True},
    ],
)
def test_invalid_config_fails(change: dict) -> None:
    with pytest.raises(ValueError):
        FinBertConfig(**change)


def test_config_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"schema_version": 1, "fine_tune": True}))
    with pytest.raises(ValueError, match="configuration"):
        FinBertConfig.load(path)


def test_batches_preserve_order_and_disable_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import contextmanager
    from types import SimpleNamespace

    import numpy as np

    from finsight.sentiment.finbert import FinBertSentiment

    calls = []
    state = {"inference": False, "eval": False, "gradients": True}

    class Inputs(dict):
        def to(self, device: str) -> "Inputs":
            assert device == "cpu"
            return self

    class Tokenizer:
        def __call__(self, texts: list[str], **kwargs: object) -> dict:
            calls.append((texts, kwargs))
            if kwargs.get("return_tensors"):
                assert kwargs["padding"] is True
                assert kwargs["truncation"] is True
                assert kwargs["max_length"] == 4
                return Inputs(texts=texts)
            return {"input_ids": [[0] * (len(text.split()) + 2) for text in texts]}

    class Model:
        config = SimpleNamespace(id2label={0: "positive", 1: "negative", 2: "neutral"})

        def to(self, **kwargs: object) -> None:
            assert kwargs == {"device": "cpu", "dtype": "float32"}

        def eval(self) -> None:
            state["eval"] = True

        def requires_grad_(self, value: bool) -> None:
            state["gradients"] = value

        def __call__(self, texts: list[str]) -> SimpleNamespace:
            assert state == {"inference": True, "eval": True, "gradients": False}
            index = {"positive": 0, "negative": 1, "neutral": 2}
            logits = [
                [5 if i == index[text.split()[0]] else 0 for i in range(3)]
                for text in texts
            ]
            return SimpleNamespace(logits=logits)

    @contextmanager
    def inference_mode():
        state["inference"] = True
        try:
            yield
        finally:
            state["inference"] = False

    def softmax(logits: list[list[float]], dim: int) -> SimpleNamespace:
        assert dim == -1
        exponent = np.exp(logits)
        values = exponent / exponent.sum(axis=1, keepdims=True)
        return SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=values.tolist))

    torch = SimpleNamespace(
        float32="float32", inference_mode=inference_mode, softmax=softmax
    )
    monkeypatch.setattr(
        "finsight.sentiment.finbert._load_checkpoint",
        lambda *args: (torch, Tokenizer(), Model(), tmp_path),
    )
    model = FinBertSentiment(
        FinBertConfig(device="cpu", batch_size=2, max_length=4), tmp_path
    )
    results = model.predict_batch(
        ["positive first", "negative many extra words", "neutral"]
    )
    assert [item.label for item in results] == ["positive", "negative", "neutral"]
    assert [item.token_count for item in results] == [4, 6, 3]
    assert [item.truncated for item in results] == [False, True, False]
    assert [
        len(texts) for texts, options in calls if options.get("return_tensors")
    ] == [2, 1]
    assert model.predict_batch([]) == []
    assert len(calls) == 4
    for texts in (["valid", ""], [123], "a string"):
        with pytest.raises(ValueError, match="nonempty strings"):
            model.predict_batch(texts)
    assert len(calls) == 4


def test_wrong_checkpoint_label_mapping_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from finsight.sentiment.finbert import FinBertSentiment

    model = SimpleNamespace(
        config=SimpleNamespace(id2label={0: "negative", 1: "neutral", 2: "positive"})
    )
    monkeypatch.setattr(
        "finsight.sentiment.finbert._load_checkpoint",
        lambda *args: (None, None, model, tmp_path),
    )
    with pytest.raises(ValueError, match="label mapping"):
        FinBertSentiment(FinBertConfig(device="cpu"), tmp_path)


def test_download_is_pinned_and_loading_rejects_missing_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock

    from finsight.sentiment.finbert import CHECKPOINT_FILES, _load_checkpoint

    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        manual_seed=Mock(),
        use_deterministic_algorithms=Mock(),
        set_float32_matmul_precision=Mock(),
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace()), cudnn=SimpleNamespace()
        ),
    )
    download = Mock(return_value=str(tmp_path))
    tokenizer = Mock()
    load = Mock(return_value=(object(), {"missing_keys": ["classifier.weight"]}))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download)
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer),
            AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=load),
        ),
    )
    monkeypatch.setattr("finsight.sentiment.finbert.CHECKPOINT_HASHES", {})
    config = FinBertConfig()
    with pytest.raises(ValueError, match="load exactly"):
        _load_checkpoint(config, tmp_path)
    download.assert_called_once_with(
        repo_id=config.model_id,
        revision=config.revision,
        cache_dir=str(tmp_path),
        allow_patterns=list(CHECKPOINT_FILES),
    )
    assert load.call_args.kwargs["weights_only"] is True
    assert load.call_args.kwargs["local_files_only"] is True
    assert load.call_args.kwargs["trust_remote_code"] is False
    torch.use_deterministic_algorithms.assert_called_once_with(True)
    torch.cuda.is_available = lambda: False
    with pytest.raises(RuntimeError, match="CUDA requested"):
        _load_checkpoint(config, tmp_path)
    assert download.call_count == 1
