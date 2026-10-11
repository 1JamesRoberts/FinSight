"""Pure training policy checks require neither GPUs nor model downloads."""

import hashlib
import json
from pathlib import Path

import pytest

from finsight.sentiment.finbert import MODEL_ID, REVISION, verify_finetuned_artifact
from finsight.sentiment.finbert_training import (
    EarlyStopping,
    FineTuneConfig,
    sentfin_partitions,
)


@pytest.mark.parametrize(
    "change",
    [
        {"learning_rate": 0},
        {"learning_rate": float("nan")},
        {"train_batch_size": True},
        {"max_epochs": 0},
        {"patience": 0},
        {"max_length": 513},
        {"warmup_ratio": 1},
        {"device": "auto"},
    ],
)
def test_invalid_training_config(change: dict) -> None:
    with pytest.raises(ValueError):
        FineTuneConfig(**change)


def test_stopping_uses_strict_improvement_and_earliest_tie() -> None:
    state = EarlyStopping(patience=2)
    assert state.observe(1, 0.7)
    assert state.observe(2, 0.8)
    assert not state.observe(3, 0.8)
    assert not state.should_stop
    assert not state.observe(4, 0.79)
    assert state.should_stop
    assert state.best_epoch == 2
    assert state.best_score == 0.8
    with pytest.raises(ValueError):
        state.observe(5, float("nan"))


def test_partitions_exclude_other_corpora_and_multiple_targets() -> None:
    rows = [
        {"corpus": "phrasebank", "partition": "train"},
        {"corpus": "fiqa", "partition": "train"},
    ]
    for partition in ("train", "validation"):
        for label in ("positive", "negative", "neutral"):
            rows.append(
                {
                    "corpus": "sentfin",
                    "partition": partition,
                    "record_id": partition + label,
                    "group_id": partition + label,
                    "native_record": {"title": label, "sentiments": [{"label": label}]},
                }
            )
    rows.append(
        {
            "corpus": "sentfin",
            "partition": "train",
            "native_record": {"sentiments": [{}, {}]},
        }
    )
    train, validation = sentfin_partitions(rows)
    assert len(train) == len(validation) == 3
    rows.append({"partition": "test_candidate"})
    with pytest.raises(ValueError, match="development records only"):
        sentfin_partitions(rows)
    rows.pop()
    rows[5]["group_id"] = rows[2]["group_id"]
    with pytest.raises(ValueError, match="overlap"):
        sentfin_partitions(rows)


def test_artifact_checksums_and_safe_paths(tmp_path: Path) -> None:
    names = ("model.safetensors", "config.json", "tokenizer_config.json", "vocab.txt")
    for name in names:
        (tmp_path / name).write_bytes(b"synthetic")
    manifest = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "files": {
            name: {"size_bytes": 9, "sha256": hashlib.sha256(b"synthetic").hexdigest()}
            for name in names
        },
    }
    path = tmp_path / "artifact-manifest.json"
    path.write_bytes(json.dumps(manifest).encode())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert verify_finetuned_artifact(tmp_path, digest) == manifest
    (tmp_path / "model.safetensors").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="file checksum"):
        verify_finetuned_artifact(tmp_path, digest)
    manifest["files"]["../outside"] = {}
    path.write_bytes(json.dumps(manifest).encode())
    # Restore the first file so the unsafe path is reached.
    (tmp_path / "model.safetensors").write_bytes(b"synthetic")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="unsafe"):
        verify_finetuned_artifact(tmp_path, digest)


def test_tiny_training_saves_reloadable_model_without_validation_gradients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from finsight.sentiment import finbert_training as module
    from finsight.sentiment.finbert import CHECKPOINT_FILES, FinBertSentiment
    from finsight.sentiment.validation import classification_metrics

    torch.manual_seed(42)
    torch.set_num_threads(1)
    source = tmp_path / "source"
    source.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "train", "validation"]
    (source / "vocab.txt").write_text("\n".join(vocab) + "\n")
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(source / "vocab.txt"))
    model = transformers.BertForSequenceClassification(
        transformers.BertConfig(
            vocab_size=len(vocab),
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=32,
            num_labels=3,
            id2label={0: "positive", 1: "negative", 2: "neutral"},
            label2id={"positive": 0, "negative": 1, "neutral": 2},
            attn_implementation="eager",
        )
    )
    for name in CHECKPOINT_FILES:
        if not (source / name).exists():
            (source / name).write_bytes(b"source unchanged")
    original = {name: (source / name).read_bytes() for name in CHECKPOINT_FILES}
    initial = model.classifier.weight.detach().clone()
    observations = []

    def observe(model, args, kwargs):
        observations.append(
            (model.training, torch.is_grad_enabled(), kwargs["input_ids"])
        )

    model.register_forward_pre_hook(observe, with_kwargs=True)
    monkeypatch.setattr(
        module, "_load_checkpoint", lambda *_: (torch, tokenizer, model, source)
    )
    rows = []
    labels = ["positive", "negative", "neutral"]
    for partition in ("train", "validation"):
        for label in labels:
            rows.append(
                {
                    "corpus": "sentfin",
                    "partition": partition,
                    "record_id": partition + label,
                    "group_id": partition + label,
                    "native_record": {
                        "title": partition,
                        "sentiments": [{"label": label}],
                    },
                }
            )
    train, validation = sentfin_partitions(rows)
    config = FineTuneConfig(
        device="cpu",
        train_batch_size=2,
        evaluation_batch_size=2,
        max_epochs=2,
        patience=1,
        max_length=16,
        learning_rate=0.01,
        warmup_ratio=0,
    )
    fit = module.fit_checkpoint(config, train, validation, tmp_path)
    assert not torch.equal(initial, model.classifier.weight)
    assert all(p.requires_grad for p in model.parameters())
    for training, gradients, ids in observations:
        assert training == gradients
        assert (5 if training else 6) in ids
        assert (6 if training else 5) not in ids
    assert original == {name: (source / name).read_bytes() for name in CHECKPOINT_FILES}
    loaded = FinBertSentiment.from_artifact(
        config.inference_config(),
        fit["artifact_directory"],
        fit["artifact_manifest_sha256"],
    )
    assert not loaded._model.training
    assert not any(p.requires_grad for p in loaded._model.parameters())
    predictions = loaded.predict_batch(["validation"] * 3)
    scores = classification_metrics(labels, [p.label.value for p in predictions])
    assert scores == fit["history"][fit["best_epoch"] - 1]["validation"]


def test_training_rejects_reserved_records_before_loading(tmp_path: Path) -> None:
    from finsight.sentiment.finbert_training import fit_checkpoint

    with pytest.raises(ValueError, match="development records only"):
        fit_checkpoint(FineTuneConfig(), [{"partition": "test"}], [], tmp_path)
