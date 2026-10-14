"""Actual tiny custom-model updates, frozen head and immutable reload."""

import json
from dataclasses import replace

import pytest

from finsight.sentiment.finbert import file_sha256
from finsight.sentiment.finbert_training import FineTuneConfig, fit_checkpoint
from finsight.sentiment.proprak_deberta import build_model
from finsight.sentiment.proprak_training import FROZEN, load_plan, reload_model
from finsight.sentiment.transformer_comparison import save_model, score_model
from finsight.sentiment.validation import LABELS, classification_metrics


def test_fixed_plan_and_template_cannot_drift(tmp_path) -> None:
    plan, config = load_plan()
    assert config.learning_rate == 1e-5
    assert plan["seeds"] == [42, 43, 44]
    assert config.max_epochs == 5
    plan["seeds"] = [42]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="plan"):
        load_plan(path)


@pytest.mark.filterwarnings(
    "ignore:.*torch.jit.script.*deprecated:DeprecationWarning:torch.jit._script"
)
def test_training_freezes_regression_updates_encoder_and_reloads(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    t = pytest.importorskip("transformers")
    from finsight.sentiment.finbert_training import membership_digest

    torch.manual_seed(42)
    torch.set_num_threads(1)
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    vocab = source / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ntrain\nvalidation\n")
    tokenizer = t.BertTokenizerFast(vocab_file=str(vocab))
    c = t.DebertaV2Config(
        vocab_size=7,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=32,
        type_vocab_size=0,
        attn_implementation="eager",
    )
    model = build_model(c)
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    asset = {
        "model_id": "synthetic",
        "revision": "test",
        "files": {
            "config.json": {
                "sha256": file_sha256(source / "config.json"),
                "size_bytes": 2,
            }
        },
    }
    rows = [
        {
            "corpus": "sentfin",
            "partition": part,
            "record_id": part + label,
            "group_id": part + label,
            "native_record": {"title": part, "sentiments": [{"label": label}]},
        }
        for part in ("train", "validation")
        for label in LABELS
    ]
    train, validation = rows[:3], rows[3:]
    cfg = FineTuneConfig(
        device="cpu",
        learning_rate=0.01,
        max_epochs=2,
        patience=1,
        train_batch_size=2,
        max_length=16,
        warmup_ratio=0,
    )

    def saver(m, tok, _root, config, epoch, membership):
        return save_model(m, tok, tmp_path, asset, config, epoch, membership)

    fitted = fit_checkpoint(
        cfg,
        train,
        validation,
        tmp_path,
        checkpoint_loader=lambda *_: (torch, tokenizer, model, source),
        artifact_saver=saver,
        checkpoint_files=("config.json",),
        index_labels=LABELS,
        frozen_parameters=FROZEN,
    )
    for name, parameter in model.named_parameters():
        if name in FROZEN:
            assert torch.equal(parameter, initial[name])
            assert not parameter.requires_grad
            assert parameter.grad is None
    assert not torch.equal(model.cls_head.weight, initial["cls_head.weight"])
    assert not torch.equal(
        model.backbone.embeddings.word_embeddings.weight,
        initial["backbone.embeddings.word_embeddings.weight"],
    )
    expected = sum(p.numel() for n, p in model.named_parameters() if n not in FROZEN)
    assert fitted["parameters_updated"] == expected
    membership = membership_digest(train)
    directory, identity = (
        fitted["artifact_directory"],
        fitted["artifact_manifest_sha256"],
    )
    tok, restored = reload_model(directory, identity, asset, cfg, membership)
    predictions, _ = score_model(torch, tok, restored, ["validation"] * 3, cfg)
    metrics = classification_metrics(list(LABELS), [p.label.value for p in predictions])
    assert metrics == fitted["history"][fitted["best_epoch"] - 1]["validation"]
    for name in FROZEN:
        assert torch.equal(dict(restored.named_parameters())[name], initial[name])
    with pytest.raises(ValueError, match="ancestry"):
        reload_model(directory, identity, asset, replace(cfg, seed=43), membership)
    with pytest.raises(ValueError, match="unknown frozen"):
        fit_checkpoint(
            cfg,
            train,
            validation,
            tmp_path,
            checkpoint_loader=lambda *_: (torch, tokenizer, model, source),
            frozen_parameters=("nonexistent",),
            index_labels=LABELS,
        )
