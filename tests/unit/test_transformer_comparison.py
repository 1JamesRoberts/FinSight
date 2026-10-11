"""Safety checks for the bounded transformer comparison."""

import pytest

from finsight.sentiment.transformer_comparison import checked_labels, validate_loading


def test_labels_are_explicit_and_cover_three_classes() -> None:
    assert checked_labels({"0": "negative", "1": "neutral", "2": "positive"}) == (
        "negative",
        "neutral",
        "positive",
    )
    with pytest.raises(ValueError, match="label"):
        checked_labels({"0": "LABEL_0", "1": "LABEL_1", "2": "LABEL_2"})


def test_inference_rejects_random_or_partial_classifier() -> None:
    with pytest.raises(ValueError, match="loading"):
        validate_loading({"missing_keys": ["classifier.weight"]}, "roberta", False)
    validate_loading({}, "roberta", False)


def test_base_training_allows_only_expected_new_head() -> None:
    validate_loading(
        {
            "missing_keys": [
                "classifier.dense.weight",
                "classifier.dense.bias",
                "classifier.out_proj.weight",
                "classifier.out_proj.bias",
            ],
            "unexpected_keys": ["lm_head.bias"],
        },
        "roberta",
        True,
    )
    with pytest.raises(ValueError, match="loading"):
        validate_loading(
            {"missing_keys": ["roberta.encoder.layer.0.weight"]}, "roberta", True
        )
    with pytest.raises(ValueError, match="loading"):
        validate_loading(
            {"unexpected_keys": ["some_other_encoder.weight"]}, "roberta", True
        )
    with pytest.raises(ValueError, match="loading"):
        validate_loading({"mismatched_keys": ["classifier.weight"]}, "deberta-v2", True)


@pytest.mark.filterwarnings(
    "ignore:.*torch.jit.script.*deprecated:DeprecationWarning:torch.jit._script"
)
@pytest.mark.parametrize(
    "cohort,index_labels",
    [
        ("sentfin_single_target", ("positive", "negative", "neutral")),
        ("sentfin_single_target", ("negative", "neutral", "positive")),
        ("phrasebank", ("positive", "negative", "neutral")),
    ],
)
@pytest.mark.parametrize("kind", ["roberta", "deberta-v2"])
def test_tiny_base_training_and_generic_artifact_reload(
    tmp_path, monkeypatch, kind, cohort, index_labels
) -> None:
    torch = pytest.importorskip("torch")
    t = pytest.importorskip("transformers")
    from finsight.sentiment.finbert_training import FineTuneConfig, fit_checkpoint
    from finsight.sentiment.transformer_comparison import (
        load_saved,
        save_model,
        score_model,
    )
    from finsight.sentiment.validation import classification_metrics

    torch.manual_seed(42)
    torch.set_num_threads(1)
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ntrain\nvalidation\n")
    tokenizer = t.BertTokenizerFast(vocab_file=str(vocab))
    kwargs = dict(
        vocab_size=7,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=32,
        num_labels=3,
        id2label=dict(enumerate(index_labels)),
        label2id={label: i for i, label in enumerate(index_labels)},
        attn_implementation="eager",
    )
    if kind == "roberta":
        model = t.RobertaForSequenceClassification(
            t.RobertaConfig(**kwargs, type_vocab_size=2)
        )
    else:
        model = t.DebertaV2ForSequenceClassification(
            t.DebertaV2Config(**kwargs, type_vocab_size=2)
        )
    config = FineTuneConfig(
        device="cpu",
        max_length=16,
        train_batch_size=2,
        evaluation_batch_size=2,
        max_epochs=1,
        warmup_ratio=0,
        learning_rate=0.01,
    )
    labels = ["positive", "negative", "neutral"]
    rows = [
        {
            "corpus": "phrasebank" if cohort == "phrasebank" else "sentfin",
            "partition": partition,
            "record_id": partition + label,
            "group_id": partition + label,
            "native_record": (
                {"sentence": partition, "label": label}
                if cohort == "phrasebank"
                else {"title": partition, "sentiments": [{"label": label}]}
            ),
        }
        for partition in ("train", "validation")
        for label in labels
    ]
    asset = {
        "model_id": "synthetic/" + kind,
        "revision": "synthetic",
        "model_type": kind,
    }
    before = next(model.parameters()).detach().clone()
    fit = fit_checkpoint(
        config,
        rows[:3],
        rows[3:],
        tmp_path,
        checkpoint_loader=lambda *_: (torch, tokenizer, model, tmp_path),
        checkpoint_files=("vocab.txt",),
        run_namespace="tiny",
        training_cohort=cohort,
        index_labels=index_labels,
        artifact_saver=lambda model, tokenizer, _root, conf, epoch, membership: (
            save_model(model, tokenizer, tmp_path, asset, conf, epoch, membership)
        ),
    )
    assert not torch.equal(before, next(model.parameters()))
    torch, tokenizer, model, _ = load_saved(
        fit["artifact_directory"], fit["artifact_manifest_sha256"], asset, config
    )
    predictions, meta = score_model(torch, tokenizer, model, ["validation"] * 3, config)
    assert (
        classification_metrics(labels, [p.label.value for p in predictions])
        == fit["history"][0]["validation"]
    )
    assert meta["architecture"]["layers"] == 1
    assert not model.training and not any(p.requires_grad for p in model.parameters())


def test_phrasebank_training_rejects_other_cohorts_and_reserved_data(tmp_path) -> None:
    from finsight.sentiment.finbert_training import FineTuneConfig, fit_checkpoint

    with pytest.raises(ValueError, match="development records only"):
        fit_checkpoint(
            FineTuneConfig(),
            [{"partition": "test_candidate"}],
            [],
            tmp_path,
            training_cohort="phrasebank",
        )
    with pytest.raises(ValueError, match="unsupported training cohort"):
        fit_checkpoint(FineTuneConfig(), [], [], tmp_path, training_cohort="fiqa")
