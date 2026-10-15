"""Matched target training keeps source groups, input policy and validation separate."""

from dataclasses import replace

import pytest

from finsight.data.target_pairs import TargetExample
from finsight.sentiment.target_training import validate_partitions


def examples(partition: str) -> tuple[TargetExample, ...]:
    return tuple(
        TargetExample(
            f"{partition}-{i}",
            f"{partition}-{i // 2}",
            f"{partition}-{i // 2}",
            partition,
            i % 2,
            ("alpha", "beta")[i % 2],
            f"{partition} {('gain', 'loss', 'flat')[i // 2]}",
            ("positive", "negative", "neutral")[i % 3],
            True,
            True,
            False,
            (),
        )
        for i in range(6)
    )


@pytest.mark.parametrize("failure", ["reserved", "group", "text", "id", "devonly"])
def test_rejects_invalid_training_or_validation_membership(failure: str) -> None:
    train, val = examples("train"), examples("validation")
    if failure == "reserved":
        val = (replace(val[0], partition="test"), *val[1:])
    elif failure == "group":
        val = (replace(val[0], group_id=train[0].group_id), *val[1:])
    elif failure == "text":
        val = (replace(val[0], headline=train[0].headline.upper()), *val[1:])
    elif failure == "id":
        train = (*train, train[0])
    else:
        val = (replace(val[0], development_only=True), *val[1:])
    with pytest.raises(ValueError):
        validate_partitions(train, val)


@pytest.mark.filterwarnings("ignore:.*torch.jit.script.*deprecated:DeprecationWarning")
@pytest.mark.parametrize("mode", ["matched_headline_only", "target_pair"])
@pytest.mark.parametrize("backbone", ["deberta", "bert", "roberta"])
def test_tiny_training_updates_encoder_and_reloads_best_without_validation_gradients(
    tmp_path, mode: str, backbone: str
) -> None:
    torch = pytest.importorskip("torch")
    t = pytest.importorskip("transformers")
    from finsight.sentiment.finbert_training import FineTuneConfig
    from finsight.sentiment.target_training import (
        collate_examples,
        fit_target,
        score_examples,
        state_sha256,
    )
    from finsight.sentiment.transformer_comparison import load_saved

    torch.set_num_threads(1)
    torch.manual_seed(42)
    vocab = tmp_path / "vocab.txt"
    vocab.write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nalpha\nbeta\ntrain\nvalidation\ngain\nloss\nflat\n"
    )
    tokenizer = t.BertTokenizerFast(vocab_file=str(vocab))
    labels = ("negative", "neutral", "positive")
    model_class, config_class = {
        "deberta": (t.DebertaV2ForSequenceClassification, t.DebertaV2Config),
        "bert": (t.BertForSequenceClassification, t.BertConfig),
        "roberta": (t.RobertaForSequenceClassification, t.RobertaConfig),
    }[backbone]
    model = model_class(
        config_class(
            vocab_size=12,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=32,
            type_vocab_size=2,
            num_labels=3,
            id2label=dict(enumerate(labels)),
            label2id={label: i for i, label in enumerate(labels)},
            attn_implementation="eager",
        )
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
    train, val = examples("train"), examples("validation")
    original = model.base_model.embeddings.word_embeddings.weight.detach().clone()
    initial_sha = state_sha256(model)
    observations = []
    handle = model.register_forward_pre_hook(
        lambda m, args, kwargs: observations.append(
            (m.training, torch.is_grad_enabled(), "labels" in kwargs)
        ),
        with_kwargs=True,
    )
    first = collate_examples(
        torch, tokenizer, train[:2], mode, config.max_length, labels=True
    )
    changed = tuple(replace(e, label="neutral") for e in train[:2])
    second = collate_examples(
        torch, tokenizer, changed, mode, config.max_length, labels=True
    )
    assert torch.equal(first["input_ids"], second["input_ids"])
    assert not torch.equal(first["labels"], second["labels"])
    asset = {
        "model_id": "synthetic/deberta",
        "revision": "synthetic",
        "model_type": model.config.model_type,
    }
    context = {"input_mode": mode, "plan_sha256": "test-plan"}
    fit = fit_target(
        torch,
        tokenizer,
        model,
        train,
        val,
        config=config,
        data_dir=tmp_path,
        asset=asset,
        context=context,
    )
    handle.remove()
    assert not torch.equal(original, model.base_model.embeddings.word_embeddings.weight)
    assert state_sha256(model) != initial_sha
    assert fit["optimizer_steps"] == 3
    assert any(
        training and grad and has_labels for training, grad, has_labels in observations
    )
    assert all(
        not grad and not has_labels
        for training, grad, has_labels in observations
        if not training
    )
    torch, tokenizer, loaded, manifest = load_saved(
        fit["artifact_directory"], fit["artifact_manifest_sha256"], asset, config
    )
    assert manifest["context"] == context
    rows, profile = score_examples(torch, tokenizer, loaded, val, mode, config)
    assert len(rows) == 6 and profile["architecture"]["layers"] == 1
    from finsight.sentiment.validation import classification_metrics

    assert (
        classification_metrics(
            [r["gold"] for r in rows], [r["prediction"] for r in rows]
        )
        == fit["history"][0]["validation"]
    )

    # Completed trial reuse checks private bytes, native gold and artifact context.
    from dataclasses import asdict

    from finsight.data.target_preparation import json_bytes
    from finsight.sentiment.target_control import evaluate_cohorts
    from finsight.sentiment.target_deberta_experiment import (
        predictions_bytes,
        read_trial,
        sha,
    )

    data = predictions_bytes(rows)
    result = {
        "context": context,
        "configuration": asdict(config),
        "final_test_evaluated": False,
        "predictions_sha256": sha(data),
        "cohorts": evaluate_cohorts(rows, {e.target for e in train}),
        "artifact_relative_path": fit["artifact_directory"]
        .relative_to(tmp_path)
        .as_posix(),
        "artifact_manifest_sha256": fit["artifact_manifest_sha256"],
        "fit": {k: v for k, v in fit.items() if k != "artifact_directory"},
    }
    body = json_bytes(result)
    path = tmp_path / "trial.json"
    path.write_bytes(body)
    root = tmp_path / "interim/target-deberta" / sha(body)
    root.mkdir(parents=True)
    (root / "run.json").write_bytes(body)
    (root / "predictions.jsonl").write_bytes(data)
    assert read_trial(path, tmp_path, asset, config, context, train + val) == result
    (root / "predictions.jsonl").write_bytes(data + b" ")
    with pytest.raises(ValueError, match="predictions differ"):
        read_trial(path, tmp_path, asset, config, context, train + val)


def fake_trials() -> list[dict]:
    from finsight.sentiment.target_training import MODES

    return [
        {
            "input_mode": mode,
            "seed": seed,
            "configuration": {"seed": seed},
            "initialization": {"state_sha256": str(seed), "rng_sha256": str(seed)},
            "fit": {
                "history": [
                    {"training_order_sha256": "first"},
                    {"training_order_sha256": "second"},
                ]
            },
            "cohorts": {
                "all_targets": {
                    "classification": {"accuracy": 0.5, "macro_f1": 0.6},
                    "calibration": {"nll": 0.8, "brier": 0.3, "ece": 0.1},
                }
            },
        }
        for mode in MODES
        for seed in (42, 43, 44)
    ]


def test_aggregate_requires_all_seeds_and_reports_variation() -> None:
    from finsight.sentiment.target_deberta_experiment import aggregate_trials

    trials = fake_trials()
    scores = aggregate_trials(trials, ["all_targets"])
    assert scores["target_pair"]["all_targets"]["macro_f1"] == {
        "mean": 0.6,
        "sample_std": 0.0,
    }
    with pytest.raises(ValueError):
        aggregate_trials(trials[:5], ["all_targets"])


@pytest.mark.parametrize("field", ["weights", "rng", "configuration", "order"])
def test_matching_audit_rejects_different_starts_or_shuffles(field: str) -> None:
    from finsight.sentiment.target_deberta_experiment import aggregate_trials

    trials = fake_trials()
    changed = trials[3]
    if field in ("weights", "rng"):
        key = "state_sha256" if field == "weights" else "rng_sha256"
        changed["initialization"][key] = "different"
    elif field == "configuration":
        changed["configuration"]["seed"] = 100
    else:
        changed["fit"]["history"][0]["training_order_sha256"] = "different"
    with pytest.raises(ValueError, match="matched"):
        aggregate_trials(trials, ["all_targets"])
