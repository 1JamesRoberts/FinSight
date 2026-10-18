"""Validation-only deployment selection and reusable, label-free prediction."""

import json
from pathlib import Path

import pytest

from finsight.sentiment.deployment import CONFIG, build_deployment, check_deployment
from finsight.sentiment.models.target_predictor import TargetSentimentPredictor

ROOT = Path(__file__).resolve().parents[2]


def test_deployment_selection_reads_validation_metadata_only(monkeypatch):
    read = Path.read_bytes

    def guarded(path):
        if "data" in path.parts or "phase4-target-final" in path.name:
            pytest.fail("deployment selection accessed private or final-test data")
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    config = build_deployment(ROOT)
    assert config == check_deployment(ROOT, ROOT / CONFIG)
    assert config["seed"] == 43
    assert config["selection"]["scores"]["43"] == pytest.approx(0.8976832820983528)
    assert config["artifact"]["manifest_sha256"].startswith("2f742f69854f")
    assert config["selection"]["source"] == "saved_development_validation_only"


@pytest.mark.parametrize("change", ["seed", "artifact", "length", "rule", "extra"])
def test_changed_deployment_config_is_rejected(tmp_path, change):
    config = build_deployment(ROOT)
    if change == "seed":
        config["seed"] = 44
    elif change == "artifact":
        config["artifact"]["relative_path"] = "../weights"
    elif change == "length":
        config["training_configuration"]["max_length"] = 64
    elif change == "rule":
        config["selection"]["metric"] = "test_accuracy"
    else:
        config["arbitrary"] = True
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="deployment"):
        check_deployment(ROOT, path)


@pytest.fixture
def tiny(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from finsight.sentiment.training.config import FineTuneConfig

    torch.set_num_threads(1)
    torch.manual_seed(43)
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nalpha\nbeta\ngains\nfalls\n")
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(vocab))
    labels = ["positive", "negative", "neutral"]  # Deliberately noncanonical order.
    model = transformers.RobertaForSequenceClassification(
        transformers.RobertaConfig(
            vocab_size=9,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=64,
            type_vocab_size=2,
            num_labels=3,
            id2label=dict(enumerate(labels)),
            label2id={v: i for i, v in enumerate(labels)},
            attn_implementation="eager",
        )
    )
    config = FineTuneConfig(device="cpu", max_length=16, evaluation_batch_size=2)
    predictor = TargetSentimentPredictor(
        torch,
        tokenizer,
        model,
        config,
        model_version="synthetic-test-only",
        deployment_sha256="synthetic-config",
    )
    return predictor, torch, tokenizer, model


def test_prediction_reuses_model_and_preserves_pair_encoding_and_order(tiny):
    predictor, torch, tokenizer, model = tiny
    before = {k: v.clone() for k, v in model.state_dict().items()}
    calls = []
    handle = model.register_forward_pre_hook(
        lambda m, args, kwargs: calls.append(kwargs), with_kwargs=True
    )
    results = predictor.predict_many(
        ["alpha gains beta falls"] * 3, ["alpha", "beta", "alpha"]
    )
    assert len(calls) == 2  # Two batches, no loading or redundant warmup per request.
    assert [r.target for r in results] == ["alpha", "beta", "alpha"]
    assert results[0] == results[2]
    first = calls[0]["input_ids"][0].tolist()
    expected = tokenizer("alpha", "alpha gains beta falls")["input_ids"]
    assert first == expected
    with torch.inference_mode():
        logits = model(**{k: v[:1] for k, v in calls[0].items()}).logits
        expected_p = torch.softmax(logits, -1)[0].tolist()
    assert results[0].probabilities == pytest.approx(
        dict(zip(["positive", "negative", "neutral"], expected_p, strict=True))
    )
    assert results[0].model_version == "synthetic-test-only"
    assert not results[0].probabilities_calibrated
    assert results[0].label == max(
        results[0].probabilities, key=results[0].probabilities.get
    )
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    handle.remove()
    assert predictor.predict("alpha gains beta falls", "alpha") == results[0]


@pytest.mark.parametrize(
    "texts,targets",
    [
        ([], []),
        ([""], ["alpha"]),
        (["x"], [" "]),
        (["x"], []),
        ([None], ["alpha"]),
        (["x"], [7]),
    ],
)
def test_invalid_inputs_fail_before_forward(tiny, texts, targets):
    predictor, _, _, model = tiny
    calls = []
    handle = model.register_forward_pre_hook(lambda *args: calls.append(True))
    with pytest.raises(ValueError):
        predictor.predict_many(texts, targets)
    assert not calls
    handle.remove()


def test_target_overflow_fails_and_long_headline_preserves_target(tiny):
    predictor, _, _, _ = tiny
    with pytest.raises(ValueError, match="target"):
        predictor.predict("alpha gains", "alpha " * 30)
    result = predictor.predict("alpha gains " * 100, "beta")
    assert result.target == "beta"


def test_company_wrapper_abstains_and_preserves_explicit_target(tiny):
    predictor, _, _, model = tiny
    from finsight.sentiment.companies import load_registry

    registry = load_registry(ROOT / "configs/companies.json")
    calls = []
    handle = model.register_forward_pre_hook(lambda *args: calls.append(True))
    for target, status in [
        ("Apple", "ambiguous"),
        ("Google", "ambiguous"),
        ("TSLA", "unsupported"),
    ]:
        result = predictor.predict_company("alpha gains", target, registry)
        assert result.resolution.status == status and result.prediction is None
    assert not calls
    accepted = predictor.predict_company(
        "alpha gains", "Apple", registry, company_context=True
    )
    assert accepted.resolution.company.primary_ticker == "AAPL"
    assert accepted.prediction.target == "Apple"
    assert predictor.predict("alpha gains", "Sebi").target == "Sebi"
    handle.remove()


def test_load_verifies_once_then_keeps_model_for_multiple_requests(tiny, monkeypatch):
    predictor, torch, tokenizer, model = tiny
    import finsight.sentiment.models.target_predictor as module

    config = build_deployment(ROOT)
    calls = []

    def loader(directory, identifier, source, training_config, *, device=None):
        calls.append((directory, identifier, training_config, device))
        return torch, tokenizer, model, {}

    monkeypatch.setattr(module, "load_saved", loader)
    loaded = TargetSentimentPredictor.load(
        root=ROOT, data_dir=Path("synthetic"), device="cpu", batch_size=1
    )
    assert len(calls) == 1 and calls[0][3] == "cpu"
    assert calls[0][2].device == "cuda"  # Original training provenance is unchanged.
    assert loaded.config.device == "cpu" and loaded.config.evaluation_batch_size == 1
    assert loaded.model_version == config["artifact"]["manifest_sha256"]
    loaded.predict("alpha gains", "alpha")
    loaded.predict("beta falls", "beta")
    assert len(calls) == 1


def test_selection_exact_ties_and_invalid_scores():
    from finsight.sentiment.deployment import select_seed

    assert select_seed({42: 0.8, 43: 0.8, 44: 0.8}) == 42
    assert select_seed({42: 0.8, 43: 0.80000001, 44: 0.8}) == 43
    for values in ({}, {42: float("nan"), 43: 0.8, 44: 0.8}):
        with pytest.raises(ValueError):
            select_seed(values)


def test_cpu_override_preserves_original_manifest_configuration(tiny, tmp_path):
    from dataclasses import replace

    from finsight.sentiment.artifacts import file_sha256, save_model
    from finsight.sentiment.models.transformer import load_saved

    predictor, _, tokenizer, model = tiny
    training = replace(predictor.config, device="cuda")
    asset = {
        "model_id": "synthetic/roberta",
        "revision": "synthetic",
        "model_type": "roberta",
    }
    directory, identifier = save_model(
        model, tokenizer, tmp_path, asset, training, 1, "synthetic-membership"
    )
    before = file_sha256(directory / "artifact-manifest.json")
    _, _, loaded, manifest = load_saved(
        directory, identifier, asset, training, device="cpu"
    )
    assert next(loaded.parameters()).device.type == "cpu"
    assert manifest["configuration"]["device"] == "cuda"
    assert file_sha256(directory / "artifact-manifest.json") == before
    # A runtime override must not hide a changed original training configuration.
    with pytest.raises(ValueError, match="configuration"):
        load_saved(
            directory, identifier, asset, replace(training, max_length=20), device="cpu"
        )
    weights = directory / "model.safetensors"
    weights.write_bytes(weights.read_bytes() + b"corrupt")
    with pytest.raises(ValueError):
        load_saved(directory, identifier, asset, training, device="cpu")


def test_nonfinite_model_outputs_fail(tiny, monkeypatch):
    from types import SimpleNamespace

    predictor, torch, _, model = tiny

    def invalid(**tensors):
        return SimpleNamespace(
            logits=torch.full((len(tensors["input_ids"]), 3), float("nan"))
        )

    monkeypatch.setattr(model, "forward", invalid)
    with pytest.raises(ValueError, match="probabilities"):
        predictor.predict("alpha gains", "alpha")


def test_cli_config_and_prediction_use_explicit_inputs(tiny, monkeypatch, capsys):
    from finsight.sentiment import predict as command

    monkeypatch.setattr("sys.argv", ["predict", "--show-config"])
    command.main()
    assert json.loads(capsys.readouterr().out)["seed"] == 43
    predictor, _, _, _ = tiny
    loaded = []

    def load(**kwargs):
        loaded.append(kwargs)
        return predictor

    monkeypatch.setattr(command.TargetSentimentPredictor, "load", load)
    monkeypatch.setattr(
        "sys.argv",
        [
            "predict",
            "--device",
            "cpu",
            "--text",
            "alpha gains beta falls",
            "--target",
            "alpha",
            "--target",
            "beta",
        ],
    )
    command.main()
    result = json.loads(capsys.readouterr().out)
    assert [r["target"] for r in result] == ["alpha", "beta"]
    assert loaded[0]["device"] == "cpu" and len(loaded) == 1
