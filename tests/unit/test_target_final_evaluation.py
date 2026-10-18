"""Synthetic runner coverage, label-free inference and independent paired intervals."""

import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from finsight.sentiment.evaluation.metrics import LABELS, classification_metrics
from finsight.sentiment.evaluation.target_data import expand_frozen
from finsight.sentiment.evaluation.target_metrics import families, summarize
from finsight.sentiment.evaluation.target_protocol import PLAN
from finsight.sentiment.experiments import target_final_evaluation as runner
from finsight.sentiment.models.target_inference import TargetText, unique_inputs

ROOT = Path(__file__).resolve().parents[2]


def plan():
    result = json.loads((ROOT / PLAN).read_bytes())
    result["uncertainty"]["replicates"] = 30  # Synthetic bootstrap test budget only.
    return result


def records():
    return [
        {
            "record_id": str(i),
            "headline_id": str(i // 2),
            "group_id": str(i // 4),
            "headline": f"PRIVATEHEADLINE {i // 2}",
            "target": f"PRIVATETARGET{i % 2}",
            "gold": LABELS[i % 3],
            "multi_target": True,
            "mixed_labels": True,
        }
        for i in range(8)
    ]


def fake_predict(spec, inputs, data):
    assert inputs and all(
        isinstance(r, TargetText) and not hasattr(r, "gold") and not hasattr(r, "label")
        for r in inputs
    )
    return [
        {
            "prediction": "neutral",
            "probabilities": {"negative": 0.1, "neutral": 0.8, "positive": 0.1},
        }
        for _ in inputs
    ], {"mode": "synthetic"}


def test_deduplication_broadcasts_headlines_but_preserves_distinct_targets():
    inputs = [
        TargetText("a", "alpha beats beta", "alpha"),
        TargetText("a", "alpha beats beta", "beta"),
    ]
    assert unique_inputs(inputs, "matched_headline_only")[1] == [0, 0]
    assert unique_inputs(inputs, "target_pair")[1] == [0, 1]
    with pytest.raises(ValueError, match="conflicting text"):
        unique_inputs([inputs[0], TargetText("a", "different", "beta")], "target_pair")


def test_all_variants_resume_without_repeating_inference(tmp_path):
    outputs, _ = runner.execute_predictions(
        tmp_path,
        "receipt",
        records(),
        plan(),
        tmp_path,
        resume=False,
        predictor=fake_predict,
    )
    assert len(outputs) == 21
    with pytest.raises(ValueError, match="explicit resume"):
        runner.execute_predictions(
            tmp_path,
            "receipt",
            records(),
            plan(),
            tmp_path,
            resume=False,
            predictor=fake_predict,
        )

    def forbidden(*args):
        raise AssertionError("completed variant inferred again")

    repeated, _ = runner.execute_predictions(
        tmp_path,
        "receipt",
        records(),
        plan(),
        tmp_path,
        resume=True,
        predictor=forbidden,
    )
    assert repeated == outputs


@pytest.mark.parametrize(
    "corruption", ["missing", "bytes", "membership", "spec", "receipt"]
)
def test_resume_rejects_corruption(tmp_path, corruption):
    runner.execute_predictions(
        tmp_path,
        "receipt",
        records(),
        plan(),
        tmp_path,
        resume=False,
        predictor=fake_predict,
    )
    selected = next((tmp_path / "predictions").glob("*.json"))
    rs, protocol, receipt = records(), plan(), "receipt"
    if corruption == "missing":
        selected.unlink()
    elif corruption == "bytes":
        selected.write_bytes(selected.read_bytes() + b" ")
    elif corruption == "membership":
        rs[0]["gold"] = "positive"
    elif corruption == "spec":
        protocol["transformers"]["deberta_base"][0]["selected_epoch"] += 1
    else:
        receipt = "different"
    with pytest.raises(ValueError):
        runner.execute_predictions(
            tmp_path,
            receipt,
            rs,
            protocol,
            tmp_path,
            resume=True,
            predictor=fake_predict,
        )


@pytest.mark.parametrize("failure", ["short", "nan", "wrong_argmax", "extra"])
def test_failed_predictions_are_logged_and_never_completed(tmp_path, failure):
    def invalid(spec, inputs, data):
        predictions, meta = fake_predict(spec, inputs, data)
        if failure == "short":
            predictions.pop()
        elif failure == "nan":
            predictions[0]["probabilities"]["negative"] = float("nan")
        elif failure == "wrong_argmax":
            predictions[0]["prediction"] = "positive"
        else:
            predictions[0]["gold"] = "negative"
        return predictions, meta

    with pytest.raises(ValueError):
        runner.execute_predictions(
            tmp_path,
            "receipt",
            records(),
            plan(),
            tmp_path,
            resume=False,
            predictor=invalid,
        )
    status = json.loads((tmp_path / "progress.json").read_bytes())
    assert status["completed"] == {}
    assert status["attempts"][-1]["status"] == "failed"


def test_interrupted_run_reuses_completed_variants(tmp_path):
    calls = []

    def interrupted(spec, inputs, data):
        calls.append(spec)
        if len(calls) == 3:
            raise RuntimeError("interrupted")
        return fake_predict(spec, inputs, data)

    with pytest.raises(RuntimeError):
        runner.execute_predictions(
            tmp_path,
            "receipt",
            records(),
            plan(),
            tmp_path,
            resume=False,
            predictor=interrupted,
        )
    completed = []

    def finish(spec, inputs, data):
        completed.append(spec)
        return fake_predict(spec, inputs, data)

    runner.execute_predictions(
        tmp_path, "receipt", records(), plan(), tmp_path, resume=True, predictor=finish
    )
    assert len(completed) == 19


def test_paired_bootstrap_matches_independent_whole_group_recalculation(tmp_path):
    rs, protocol = records(), plan()
    outputs, _ = runner.execute_predictions(
        tmp_path,
        "receipt",
        rs,
        protocol,
        tmp_path,
        resume=False,
        predictor=fake_predict,
    )
    # Distinct seed behavior makes averaging seed metrics observable.
    for seed in (42, 43, 44):
        key = f"roberta_base.target_pair.{seed}"
        for i, row in enumerate(outputs[key]):
            label = rs[i]["gold"] if seed == 42 else "neutral"
            row.update(
                prediction=label,
                probabilities={c: 0.8 if c == label else 0.1 for c in LABELS},
            )
    report, private = summarize(rs, outputs, {"PRIVATETARGET0"}, protocol)
    values = []
    groups = private["bootstrap_draws"]["multi_target"]["groups"]
    for draw in private["bootstrap_draws"]["multi_target"]["draws"]:
        indices = [
            i for g in draw for i, r in enumerate(rs) if r["group_id"] == groups[g]
        ]
        gold = [rs[i]["gold"] for i in indices]
        left = np.mean(
            [
                classification_metrics(
                    gold, [outputs[k][i]["prediction"] for i in indices]
                )["macro_f1"]
                for k in families()["roberta_base.target_pair"]
            ]
        )
        right = np.mean(
            [
                classification_metrics(
                    gold, [outputs[k][i]["prediction"] for i in indices]
                )["macro_f1"]
                for k in families()["roberta_base.matched_headline_only"]
            ]
        )
        values.append(left - right)
    actual = report["cohorts"]["multi_target"]["paired_differences"][
        "roberta_conditioning"
    ]["interval_95"]
    assert actual == pytest.approx(np.quantile(values, [0.025, 0.975]))
    assert report["cohorts"]["same_label_headlines"]["metrics"] is None
    assert "PRIVATE" not in json.dumps(report)
    assert "PRIVATETARGET0" in private["per_native_target"]


def frozen_fixture():
    rows = []
    for i in range(2):
        annotations = [
            {"target": "alpha", "label": "positive"},
            {"target": "beta", "label": "negative"},
        ]
        rows.append(
            {
                "record_id": str(i),
                "group_id": f"test-{i}",
                "corpus": "sentfin",
                "partition": "test_candidate",
                "development_only": False,
                "review_reasons": [],
                "native_record": {"title": f"headline {i}", "sentiments": annotations},
            }
        )
    expected = {
        "headlines": 2,
        "target_annotations": 4,
        "class_counts": dict(
            Counter(a["label"] for r in rows for a in r["native_record"]["sentiments"])
        ),
    }
    return rows, {"multi_target": expected, "mixed_label_headlines": expected}


@pytest.mark.parametrize(
    "corruption", ["group", "text", "consumed", "label", "duplicate", "count", "flag"]
)
def test_frozen_membership_audit_rejects_overlap_and_scope_changes(corruption):
    rows, expected = frozen_fixture()
    development = [{"group_id": "train", "headline": "unrelated"}]
    if corruption == "group":
        development[0]["group_id"] = rows[0]["group_id"]
    elif corruption == "text":
        development[0]["headline"] = "HEADLINE 0"
    elif corruption == "consumed":
        consumed = copy.deepcopy(rows[0])
        consumed["record_id"] = "consumed"
        consumed["native_record"]["sentiments"].pop()
        rows.append(consumed)
    elif corruption == "label":
        rows[0]["native_record"]["sentiments"][0]["label"] = "unknown"
    elif corruption == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif corruption == "count":
        expected["multi_target"]["headlines"] += 1
    else:
        rows[0]["development_only"] = True
    with pytest.raises(ValueError):
        expand_frozen(rows, development, expected)


def test_frozen_expansion_keeps_original_target_labels_and_identity():
    rows, expected = frozen_fixture()
    examples = expand_frozen(rows, [], expected)
    assert len(examples) == len({r["record_id"] for r in examples}) == 4
    assert [r["gold"] for r in examples] == ["positive", "negative"] * 2


def test_preflight_failure_prevents_frozen_decode(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("artifact verification failed")

    monkeypatch.setattr(runner, "verify_plan", fail)
    monkeypatch.setattr(
        runner, "frozen_inputs", lambda *args: pytest.fail("test decoded")
    )
    with pytest.raises(ValueError, match="artifact verification"):
        runner.run_final(tmp_path, tmp_path, tmp_path / "receipt.json")


@pytest.mark.filterwarnings("ignore:.*torch.jit.script.*deprecated:DeprecationWarning")
@pytest.mark.parametrize("backbone", ["bert", "roberta", "deberta"])
def test_tiny_backbones_predict_without_labels_gradients_or_weight_changes(
    tmp_path, backbone
):
    torch = pytest.importorskip("torch")
    t = pytest.importorskip("transformers")
    from finsight.sentiment.models.target_inference import score_inputs
    from finsight.sentiment.training.config import FineTuneConfig

    torch.set_num_threads(1)
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nalpha\nbeta\ngains\n")
    tokenizer = t.BertTokenizerFast(vocab_file=str(vocab))
    model_class, config_class = {
        "bert": (t.BertForSequenceClassification, t.BertConfig),
        "roberta": (t.RobertaForSequenceClassification, t.RobertaConfig),
        "deberta": (t.DebertaV2ForSequenceClassification, t.DebertaV2Config),
    }[backbone]
    model = model_class(
        config_class(
            vocab_size=8,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=32,
            pad_token_id=0,
            num_labels=3,
            id2label=dict(enumerate(LABELS)),
            attn_implementation="eager",
        )
    )
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    seen = []

    def inspect(module, args, kwargs):
        assert not module.training and not torch.is_grad_enabled()
        assert "labels" not in kwargs
        seen.append(kwargs["input_ids"])

    model.register_forward_pre_hook(inspect, with_kwargs=True)
    inputs = [
        TargetText("a", "alpha gains beta", "alpha"),
        TargetText("a", "alpha gains beta", "beta"),
    ]
    config = FineTuneConfig(device="cpu", max_length=16)
    for arm in ("matched_headline_only", "target_pair"):
        predictions, metadata = score_inputs(
            torch, tokenizer, model, inputs, arm, config
        )
        assert len(predictions) == 2
        assert metadata["unique_inputs_scored"] == (
            1 if arm == "matched_headline_only" else 2
        )
        if arm == "matched_headline_only":
            assert predictions[0] == predictions[1]
    assert seen
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


@pytest.fixture
def execution(tmp_path, monkeypatch):
    root, data = tmp_path / "repo", tmp_path / "data"
    (root / "configs").mkdir(parents=True)
    (root / runner.PLAN).write_text(json.dumps(plan()))
    (root / runner.LOCK).write_text("synthetic lock")
    monkeypatch.setattr(runner, "verify_plan", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        runner, "code_identity", lambda *args: {"revision": "synthetic", "sources": {}}
    )
    monkeypatch.setattr(runner, "environment", lambda: {"gpu": "synthetic"})
    monkeypatch.setattr(
        runner,
        "development_inputs",
        lambda *args: (
            [{"target": "PRIVATETARGET0"}],
            [],
            {"manifest_sha256": "synthetic"},
        ),
    )
    receipt = runner.prepare_execution(root, data)
    calls = []
    monkeypatch.setattr(
        runner, "frozen_inputs", lambda *args: calls.append("decode") or records()
    )
    original = runner.execute_predictions

    def execute(*args, **kwargs):
        return original(*args, **kwargs, predictor=fake_predict)

    monkeypatch.setattr(runner, "execute_predictions", execute)
    return root, data, receipt, calls


def test_complete_execution_and_resume_do_not_repeat_test_decode(
    execution, monkeypatch
):
    root, data, receipt, calls = execution
    path = runner.run_final(root, data, receipt)
    report = json.loads(path.read_bytes())
    assert report["test_scored"] is True
    assert report["selection_changed"] is False
    assert "PRIVATE" not in path.read_text()
    assert calls == ["decode"]

    def forbidden(*args):
        pytest.fail("completed execution decoded test again")

    monkeypatch.setattr(runner, "frozen_inputs", forbidden)
    assert runner.run_final(root, data, receipt, resume=True) == path
    with pytest.raises(ValueError, match="explicit resume"):
        runner.run_final(root, data, receipt)


def test_postprocessing_failure_resumes_verified_predictions(execution, monkeypatch):
    root, data, receipt, _ = execution
    original = runner.summarize

    def fail(*args):
        raise RuntimeError("postprocessing interrupted")

    monkeypatch.setattr(runner, "summarize", fail)
    with pytest.raises(RuntimeError):
        runner.run_final(root, data, receipt)
    status = json.loads((receipt.parent / "progress.json").read_bytes())
    assert status["attempts"][-1]["status"] == "failed_postprocessing"
    assert len(status["completed"]) == 21
    monkeypatch.setattr(runner, "summarize", original)
    assert runner.run_final(root, data, receipt, resume=True).exists()
    status = json.loads((receipt.parent / "progress.json").read_bytes())
    assert status["attempts"][-1]["computed"] == []


def test_changed_code_and_active_session_block_decoding(execution, monkeypatch):
    root, data, receipt, calls = execution
    with monkeypatch.context() as patch:
        patch.setattr(runner, "code_identity", lambda *args: {"revision": "different"})
        with pytest.raises(ValueError, match="context"):
            runner.run_final(root, data, receipt)
    (receipt.parent / "evaluation.lock").write_text("other process")
    with pytest.raises(FileExistsError):
        runner.run_final(root, data, receipt)
    assert calls == []


def test_wrong_receipt_location_is_rejected_before_decode(execution):
    root, data, receipt, calls = execution
    moved = receipt.parent / "unbound.json"
    moved.write_bytes(receipt.read_bytes())
    with pytest.raises(ValueError, match="content-addressed"):
        runner.run_final(root, data, moved)
    assert calls == []


def test_verified_export_contains_only_aggregates_and_never_overwrites(execution):
    root, data, receipt, _ = execution
    runner.run_final(root, data, receipt)
    json_path, markdown = runner.export_results(
        root, receipt.parent, root / "reports/final"
    )
    assert "PRIVATE" not in json_path.read_text() + markdown.read_text()
    assert "Seed SD" in markdown.read_text()
    with pytest.raises(FileExistsError):
        runner.export_results(root, receipt.parent, root / "reports/final")
    private = receipt.parent / "results-private.json"
    private.write_bytes(private.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksums"):
        runner.export_results(root, receipt.parent, root / "reports/other")


def exclusion_fixture():
    """A consumed near-duplicate of one reserved group; no real test data."""
    from finsight.data.review import text_hash

    rows, _ = frozen_fixture()
    reserved = rows[0]
    reserved["native_record"]["sentiments"] = [
        {"target": "alpha", "label": "neutral"},
        {"target": "beta", "label": "neutral"},
    ]
    consumed = copy.deepcopy(reserved)
    consumed["record_id"] = "previously-evaluated"
    consumed["native_record"]["title"] = "a shortened related headline"
    consumed["native_record"]["sentiments"].pop()
    rows.append(consumed)
    registry = [
        {
            "group_id": reserved["group_id"],
            "reserved_text_sha256": text_hash(reserved["native_record"]["title"]),
            "consumed_text_sha256": text_hash(consumed["native_record"]["title"]),
            "reserved_class_counts": {"neutral": 2},
            "reason": "near_duplicate_group_previously_evaluated_in_phase3",
        }
    ]
    counts = {
        "headlines": 1,
        "target_annotations": 2,
        "class_counts": {"positive": 1, "negative": 1},
    }
    return rows, {"multi_target": counts, "mixed_label_headlines": counts}, registry


def test_versioned_exclusion_removes_only_reviewed_reserved_group():
    rows, expected, registry = exclusion_fixture()
    original = copy.deepcopy(rows)
    with pytest.raises(ValueError, match="overlaps"):
        expand_frozen(rows, [], expected)  # v1 retains its original abort behavior.
    actual = expand_frozen(rows, [], expected, exclusions=registry)
    assert len(actual) == 2
    assert {r["headline_id"] for r in actual} == {"1"}
    assert [r["gold"] for r in actual] == ["positive", "negative"]
    assert rows == original  # Never edit the frozen benchmark in place.


@pytest.mark.parametrize(
    "change",
    [
        "missing_member",
        "reserved_text",
        "consumed_text",
        "labels",
        "reason",
        "duplicate_exclusion",
        "development_overlap",
        "unexpected_consumed_group",
    ],
)
def test_versioned_exclusion_rejects_unreviewed_scope_changes(change):
    rows, expected, registry = exclusion_fixture()
    development = []
    if change == "missing_member":
        rows.pop()
    elif change == "reserved_text":
        registry[0]["reserved_text_sha256"] = "0" * 64
    elif change == "consumed_text":
        registry[0]["consumed_text_sha256"] = "0" * 64
    elif change == "labels":
        rows[0]["native_record"]["sentiments"][0]["label"] = "positive"
    elif change == "reason":
        registry[0]["reason"] = "model_performance"
    elif change == "duplicate_exclusion":
        registry.append(copy.deepcopy(registry[0]))
    elif change == "development_overlap":
        development = [{"group_id": rows[0]["group_id"], "headline": "training"}]
    else:
        extra = copy.deepcopy(rows[1])
        extra["record_id"] = "unexpected-consumed"
        extra["native_record"]["sentiments"].pop()
        rows.append(extra)
    with pytest.raises(ValueError):
        expand_frozen(rows, development, expected, exclusions=registry)
