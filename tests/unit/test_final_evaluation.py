"""Synthetic locked evaluation, bootstrap, retry and no-test-access boundaries."""

import json
from pathlib import Path

import numpy as np
import pytest

from finsight.sentiment import final_evaluation as runner
from finsight.sentiment.final_metrics import bootstrap, evaluate_outputs, validate_rows
from finsight.sentiment.transformer_analysis import json_body
from finsight.sentiment.validation import LABELS, classification_metrics


def records():
    return [
        dict(
            record_id=str(i),
            group_id=str(i // 2),
            text="synthetic text",
            gold=LABELS[i % 3],
        )
        for i in range(6)
    ]


def predictions(correct=True):
    return [
        {
            "record_id": r["record_id"],
            "gold": r["gold"],
            "prediction": r["gold"] if correct else LABELS[(i + 1) % 3],
            "probabilities": {
                c: 0.8 if c == (r["gold"] if correct else LABELS[(i + 1) % 3]) else 0.1
                for c in LABELS
            },
        }
        for i, r in enumerate(records())
    ]


@pytest.mark.parametrize(
    "corruption", ["order", "gold", "missing", "label", "probabilities"]
)
def test_prediction_validation_rejects_corruption(corruption):
    rows = predictions()
    if corruption == "order":
        rows.reverse()
    elif corruption == "gold":
        rows[0]["gold"] = "positive"
    elif corruption == "missing":
        rows.pop()
    elif corruption == "label":
        rows[0]["prediction"] = "other"
    else:
        rows[0]["probabilities"]["negative"] = float("nan")
    with pytest.raises(ValueError):
        validate_rows(records(), rows)


def test_metric_families_average_scores_not_probabilities():
    outputs = {"a": predictions(), "b": predictions(False)}
    result = evaluate_outputs(records(), outputs, families={"model": ["a", "b"]})
    assert result["statistics"]["model"]["accuracy"]["mean"] == 0.5
    assert result["statistics"]["model"]["macro_f1"]["mean"] == 0.5
    assert result["statistics"]["model"]["accuracy"]["sample_std"] == pytest.approx(
        2**-0.5
    )


def test_group_bootstrap_matches_brute_force_and_preserves_duplicate_draws():
    rows = predictions()
    result, private = bootstrap(
        records(), {"a": rows}, families={"model": ["a"]}, replicates=30
    )
    assert result["groups"] == 3
    assert result["group_sizes"] == {2: 3}
    f1 = []
    for draw in private["draws"]:
        indices = [
            i
            for group in draw
            for i, r in enumerate(records())
            if r["group_id"] == private["groups"][group]
        ]
        assert len(indices) == 6
        f1.append(
            classification_metrics(
                [rows[i]["gold"] for i in indices],
                [rows[i]["prediction"] for i in indices],
            )["macro_f1"]
        )
    assert result["intervals"]["model"]["macro_f1"] == pytest.approx(
        np.quantile(f1, [0.025, 0.975]).tolist()
    )
    assert bootstrap(
        records(), {"a": rows}, families={"model": ["a"]}, replicates=30
    ) == (result, private)


def test_paired_bootstrap_exactly_equal_variants_have_zero_difference():
    outputs = {"a": predictions(), "b": predictions()}
    result, _ = bootstrap(
        records(),
        outputs,
        families={"deberta_base.raw": ["a"], "finbert.raw": ["b"]},
        replicates=20,
    )
    assert result["paired_differences"][
        "deberta_base.raw minus finbert.raw: macro_f1"
    ] == [0, 0]
    assert result["intervals"]["deberta_base.raw"]["nll"] == pytest.approx(
        [-__import__("math").log(0.8)] * 2
    )


def test_label_only_outputs_have_no_probability_metrics():
    rows = [{k: v for k, v in r.items() if k != "probabilities"} for r in predictions()]
    result = evaluate_outputs(records(), {"hard": rows}, families={"hard": ["hard"]})
    assert result["runs"]["hard"]["probability_metrics"] is None
    result, _ = bootstrap(
        records(), {"hard": rows}, families={"hard": ["hard"]}, replicates=10
    )
    assert set(result["intervals"]["hard"]) == {"macro_f1", "accuracy"}


def simple_plan():
    return {"baselines": {"vader": {}}, "transformers": {}}


def test_saved_predictions_resume_without_reinference(tmp_path):
    calls = []

    def predict(name, spec, texts, data_dir):
        calls.append(texts)
        return [{"prediction": "neutral"} for _ in texts], {"synthetic": True}

    first = runner.execute_predictions(
        tmp_path,
        "receipt",
        records(),
        simple_plan(),
        tmp_path,
        resume=False,
        predict=predict,
    )
    assert len(calls) == 1
    second = runner.execute_predictions(
        tmp_path,
        "receipt",
        records(),
        simple_plan(),
        tmp_path,
        resume=True,
        predict=predict,
    )
    assert first == second and len(calls) == 1
    progress = json.loads((tmp_path / "progress.json").read_bytes())
    assert progress["attempts"][1]["reused"] == ["vader"]
    assert not (tmp_path / "active.lock").exists()
    with pytest.raises(ValueError, match="resume"):
        runner.execute_predictions(
            tmp_path,
            "receipt",
            records(),
            simple_plan(),
            tmp_path,
            resume=False,
            predict=predict,
        )


@pytest.mark.parametrize("corruption", ["bytes", "missing"])
def test_resume_refuses_changed_or_missing_completed_predictions(tmp_path, corruption):
    def predict(*_):
        return [{"prediction": "neutral"}] * 6, {}

    runner.execute_predictions(
        tmp_path,
        "receipt",
        records(),
        simple_plan(),
        tmp_path,
        resume=False,
        predict=predict,
    )
    path = tmp_path / "predictions/vader.json"
    if corruption == "bytes":
        path.write_bytes(b"corrupted")
    else:
        path.unlink()
    with pytest.raises(ValueError):
        runner.execute_predictions(
            tmp_path,
            "receipt",
            records(),
            simple_plan(),
            tmp_path,
            resume=True,
            predict=lambda *_: pytest.fail("must not rescore"),
        )


def test_failure_is_logged_and_completed_models_reused_on_retry(tmp_path):
    plan = {"baselines": {"first": {}, "second": {}}, "transformers": {}}
    calls = []

    def failing(name, *_):
        calls.append(name)
        if name == "second":
            raise RuntimeError("synthetic failure")
        return [{"prediction": "neutral"}] * 6, {}

    with pytest.raises(RuntimeError):
        runner.execute_predictions(
            tmp_path,
            "receipt",
            records(),
            plan,
            tmp_path,
            resume=False,
            predict=failing,
        )
    status = json.loads((tmp_path / "progress.json").read_bytes())
    assert status["attempts"][0]["status"] == "failed"
    assert "first" in status["completed"] and "second" not in status["completed"]

    def retry(name, *_):
        assert name == "second"
        return [{"prediction": "neutral"}] * 6, {}

    runner.execute_predictions(
        tmp_path, "receipt", records(), plan, tmp_path, resume=True, predict=retry
    )


def test_incomplete_model_output_aborts_without_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="incomplete"):
        runner.execute_predictions(
            tmp_path,
            "receipt",
            records(),
            simple_plan(),
            tmp_path,
            resume=False,
            predict=lambda *_: ([{"prediction": "neutral"}], {}),
        )
    assert not (tmp_path / "predictions/vader.json").exists()


def test_active_execution_lock_blocks_second_writer(tmp_path):
    (tmp_path / "active.lock").write_bytes(b"1")
    with pytest.raises(FileExistsError):
        runner.execute_predictions(
            tmp_path,
            "receipt",
            records(),
            simple_plan(),
            tmp_path,
            resume=False,
            predict=lambda *_: pytest.fail("concurrent writer"),
        )


def test_preparation_never_decodes_test_records(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "preflight", lambda *_: {})
    monkeypatch.setattr(
        runner, "code_identity", lambda *_: {"revision": "synthetic", "sources": {}}
    )
    monkeypatch.setattr(runner, "environment", lambda: {"synthetic": True})
    monkeypatch.setattr(
        runner,
        "frozen_records",
        lambda *_: pytest.fail("test decoding during preparation"),
    )
    (tmp_path / "configs").mkdir()
    for name in ("final-evaluation-plan.json", "final-evaluation-ensembles-v2.json"):
        (tmp_path / "configs" / name).write_bytes(b"{}")
    path = runner.prepare_execution(tmp_path, tmp_path / "data")
    assert json.loads(path.read_bytes())["test_records_decoded_at_preparation"] is False
    assert runner.prepare_execution(tmp_path, tmp_path / "data") == path


def test_preflight_and_receipt_fail_before_frozen_decode(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "preflight", lambda *_: {})
    monkeypatch.setattr(
        runner,
        "checked_receipt",
        lambda *_: (_ for _ in ()).throw(ValueError("changed code")),
    )
    monkeypatch.setattr(
        runner, "frozen_records", lambda *_: pytest.fail("premature test access")
    )
    with pytest.raises(ValueError, match="changed code"):
        runner.run_final(tmp_path, tmp_path, tmp_path / "receipt.json")


def test_receipt_detects_changed_execution_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "preflight", lambda *_: {})
    identity = {"revision": "synthetic", "sources": {"a.py": "original"}}
    monkeypatch.setattr(runner, "code_identity", lambda *_: identity)
    monkeypatch.setattr(runner, "environment", lambda: {"synthetic": True})
    (tmp_path / "configs").mkdir()
    for name in ("final-evaluation-plan.json", "final-evaluation-ensembles-v2.json"):
        (tmp_path / "configs" / name).write_bytes(b"{}")
    path = runner.prepare_execution(tmp_path, tmp_path / "data")
    identity["sources"]["a.py"] = "changed"
    with pytest.raises(ValueError, match="differs"):
        runner.checked_receipt(tmp_path, tmp_path / "data", path)


def test_gpu_loader_only_gets_local_paths_and_texts(tmp_path, monkeypatch):
    from finsight.sentiment import saved_inference as module

    monkeypatch.setattr(module, "verify_artifact", lambda *_args, **_kwargs: 0)
    seen = []
    monkeypatch.setattr(
        module,
        "_gpu_inference",
        lambda name, spec, texts, data, started: (seen.append(texts) or [], {}),
    )
    module.infer_candidate("finbert", {"artifact": {}}, ["invented text"], tmp_path)
    assert seen == [["invented text"]]


def test_prior_uses_locked_counts_without_fitting(tmp_path):
    from finsight.sentiment.saved_inference import infer_candidate

    spec = {
        "prediction": "positive",
        "probabilities": dict(negative=0.2, neutral=0.3, positive=0.5),
    }
    rows, meta = infer_candidate("training_prior", spec, ["invented text"], tmp_path)
    assert rows == [{"prediction": "positive", "probabilities": spec["probabilities"]}]
    assert meta["loading_seconds"] == 0


def test_complete_synthetic_execution_resume_and_public_export(tmp_path, monkeypatch):
    plan = json.loads(Path("configs/final-evaluation-plan.json").read_bytes())
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    for name in ("final-evaluation-plan.json", "final-evaluation-ensembles-v2.json"):
        (tmp_path / "configs" / name).write_bytes(b"{}")
    directory = tmp_path / "execution"
    directory.mkdir()
    receipt_path = directory / "receipt.json"
    receipt_path.write_bytes(b"{}")
    monkeypatch.setattr(runner, "preflight", lambda *_: plan)
    monkeypatch.setattr(
        runner,
        "checked_receipt",
        lambda *_: {
            "code": {"revision": "synthetic"},
            "environment": {"synthetic": True},
        },
    )
    monkeypatch.setattr(runner, "frozen_records", lambda *_: records())
    real_execute = runner.execute_predictions
    calls = []

    def predict(name, spec, texts, data):
        calls.append(name)
        rows = [
            {
                "prediction": "neutral",
                "probabilities": dict(negative=0.2, neutral=0.6, positive=0.2),
            }
            for _ in texts
        ]
        if name in ("vader", "tfidf_svm"):
            for r in rows:
                r.pop("probabilities")
        return rows, {"synthetic": True}

    monkeypatch.setattr(
        runner,
        "execute_predictions",
        lambda *args, **kwargs: real_execute(*args, **kwargs, predict=predict),
    )
    path = runner.run_final(tmp_path, tmp_path, receipt_path)
    assert len(calls) == 14
    value = json.loads(path.read_bytes())
    assert len(value["runs"]) == 29
    assert len(value["statistics"]) == 13
    assert value["final_test_evaluated"] is True  # Synthetic test only.
    assert value["uncertainty"]["replicates"] == 1000
    assert len(value["uncertainty"]["paired_differences"]) == 11
    monkeypatch.setattr(
        runner,
        "frozen_records",
        lambda *_: pytest.fail("completed run must not decode again"),
    )
    assert runner.run_final(tmp_path, tmp_path, receipt_path, resume=True) == path
    assert len(calls) == 14
    destination = tmp_path / "public/results"
    exported = runner.export_results(directory, destination)
    assert "record_id" not in exported.read_text()
    assert "synthetic text" not in exported.read_text()
    assert runner.export_results(directory, destination) == exported
    exported.write_bytes(b"different published report")
    with pytest.raises(ValueError, match="overwrite"):
        runner.export_results(directory, destination)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        runner.run_final(tmp_path, tmp_path, receipt_path, resume=True)


def test_export_refuses_incomplete_execution(tmp_path):
    (tmp_path / "progress.json").write_bytes(json_body({"completed": {}}))
    (tmp_path / "aggregate.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="completed"):
        runner.export_results(tmp_path, tmp_path / "public")
