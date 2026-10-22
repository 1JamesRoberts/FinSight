"""Fixed trial aggregation, matching guards and native-cohort preservation."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from finsight.data.target_pairs import TargetExample, TargetText
from finsight.sentiment.experiments import target_format_training as experiment
from finsight.sentiment.experiments.target_format_training import (
    aggregate,
    render_report,
    verify_matched,
    write_state,
)
from finsight.sentiment.models.target import collate_examples


def trials():
    result = []
    for seed in (42, 43, 44):
        for i, variant in enumerate(
            ("target_pair", "target_markers", "target_replacement")
        ):
            result.append(
                {
                    "seed": seed,
                    "input_variant": variant,
                    "configuration": {"seed": seed},
                    "initialization": {
                        "state_sha256": str(seed),
                        "rng_sha256": str(seed),
                    },
                    "fit": {
                        "history": [{"training_order_sha256": str(seed)}],
                        "best_epoch": 1,
                        "training_seconds": 1,
                    },
                    "cohorts": {
                        "all_targets": {
                            "classification": {
                                "macro_f1": 0.7 + i * 0.01,
                                "accuracy": 0.8,
                            },
                            "calibration": {"nll": 1.0, "brier": 0.5, "ece": 0.1},
                        }
                    },
                    "model": {
                        "architecture": {
                            "parameters": 100,
                            "layers": 1,
                            "hidden_size": 8,
                            "attention_heads": 2,
                        }
                    },
                    "model_file_bytes": 400,
                }
            )
    return result


def test_aggregate_all_seeds_and_render_explanations():
    runs = trials()
    summary = aggregate(runs, ["all_targets"])
    assert summary["target_markers"]["all_targets"]["macro_f1"][
        "mean"
    ] == pytest.approx(0.71)
    assert summary["target_markers"]["all_targets"]["macro_f1"]["sample_std"] == 0
    report = render_report({"aggregate": summary, "trials": runs})
    assert "not confidence intervals" in report
    assert "Seed fixes" in report
    assert "Saved bytes" in report


@pytest.mark.parametrize(
    "failure", ["missing", "duplicate", "configuration", "initialization", "order"]
)
def test_incomplete_or_unmatched_trials_rejected(failure):
    runs = trials()
    if failure == "missing":
        runs.pop()
    elif failure == "duplicate":
        runs[-1] = deepcopy(runs[0])
    elif failure == "configuration":
        runs[1]["configuration"]["seed"] = 99
    elif failure == "initialization":
        runs[1]["initialization"]["rng_sha256"] = "different"
    else:
        runs[1]["fit"]["history"][0]["training_order_sha256"] = "different"
    with pytest.raises(ValueError):
        aggregate(runs, ["all_targets"])


def test_early_stopping_allows_different_history_lengths():
    a, b = trials()[:2]
    b["fit"]["history"].append({"training_order_sha256": "next-epoch"})
    verify_matched(a, b)


def test_wrong_view_identity_rejected_before_tokenization():
    e = TargetExample(
        "id",
        "headline",
        "group",
        "train",
        0,
        "Acme",
        "Acme rose",
        "positive",
        False,
        False,
        False,
        (),
    )
    views = {"id": TargetText("wrong-id", "[TARGET] rose", "[TARGET]")}
    with pytest.raises(ValueError, match="identity"):
        collate_examples(
            SimpleNamespace(),
            None,
            [e],
            "target_pair",
            128,
            labels=False,
            input_views=views,
        )
    with pytest.raises(ValueError, match="target-pair"):
        collate_examples(
            SimpleNamespace(),
            None,
            [e],
            "matched_headline_only",
            128,
            labels=False,
            input_views=views,
        )
    with pytest.raises(ValueError, match="development"):
        collate_examples(
            SimpleNamespace(),
            None,
            [replace(e, partition="test")],
            "target_pair",
            128,
            labels=False,
            input_views=views,
        )


def test_state_updates_are_atomic_and_readable(tmp_path):
    import json

    path = tmp_path / "status.json"
    write_state(path, {"status": "running", "completed_trials": 0})
    write_state(path, {"status": "complete", "completed_trials": 9})
    assert json.loads(path.read_text())["completed_trials"] == 9
    assert not path.with_suffix(".tmp").exists()


@pytest.mark.parametrize("fail", [False, True])
def test_orchestrator_exports_nine_trials_or_records_failure(
    tmp_path, monkeypatch, fail
):
    import json

    plan = {
        "experiment": "synthetic",
        "seeds": [42, 43, 44],
        "evaluation_cohorts": ["all_targets"],
        "trials": [
            {"variant": r["input_variant"], "seed": r["seed"]} for r in trials()
        ],
    }
    (tmp_path / "configs").mkdir()
    (tmp_path / "reports/experiments").mkdir(parents=True)
    (tmp_path / experiment.PLAN).write_text("{}")
    (tmp_path / experiment.LOCK).write_text("{}")
    monkeypatch.setattr(experiment, "load_plan", lambda root: (plan, {}))
    monkeypatch.setattr(
        experiment,
        "locked_inputs",
        lambda *a: (
            {},
            (),
            {v: {} for v in ("target_pair", "target_markers", "target_replacement")},
        ),
    )
    monkeypatch.setattr(experiment, "check_preparation", lambda *a: {})
    monkeypatch.setattr(experiment, "source_manifest", lambda: {"source": "hash"})
    monkeypatch.setattr(
        experiment.subprocess, "check_output", lambda *a, **kw: "revision"
    )
    calls = []

    def run(*args, **kwargs):
        calls.append(args[6:8])
        if fail and len(calls) == 3:
            raise RuntimeError("synthetic training failure")
        variant, seed = args[6:8]
        trial = next(
            r for r in trials() if r["input_variant"] == variant and r["seed"] == seed
        )
        trial["context"] = {"implementation_sha256": {"source": "hash"}}
        return trial

    monkeypatch.setattr(experiment, "run_trial", run)
    state = tmp_path / "job/state.json"
    if fail:
        with pytest.raises(RuntimeError, match="synthetic training failure"):
            experiment.run_all(tmp_path, tmp_path / "data", tmp_path, state)
        saved = json.loads(state.read_text())
        assert saved["status"] == "failed" and saved["completed_trials"] == 2
        assert not (tmp_path / experiment.REPORT.with_suffix(".json")).exists()
    else:
        result = experiment.run_all(tmp_path, tmp_path / "data", tmp_path, state)
        assert len(result["trials"]) == 9 and len(calls) == 9
        assert json.loads(state.read_text())["status"] == "complete"
        differences = result["paired_seed_differences"]["target_markers"]["all_targets"]
        assert differences == pytest.approx([0.01, 0.01, 0.01])
