"""Calibration behavior and fit/assessment separation on synthetic examples."""

import json
import math
from pathlib import Path

import pytest

from finsight.sentiment.temperature_scaling import (
    apply_temperature,
    cross_fit,
    fit_temperature,
    validation_folds,
)
from finsight.sentiment.transformer_analysis import calibration_metrics
from finsight.sentiment.validation import LABELS


def rows():
    # 80% correct but confidence 98%; finite interior optimum above T=1.
    return [
        {
            "record_id": str(i),
            "gold": LABELS[i % 3],
            "prediction": LABELS[i % 3] if i % 5 else LABELS[(i + 1) % 3],
            "probabilities": {
                label: 0.98
                if label == (LABELS[i % 3] if i % 5 else LABELS[(i + 1) % 3])
                else 0.01
                for label in LABELS
            },
        }
        for i in range(90)
    ]


def metadata():
    return [
        dict(
            record_id=r["record_id"],
            group_id=str(i // 2),
            partition="validation",
            corpus="sentfin",
            native_record={"title": "synthetic", "sentiments": [{"label": r["gold"]}]},
        )
        for i, r in enumerate(rows())
    ]


def test_identity_normalization_and_class_invariance():
    original = rows()
    assert apply_temperature(original, 1) == original
    for temperature in (0.05, 2, 20):
        adjusted = apply_temperature(original, temperature)
        assert [r["prediction"] for r in adjusted] == [
            r["prediction"] for r in original
        ]
        assert all(
            sum(r["probabilities"].values()) == pytest.approx(1) for r in adjusted
        )
    assert original == rows()  # Input is never mutated.


def test_fit_improves_nll_and_matches_analytic_solution():
    fitted = fit_temperature(rows())
    expected = math.log(98) / math.log(8)  # 0.8 = 98**(1/T)/(98**(1/T)+2).
    assert fitted["temperature"] == pytest.approx(expected, abs=1e-10)
    assert fitted["boundary"] is None
    assert (
        calibration_metrics(apply_temperature(rows(), fitted["temperature"]))["nll"]
        < calibration_metrics(rows())["nll"]
    )


def test_optimizer_boundaries_and_flat_objective():
    correct = [r for i, r in enumerate(rows()) if i % 5]
    wrong = [r for i, r in enumerate(rows()) if not i % 5]
    assert fit_temperature(correct)["boundary"] == "lower_temperature"
    assert fit_temperature(wrong)["boundary"] == "upper_temperature"
    uniform = [
        {**r, "prediction": "negative", "probabilities": dict.fromkeys(LABELS, 1 / 3)}
        for r in rows()
    ]
    assert fit_temperature(uniform)["temperature"] == 1


@pytest.mark.parametrize("temperature", [0, -1, True, float("nan"), float("inf")])
def test_invalid_temperature_rejected(temperature):
    with pytest.raises(ValueError):
        apply_temperature(rows(), temperature)


def test_invalid_distributions_and_empty_fit_rejected():
    for bad in ([], [{**rows()[0], "probabilities": dict.fromkeys(LABELS, 0.1)}]):
        with pytest.raises(ValueError):
            fit_temperature(bad)
    with pytest.raises(ValueError):
        apply_temperature(rows(), 1, floor=0.5)


def test_grouped_folds_deterministic_cover_once_and_reject_test():
    records = metadata()
    folds = validation_folds(records)
    assert folds == validation_folds(records)
    assert sorted(i for _, assessed in folds for i in assessed) == list(range(90))
    for fitted, assessed in folds:
        assert not set(fitted) & set(assessed)
        assert not {records[i]["group_id"] for i in fitted} & {
            records[i]["group_id"] for i in assessed
        }
    with pytest.raises(ValueError, match="validation"):
        validation_folds([{**records[0], "partition": "test"}, *records[1:]])
    with pytest.raises(ValueError, match="unique"):
        validation_folds([*records, records[0]])


def test_cross_fit_assessment_labels_do_not_influence_own_temperature():
    original = rows()
    folds = validation_folds(metadata())
    result, adjusted = cross_fit(original, folds)
    mutated = [dict(r) for r in original]
    for i in folds[0][1]:
        mutated[i]["gold"] = LABELS[(LABELS.index(mutated[i]["gold"]) + 1) % 3]
    changed, _ = cross_fit(mutated, folds)
    assert result["folds"][0]["fit"] == changed["folds"][0]["fit"]
    assert len(adjusted) == len(original)
    assert result["before"]["accuracy"] == result["after"]["accuracy"]
    assert result["classification_before"] == result["classification_after"]


def test_malformed_fold_cover_or_overlap_rejected():
    folds = validation_folds(metadata())
    for bad in (folds[:-1], [(folds[0][0] + folds[0][1], folds[0][1]), *folds[1:]]):
        with pytest.raises(ValueError):
            cross_fit(rows(), bad)


def test_zero_probability_is_finite_and_preserves_tied_prediction():
    original = [
        dict(
            gold="positive",
            prediction="neutral",
            probabilities={
                "negative": 0.0,
                "neutral": 0.5,
                "positive": 0.5,
            },
        )
    ]
    scaled = apply_temperature(original, 2)
    assert scaled[0]["prediction"] == "neutral"
    assert math.isfinite(calibration_metrics(scaled)["nll"])


@pytest.mark.parametrize("corruption", [None, "study", "freeze", "scope", "seeds"])
def test_experiment_audits_inputs_and_keeps_memberships_private(
    tmp_path, monkeypatch, corruption
):
    from finsight.sentiment import temperature_scaling as module
    from finsight.sentiment.error_analysis import sha
    from finsight.sentiment.transformer_analysis import json_body

    plan = json.loads(Path("configs/temperature-scaling.json").read_bytes())
    monkeypatch.chdir(tmp_path)
    Path("configs").mkdir()
    receipt_body = json_body({"synthetic": True})
    Path("configs/benchmark-freeze.json").write_bytes(receipt_body)
    plan["freeze_receipt_sha256"] = sha(receipt_body)
    for source in plan["studies"]:
        study = {
            "complete": True,
            "final_test_evaluated": corruption == "scope",
            "plan": {"freeze_receipt_sha256": sha(receipt_body)},
            "selected": {m: {"learning_rate": 1e-5} for m in source["models"]},
            "seed_runs": {
                m: [
                    {"model": m, "seed": s, "learning_rate": 1e-5}
                    for s in ((42, 43) if corruption == "seeds" else (42, 43, 44))
                ]
                for m in source["models"]
            },
        }
        body = json_body(study)
        path = Path(source["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        source["sha256"] = sha(body)
    if corruption == "study":
        Path(plan["studies"][0]["path"]).write_bytes(b"corrupted")
    if corruption == "freeze":
        Path("configs/benchmark-freeze.json").write_bytes(b"corrupted")
    Path("plan.json").write_bytes(json_body(plan))
    monkeypatch.setattr(module, "load_development_records", lambda *_: metadata())
    calls = []

    def checked(entry, *_):
        calls.append((entry["model"], entry["seed"]))
        return rows(), {"synthetic_checked_input": True}

    monkeypatch.setattr(module, "checked_run", checked)
    monkeypatch.setattr(
        module.subprocess, "check_output", lambda *_, **__: "synthetic\n"
    )
    Path("uv.lock").write_bytes(b"synthetic lock")
    report_path = Path("report.json")
    if corruption:
        with pytest.raises(ValueError):
            module.run_experiment(Path("data"), Path("plan.json"), report_path)
        assert not report_path.exists()
        assert not Path("data/interim/temperature-scaling").exists()
        return
    result = module.run_experiment(Path("data"), Path("plan.json"), report_path)
    assert len(calls) == 9
    assert result["final_test_evaluated"] is False
    assert "record_id" not in report_path.read_text()
    assert "group_id" not in report_path.read_text()
    directory = (
        Path("data/interim/temperature-scaling") / result["private_packet_sha256"]
    )
    packet_bytes = (directory / "packet.json").read_bytes()
    assert sha(packet_bytes) == result["private_packet_sha256"]
    assert (
        sha((directory / "calibrators.json").read_bytes())
        == result["calibrators_sha256"]
    )
    assert len(json.loads(packet_bytes)["out_of_fold_predictions"]) == 9
    repeated = module.run_experiment(Path("data"), Path("plan.json"), report_path)
    assert repeated == result
