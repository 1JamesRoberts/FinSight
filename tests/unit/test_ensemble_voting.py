"""Equal-weight voting rules, membership alignment and supplementary lock guards."""

import json
from pathlib import Path

import pytest

from finsight.sentiment.ensemble_voting import combine, evaluate_seed, verify_extension
from finsight.sentiment.transformer_analysis import json_body
from finsight.sentiment.validation import LABELS


def row(label, probs=None, *, record_id="synthetic", gold="positive"):
    return {
        "record_id": record_id,
        "gold": gold,
        "prediction": label,
        "probabilities": probs or {c: 0.8 if c == label else 0.1 for c in LABELS},
    }


def test_soft_average_can_disagree_with_hard_majority():
    members = [
        row("positive", dict(negative=0.45, neutral=0.05, positive=0.5)),
        row("positive", dict(negative=0.45, neutral=0.05, positive=0.5)),
        row("negative", dict(negative=0.98, neutral=0.01, positive=0.01)),
    ]
    result = combine(members)
    assert result["soft"]["prediction"] == "negative"
    assert result["soft"]["probabilities"]["negative"] == pytest.approx(1.88 / 3)
    assert result["hard"]["prediction"] == "positive"
    assert result["hard"]["votes"] == dict(negative=1, neutral=0, positive=2)
    assert result["hard"]["tie"] is False
    assert "probabilities" not in result["hard"]


def test_hard_three_way_tie_and_soft_exact_tie_rules():
    result = combine([row(c) for c in LABELS])
    assert result["hard"]["prediction"] == "neutral"
    assert result["hard"]["tie"] is True
    assert result["soft"]["prediction"] == "negative"


@pytest.mark.parametrize(
    "labels", [("negative",) * 3, ("positive", "neutral", "neutral")]
)
def test_unanimous_and_majority_decisions(labels):
    result = combine([row(c) for c in labels])
    assert result["hard"]["prediction"] == max(LABELS, key=labels.count)


@pytest.mark.parametrize("bad", ["size", "id", "gold", "probabilities", "argmax"])
def test_invalid_or_misaligned_members_rejected(bad):
    members = [row("positive") for _ in range(3)]
    if bad == "size":
        members.pop()
    elif bad == "id":
        members[1]["record_id"] = "different"
    elif bad == "gold":
        members[1]["gold"] = "negative"
    elif bad == "argmax":
        members[1]["prediction"] = "negative"
    else:
        members[1]["probabilities"]["positive"] = float("nan")
    with pytest.raises(ValueError):
        combine(members)


def test_seed_comparison_paired_corrections_and_hard_probability_boundary():
    members = {
        "finbert": [row("negative"), row("neutral", record_id="b", gold="neutral")],
        "roberta_base": [
            row("positive"),
            row("positive", record_id="b", gold="neutral"),
        ],
        "deberta_base": [
            row("positive"),
            row("positive", record_id="b", gold="neutral"),
        ],
    }
    result, private = evaluate_seed(members, seed=42)
    assert result["hard"]["probability_metrics"] is None
    assert result["hard"]["classification"]["samples"] == 2
    assert result["hard"]["versus_deberta"]["both_correct"] == 1
    assert result["hard"]["versus_deberta"]["both_wrong"] == 1
    assert result["soft"]["probability_metrics"]["samples"] == 2
    assert len(private) == 2
    assert "record_id" not in json.dumps(result)
    with pytest.raises(ValueError):
        evaluate_seed({**members, "finbert": members["finbert"][:1]}, seed=42)


def test_extension_checksum_protects_parent_and_recipe(tmp_path, monkeypatch):
    from finsight.sentiment import ensemble_voting as module
    from finsight.sentiment.finbert import file_sha256

    (tmp_path / "configs").mkdir()
    (tmp_path / "recipe.json").write_bytes(b"recipe")
    (tmp_path / "parent.json").write_bytes(b"parent")
    extension = {
        "protocol_id": "phase3-sentfin-single-target-v2",
        "parent_plan_path": "parent.json",
        "parent_plan_sha256": file_sha256(tmp_path / "parent.json"),
        "ensemble_plan_path": "recipe.json",
        "ensemble_plan_sha256": file_sha256(tmp_path / "recipe.json"),
        "methods": ["soft", "hard"],
        "test_evaluated": False,
    }
    path = tmp_path / "configs/final-evaluation-ensembles-v2.json"
    path.write_bytes(json_body(extension))
    lock = {"protocol_sha256": file_sha256(path)}
    (tmp_path / "configs/final-evaluation-ensembles-v2-lock.json").write_bytes(
        json_body(lock)
    )
    monkeypatch.setattr(module, "verify_locked_plan", lambda *_args, **_kwargs: {})
    assert verify_extension(tmp_path, Path("unused"))["test_evaluated"] is False
    (tmp_path / "recipe.json").write_bytes(b"changed")
    with pytest.raises(ValueError):
        verify_extension(tmp_path, Path("unused"))
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        verify_extension(tmp_path, Path("unused"))


@pytest.mark.parametrize(
    "corruption", [None, "packet_bytes", "membership", "folds", "probabilities"]
)
def test_validation_audits_oof_and_keeps_votes_private(
    tmp_path, monkeypatch, corruption
):
    from finsight.sentiment import ensemble_voting as module
    from finsight.sentiment.error_analysis import sha
    from finsight.sentiment.temperature_scaling import (
        apply_temperature,
        validation_folds,
    )
    from finsight.sentiment.validation import classification_metrics

    recipe = json.loads(Path("configs/ensemble-voting.json").read_bytes())
    monkeypatch.chdir(tmp_path)
    Path("configs").mkdir()
    Path("configs/benchmark-freeze.json").write_bytes(b"{}")
    Path("uv.lock").write_bytes(b"synthetic lock")
    validation = [
        dict(
            record_id=str(i),
            group_id=str(i),
            partition="validation",
            corpus="sentfin",
            native_record={
                "title": "synthetic",
                "sentiments": [{"label": LABELS[i % 3]}],
            },
        )
        for i in range(30)
    ]
    raw = [row(LABELS[i % 3], record_id=str(i), gold=LABELS[i % 3]) for i in range(30)]
    scaled = apply_temperature(raw, 2)
    folds = validation_folds(validation)
    packet = {
        "records": [{k: r[k] for k in ("record_id", "group_id")} for r in validation],
        "folds": [{"fit": f, "assess": a} for f, a in folds],
        "out_of_fold_predictions": {
            f"{m}-{s}": [dict(r) for r in scaled]
            for m in module.MODELS
            for s in module.SEEDS
        },
    }
    if corruption == "membership":
        packet["records"][0]["record_id"] = "different"
    if corruption == "folds":
        packet["folds"][0]["fit"] = []
    if corruption == "probabilities":
        key = "finbert-42"
        packet["out_of_fold_predictions"][key][0]["probabilities"] = dict(
            negative=0.9, neutral=0.05, positive=0.05
        )
    packet_bytes = json_body(packet)
    identifier = sha(packet_bytes)
    directory = Path("data/interim/temperature-scaling") / identifier
    directory.mkdir(parents=True)
    (directory / "packet.json").write_bytes(
        packet_bytes + (b" " if corruption == "packet_bytes" else b"")
    )
    calibration = {
        "private_packet_sha256": identifier,
        "models": {
            m: {
                "runs": [
                    {"seed": s, "folds": [{"fit": {"temperature": 2}} for _ in folds]}
                    for s in module.SEEDS
                ]
            }
            for m in module.MODELS
        },
    }
    Path(recipe["calibration_report_path"]).parent.mkdir(parents=True)
    Path(recipe["calibration_report_path"]).write_bytes(json_body(calibration))
    recipe["calibration_report_sha256"] = sha(json_body(calibration))
    Path("configs/ensemble-voting.json").write_bytes(json_body(recipe))
    primary = {"transformers": {}}
    for model in module.MODELS:
        primary["transformers"][model] = []
        for seed in module.SEEDS:
            report_path = Path(f"{model}-{seed}.json")
            report_path.write_bytes(
                json_body(
                    {
                        "metrics": classification_metrics(
                            [r["gold"] for r in raw], [r["prediction"] for r in raw]
                        )
                    }
                )
            )
            primary["transformers"][model].append(
                {
                    "seed": seed,
                    "report_path": str(report_path),
                    "report_sha256": sha(report_path.read_bytes()),
                    "configuration": {"learning_rate": 1e-5},
                }
            )
    Path(recipe["parent_final_plan_path"]).write_bytes(json_body(primary))
    Path("configs/final-evaluation-ensembles-v2.json").write_bytes(b"{}")
    monkeypatch.setattr(
        module, "verify_extension", lambda *_: {"protocol_id": "synthetic"}
    )
    monkeypatch.setattr(module, "load_development_records", lambda *_: validation)
    monkeypatch.setattr(
        module, "checked_run", lambda *_: (raw, {"synthetic_checked_source": True})
    )
    monkeypatch.setattr(
        module.subprocess, "check_output", lambda *_, **__: "synthetic\n"
    )
    path = Path("result.json")
    if corruption:
        with pytest.raises(ValueError):
            module.run_validation(Path("data"), path)
        assert not path.exists()
        assert not Path("data/interim/ensemble-voting").exists()
        return
    result = module.run_validation(Path("data"), path)
    assert result["final_test_evaluated"] is False
    assert "record_id" not in path.read_text()
    private = (
        Path("data/interim/ensemble-voting")
        / result["private_predictions_sha256"]
        / "predictions.json"
    )
    assert sha(private.read_bytes()) == result["private_predictions_sha256"]
    assert module.run_validation(Path("data"), path) == result
