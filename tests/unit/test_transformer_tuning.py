"""Selection and scope checks for the bounded learning-rate study."""

from dataclasses import replace

import pytest

from finsight.sentiment.finbert_training import FineTuneConfig
from finsight.sentiment.transformer_tuning import (
    choose_rate,
    seed_statistics,
    validate_config,
)


def test_rate_selection_uses_macro_f1_and_declared_tie_order() -> None:
    candidates = [
        {"learning_rate": 1e-5, "metrics": {"macro_f1": 0.8, "accuracy": 0.99}},
        {"learning_rate": 2e-5, "metrics": {"macro_f1": 0.9, "accuracy": 0.8}},
        {"learning_rate": 3e-5, "metrics": {"macro_f1": 0.9, "accuracy": 0.95}},
    ]
    assert choose_rate(candidates)["learning_rate"] == 2e-5
    with pytest.raises(ValueError, match="candidates"):
        choose_rate([])


def test_seed_statistics_include_all_seeds_and_sample_standard_deviation() -> None:
    rows = [
        {"seed": seed, "metrics": {"macro_f1": value, "accuracy": value}}
        for seed, value in [(42, 0.8), (43, 0.9), (44, 1.0)]
    ]
    stats = seed_statistics(rows)
    assert stats["macro_f1"]["mean"] == pytest.approx(0.9)
    assert stats["macro_f1"]["sample_std"] == pytest.approx(0.1)
    assert stats["seeds"] == [42, 43, 44]
    with pytest.raises(ValueError, match="seeds"):
        seed_statistics(rows[:2])
    with pytest.raises(ValueError, match="seeds"):
        seed_statistics([rows[0], rows[0], rows[2]])


def test_grid_changes_only_learning_rate_and_seed() -> None:
    baseline = FineTuneConfig()
    validate_config(replace(baseline, learning_rate=1e-5, seed=43), baseline, 1e-5, 43)
    with pytest.raises(ValueError, match="fixed"):
        validate_config(replace(baseline, max_length=64), baseline, 2e-5, 42)
    with pytest.raises(ValueError, match="fixed"):
        validate_config(replace(baseline, seed=44), baseline, 2e-5, 42)


def test_study_freezes_rates_before_repeats_and_reuses_seed_42(
    tmp_path, monkeypatch
) -> None:
    import json

    from finsight.sentiment import transformer_tuning as module

    (tmp_path / "plan.json").write_text("{}")
    monkeypatch.setattr(module, "CONFIGS", tmp_path)
    monkeypatch.setattr(module, "checked_plan", lambda: {})
    calls = []

    def execute(model, rate, seed, data_dir, root, resume):
        calls.append((model, rate, seed))
        if seed != 42:
            selection = json.loads((root / "selection.json").read_bytes())
            assert set(selection["selected"]) == {"roberta_base", "finbert"}
            assert not selection["repeat_seeds_observed_before_selection"]
        return {
            "model": model,
            "learning_rate": rate,
            "seed": seed,
            "training_membership_sha256": "train",
            "validation_membership_sha256": "val",
            "metrics": {"macro_f1": 0.9 if rate == 2e-5 else 0.8, "accuracy": 0.9},
        }

    monkeypatch.setattr(module, "execute_run", execute)
    result = module.run_study(tmp_path, tmp_path / "reports")
    assert len(calls) == 10
    assert all(seed == 42 for _, _, seed in calls[:6])
    assert all(rate == 2e-5 for _, rate, _ in calls[6:])
    assert result["complete"] and not result["final_test_evaluated"]
    assert result["statistics"]["finbert"]["seeds"] == [42, 43, 44]


def test_study_rejects_different_candidate_memberships(tmp_path, monkeypatch) -> None:
    from finsight.sentiment import transformer_tuning as module

    (tmp_path / "plan.json").write_text("{}")
    monkeypatch.setattr(module, "CONFIGS", tmp_path)
    monkeypatch.setattr(module, "checked_plan", lambda: {})
    monkeypatch.setattr(
        module,
        "execute_run",
        lambda model, rate, seed, *_: {
            "learning_rate": rate,
            "metrics": {"macro_f1": 0.9},
            "training_membership_sha256": "train",
            "validation_membership_sha256": str(rate),
        },
    )
    with pytest.raises(ValueError, match="memberships"):
        module.run_study(tmp_path, tmp_path / "reports")


def test_existing_report_requires_explicit_resume_before_training(
    tmp_path, monkeypatch
) -> None:
    from finsight.sentiment import transformer_tuning as module

    destination = tmp_path / "reports/finbert/lr-2e-5-seed-42"
    destination.mkdir(parents=True)
    (destination / "finbert.json").write_text("{}")
    monkeypatch.setattr(module.FineTuneConfig, "load", lambda _: FineTuneConfig())
    monkeypatch.setattr(
        module, "run_finetuning", lambda *_: pytest.fail("must not retrain")
    )
    with pytest.raises(ValueError, match="already exists"):
        module.execute_run("finbert", 2e-5, 42, tmp_path, tmp_path / "reports", False)
