"""Selection and train-only search boundaries on synthetic data."""

import json
from pathlib import Path

import pytest

from finsight.sentiment.tfidf import TfidfConfig, TfidfSentiment
from finsight.sentiment.tuning import candidate_configs, run_tuning, select_candidate


def test_grid_preserves_features_and_rejects_invalid_candidates() -> None:
    base = TfidfConfig(min_df=7, max_iter=9000, seed=19)
    grid = candidate_configs(
        {
            "schema_version": 1,
            "c_values": [0.1, 1, 10],
            "class_weights": [None, "balanced"],
        },
        base,
    )
    assert [(c.c, c.class_weight) for c in grid] == [
        (0.1, None),
        (0.1, "balanced"),
        (1, None),
        (1, "balanced"),
        (10, None),
        (10, "balanced"),
    ]
    assert all(c.min_df == 7 and c.max_iter == 9000 and c.seed == 19 for c in grid)
    for values, weights in [
        ([], [None]),
        ([True, 1], [None]),
        ([0, 1], [None]),
        ([float("nan"), 1], [None]),
        ([1, 1], [None]),
        ([1], ["unsupported"]),
        ([0.1], [None]),
    ]:
        with pytest.raises(ValueError):
            candidate_configs(
                {"schema_version": 1, "c_values": values, "class_weights": weights},
                base,
            )
    with pytest.raises(ValueError, match="plan"):
        candidate_configs(
            {"schema_version": True, "c_values": [1], "class_weights": [None]}, base
        )


def test_selection_uses_macro_f1_not_accuracy_and_stable_ties() -> None:
    candidates = [
        {
            "model_kind": "linear_svm",
            "run": {
                "metrics": {
                    "phrasebank": {"linear_svm": {"macro_f1": f1, "accuracy": accuracy}}
                }
            },
        }
        for f1, accuracy in [(0.4, 0.99), (0.8, 0.7), (0.8, 0.95)]
    ]
    assert select_candidate(candidates, "phrasebank") == 1
    with pytest.raises(ValueError, match="no candidates"):
        select_candidate([], "phrasebank")


def synthetic_records() -> list[dict]:
    rows = []
    for corpus in ("phrasebank", "sentfin"):
        for partition in ("train", "validation"):
            for label, text in [
                ("negative", "losses decline"),
                ("neutral", "meeting scheduled"),
                ("positive", "profits increase"),
            ]:
                if partition == "validation":
                    text = "validationcanary " + text
                native = (
                    {"sentence": text, "label": label}
                    if corpus == "phrasebank"
                    else {
                        "title": text,
                        "sentiments": [{"target": "Example", "label": label}],
                    }
                )
                rows.append(
                    {
                        "record_id": f"{corpus}/{partition}/{label}",
                        "corpus": corpus,
                        "partition": partition,
                        "native_record": native,
                    }
                )
    return rows


@pytest.mark.parametrize("corrupt_membership", [False, True])
def test_search_preserves_boundaries_artifacts_and_baselines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corrupt_membership: bool,
) -> None:
    monkeypatch.setattr(
        "finsight.sentiment.tfidf_validation.load_development_records",
        lambda *args: synthetic_records(),
    )
    original_fit = TfidfSentiment.fit
    fits = []

    def observed_fit(self: TfidfSentiment, texts: list[str], labels: list[str]) -> None:
        assert all("validationcanary" not in text for text in texts)
        assert len(texts) == 3  # Separate corpus fits, never pooled.
        fits.append(texts)
        original_fit(self, texts, labels)

    monkeypatch.setattr(TfidfSentiment, "fit", observed_fit)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "benchmark_manifest_sha256": "synthetic",
                "parent_split_manifest_sha256": "synthetic-parent",
            }
        )
    )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {"schema_version": 1, "c_values": [1], "class_weights": [None, "balanced"]}
        )
    )
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"schema_version": 1, "min_df": 1, "max_iter": 10000}))
    report = tmp_path / "tuning.md"
    old_report = tmp_path / "baseline.md"
    old_report.write_text("preserved")
    if corrupt_membership:
        from finsight.sentiment import tuning

        original_run = tuning.run_tfidf_validation

        def changed_run(*args: object, **kwargs: object) -> dict:
            run = original_run(*args, **kwargs)
            if kwargs["model_kind"] == "linear_svm":
                run["metrics"]["phrasebank"]["validation_membership_sha256"] = "changed"
            return run

        monkeypatch.setattr(tuning, "run_tfidf_validation", changed_run)
        with pytest.raises(ValueError, match="validation_membership"):
            run_tuning(
                tmp_path / "data",
                receipt,
                plan,
                report,
                logreg_config=base,
                svm_config=base,
            )
        assert not report.exists()
        return
    summary = run_tuning(
        tmp_path / "data", receipt, plan, report, logreg_config=base, svm_config=base
    )
    assert len(fits) == 4
    assert len(summary["candidates"]) == 4
    assert summary["final_test_evaluated"] is False
    assert summary["hyperparameters_tuned"] is True
    assert old_report.read_text() == "preserved"
    for candidate in summary["candidates"]:
        run = candidate["run"]
        assert run["final_test_evaluated"] is False
        for result in run["metrics"].values():
            assert result["training_samples"] == 3
            assert (tmp_path / "data" / result["artifact_relative_path"]).exists()
    for cohort, selected in summary["selected_run_ids"].items():
        assert set(selected) == {"logistic_regression", "linear_svm", "overall"}
        winner = next(
            item
            for item in summary["candidates"]
            if item["run_id"] == selected["overall"]
        )
        assert winner["run"]["metrics"][cohort][winner["model_kind"]][
            "macro_f1"
        ] == max(
            item["run"]["metrics"][cohort][item["model_kind"]]["macro_f1"]
            for item in summary["candidates"]
        )
    assert "validationcanary" not in report.read_text()
    public_json = report.with_suffix(".json").read_text()
    assert "validationcanary" not in public_json
    assert "phrasebank/train/negative" not in public_json
