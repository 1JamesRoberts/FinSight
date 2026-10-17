"""Target error review uses all seeds, private examples and aggregate-only outputs."""

from copy import deepcopy

import pytest

from finsight.sentiment.target_error_analysis import (
    CATEGORIES,
    MODELS,
    analyze_rows,
    review_sample,
    summarize_annotations,
)


def row(i: int, *, headline_correct: bool, target_correct: bool) -> dict:
    return {
        "example_id": str(i),
        "record_id": str(i // 2),
        "group_id": str(i // 2),
        "partition": "validation",
        "development_only": False,
        "headline": "Private alpha gains while beta falls",
        "target": "alpha",
        "gold": "positive",
        "multi_target": True,
        "mixed_labels": True,
        "seen_native_target": True,
        "review_strata": ["conditioning_harm"],
        "predictions": {
            m: {
                "matched_headline_only": [
                    "positive" if headline_correct else "negative"
                ]
                * 3,
                "target_pair": ["positive" if target_correct else "neutral"] * 3,
            }
            for m in MODELS
        },
    }


def test_counts_persistence_disagreement_and_both_transition_directions():
    rows = [
        row(0, headline_correct=False, target_correct=True),
        row(1, headline_correct=True, target_correct=False),
        row(2, headline_correct=False, target_correct=False),
    ]
    rows[2]["predictions"][MODELS[0]]["target_pair"] = [
        "positive",
        "neutral",
        "positive",
    ]
    result = analyze_rows(rows)
    all_rows = result["cohorts"]["all_targets"]
    assert all_rows["examples"] == 3
    assert all_rows["all_nine_target_runs_wrong"] == 1
    m = all_rows["models"][MODELS[0]]
    assert m["errors_in_k_of_three_seeds"] == {"0": 1, "1": 1, "2": 0, "3": 1}
    assert m["prediction_varies_across_seeds"] == 1
    assert m["persistent_conditioning_fixes"] == 1
    assert m["persistent_conditioning_harms"] == 1
    assert m["conditioning_by_seed"][0]["fixed"] == 2
    assert m["conditioning_by_seed"][0]["harmed"] == 1
    assert all_rows["pairwise"][0]["by_seed"][0]["prediction_disagreements"] == 1


@pytest.mark.parametrize("bad", ["reserved", "duplicate", "class", "seed", "devonly"])
def test_analysis_rejects_wrong_scope_or_malformed_predictions(bad):
    rows = [row(0, headline_correct=False, target_correct=False)]
    if bad == "reserved":
        rows[0]["partition"] = "test"
    elif bad == "duplicate":
        rows *= 2
    elif bad == "class":
        rows[0]["predictions"][MODELS[0]]["target_pair"][0] = "other"
    elif bad == "seed":
        rows[0]["predictions"][MODELS[0]]["target_pair"].pop()
    else:
        rows[0]["development_only"] = True
    with pytest.raises(ValueError):
        analyze_rows(rows)


def test_sampling_deterministic_deduplicated_and_retains_strata():
    rows = [row(i, headline_correct=False, target_correct=False) for i in range(20)]
    rows += [row(i, headline_correct=True, target_correct=False) for i in range(20, 40)]
    rows += [row(i, headline_correct=False, target_correct=True) for i in range(40, 60)]
    original = deepcopy(rows)
    selected = review_sample(rows, per_stratum=2)
    assert selected == review_sample(rows[::-1], per_stratum=2)
    assert len({r["example_id"] for r in selected}) == len(selected)
    assert rows == original
    strata = {s for r in selected for s in r["review_strata"]}
    assert {"all_nine_wrong", "conditioning_harm", "conditioning_fix"} <= strata
    assert all(isinstance(s, str) for s in strata)
    with pytest.raises(ValueError):
        review_sample(rows, per_stratum=0)


def test_manual_review_exact_coverage_multilabel_privacy_and_no_mutation():
    packet = {
        "protocol": "fixed",
        "examples": [row(0, headline_correct=True, target_correct=False)],
    }
    original = deepcopy(packet)
    annotation = {
        "example_id": "0",
        "categories": [CATEGORIES[0], CATEGORIES[1]],
        "gold_ambiguous": True,
        "note": "Private reviewer reasoning",
    }
    result = summarize_annotations(packet, [annotation])
    assert result["reviewed_annotations"] == 1
    assert result["suspected_gold_ambiguity"] == 1
    assert result["category_counts"] == {CATEGORIES[0]: 1, CATEGORIES[1]: 1}
    assert packet == original
    assert not result["gold_labels_changed"]
    import json

    public = json.dumps(result)
    assert all(
        secret not in public
        for secret in ("Private alpha", "Private reviewer", '"example_id"', '"target"')
    )
    for annotations in (
        [],
        [annotation] * 2,
        [{**annotation, "categories": ["other"]}],
        [{**annotation, "note": ""}],
        [{**annotation, "gold_ambiguous": 1}],
        [{**annotation, "categories": [CATEGORIES[0]] * 2}],
    ):
        with pytest.raises(ValueError):
            summarize_annotations(packet, annotations)


def test_metric_roundoff_is_tight_and_counts_still_exact():
    from finsight.sentiment.target_error_analysis import metrics_equal

    assert metrics_equal({"brier": 0.2 + 5e-17, "count": 8}, {"brier": 0.2, "count": 8})
    assert not metrics_equal({"brier": 0.2 + 1e-8}, {"brier": 0.2})
    assert not metrics_equal({"count": 8}, {"count": 9})
    assert not metrics_equal(float("nan"), float("nan"))
    assert not metrics_equal([1, 2], [1])


@pytest.mark.parametrize(
    "corruption",
    [None, "checksum", "gold", "order", "private", "scope", "manifest", "metrics"],
)
def test_receipts_verify_native_membership_gold_metrics_and_manifest(
    tmp_path, monkeypatch, corruption
):
    import json
    from dataclasses import asdict

    from finsight.data.target_pairs import TargetExample
    from finsight.data.target_preparation import json_bytes
    from finsight.sentiment import target_error_analysis as a
    from finsight.sentiment.target_control import evaluate_cohorts
    from finsight.sentiment.target_deberta_experiment import predictions_bytes, sha
    from finsight.sentiment.target_training import membership_sha256

    final = tmp_path / "final.json"
    final.write_bytes(b"original final receipt")
    monkeypatch.setattr(a, "FINAL", final)
    examples = tuple(
        TargetExample(
            str(i),
            str(i),
            str(i),
            "train" if i < 3 else "validation",
            0,
            "alpha",
            f"Synthetic {i}",
            ("negative", "neutral", "positive")[i % 3],
            False,
            False,
            False,
            (),
        )
        for i in range(5)
    )
    val = [e for e in examples if e.partition == "validation"]
    rows = [
        {
            **asdict(e),
            "gold": e.label,
            "prediction": e.label,
            "probabilities": {label: float(label == e.label) for label in a.LABELS},
        }
        for e in val
    ]
    cohorts = evaluate_cohorts(rows, {"alpha"})
    if corruption == "gold":
        rows[0]["gold"] = "positive"
    if corruption == "order":
        rows.reverse()
    data = predictions_bytes(rows)
    context = {"input_mode": "target_pair"}
    manifest = {
        "source": {},
        "context": context,
        "configuration": {"seed": 42},
        "epoch": 1,
        "training_membership_sha256": membership_sha256(examples[:3]),
    }
    if corruption == "manifest":
        manifest["source"] = {"different": True}
    artifact = tmp_path / "model"
    artifact.mkdir()
    manifest_body = json_bytes(manifest)
    (artifact / "artifact-manifest.json").write_bytes(manifest_body)
    trial = {
        "context": context,
        "configuration": {"seed": 42},
        "seed": 42,
        "final_test_evaluated": corruption == "scope",
        "original_final_aggregate_sha256": sha(final.read_bytes()),
        "predictions_sha256": sha(data),
        "cohorts": cohorts,
        "artifact_relative_path": "model",
        "artifact_manifest_sha256": sha(manifest_body),
        "fit": {
            "best_epoch": 1,
            "history": [
                {"epoch": 1, "validation": cohorts["all_targets"]["classification"]}
            ],
        },
    }
    if corruption == "metrics":
        trial["cohorts"]["all_targets"]["examples"] += 1
    body = json_bytes(trial)
    path = tmp_path / "trial.json"
    path.write_bytes(body)
    root = tmp_path / "interim/target-deberta" / sha(body)
    root.mkdir(parents=True)
    (root / "run.json").write_bytes(body + (b" " if corruption == "private" else b""))
    (root / "predictions.jsonl").write_bytes(
        data + (b" " if corruption == "checksum" else b"")
    )
    if corruption:
        with pytest.raises(ValueError):
            a.checked_predictions(trial, path, tmp_path, examples, {}, context)
    else:
        checked = a.checked_predictions(trial, path, tmp_path, examples, {}, context)
        assert checked == json.loads("[" + ",".join(data.decode().splitlines()) + "]")


def test_public_renderer_does_not_include_private_review_fields():
    from finsight.sentiment.target_error_analysis import render_report

    source = row(0, headline_correct=True, target_correct=False)
    packet = {"protocol": "fixed sample", "examples": [source]}
    annotations = [
        {
            "example_id": "0",
            "categories": [CATEGORIES[0]],
            "gold_ambiguous": False,
            "note": "Private reasoning to exclude",
        }
    ]
    result = analyze_rows([source])
    result["qualitative_review"] = summarize_annotations(packet, annotations)
    body = render_report(result)
    assert all(
        secret not in body
        for secret in (source["headline"], "Private reasoning", '"example_id"')
    )
    assert "**1** wrong" in body
    assert "Every model/seed fixes more" not in body
    assert "Some matched seeds harm" in body
    assert "not corpus prevalence" in body
