"""Audit target validation predictions and prepare a private error review."""

import argparse
import json
import math
import platform
import re
import subprocess
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

from finsight.data.company_coverage import literal_match
from finsight.data.target_pairs import LABELS, TargetExample
from finsight.data.target_preparation import json_bytes
from finsight.sentiment.finbert import file_sha256
from finsight.sentiment.target_backbones import context_for as backbone_context
from finsight.sentiment.target_backbones import load_plan as backbone_plan
from finsight.sentiment.target_control import FINAL, evaluate_cohorts, verified_inputs
from finsight.sentiment.target_deberta_experiment import (
    aggregate_trials,
    predictions_bytes,
    sha,
)
from finsight.sentiment.target_deberta_experiment import (
    context_for as deberta_context,
)
from finsight.sentiment.target_training import MODES, membership_sha256

MODELS = ("deberta_base", "roberta_base", "finbert")
SEEDS = (42, 43, 44)
REPORT = Path("reports/experiments/target-error-analysis")
SOURCES = {
    "deberta": (
        Path("reports/experiments/target-deberta.json"),
        "bb564efcff73577393e8be002c1cc50315e6950db7ae6eb699f7cdbc592e629f",
    ),
    "backbones": (
        Path("reports/experiments/target-backbones.json"),
        "3f5e32fbfcfe714cb63efd796713ffe4f3d0bf0e4309731e55bc94acca90930c",
    ),
}
CATEGORIES = (
    "target_binding",
    "alias_or_nonliteral_target",
    "comparison",
    "neutral_boundary",
    "conflicting_cues",
    "financial_context",
    "event_interpretation",
    "insufficient_context",
)
PROTOCOL = (
    "Up to five annotation examples per declared stratum, distinct headline duplicate "
    "groups within a stratum; SHA-256(target-review-v1:42:example_id) order. "
    "Deduplicate example IDs across strata, preserving all selection reasons. "
    "All nine wrong = all three target-pair seeds of all three models wrong. "
    "Persistent fix/harm requires all three headline seeds wrong/correct and all "
    "three paired seeds correct/wrong within at least one model. Comparison regex "
    "and nonliteral matching select review candidates only, not automatic error tags. "
    "Selected annotation examples may share headlines across strata. Purposive, "
    "not representative; no error-category prevalence estimate. No ensemble model."
)
COMPARATIVE = re.compile(
    r"\b(?:vs\.?|versus|outperform\w*|underperform\w*|than|while|but|despite|compared)\b",
    re.I,
)


def metrics_equal(actual: object, expected: object) -> bool:
    """Allow 1e-12 arithmetic roundoff; saved prediction bytes must match exactly."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            metrics_equal(actual[k], expected[k]) for k in actual
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            metrics_equal(a, b) for a, b in zip(actual, expected, strict=True)
        )
    if type(actual) is float and type(expected) in (int, float):
        return math.isfinite(actual) and math.isclose(
            actual, expected, rel_tol=0, abs_tol=1e-12
        )
    return type(actual) is type(expected) and actual == expected


def checked_predictions(
    trial: dict,
    path: Path,
    data_dir: Path,
    examples: tuple[TargetExample, ...],
    asset: dict,
    context: dict,
) -> list[dict]:
    """Verify native rows and saved metadata; load no model weights."""
    body = path.read_bytes()
    validation = tuple(e for e in examples if e.partition == "validation")
    if (
        not validation
        or any(e.development_only for e in validation)
        or any(e.partition not in ("train", "validation") for e in examples)
    ):
        raise ValueError("development membership required")
    if (
        json.loads(body) != trial
        or trial["context"] != context
        or trial["final_test_evaluated"] is not False
        or trial["configuration"]["seed"] != trial["seed"]
        or trial["original_final_aggregate_sha256"] != file_sha256(FINAL)
    ):
        raise ValueError("trial report context/scope differs")
    root = data_dir / "interim/target-deberta" / sha(body)
    if (root / "run.json").read_bytes() != body:
        raise ValueError("private trial receipt differs")
    data = (root / "predictions.jsonl").read_bytes()
    if sha(data) != trial["predictions_sha256"]:
        raise ValueError("prediction checksum differs")
    rows = [json.loads(line) for line in data.splitlines()]
    if len(rows) != len(validation):
        raise ValueError("validation length differs")
    expected = [
        {
            **asdict(e),
            "gold": e.label,
            "prediction": r["prediction"],
            "probabilities": r["probabilities"],
        }
        for e, r in zip(validation, rows, strict=True)
    ]
    if predictions_bytes(expected) != data:
        raise ValueError("native membership/order/gold differs")
    cohorts = evaluate_cohorts(
        rows, {e.target for e in examples if e.partition == "train"}
    )
    if not metrics_equal(cohorts, trial["cohorts"]):
        raise ValueError("saved cohort metrics differ")
    relative = Path(trial["artifact_relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe artifact path")
    manifest_body = (data_dir / relative / "artifact-manifest.json").read_bytes()
    manifest = json.loads(manifest_body)
    if (
        sha(manifest_body) != trial["artifact_manifest_sha256"]
        or manifest["source"] != asset
        or manifest["context"] != context
        or manifest["configuration"] != trial["configuration"]
        or manifest["epoch"] != trial["fit"]["best_epoch"]
        or manifest["training_membership_sha256"]
        != membership_sha256([e for e in examples if e.partition == "train"])
    ):
        raise ValueError("artifact manifest differs")
    selected = next(
        e for e in trial["fit"]["history"] if e["epoch"] == manifest["epoch"]
    )
    if selected["validation"] != cohorts["all_targets"]["classification"]:
        raise ValueError("selected checkpoint metrics differ")
    return rows


def validate_rows(rows: list[dict]) -> None:
    if not rows or len({r["example_id"] for r in rows}) != len(rows):
        raise ValueError("nonempty unique example IDs required")
    for r in rows:
        if (
            r["partition"] != "validation"
            or r["development_only"]
            or r["gold"] not in LABELS
            or set(r["predictions"]) != set(MODELS)
        ):
            raise ValueError("validation-only known models/labels required")
        for model in MODELS:
            if set(r["predictions"][model]) != set(MODES):
                raise ValueError("both matched input arms required")
            for labels in r["predictions"][model].values():
                if len(labels) != 3 or any(p not in LABELS for p in labels):
                    raise ValueError("three known seed predictions required")


def correctness(row: dict, model: str, mode: str) -> list[bool]:
    return [p == row["gold"] for p in row["predictions"][model][mode]]


def persistent_transition(row: dict, model: str, *, fix: bool) -> bool:
    head, pair = [correctness(row, model, m) for m in MODES]
    return (not any(head) and all(pair)) if fix else (all(head) and not any(pair))


def _cohort_counts(rows: list[dict]) -> dict:
    models = {}
    for model in MODELS:
        errors = Counter(
            sum(not ok for ok in correctness(r, model, MODES[1])) for r in rows
        )
        transitions = []
        for i, seed in enumerate(SEEDS):
            counts = Counter()
            for r in rows:
                head, pair = [correctness(r, model, m)[i] for m in MODES]
                counts[
                    "both_correct"
                    if head and pair
                    else "fixed"
                    if pair
                    else "harmed"
                    if head
                    else "both_wrong"
                ] += 1
            transitions.append(
                {
                    "seed": seed,
                    **{
                        k: counts[k]
                        for k in ("both_correct", "fixed", "harmed", "both_wrong")
                    },
                }
            )
        models[model] = {
            "errors_in_k_of_three_seeds": {str(k): errors[k] for k in range(4)},
            "prediction_varies_across_seeds": sum(
                len(set(r["predictions"][model][MODES[1]])) > 1 for r in rows
            ),
            "persistent_conditioning_fixes": sum(
                persistent_transition(r, model, fix=True) for r in rows
            ),
            "persistent_conditioning_harms": sum(
                persistent_transition(r, model, fix=False) for r in rows
            ),
            "recurring_error_gold_counts": {
                label: sum(
                    r["gold"] == label and not any(correctness(r, model, MODES[1]))
                    for r in rows
                )
                for label in LABELS
            },
            "conditioning_by_seed": transitions,
        }
    pairwise = []
    for a, b in combinations(MODELS, 2):
        seeds = []
        for i, seed in enumerate(SEEDS):
            counts = Counter()
            for r in rows:
                first, second = [correctness(r, m, MODES[1])[i] for m in (a, b)]
                counts[
                    "both_correct"
                    if first and second
                    else "a_only_correct"
                    if first
                    else "b_only_correct"
                    if second
                    else "both_wrong"
                ] += 1
                counts["prediction_disagreements"] += (
                    r["predictions"][a][MODES[1]][i] != r["predictions"][b][MODES[1]][i]
                )
            seeds.append(
                {
                    "seed": seed,
                    **{
                        k: counts[k]
                        for k in (
                            "both_correct",
                            "a_only_correct",
                            "b_only_correct",
                            "both_wrong",
                            "prediction_disagreements",
                        )
                    },
                }
            )
        pairwise.append({"a": a, "b": b, "by_seed": seeds})
    return {
        "examples": len(rows),
        "headlines": len({r["record_id"] for r in rows}),
        "duplicate_groups": len({r["group_id"] for r in rows}),
        "models": models,
        "all_nine_target_runs_wrong": sum(
            not any(correctness(r, m, MODES[1])[i] for m in MODELS for i in range(3))
            for r in rows
        ),
        "all_nine_target_runs_correct": sum(
            all(correctness(r, m, MODES[1])[i] for m in MODELS for i in range(3))
            for r in rows
        ),
        "any_target_run_wrong": sum(
            any(not ok for m in MODELS for ok in correctness(r, m, MODES[1]))
            for r in rows
        ),
        "pairwise": pairwise,
    }


def analyze_rows(rows: list[dict]) -> dict:
    """Compute annotation counts, never pool repeated seeds as independent examples."""
    validate_rows(rows)
    cohorts = {
        "all_targets": rows,
        "single_target": [r for r in rows if not r["multi_target"]],
        "multi_target": [r for r in rows if r["multi_target"]],
        "mixed_label_headlines": [r for r in rows if r["mixed_labels"]],
        "seen_native_target": [r for r in rows if r["seen_native_target"]],
        "unseen_native_target": [r for r in rows if not r["seen_native_target"]],
        "nonliteral_target": [
            r for r in rows if not literal_match(r["headline"], r["target"])
        ],
    }
    return {
        "cohorts": {
            name: _cohort_counts(selected) for name, selected in cohorts.items()
        }
    }


def review_sample(rows: list[dict], per_stratum: int = 5) -> list[dict]:
    validate_rows(rows)
    if type(per_stratum) is not int or per_stratum < 1:
        raise ValueError("positive integer sample size required")

    def error(r: dict) -> bool:
        return any(not ok for m in MODELS for ok in correctness(r, m, MODES[1]))

    strata = {
        "all_nine_wrong": lambda r: all(
            not any(correctness(r, m, MODES[1])) for m in MODELS
        ),
        "stable_model_disagreement": lambda r: (
            all(len(set(r["predictions"][m][MODES[1]])) == 1 for m in MODELS)
            and len({r["predictions"][m][MODES[1]][0] for m in MODELS}) > 1
        ),
        "seed_instability": lambda r: any(
            len(set(r["predictions"][m][MODES[1]])) > 1 for m in MODELS
        ),
        "conditioning_harm": lambda r: any(
            persistent_transition(r, m, fix=False) for m in MODELS
        ),
        "conditioning_fix": lambda r: any(
            persistent_transition(r, m, fix=True) for m in MODELS
        ),
        "mixed_label_error": lambda r: r["mixed_labels"] and error(r),
        "comparative_candidate_error": lambda r: (
            bool(COMPARATIVE.search(r["headline"])) and error(r)
        ),
        "nonliteral_target_error": lambda r: (
            not literal_match(r["headline"], r["target"]) and error(r)
        ),
        "neutral_boundary_error": lambda r: (
            error(r)
            and (
                r["gold"] == "neutral"
                or any("neutral" in r["predictions"][m][MODES[1]] for m in MODELS)
            )
        ),
    }
    ordered = sorted(
        rows, key=lambda r: sha(f"target-review-v1:42:{r['example_id']}".encode())
    )
    selected = {}
    for name, predicate in strata.items():
        groups = set()
        for r in ordered:
            if not predicate(r) or r["group_id"] in groups:
                continue
            groups.add(r["group_id"])
            if r["example_id"] not in selected:
                selected[r["example_id"]] = {**r, "review_strata": []}
            selected[r["example_id"]]["review_strata"].append(name)
            if len(groups) == per_stratum:
                break
    return sorted(selected.values(), key=lambda r: r["example_id"])


def summarize_annotations(packet: dict, annotations: list[dict]) -> dict:
    ids = {r["example_id"] for r in packet["examples"]}
    if (
        len(ids) != len(packet["examples"])
        or len(annotations) != len(ids)
        or {r["example_id"] for r in annotations} != ids
    ):
        raise ValueError("annotations must cover the exact sample once")
    for r in annotations:
        tags = r["categories"]
        if (
            not isinstance(tags, list)
            or not tags
            or len(tags) != len(set(tags))
            or any(t not in CATEGORIES for t in tags)
            or type(r["gold_ambiguous"]) is not bool
            or not isinstance(r["note"], str)
            or not r["note"].strip()
        ):
            raise ValueError("invalid manual annotation")
    return {
        "reviewed_annotations": len(annotations),
        "reviewed_headlines": len({r["record_id"] for r in packet["examples"]}),
        "category_counts": {
            tag: sum(tag in r["categories"] for r in annotations)
            for tag in CATEGORIES
            if any(tag in r["categories"] for r in annotations)
        },
        "suspected_gold_ambiguity": sum(r["gold_ambiguous"] for r in annotations),
        "stratum_counts": dict(
            sorted(
                Counter(
                    s for r in packet["examples"] for s in r["review_strata"]
                ).items()
            )
        ),
        "protocol": packet["protocol"],
        "packet_sha256": sha(json_bytes(packet)),
        "annotations_sha256": sha(json_bytes(annotations)),
        "reviewer": (
            "Codex exploratory qualitative review; not independent human adjudication"
        ),
        "gold_labels_changed": False,
    }


def prepare(data_dir: Path) -> tuple[dict, dict]:
    plan, deberta_asset, examples, _ = verified_inputs(data_dir)
    supplement = backbone_plan()
    studies = {}
    for key, (path, digest) in SOURCES.items():
        if file_sha256(path) != digest:
            raise ValueError("pinned source comparison differs")
        studies[key] = json.loads(path.read_bytes())
        if studies[key]["final_test_evaluated"] is not False:
            raise ValueError("development comparisons only")
    validation = tuple(e for e in examples if e.partition == "validation")
    training_targets = {e.target for e in examples if e.partition == "train"}
    joined = [
        {
            **asdict(e),
            "gold": e.label,
            "seen_native_target": e.target in training_targets,
            "predictions": {m: {} for m in MODELS},
            "confidences": {m: {} for m in MODELS},
        }
        for e in validation
    ]
    receipts = []
    for model in MODELS:
        study = (
            studies["deberta"]
            if model == "deberta_base"
            else studies["backbones"]["models"][model]
        )
        if (
            aggregate_trials(study["trials"], plan["evaluation_cohorts"])
            != study["aggregate"]
        ):
            raise ValueError("aggregate comparison differs")
        for mode in MODES:
            for seed in SEEDS:
                trial = next(
                    t
                    for t in study["trials"]
                    if t["input_mode"] == mode and t["seed"] == seed
                )
                if model == "deberta_base":
                    path = (
                        Path("reports/experiments/target-deberta-runs")
                        / mode
                        / f"seed-{seed}.json"
                    )
                    asset, context = deberta_asset, deberta_context(mode, examples)
                else:
                    path = (
                        Path("reports/experiments/target-backbones-runs")
                        / model
                        / mode
                        / f"seed-{seed}.json"
                    )
                    asset, context = (
                        supplement["models"][model]["asset"],
                        backbone_context(model, mode, examples),
                    )
                rows = checked_predictions(
                    trial, path, data_dir, examples, asset, context
                )
                receipts.append(
                    {
                        "model": model,
                        "mode": mode,
                        "seed": seed,
                        "report_sha256": file_sha256(path),
                        "predictions_sha256": trial["predictions_sha256"],
                        "artifact_manifest_sha256": trial["artifact_manifest_sha256"],
                    }
                )
                for source, prediction in zip(joined, rows, strict=True):
                    source["predictions"][model].setdefault(mode, []).append(
                        prediction["prediction"]
                    )
                    source["confidences"][model].setdefault(mode, []).append(
                        max(prediction["probabilities"].values())
                    )
    summary = analyze_rows(joined)
    selected = review_sample(joined)
    for r in selected:
        r["headline_annotations"] = [
            {"target": e.target, "label": e.label}
            for e in validation
            if e.record_id == r["record_id"]
        ]
    packet = {"protocol": PROTOCOL, "inputs": receipts, "examples": selected}
    summary.update(
        {
            "schema_version": 1,
            "scope": "native SEntFiN development validation only",
            "final_test_evaluated": False,
            "models_trained": False,
            "new_inference": False,
            "labels_changed": False,
            "ensemble_created": False,
            "inputs": receipts,
            "source_reports": {
                k: {"path": p.as_posix(), "sha256": s} for k, (p, s) in SOURCES.items()
            },
            "original_final_aggregate_sha256": file_sha256(FINAL),
            "implementation_sha256": file_sha256(Path(__file__)),
            "code_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "working_tree_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], text=True
                ).strip()
            ),
            "lockfile_sha256": file_sha256(Path("uv.lock")),
            "environment": {
                "python": platform.python_version(),
                "numpy": version("numpy"),
                "scikit-learn": version("scikit-learn"),
            },
            "verification": (
                "All 18 original private reports/prediction hashes, native ordered "
                "validation rows/gold, 108 cohort metric sets, artifact manifests "
                "and matched seed initialization/order. Metric tolerance is 1e-12; "
                "prediction bytes match exactly. Full weight bytes are not "
                "loaded in this analysis; prior Windows training verified them."
            ),
            "definitions": {
                "persistence": (
                    "Wrong in k/3 separately trained target-pair seeds; "
                    "predictions may "
                    "differ even when all seeds are wrong."
                ),
                "conditioning": (
                    "Per matched seed: fixed = headline wrong, pair correct; harmed = "
                    "headline correct, pair wrong. Persistent requires all three seeds "
                    "in each direction."
                ),
                "cohorts": (
                    "Annotation counts with separate headline/group counts; "
                    "overlapping "
                    "cohorts and repeated seeds are not independent observations. "
                    "Seen/unseen means exact native string."
                ),
                "manual_review": (
                    "Purposive qualitative sample; overlapping tags; "
                    "suspected ambiguity"
                    " is reviewer judgment, not relabeling or independent adjudication."
                ),
            },
        }
    )
    return summary, packet


def render_report(result: dict) -> str:
    """Render only explicitly allowed aggregate counts and general review findings."""
    c = result["cohorts"]["all_targets"]
    q = result["qualitative_review"]
    lines = [
        "# Phase 4 target-sentiment error analysis",
        "",
        f"Saved validation predictions only: {c['examples']:,} native target "
        f"annotations from {c['headlines']:,} headlines; DeBERTa Base, "
        "RoBERTa Base and FinBERT; seeds 42/43/44.",
        "Both matched headline-only and target-pair arms are checked (18 trials). "
        "No new inference, training, calibration fitting, ensemble or test scoring.",
        "",
        "## Shared errors and disagreement",
        "",
        "All nine target-pair runs get "
        f"**{c['all_nine_target_runs_correct']}** annotations correct and "
        f"**{c['all_nine_target_runs_wrong']}** wrong. At least one run misses "
        f"**{c['any_target_run_wrong']}** annotations.",
        "These are annotation counts, not nine independent copies of the data.",
        "",
        "| Cohort | Annotations | Headlines | All nine wrong | Any run wrong |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in (
        "all_targets",
        "single_target",
        "multi_target",
        "mixed_label_headlines",
        "seen_native_target",
        "unseen_native_target",
        "nonliteral_target",
    ):
        x = result["cohorts"][name]
        lines.append(
            f"| {name} | {x['examples']} | {x['headlines']} | "
            f"{x['all_nine_target_runs_wrong']} | {x['any_target_run_wrong']} |"
        )
    lines += [
        "",
        "Cohorts overlap. Seen/unseen refers to exact native target strings, "
        "not verified company identities. Nonliteral uses the existing whole-token "
        "matching diagnostic and does not automatically imply an alias error.",
        "",
        "| Backbone | Wrong in 1/3 seeds | Wrong in 2/3 | Wrong in 3/3 | "
        "Prediction varies | Persistent fixes | Persistent harms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in MODELS:
        x = c["models"][m]
        errors = x["errors_in_k_of_three_seeds"]
        lines.append(
            f"| {m} | {errors['1']} | {errors['2']} | {errors['3']} | "
            f"{x['prediction_varies_across_seeds']} | "
            f"{x['persistent_conditioning_fixes']} | "
            f"{x['persistent_conditioning_harms']} |"
        )
    lines += [
        "",
        "Prediction variation includes changing wrong labels. A recurring error "
        "is wrong in all three paired seeds, even if its wrong label changes. "
        "A persistent fix requires all three headline seeds wrong and all three "
        "paired seeds correct within one model; persistent harm reverses this.",
        "",
        "## What conditioning fixes and breaks",
        "",
        "| Backbone | Seed | Fixed | Harmed | Both correct | Both wrong |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in MODELS:
        for x in c["models"][m]["conditioning_by_seed"]:
            lines.append(
                f"| {m} | {x['seed']} | {x['fixed']} | {x['harmed']} | "
                f"{x['both_correct']} | {x['both_wrong']} |"
            )
    lines += [
        "",
        f"Each matched-seed row partitions the same {c['examples']:,} annotations. "
        "These "
        "correctness counts are not macro F1 and should not be pooled as independent "
        "observations. "
        + (
            "Every model/seed fixes more examples than it harms."
            if all(
                x["fixed"] > x["harmed"]
                for m in MODELS
                for x in c["models"][m]["conditioning_by_seed"]
            )
            else "Some matched seeds harm at least as many examples as they fix."
        ),
        "",
        "## Pairwise target-model disagreements",
        "",
        "| A | B | Seed | Predictions differ | A only correct | "
        "B only correct | Both wrong |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair in c["pairwise"]:
        for x in pair["by_seed"]:
            lines.append(
                f"| {pair['a']} | {pair['b']} | {x['seed']} | "
                f"{x['prediction_disagreements']} | {x['a_only_correct']} | "
                f"{x['b_only_correct']} | {x['both_wrong']} |"
            )
    lines += [
        "",
        "Matched seed numbers are descriptive comparisons, not a formal "
        "significance test. No voting or new ensemble prediction is constructed.",
        "",
        "## Qualitative review",
        "",
        f"Reviewed **{q['reviewed_annotations']} annotations from "
        f"{q['reviewed_headlines']} headlines**. Category tags overlap; counts "
        "are descriptions of the selected sample, not corpus prevalence.",
        "",
        "| Review tag | Sample annotations |",
        "| --- | ---: |",
    ]
    for tag in CATEGORIES:
        lines.append(f"| {tag} | {q['category_counts'].get(tag, 0)} |")
    lines += [
        "",
        f"**{q['suspected_gold_ambiguity']}** selected cases have suspected "
        "gold/context ambiguity. This is exploratory Codex judgment, not independent "
        "human adjudication, an estimated annotation-error rate, or a reason to "
        "drop/relabel examples after observing model failures. "
        "All gold stays unchanged.",
        "",
        "Observed patterns, paraphrased without source text:",
        "",
        "- Conditioning often separates the affected business from a quoted adviser, "
        "regulator, transaction vehicle or location that is only mentioned factually.",
        "- It can also introduce sentiment into factual mentions: advice, legal "
        "events and financing substitutions can be interpreted more strongly than "
        "the source label. There are persistent harms in every backbone.",
        "- Relative underperformance need not mean an absolute loss; resilience "
        "need not imply a price increase. Contrasting actors, historical declines, "
        "current reversals and negated advice remain difficult.",
        "- Neutral-versus-directional labels are sometimes unclear from a headline "
        "alone. Some headings omit the actual recommendation; some source labels "
        "differ from the apparent price direction. Agreement among models does "
        "not adjudicate the gold label.",
        "- Nonliteral targets in the sample include singular/plural differences. "
        "Some mismatches occur in both input formats, so they are not evidence that "
        "an alias caused the error. An unexpanded abbreviation also leaves identity "
        "uncertain. Mechanisms suggested in notes are hypotheses, not causal tests.",
        "",
        "## Method, privacy and next decisions",
        "",
        q["protocol"],
        "",
        "Sampling covers shared failures, stable inter-model disagreement, seed "
        "instability, persistent fixes/harms, mixed-label errors, comparison-regex "
        "candidates, nonliteral errors and neutral-boundary errors. Review uses "
        "all three seeds plus sibling target labels. Native text, names, IDs, row "
        "predictions and free-text reasoning remain in ignored local storage.",
        "The selection is purposive and the tags describe apparent difficulties, "
        "not proven model mechanisms. Comparison-regex hits are reviewed manually; "
        "a contrastive conjunction is not automatically a comparison category.",
        "",
        "Keep this benchmark unchanged. Record an annotation rubric and seek "
        "independent adjudication before any separately versioned label repairs. "
        "Next, predeclare target-level model selection and reserved-test comparisons; "
        "this review does not select a best seed, unlock tests or establish a winner.",
        "Existing validation F1/calibration/architecture comparisons remain in "
        "[the backbone report](target-backbones.md). Five-company coverage and "
        "automatic entity identification are still unresolved.",
        "",
        "Receipt verification covers all 18 prediction hashes, native membership/"
        "gold, 108 cohort metric sets and artifact manifests. Metric roundoff "
        "tolerance is 1e-12; prediction bytes must match exactly. Full weight bytes "
        "are not loaded in this analysis; prior Windows training verified them.",
        "",
        f"Private packet SHA-256: `{q['packet_sha256']}`.",
        "",
        f"Private annotations SHA-256: `{q['annotations_sha256']}`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--annotations", type=Path)
    args = parser.parse_args()
    summary, packet = prepare(args.data_dir)
    directory = args.data_dir / "interim/target-error-review" / sha(json_bytes(packet))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "packet.json").write_bytes(json_bytes(packet))
    (directory / "counts.json").write_bytes(json_bytes(summary))
    if args.annotations:
        annotations = json.loads(args.annotations.read_bytes())
        summary["qualitative_review"] = summarize_annotations(packet, annotations)
        (directory / "annotations.json").write_bytes(json_bytes(annotations))
        if any(REPORT.with_suffix(s).exists() for s in (".json", ".md")):
            raise FileExistsError("aggregate review already published")
        REPORT.with_suffix(".json").write_bytes(json_bytes(summary))
        REPORT.with_suffix(".md").write_bytes(render_report(summary).encode())
        print(f"Public aggregate: {REPORT.with_suffix('.json')}")
    print(f"Private review: {directory / 'packet.json'}")


if __name__ == "__main__":
    main()
