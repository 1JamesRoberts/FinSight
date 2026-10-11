"""Bounded validation search using the existing train-only experiment runner."""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from finsight.config import Settings
from finsight.sentiment.tfidf import TfidfConfig
from finsight.sentiment.tfidf_validation import run_tfidf_validation


def digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def candidate_configs(plan: dict[str, Any], base: TfidfConfig) -> list[TfidfConfig]:
    """Validate a small explicit grid; retain all other training settings."""
    if (
        not isinstance(plan, dict)
        or set(plan) != {"schema_version", "c_values", "class_weights"}
        or type(plan["schema_version"]) is not int
        or plan["schema_version"] != 1
    ):
        raise ValueError("unsupported tuning plan")
    values, weights = plan["c_values"], plan["class_weights"]
    if (
        not isinstance(values, list)
        or not values
        or not isinstance(weights, list)
        or not weights
    ):
        raise ValueError("grid requires nonempty lists")
    candidates = [replace(base, c=c, class_weight=w) for c in values for w in weights]
    identities = [(item.c, item.class_weight) for item in candidates]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate grid candidates")
    if (1.0, None) not in identities:
        raise ValueError("grid must include the C=1 unweighted baseline")
    return candidates


def select_candidate(candidates: list[dict[str, Any]], cohort: str) -> int:
    """Highest validation macro F1; exact ties retain declared grid order."""
    if not candidates:
        raise ValueError("no candidates to select")
    return max(
        range(len(candidates)),
        key=lambda i: candidates[i]["run"]["metrics"][cohort][
            candidates[i]["model_kind"]
        ]["macro_f1"],
    )


def run_tuning(
    data_dir: Path,
    receipt_path: Path,
    plan_path: Path,
    report_path: Path,
    *,
    logreg_config: Path = Path("configs/tfidf-logreg.json"),
    svm_config: Path = Path("configs/tfidf-svm.json"),
    vader_config: Path = Path("configs/vader.json"),
) -> dict[str, Any]:
    """Fit separate corpus models; publish only aggregates and checked provenance."""
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    bases = {
        "logistic_regression": TfidfConfig.load(logreg_config),
        "linear_svm": TfidfConfig.load(svm_config),
    }
    # Validate the entire grid before starting any experiment.
    grids = {kind: candidate_configs(plan, base) for kind, base in bases.items()}
    source_paths = {
        "plan": plan_path,
        "logistic_regression_base": logreg_config,
        "linear_svm_base": svm_config,
        "tuning_implementation": Path(__file__),
    }
    source_hashes = {
        name: digest(path.read_bytes()) for name, path in source_paths.items()
    }
    directory = (
        data_dir
        / "interim/tfidf-tuning"
        / digest(json.dumps(source_hashes, sort_keys=True).encode())
    )
    directory.mkdir(parents=True, exist_ok=True)
    candidates = []
    for kind, grid in grids.items():
        for index, config in enumerate(grid):
            config_path = directory / f"{kind}-{index}.json"
            config_path.write_text(
                json.dumps({"schema_version": 1, **asdict(config)}, indent=2) + "\n"
            )
            print(
                f"Fitting {kind}: C={config.c}, class_weight={config.class_weight}",
                flush=True,
            )
            # Candidate reports remain local; preserve original public reports.
            run = run_tfidf_validation(
                data_dir,
                receipt_path,
                config_path,
                directory / f"{kind}-{index}.md",
                vader_config,
                model_kind=kind,
            )
            run_id = digest((directory / f"{kind}-{index}.json").read_bytes())
            candidates.append({"model_kind": kind, "run_id": run_id, "run": run})
    cohorts = ("phrasebank", "sentfin_single_target")
    # Reject comparisons if membership, feature setup, controls, or split changed.
    for cohort in cohorts:
        reference = candidates[0]["run"]
        for candidate in candidates:
            run = candidate["run"]
            result, original = run["metrics"][cohort], reference["metrics"][cohort]
            for key in (
                "training_membership_sha256",
                "validation_membership_sha256",
                "training_samples",
                "training_class_counts",
                "vader",
                "training_majority",
            ):
                if result[key] != original[key]:
                    raise ValueError(f"inconsistent candidate {key}")
            for key in (
                "parent_split_manifest_sha256",
                "freeze_receipt_sha256",
                "lockfile_sha256",
            ):
                if run[key] != reference[key]:
                    raise ValueError(f"inconsistent candidate {key}")
    selected = {}
    for cohort in cohorts:
        selected[cohort] = {}
        for kind in bases:
            matching = [item for item in candidates if item["model_kind"] == kind]
            winner = matching[select_candidate(matching, cohort)]
            selected[cohort][kind] = winner["run_id"]
        selected[cohort]["overall"] = candidates[select_candidate(candidates, cohort)][
            "run_id"
        ]
    summary = {
        "schema_version": 1,
        "experiment": "tfidf-validation-tuning-v1",
        "hyperparameters_tuned": True,
        "final_test_evaluated": False,
        "selection_metric": "validation macro_f1",
        "tie_rule": "Exact ties retain declared order: LR before SVM, then grid order.",
        "scope": "Separate training-only corpus fits; repeated validation "
        "selection, no CV or pooled training.",
        "plan": plan,
        "source_sha256": source_hashes,
        "candidates": candidates,
        "selected_run_ids": selected,
    }
    body = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    summary_id = digest(body.encode())
    local = data_dir / "interim/tfidf-tuning" / summary_id
    local.mkdir(parents=True, exist_ok=True)
    (local / "search.json").write_text(body)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.with_suffix(".json").write_text(body)
    report_path.write_text(render_tuning_report(summary, summary_id))
    return summary


def render_tuning_report(summary: dict[str, Any], summary_id: str) -> str:
    lines = [
        "# TF-IDF validation tuning",
        "",
        summary["scope"],
        "",
        "Selection: validation macro F1. " + summary["tie_rule"],
        "TF-IDF settings remain fixed; only C and class weighting vary.",
        "These validation scores informed selection and are not "
        "unbiased final performance estimates.",
        "The frozen test was not scored. Existing baseline reports are preserved.",
    ]
    for cohort, selected in summary["selected_run_ids"].items():
        lines += [
            "",
            f"## {cohort}",
            "",
            "| Model | C | Weight | Accuracy | Macro F1 | ms/text | KiB "
            "| Selected within model |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for candidate in summary["candidates"]:
            kind, run = candidate["model_kind"], candidate["run"]
            config, result = run["configuration"], run["metrics"][cohort]
            score = result[kind]
            lines.append(
                f"| {kind} | {config['c']} | {config['class_weight']} | "
                f"{score['accuracy']:.4f} | {score['macro_f1']:.4f} | "
                f"{result['mean_inference_ms_per_text']:.3f} | "
                f"{result['model_profile']['artifact_kib']:.2f} | "
                f"{'yes' if selected[kind] == candidate['run_id'] else ''} |"
            )
        winner = next(
            item
            for item in summary["candidates"]
            if item["run_id"] == selected["overall"]
        )
        result = winner["run"]["metrics"][cohort]
        kind = winner["model_kind"]
        baseline = next(
            item
            for item in summary["candidates"]
            if item["model_kind"] == kind
            and item["run"]["configuration"]["c"] == 1
            and item["run"]["configuration"]["class_weight"] is None
        )
        delta = (
            result[kind]["macro_f1"]
            - baseline["run"]["metrics"][cohort][kind]["macro_f1"]
        )
        lines += [
            "",
            f"Overall selected: **{kind}**, `{winner['run']['configuration']}`.",
            f"Macro F1 change versus its C=1 unweighted baseline: {delta:+.4f}.",
            f"Controls: VADER {result['vader']['macro_f1']:.4f}; training majority "
            f"{result['training_majority']['macro_f1']:.4f} macro F1.",
            "",
            "| Class | Precision | Recall | F1 |",
            "| --- | ---: | ---: | ---: |",
        ]
        for label, score in result[kind]["per_class"].items():
            lines.append(
                f"| {label} | {score['precision']:.4f} | "
                f"{score['recall']:.4f} | {score['f1']:.4f} |"
            )
        lines += [
            "",
            "Selected confusion matrix (gold rows/predicted columns; "
            "negative/neutral/positive):",
            "",
            "```text",
            *(str(row) for row in result[kind]["confusion_matrix"]),
            "```",
            "",
            f"Architecture: {result['model_profile']['pipeline']}; "
            f"{result['model_profile']['input_features']:,} features, "
            f"{result['model_profile']['classifier_parameters']:,} stored "
            "classifier parameters.",
            f"Selected artifact SHA-256: `{result['artifact_sha256']}`.",
            f"Selected original run ID: `{winner['run_id']}`.",
        ]
    lines += [
        "",
        f"Search ID: `{summary_id}`.",
        "Complete candidate configs, per-class metrics, confusion "
        "matrices, model sizes, timings,",
        "split identities, dependency versions, code provenance, and "
        "artifact hashes are in JSON.",
        "Models, vocabulary, record IDs, and predictions remain in "
        "ignored local storage.",
        "Corpus styles/annotation schemes differ; scores do not "
        "measure cross-corpus transfer.",
        "See `docs/tfidf-tuning.md` for setup, commands, selection "
        "rules, and limitations.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Settings.from_env().data_dir)
    parser.add_argument(
        "--freeze-receipt", type=Path, default=Path("configs/benchmark-freeze.json")
    )
    parser.add_argument("--plan", type=Path, default=Path("configs/tfidf-tuning.json"))
    parser.add_argument(
        "--report-path", type=Path, default=Path("reports/experiments/tfidf-tuning.md")
    )
    args = parser.parse_args()
    run_tuning(args.data_dir, args.freeze_receipt, args.plan, args.report_path)
    print(f"Tuning report saved to {args.report_path}")


if __name__ == "__main__":
    main()
