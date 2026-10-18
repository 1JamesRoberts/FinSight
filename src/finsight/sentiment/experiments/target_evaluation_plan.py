"""Create or verify an immutable target evaluation contract without test inference."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from finsight.sentiment.artifacts import file_sha256, json_body, source_manifest
from finsight.sentiment.evaluation.metrics import LABELS
from finsight.sentiment.evaluation.target_protocol import (
    ARMS,
    COMPARISONS,
    INFERENCE,
    LOCK,
    MODELS,
    PLAN,
    PROTOCOL,
    SEED_POLICY,
    SEEDS,
    SELECTION,
    SOURCE_PATHS,
    UNCERTAINTY,
    cohort_contract,
    selected_model,
    source_studies,
    trial_path,
    validate_plan,
    verify_plan,
)


def build_plan(root: Path, data_dir: Path) -> dict:
    "Bind validation-selected checkpoints and published holdout counts; no test read."
    # Use published development results and manifests only. Calling a dataset
    # loader here would unnecessarily read reserved bytes for integrity checks.
    from finsight.sentiment.experiments.target_deberta import aggregate_trials

    studies = source_studies(root)
    published = json.loads(
        (root / "reports/experiments/target-error-analysis.json").read_bytes()
    )
    for source in published["source_reports"].values():
        if file_sha256(root / source["path"]) != source["sha256"]:
            raise ValueError("reviewed comparison source differs")
    original = json.loads((root / "configs/target-sentiment-plan.json").read_bytes())
    for study in studies.values():
        if (
            aggregate_trials(study["trials"], original["evaluation_cohorts"])
            != study["aggregate"]
        ):
            raise ValueError("development aggregate differs from its seed trials")
    receipt = json.loads((root / "configs/benchmark-freeze.json").read_bytes())
    scores = {
        m: studies[m]["aggregate"]["target_pair"]["all_targets"]["macro_f1"]["mean"]
        for m in MODELS
    }
    transformers = {}
    for model in MODELS:
        runs = []
        for arm in ARMS:
            for seed in SEEDS:
                trial = next(
                    t
                    for t in studies[model]["trials"]
                    if (t["input_mode"], t["seed"]) == (arm, seed)
                )
                manifest = json.loads(
                    (
                        data_dir
                        / trial["artifact_relative_path"]
                        / "artifact-manifest.json"
                    ).read_bytes()
                )
                runs.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "report_path": trial_path(model, arm, seed),
                        "report_sha256": file_sha256(
                            root / trial_path(model, arm, seed)
                        ),
                        "configuration": trial["configuration"],
                        "selected_epoch": trial["fit"]["best_epoch"],
                        "artifact": {
                            "kind": "transformer",
                            "relative_path": trial["artifact_relative_path"],
                            "manifest_sha256": trial["artifact_manifest_sha256"],
                            "files": manifest["files"],
                        },
                    }
                )
        transformers[model] = runs
    sparse = json.loads(
        (root / "reports/experiments/target-tfidf-logreg.json").read_bytes()
    )
    counts = sparse["training_class_counts"]
    baselines = {
        f"tfidf.{arm}": {
            "relative_path": sparse["models"][arm]["artifact_relative_path"],
            "sha256": sparse["models"][arm]["artifact_sha256"],
            "configuration_sha256": sparse["configuration_sha256"],
            "seed": 42,
        }
        for arm in ("headline_only", "headline_target")
    }
    baselines["training_prior"] = {
        "counts": counts,
        "probabilities": {c: counts[c] / sum(counts.values()) for c in LABELS},
        "prediction": max(LABELS, key=counts.__getitem__),
    }
    plan = {
        "schema_version": 1,
        "protocol_id": PROTOCOL,
        "locked_date": "2026-10-17",
        "test_evaluated": False,
        "selection": {
            "rule": SELECTION,
            "validation_scores": scores,
            "selected_model": selected_model(scores),
        },
        "class_order": list(LABELS),
        "primary_metric": "macro_f1",
        "classification_metrics": [
            "accuracy",
            "macro_f1",
            "per_class_precision_recall_f1_support",
            "confusion_matrix",
        ],
        "probability_metrics": [
            "nll_clip_1e-15",
            "brier_unnormalized",
            "ece_10_equal_width_bins",
        ],
        "seed_policy": SEED_POLICY,
        "inference": INFERENCE,
        "uncertainty": UNCERTAINTY,
        "paired_comparisons": COMPARISONS,
        "difference_direction": "left_minus_right",
        "cohorts": cohort_contract(receipt),
        "transformers": transformers,
        "baselines": baselines,
        "repository_inputs": {p: file_sha256(root / p) for p in SOURCE_PATHS},
        "frozen_files": {
            (
                f"processed/benchmarks/{receipt['benchmark_manifest_sha256']}"
                "/benchmark-manifest.json"
            ): receipt["benchmark_manifest_sha256"],
            (
                f"processed/benchmarks/{receipt['benchmark_manifest_sha256']}"
                "/test-records.jsonl"
            ): receipt["records_sha256"],
        },
        "execution_gate": (
            "separate_runner_required; "
            "clean_committed_execution_code; full_artifact_and_group_"
            "disjointness_audit_before_scoring; "
            "never_overwrite_original_phase3_results"
        ),
        "failure_policy": (
            "abort_on_missing_artifacts_invalid_probabilities_members"
            "hip_or_checksum_mismatch; never_drop_failed_predictions"
        ),
        "timing": (
            "record_loading_and_prediction_separately; "
            "warmup_before_timing; "
            "log_hardware_batch_size_token_lengths_and_peak_memory"
        ),
        "limitations": [
            "validation_reused_for_epoch_selection_and_model_choice",
            "small_validation_gap_is_not_evidence_of_superiority",
            "native_financial_entities_not_verified_companies",
            "five_company_coverage_insufficient",
            "checkpoint_pretraining_exposure_unknown",
            "raw_probabilities_not_calibrated",
            "test_results_cannot_change_model_selection",
        ],
    }
    validate_plan(plan)
    return plan


def write_plan(root: Path, data_dir: Path) -> dict:
    """Create once; refuse even a partial existing contract instead of replacing it."""
    if any((root / p).exists() for p in (PLAN, LOCK)):
        raise FileExistsError("target contract already exists; verify it instead")
    plan = build_plan(root, data_dir)
    body = json_body(plan)
    lock = {
        "schema_version": 1,
        "protocol_id": PROTOCOL,
        "plan_sha256": hashlib.sha256(body).hexdigest(),
        "creation_code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "creation_working_tree_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            ).strip()
        ),
        "creation_source_sha256": source_manifest(),
        "test_scored": False,
    }
    (root / PLAN).write_bytes(body)
    (root / LOCK).write_bytes(json_body(lock))
    return verify_plan(root, data_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--write",
        action="store_true",
        help="Create once from saved validation metadata",
    )
    parser.add_argument(
        "--full-artifacts",
        action="store_true",
        help="Also hash weights and opaque frozen files; no inference",
    )
    args = parser.parse_args()
    if args.write and args.full_artifacts:
        parser.error(
            "create metadata lock first; full verification is a separate check"
        )
    result = (
        write_plan(Path.cwd(), args.data_dir)
        if args.write
        else verify_plan(Path.cwd(), args.data_dir, full_artifacts=args.full_artifacts)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
