"""Validate the Phase 4 target evaluation contract without reading test records."""

import copy
import json
from pathlib import Path
from typing import Any

from finsight.sentiment.artifacts import file_sha256
from finsight.sentiment.evaluation.metrics import LABELS
from finsight.sentiment.evaluation.protocol import safe_path, verify_artifact

PROTOCOL = "phase4-sentfin-multi-target-v2"
PLAN = "configs/target-evaluation-plan-v2.json"
LOCK = "configs/target-evaluation-lock-v2.json"
MODELS = ("deberta_base", "roberta_base", "finbert")
ARMS = ("matched_headline_only", "target_pair")
SEEDS = (42, 43, 44)
# Reviewed before any Phase 4 test prediction. Hashes identify existing frozen
# records without publishing source text or changing the original benchmark.
PREVIOUSLY_CONSUMED_GROUP = {
    "group_id": "6557c8658a8e203775339dd7fb516ed4a84d25cde55cd4797a3350e98f2ed634",
    "reserved_text_sha256": (
        "3de3fd4443ad28f61c1da3f11f1ed3e80645288b26dc86675a08270318fbf2fc"
    ),
    "consumed_text_sha256": (
        "b34021cdc85cd1f377fc7385a634906072a88ae79b2989c8b8a56e41aba97919"
    ),
    "reserved_class_counts": {"neutral": 2},
    "reason": "near_duplicate_group_previously_evaluated_in_phase3",
}
SOURCE_PATHS = (
    "configs/target-evaluation-plan.json",
    "configs/target-evaluation-lock.json",
    "configs/benchmark-freeze.json",
    "configs/target-sentiment-plan.json",
    "configs/target-sentiment-lock.json",
    "configs/target-backbones-plan.json",
    "configs/transformer-assets.json",
    "configs/final-evaluation-plan.json",
    "configs/final-evaluation-lock.json",
    "reports/experiments/phase3-final.json",
    "reports/experiments/target-deberta.json",
    "reports/experiments/target-backbones.json",
    "reports/experiments/target-tfidf-logreg.json",
    "reports/experiments/target-error-analysis.json",
    "uv.lock",
)
SELECTION = {
    "data": "development_validation_only",
    "cohort": "all_targets",
    "metric": "macro_f1",
    "arm": "target_pair",
    "aggregation": "arithmetic_mean_of_three_seed_metrics",
    "tie_order": list(MODELS),
    "best_seed_selection": False,
    "interpretation": "validation-selected candidate; not an independent test winner",
}
SEED_POLICY = {
    "seeds": list(SEEDS),
    "report_each_seed": True,
    "metric_aggregation": "mean_and_sample_standard_deviation",
    "ensemble": False,
    "retrain": False,
    "train_plus_validation": False,
}
INFERENCE = {
    "device": "cuda",
    "batch_size": 32,
    "max_length": 128,
    "precision": "float32",
    "attention": "eager",
    "deterministic": True,
    "tf32": False,
    "calibration": "none_raw_probabilities",
    "target_pair": "native_target_first_headline_second_only_second_truncation",
    "headline_only": "one_prediction_per_headline_broadcast_to_native_targets",
    "class_ties": list(LABELS),
}
UNCERTAINTY = {
    "unit": "headline_duplicate_group",
    "paired": True,
    "replicates": 1000,
    "seed": 42,
    "interval": "percentile_95",
    "resample_training_seeds": False,
    "seed_aggregation": "mean_of_seed_metrics_within_each_replicate",
    "subgroups": "retain_all_targets_from_each_sampled_group",
    "secondary_intervals": "exploratory_without_multiplicity_correction",
}
COMPARISONS = [
    {
        "id": "roberta_conditioning",
        "role": "primary",
        "left": "roberta_base.target_pair",
        "right": "roberta_base.matched_headline_only",
    },
    {
        "id": "deberta_conditioning",
        "role": "secondary",
        "left": "deberta_base.target_pair",
        "right": "deberta_base.matched_headline_only",
    },
    {
        "id": "finbert_conditioning",
        "role": "secondary",
        "left": "finbert.target_pair",
        "right": "finbert.matched_headline_only",
    },
    {
        "id": "roberta_vs_deberta",
        "role": "secondary",
        "left": "roberta_base.target_pair",
        "right": "deberta_base.target_pair",
    },
    {
        "id": "roberta_vs_finbert",
        "role": "secondary",
        "left": "roberta_base.target_pair",
        "right": "finbert.target_pair",
    },
    {
        "id": "roberta_vs_sparse",
        "role": "secondary",
        "left": "roberta_base.target_pair",
        "right": "tfidf.headline_target",
    },
    {
        "id": "sparse_conditioning",
        "role": "secondary",
        "left": "tfidf.headline_target",
        "right": "tfidf.headline_only",
    },
]


def selected_model(scores: dict[str, float]) -> str:
    """Use unrounded validation means; exact ties follow the fixed incumbent order."""
    if set(scores) != set(MODELS) or any(
        type(v) not in (int, float) or not 0 <= v <= 1 for v in scores.values()
    ):
        raise ValueError("three finite validation scores required")
    return max(MODELS, key=scores.__getitem__)


def source_studies(root: Path) -> dict[str, dict]:
    deberta = json.loads(
        (root / "reports/experiments/target-deberta.json").read_bytes()
    )
    backbones = json.loads(
        (root / "reports/experiments/target-backbones.json").read_bytes()
    )
    return {"deberta_base": deberta, **backbones["models"]}


def trial_path(model: str, arm: str, seed: int) -> str:
    namespace = (
        "target-deberta-runs"
        if model == "deberta_base"
        else f"target-backbones-runs/{model}"
    )
    return f"reports/experiments/{namespace}/{arm}/seed-{seed}.json"


def cohort_contract(receipt: dict) -> dict:
    """Copy already-published counts; do not open frozen JSONL to derive new cohorts."""
    summary = copy.deepcopy(receipt["summary"]["cohorts"])
    # The reviewed correction removes one same-label headline, not mixed labels.
    summary["multi_target"]["headlines"] -= 1
    summary["multi_target"]["target_annotations"] -= 2
    summary["multi_target"]["class_counts"]["neutral"] -= 2
    return {
        "primary": "multi_target",
        "expected": {
            "multi_target": summary["multi_target"],
            "mixed_label_headlines": summary["mixed_target_sentiment"],
        },
        "secondary": [
            "mixed_label_headlines",
            "same_label_headlines",
            "seen_native_target",
            "unseen_native_target",
        ],
        "seen_target_rule": "exact_native_target_string_present_as_training_target",
        "single_target": "excluded_already_evaluated_in_phase3",
        "phrasebank": "excluded_remains_reserved",
        "membership": "frozen_multi_target_excluding_locked_previously_consumed_group",
        "excluded_previously_consumed_groups": [
            copy.deepcopy(PREVIOUSLY_CONSUMED_GROUP)
        ],
        "labels": "preserve_all_native_labels_no_post_prediction_exclusions",
        "group_audit": (
            "before_scoring_fail_on_overlap_with_development_or_consu"
            "med_single_target_groups"
        ),
        "per_target": (
            "private_descriptive_support_and_fixed_class_metrics; "
            "flag_support_below_20; no_company_generalization_claim"
        ),
    }


def validate_plan(plan: dict[str, Any]) -> None:
    """Reject changes to test scope, comparisons, seeds or confidence policy."""
    if (
        plan["schema_version"] != 1
        or plan["protocol_id"] != PROTOCOL
        or plan["class_order"] != list(LABELS)
        or plan["selection"]["rule"] != SELECTION
        or plan["selection"]["selected_model"]
        != selected_model(plan["selection"]["validation_scores"])
        or plan["seed_policy"] != SEED_POLICY
        or plan["inference"] != INFERENCE
        or plan["uncertainty"] != UNCERTAINTY
        or plan["paired_comparisons"] != COMPARISONS
        or plan["primary_metric"] != "macro_f1"
        or plan["difference_direction"] != "left_minus_right"
        or plan["test_evaluated"] is not False
        or set(plan["transformers"]) != set(MODELS)
        or set(plan["baselines"])
        != {"tfidf.headline_only", "tfidf.headline_target", "training_prior"}
        or set(plan["repository_inputs"]) != set(SOURCE_PATHS)
    ):
        raise ValueError("unsupported target evaluation contract")
    artifacts = set()
    for model in MODELS:
        runs = plan["transformers"][model]
        if [(r["arm"], r["seed"]) for r in runs] != [
            (a, s) for a in ARMS for s in SEEDS
        ]:
            raise ValueError("both arms and all three seeds required")
        for run in runs:
            if (
                run["configuration"]["seed"] != run["seed"]
                or run["configuration"]["max_length"] != 128
                or run["report_path"] != trial_path(model, run["arm"], run["seed"])
                or run["artifact"]["kind"] != "transformer"
            ):
                raise ValueError("checkpoint identity differs")
            artifacts.add(run["artifact"]["manifest_sha256"])
    if len(artifacts) != 18:
        raise ValueError("18 distinct selected checkpoints required")


def verify_plan(root: Path, data_dir: Path, *, full_artifacts: bool = False) -> dict:
    "Verify public receipts and checkpoint metadata; never decode or score test data."
    plan_path = root / PLAN
    lock = json.loads((root / LOCK).read_bytes())
    if (
        lock["schema_version"] != 1
        or lock["test_scored"] is not False
        or lock["protocol_id"] != PROTOCOL
        or lock["plan_sha256"] != file_sha256(plan_path)
    ):
        raise ValueError("target evaluation lock differs")
    plan = json.loads(plan_path.read_bytes())
    validate_plan(plan)
    for relative, digest in plan["repository_inputs"].items():
        if file_sha256(safe_path(root, relative)) != digest:
            raise ValueError(f"source input differs: {relative}")
    receipt = json.loads((root / "configs/benchmark-freeze.json").read_bytes())
    if plan["cohorts"] != cohort_contract(receipt):
        raise ValueError("reserved cohort contract differs")
    expected_frozen = {
        (
            f"processed/benchmarks/{receipt['benchmark_manifest_sha256']}"
            "/benchmark-manifest.json"
        ): receipt["benchmark_manifest_sha256"],
        (
            f"processed/benchmarks/{receipt['benchmark_manifest_sha256']}"
            "/test-records.jsonl"
        ): receipt["records_sha256"],
    }
    if plan["frozen_files"] != expected_frozen:
        raise ValueError("frozen file identities differ")
    studies = source_studies(root)
    scores = {
        m: studies[m]["aggregate"]["target_pair"]["all_targets"]["macro_f1"]["mean"]
        for m in MODELS
    }
    if plan["selection"]["validation_scores"] != scores:
        raise ValueError("validation selection evidence differs")
    checked = 0
    for model, runs in plan["transformers"].items():
        for run in runs:
            path = safe_path(root, run["report_path"])
            trial = next(
                t
                for t in studies[model]["trials"]
                if (t["input_mode"], t["seed"]) == (run["arm"], run["seed"])
            )
            if (
                json.loads(path.read_bytes()) != trial
                or file_sha256(path) != run["report_sha256"]
            ):
                raise ValueError("source trial differs")
            directory = safe_path(data_dir, run["artifact"]["relative_path"])
            checked += verify_artifact(data_dir, run["artifact"], full=full_artifacts)
            manifest = json.loads((directory / "artifact-manifest.json").read_bytes())
            if (
                run["configuration"] != trial["configuration"]
                or run["artifact"]["relative_path"] != trial["artifact_relative_path"]
                or run["artifact"]["manifest_sha256"]
                != trial["artifact_manifest_sha256"]
                or run["selected_epoch"] != trial["fit"]["best_epoch"]
                or manifest["epoch"] != run["selected_epoch"]
                or manifest["configuration"] != run["configuration"]
                or manifest["context"] != trial["context"]
                or manifest["training_membership_sha256"]
                != trial["training_membership_sha256"]
            ):
                raise ValueError("selected checkpoint ancestry differs")
    sparse = json.loads(
        (root / "reports/experiments/target-tfidf-logreg.json").read_bytes()
    )
    for arm in ("headline_only", "headline_target"):
        source = sparse["models"][arm]
        baseline = plan["baselines"][f"tfidf.{arm}"]
        if baseline != {
            "relative_path": source["artifact_relative_path"],
            "sha256": source["artifact_sha256"],
            "configuration_sha256": sparse["configuration_sha256"],
            "seed": 42,
        }:
            raise ValueError("sparse baseline identity differs")
        if (
            full_artifacts
            and file_sha256(safe_path(data_dir, baseline["relative_path"]))
            != baseline["sha256"]
        ):
            raise ValueError("sparse baseline checksum differs")
    counts = sparse["training_class_counts"]
    if plan["baselines"]["training_prior"] != {
        "counts": counts,
        "probabilities": {c: counts[c] / sum(counts.values()) for c in LABELS},
        "prediction": max(LABELS, key=counts.__getitem__),
    }:
        raise ValueError("training-only prior differs")
    if full_artifacts:
        # Hash frozen bytes as opaque files: this still never parses their text/gold.
        for relative, digest in plan["frozen_files"].items():
            if file_sha256(safe_path(data_dir, relative)) != digest:
                raise ValueError("frozen file checksum differs")
    return {
        "protocol_id": PROTOCOL,
        "selected_model": plan["selection"]["selected_model"],
        "transformer_checkpoints": 18,
        "full_artifacts_verified": full_artifacts,
        "weight_files_checked": checked,
        "test_records_decoded": False,
        "test_scored": False,
    }
