"""Pin an inference checkpoint from saved validation evidence, never test scores."""

import json
import math
from pathlib import Path

from finsight.sentiment.artifacts import file_sha256

CONFIG = "configs/target-roberta-inference.json"
STUDY = "reports/experiments/target-backbones.json"
ASSETS = "configs/target-backbones-plan.json"
SEEDS = (42, 43, 44)


def select_seed(scores: dict[int, float]) -> int:
    """Highest unrounded validation F1; lowest seed breaks an exact tie."""
    if set(scores) != set(SEEDS) or any(
        type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
        for v in scores.values()
    ):
        raise ValueError("three finite validation scores required")
    return max(SEEDS, key=lambda seed: (scores[seed], -seed))


def build_deployment(root: Path) -> dict:
    """Build public metadata only; do not read datasets, weights or final results."""
    study = json.loads((root / STUDY).read_bytes())["models"]["roberta_base"]
    trials = [t for t in study["trials"] if t["input_mode"] == "target_pair"]
    if sorted(t["seed"] for t in trials) != list(SEEDS):
        raise ValueError("three saved RoBERTa target-pair trials required")
    sources = {p: file_sha256(root / p) for p in (STUDY, ASSETS)}
    for trial in trials:
        relative = (
            "reports/experiments/target-backbones-runs/roberta_base/"
            f"target_pair/seed-{trial['seed']}.json"
        )
        if json.loads((root / relative).read_bytes()) != trial:
            raise ValueError("saved validation trial differs from comparison")
        sources[relative] = file_sha256(root / relative)
    scores = {
        t["seed"]: t["cohorts"]["all_targets"]["classification"]["macro_f1"]
        for t in trials
    }
    seed = select_seed(scores)
    trial = next(t for t in trials if t["seed"] == seed)
    return {
        "schema_version": 1,
        "model_key": "roberta_base",
        "input_mode": "target_pair",
        "seed": seed,
        "selected_epoch": trial["fit"]["best_epoch"],
        "selection": {
            "source": "saved_development_validation_only",
            "cohort": "all_targets",
            "metric": "macro_f1",
            "rule": "highest_unrounded_validation_macro_f1_then_lowest_seed",
            "scores": {str(s): scores[s] for s in SEEDS},
            "timing": "deployment_choice_after_final_evaluation_using_validation_only",
            "benchmark_unchanged": "published_results_remain_three_seed_averages",
        },
        "artifact": {
            "relative_path": trial["artifact_relative_path"],
            "manifest_sha256": trial["artifact_manifest_sha256"],
        },
        "source": json.loads((root / ASSETS).read_bytes())["models"]["roberta_base"][
            "asset"
        ],
        "training_configuration": trial["configuration"],
        "probabilities_calibrated": False,
        "repository_inputs": sources,
    }


def check_deployment(root: Path, config_path: Path) -> dict:
    """An edited seed, source, input mode or artifact requires a new reviewed config."""
    actual = json.loads(config_path.read_bytes())
    if actual != build_deployment(root):
        raise ValueError(
            "RoBERTa deployment configuration differs from validation evidence"
        )
    return actual
