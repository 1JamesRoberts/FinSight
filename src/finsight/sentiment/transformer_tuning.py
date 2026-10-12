"""Bounded SEntFiN learning-rate selection followed by fixed-rate seed repeats."""

import argparse
import gc
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from finsight.sentiment.finbert import file_sha256
from finsight.sentiment.finbert_training import FineTuneConfig, run_finetuning
from finsight.sentiment.transformer_comparison import run_experiment, verify_files

CONFIGS = Path("configs/transformer-tuning")
REPORT_ROOT = Path("reports/experiments/transformer-tuning-runs")
RATES = (1e-5, 2e-5, 3e-5)
SEEDS = (42, 43, 44)
MODELS = ("roberta_base", "finbert")


def validate_config(
    config: FineTuneConfig, baseline: FineTuneConfig, rate: float, seed: int
) -> None:
    """Reject accidental changes to the controlled settings."""
    if config != replace(baseline, learning_rate=rate, seed=seed):
        raise ValueError("fixed training settings differ")


def choose_rate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Strict macro-F1 improvement; exact ties keep declared candidate order."""
    if not candidates:
        raise ValueError("no learning-rate candidates")
    return max(candidates, key=lambda row: row["metrics"]["macro_f1"])


def seed_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe all three predeclared seeds, never choose the best seed."""
    if sorted(row["seed"] for row in rows) != list(SEEDS):
        raise ValueError("expected exactly the three declared seeds")
    return {
        "seeds": list(SEEDS),
        **{
            metric: {
                "mean": mean(row["metrics"][metric] for row in rows),
                "sample_std": stdev(row["metrics"][metric] for row in rows),
                "minimum": min(row["metrics"][metric] for row in rows),
                "maximum": max(row["metrics"][metric] for row in rows),
            }
            for metric in ("accuracy", "macro_f1")
        },
    }


def checked_plan() -> dict[str, Any]:
    plan = json.loads((CONFIGS / "plan.json").read_bytes())
    if (
        plan["schema_version"] != 1
        or plan["models"] != list(MODELS)
        or plan["learning_rates"] != list(RATES)
        or plan["selection_seed"] != 42
        or plan["repeat_seeds"] != list(SEEDS)
        or plan["selection_metric"] != "macro_f1"
        or plan["cohort"] != "sentfin_single_target"
    ):
        raise ValueError("unsupported bounded study plan")
    for field, path in (
        ("baseline_config_sha256", "configs/transformer-training.json"),
        ("asset_manifest_sha256", "configs/transformer-assets.json"),
        ("freeze_receipt_sha256", "configs/benchmark-freeze.json"),
    ):
        if plan[field] != file_sha256(Path(path)):
            raise ValueError("study plan input checksum differs")
    return plan


def execute_run(
    model: str, rate: float, seed: int, data_dir: Path, root: Path, resume: bool
) -> dict[str, Any]:
    tag = f"lr-{rate:.0e}-seed-{seed}".replace("e-0", "e-")
    config_path = CONFIGS / f"{tag}.json"
    config = FineTuneConfig.load(config_path)
    validate_config(
        config,
        FineTuneConfig.load(Path("configs/transformer-training.json")),
        rate,
        seed,
    )
    destination = root / model / tag
    report = destination / f"{model}.json"
    if report.exists():
        if not resume:
            raise ValueError(f"report already exists; use --resume: {report}")
        run = json.loads(report.read_bytes())
    elif model == "finbert":
        run = run_finetuning(
            data_dir,
            Path("configs/benchmark-freeze.json"),
            config_path,
            destination / "finbert.md",
        )
    else:
        run = run_experiment(model, data_dir, config_path, report_dir=destination)
    if run["configuration"] != asdict(config) or run["final_test_evaluated"]:
        raise ValueError("run configuration/test scope differs")
    receipt = json.loads(Path("configs/benchmark-freeze.json").read_bytes())
    for field in ("parent_split_manifest_sha256", "benchmark_manifest_sha256"):
        if run[field] != receipt[field]:
            raise ValueError("run split lineage differs")
    artifact = data_dir / run["artifact_relative_path"]
    if (
        file_sha256(artifact / "artifact-manifest.json")
        != run["fit"]["artifact_manifest_sha256"]
    ):
        raise ValueError("selected artifact manifest differs")
    manifest = json.loads((artifact / "artifact-manifest.json").read_bytes())
    verify_files(artifact, manifest["files"])
    identifier = file_sha256(report)
    namespace = "finbert-finetuning" if model == "finbert" else "transformer-comparison"
    private = data_dir / "interim" / namespace / identifier
    if (private / "run.json").read_bytes() != report.read_bytes() or file_sha256(
        private / "predictions.jsonl"
    ) != run["predictions_sha256"]:
        raise ValueError("private run/predictions differ")
    implementation = run.get(
        "implementation_sha256", run.get("provenance", {}).get("implementation_sha256")
    )
    for name, digest in implementation.items():
        path = (
            Path("src/finsight") / name
            if model == "finbert"
            else Path("src/finsight/sentiment") / name
        )
        if file_sha256(path) != digest:
            raise ValueError("run implementation differs; cannot resume changed code")
    entry = {
        "model": model,
        "learning_rate": rate,
        "seed": seed,
        "report_path": report.as_posix(),
        "report_sha256": identifier,
        "configuration_sha256": file_sha256(config_path),
        "metrics": run["metrics"],
        "best_epoch": run["fit"]["best_epoch"],
        "artifact_relative_path": run["artifact_relative_path"],
        "artifact_manifest_sha256": run["fit"]["artifact_manifest_sha256"],
        "training_membership_sha256": run["training_membership_sha256"],
        "validation_membership_sha256": run["validation_membership_sha256"],
    }
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    return entry


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(summary, indent=2, sort_keys=True) + "\n").encode())


def run_study(
    data_dir: Path, root: Path = REPORT_ROOT, resume: bool = False
) -> dict[str, Any]:
    plan = checked_plan()
    summary: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "transformer-learning-rate-seed-study",
        "plan": plan,
        "plan_sha256": file_sha256(CONFIGS / "plan.json"),
        "implementation_sha256": file_sha256(Path(__file__)),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "final_test_evaluated": False,
        "grid": {},
        "selected": {},
        "seed_runs": {},
        "statistics": {},
        "complete": False,
        "scope": (
            "SEntFiN validation selection; three rates at seed 42, followed by "
            "seeds 43/44 at each selected rate. Seed 42 selected grid run is "
            "reused. No seed selection, test scoring or superiority claim."
        ),
    }
    for model in MODELS:
        summary["grid"][model] = [
            execute_run(model, rate, 42, data_dir, root, resume) for rate in RATES
        ]
        summary["selected"][model] = choose_rate(summary["grid"][model])
        write_summary(root / "study.json", summary)
    # Persist both selections before any repeat-seed results are observed.
    write_summary(
        root / "selection.json",
        {
            "plan_sha256": summary["plan_sha256"],
            "selected": summary["selected"],
            "selection_seed": 42,
            "repeat_seeds_observed_before_selection": False,
        },
    )
    memberships = {
        (row["training_membership_sha256"], row["validation_membership_sha256"])
        for rows in summary["grid"].values()
        for row in rows
    }
    if len(memberships) != 1:
        raise ValueError("candidate training/validation memberships differ")
    for model in MODELS:
        selected = summary["selected"][model]
        repeats = [selected]
        for seed in SEEDS[1:]:
            entry = execute_run(
                model, selected["learning_rate"], seed, data_dir, root, resume
            )
            if (
                entry["training_membership_sha256"],
                entry["validation_membership_sha256"],
            ) not in memberships:
                raise ValueError("repeat split membership differs")
            repeats.append(entry)
            summary["seed_runs"][model] = repeats
            write_summary(root / "study.json", summary)
        summary["statistics"][model] = seed_statistics(repeats)
    summary["complete"] = True
    write_summary(root / "study.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    summary = run_study(args.data_dir, resume=args.resume)
    for model, stats in summary["statistics"].items():
        print(
            f"{model}: macro F1 mean {stats['macro_f1']['mean']:.4f}, "
            f"sample SD {stats['macro_f1']['sample_std']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
