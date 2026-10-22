"""Execute the locked nine RoBERTa trials; native development data only."""

import argparse
import gc
import json
import platform
import subprocess
from dataclasses import asdict, replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from statistics import mean, stdev

from finsight.data.target_pairs import TargetExample, TargetText
from finsight.data.target_preparation import json_bytes
from finsight.data.target_spans import VARIANTS
from finsight.sentiment.artifacts import file_sha256, sha, source_manifest
from finsight.sentiment.evaluation.target_data import development_inputs
from finsight.sentiment.experiments.target_control import evaluate_cohorts
from finsight.sentiment.experiments.target_deberta import predictions_bytes, read_trial
from finsight.sentiment.experiments.target_format_preparation import (
    LOCK,
    PLAN,
    check_preparation,
    load_plan,
)
from finsight.sentiment.models.target import load_fresh, score_examples
from finsight.sentiment.models.transformer import load_saved
from finsight.sentiment.training.config import FineTuneConfig
from finsight.sentiment.training.target import (
    fit_target,
    membership_sha256,
    validate_partitions,
)

TRIALS = Path("reports/experiments/target-format-runs")
REPORT = Path("reports/experiments/target-format")


def locked_inputs(
    root: Path, data_dir: Path, tokenizer_dir: Path
) -> tuple[dict, tuple[TargetExample, ...], dict[str, dict[str, TargetText]]]:
    """Keep native labels/identity separate from the two model text sequences."""
    report = check_preparation(root, data_dir, tokenizer_dir)
    original_train, original_val, _ = development_inputs(root, data_dir)
    original = {r["example_id"]: r for r in original_train + original_val}
    directory = data_dir / "interim/target-format" / report["data_manifest_sha256"]
    examples = []
    views = {v: {} for v in VARIANTS}
    for partition in ("train", "validation"):
        identifiers = []
        for line in (directory / f"{partition}.jsonl").read_bytes().splitlines():
            row = json.loads(line)
            native = row["example"]
            identifier = native["example_id"]
            if original.get(identifier) != native or native["partition"] != partition:
                raise ValueError("prepared native annotation differs")
            if set(row["inputs"]) != set(VARIANTS):
                raise ValueError("prepared input variants differ")
            examples.append(
                TargetExample(
                    **{**native, "review_reasons": tuple(native["review_reasons"])}
                )
            )
            identifiers.append(identifier)
            for variant in VARIANTS:
                item = row["inputs"][variant]
                views[variant][identifier] = TargetText(
                    identifier, item["headline"], item["target"]
                )
        if (
            sha(json_bytes(identifiers))
            != report["partitions"][partition]["membership_sha256"]
        ):
            raise ValueError("prepared membership/order differs")
    native_examples = tuple(examples)
    validate_partitions(
        tuple(e for e in native_examples if e.partition == "train"),
        tuple(e for e in native_examples if e.partition == "validation"),
    )
    return report, native_examples, views


def verify_matched(first: dict, second: dict) -> None:
    """The same seed must start identically and shuffle common epochs equally."""
    if first["configuration"] != second["configuration"] or any(
        first["initialization"][key] != second["initialization"][key]
        for key in ("state_sha256", "rng_sha256")
    ):
        raise ValueError("matched configuration or initialization differs")
    for a, b in zip(first["fit"]["history"], second["fit"]["history"], strict=False):
        if a["training_order_sha256"] != b["training_order_sha256"]:
            raise ValueError("matched training order differs")


def run_trial(
    root: Path,
    data_dir: Path,
    plan: dict,
    report: dict,
    examples: tuple[TargetExample, ...],
    views: dict[str, TargetText],
    variant: str,
    seed: int,
    *,
    reference: dict | None = None,
) -> dict:
    if {"variant": variant, "seed": seed} not in plan["trials"]:
        raise ValueError("trial outside locked study")
    config = replace(
        FineTuneConfig.load(root / plan["training_config_path"]), seed=seed
    )
    context = {
        "experiment": plan["experiment"],
        "model_key": "roberta_base",
        "input_mode": "target_pair",
        "input_variant": variant,
        "plan_sha256": file_sha256(root / PLAN),
        "lock_sha256": file_sha256(root / LOCK),
        "data_manifest_sha256": report["data_manifest_sha256"],
        "validation_membership_sha256": report["partitions"]["validation"][
            "membership_sha256"
        ],
        "implementation_sha256": source_manifest(),
        "lockfile_sha256": file_sha256(root / "uv.lock"),
    }
    path = root / TRIALS / variant / f"seed-{seed}.json"
    asset = plan["asset"]
    if path.exists():
        result = read_trial(path, data_dir, asset, config, context, examples)
        if result["input_variant"] != variant or result["seed"] != seed:
            raise ValueError("completed trial identity differs")
        print(f"Reusing verified {variant} seed {seed}", flush=True)
    else:
        training = tuple(e for e in examples if e.partition == "train")
        validation = tuple(e for e in examples if e.partition == "validation")
        print(f"Starting {variant} seed {seed}", flush=True)
        torch, tokenizer, model, initial = load_fresh(asset, config, data_dir)
        if reference and any(
            initial[key] != reference["initialization"][key]
            for key in ("state_sha256", "rng_sha256")
        ):
            raise ValueError("fresh matched initialization differs")
        fit = fit_target(
            torch,
            tokenizer,
            model,
            training,
            validation,
            config=config,
            data_dir=data_dir,
            asset=asset,
            context=context,
            input_views=views,
        )
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        directory = fit.pop("artifact_directory")
        torch, tokenizer, model, manifest = load_saved(
            directory, fit["artifact_manifest_sha256"], asset, config
        )
        if manifest["context"] != context:
            raise ValueError("saved selected-checkpoint context differs")
        rows, profile = score_examples(
            torch,
            tokenizer,
            model,
            validation,
            "target_pair",
            config,
            input_views=views,
        )
        cohorts = evaluate_cohorts(rows, {e.target for e in training})
        selected = next(e for e in fit["history"] if e["epoch"] == fit["best_epoch"])
        if cohorts["all_targets"]["classification"] != selected["validation"]:
            raise ValueError("saved/reloaded selected metrics differ")
        body = predictions_bytes(rows)
        result = {
            "schema_version": 1,
            "experiment": plan["experiment"],
            "model_key": "roberta_base",
            "input_mode": "target_pair",
            "input_variant": variant,
            "seed": seed,
            "context": context,
            "configuration": asdict(config),
            "initialization": initial,
            "fit": fit,
            "cohorts": cohorts,
            "model": profile,
            "artifact_relative_path": directory.relative_to(data_dir).as_posix(),
            "artifact_manifest_sha256": fit["artifact_manifest_sha256"],
            "model_file_bytes": sum(
                f["size_bytes"] for f in manifest["files"].values()
            ),
            "predictions_sha256": sha(body),
            "training_examples": len(training),
            "training_membership_sha256": membership_sha256(training),
            "final_test_evaluated": False,
            "code_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "environment": {
                "python": platform.python_version(),
                "torch": version("torch"),
                "transformers": version("transformers"),
                "cuda_runtime": torch.version.cuda,
            },
        }
        # Reuse the existing checksummed trial/prediction storage and reader.
        run_body = json_bytes(result)
        private = data_dir / "interim/target-deberta" / sha(run_body)
        private.mkdir(parents=True, exist_ok=True)
        (private / "predictions.jsonl").write_bytes(body)
        (private / "run.json").write_bytes(run_body)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(run_body)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        result = read_trial(path, data_dir, asset, config, context, examples)
    if reference:
        verify_matched(reference, result)
    return result


def aggregate(trials: list[dict], cohorts: list[str]) -> dict:
    expected = {(v, s) for v in VARIANTS for s in (42, 43, 44)}
    if (
        len(trials) != 9
        or {(r["input_variant"], r["seed"]) for r in trials} != expected
    ):
        raise ValueError("all nine unique trials required")
    for seed in (42, 43, 44):
        same_seed = [r for r in trials if r["seed"] == seed]
        for trial in same_seed[1:]:
            verify_matched(same_seed[0], trial)
    result = {}
    for variant in VARIANTS:
        selected = [r for r in trials if r["input_variant"] == variant]
        result[variant] = {}
        for cohort in cohorts:
            cells = [r["cohorts"][cohort] for r in selected]
            if any(c["classification"] is None for c in cells):
                result[variant][cohort] = None
                continue
            metrics = {
                "macro_f1": [c["classification"]["macro_f1"] for c in cells],
                "accuracy": [c["classification"]["accuracy"] for c in cells],
                **{
                    k: [c["calibration"][k] for c in cells]
                    for k in ("nll", "brier", "ece")
                },
            }
            result[variant][cohort] = {
                k: {"mean": mean(values), "sample_std": stdev(values)}
                for k, values in metrics.items()
            }
    return result


def render_report(result: dict) -> str:
    lines = [
        "# RoBERTa target-format development comparison",
        "",
        "Three seeds per format; identical native train/validation annotations. "
        "No frozen test scoring.",
        "",
        "| Format | Cohort | Macro F1 % ± pp | Difference vs pair (pp) |",
        "| --- | --- | ---: | ---: |",
    ]
    for variant in VARIANTS:
        for cohort, cell in result["aggregate"][variant].items():
            if cell is None:
                continue
            metric = cell["macro_f1"]
            baseline = result["aggregate"]["target_pair"][cohort]["macro_f1"]["mean"]
            lines.append(
                f"| {variant} | {cohort} | {100 * metric['mean']:.2f} ± "
                f"{100 * metric['sample_std']:.2f} | "
                f"{100 * (metric['mean'] - baseline):+.2f} |"
            )
    lines += [
        "",
        "Format is the original pair, name-preserving markers, or replacement "
        "in both sequences. Cohort denotes all targets, single-/multi-target "
        "headlines, mixed-label headlines, or exact original targets seen/unseen "
        "in retained training annotations; cohorts overlap. Macro F1 equally "
        "averages the three class F1 scores; values show mean percent ± sample "
        "standard deviation in percentage points across seeds, not confidence "
        "intervals. Difference is mean F1 minus the fresh pair baseline in "
        "percentage points (pp).",
        "",
        "| Format | Seed | Selected epoch | Parameters | Layers / hidden / heads "
        "| Saved bytes | Training seconds |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for trial in result["trials"]:
        a = trial["model"]["architecture"]
        lines.append(
            f"| {trial['input_variant']} | {trial['seed']} | "
            f"{trial['fit']['best_epoch']} | {a['parameters']:,} | {a['layers']} / "
            f"{a['hidden_size']} / {a['attention_heads']} | "
            f"{trial['model_file_bytes']:,} | {trial['fit']['training_seconds']:.1f} |"
        )
    lines += [
        "",
        "Format uses the definitions above. Seed fixes initialization and "
        "shuffling. Selected epoch maximizes validation macro F1, keeping the "
        "earliest tie. Parameters counts all model parameters. Layers/hidden/heads "
        "are encoder depth, hidden-vector width and attention heads per layer. "
        "Saved bytes includes selected model/tokenizer files. Training seconds "
        "includes optimization, epoch validation and checkpoint saving, excluding "
        "initial loading and final reload scoring.",
        "",
        "The JSON includes all histories, class metrics/confusion matrices, "
        "raw NLL/Brier/10-bin ECE, same-seed F1 differences, architecture, timing "
        "and artifact/prediction provenance. Identical starts and common-epoch "
        "row orders are verified. Full predictions and weights stay private. "
        "This is reused development validation, not independent evaluation; "
        "masking does not remove all identity clues. No production model is replaced.",
        "",
    ]
    return "\n".join(lines)


def write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(
        json_bytes({**state, "updated_utc": datetime.now(UTC).isoformat()})
    )
    temporary.replace(path)


def run_all(root: Path, data_dir: Path, tokenizer_dir: Path, state_path: Path) -> dict:
    plan, _ = load_plan(root)
    report, examples, views = locked_inputs(root, data_dir, tokenizer_dir)
    state = {
        "status": "running",
        "completed_trials": 0,
        "total_trials": 9,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
    }
    trials = []
    try:
        for entry in plan["trials"]:
            state["current_trial"] = entry
            write_state(state_path, state)
            reference = next((r for r in trials if r["seed"] == entry["seed"]), None)
            trials.append(
                run_trial(
                    root,
                    data_dir,
                    plan,
                    report,
                    examples,
                    views[entry["variant"]],
                    entry["variant"],
                    entry["seed"],
                    reference=reference,
                )
            )
            state["completed_trials"] = len(trials)
            write_state(state_path, state)
        result = {
            "experiment": plan["experiment"],
            "plan_sha256": file_sha256(root / PLAN),
            "lock_sha256": file_sha256(root / LOCK),
            "final_test_evaluated": False,
            "aggregate": aggregate(trials, plan["evaluation_cohorts"]),
            "paired_seed_differences": {
                variant: {
                    cohort: [
                        next(
                            r
                            for r in trials
                            if r["seed"] == seed and r["input_variant"] == variant
                        )["cohorts"][cohort]["classification"]["macro_f1"]
                        - next(
                            r
                            for r in trials
                            if r["seed"] == seed and r["input_variant"] == "target_pair"
                        )["cohorts"][cohort]["classification"]["macro_f1"]
                        for seed in plan["seeds"]
                    ]
                    for cohort in plan["evaluation_cohorts"]
                }
                for variant in VARIANTS[1:]
            },
            "trials": trials,
        }
        # Revalidate immutable preparation and code after the last trial.
        check_preparation(root, data_dir, tokenizer_dir)
        if any(
            r["context"]["implementation_sha256"] != source_manifest() for r in trials
        ):
            raise ValueError("execution code changed during training")
        (root / REPORT.with_suffix(".json")).write_bytes(json_bytes(result))
        (root / REPORT.with_suffix(".md")).write_text(
            render_report(result), encoding="utf-8"
        )
        state.update(status="complete", report=str(root / REPORT.with_suffix(".json")))
        write_state(state_path, state)
        return result
    except Exception as error:
        state.update(status="failed", error=repr(error))
        write_state(state_path, state)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    # Existing source_manifest() and checkpoint helpers resolve the active checkout.
    run_all(
        Path.cwd(),
        args.data_dir.resolve(),
        args.tokenizer_dir.resolve(),
        args.state.resolve(),
    )


if __name__ == "__main__":
    main()
