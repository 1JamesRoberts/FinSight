"""Execute the fixed six-run DeBERTa target-conditioning comparison, validation only."""

import argparse
import gc
import json
import platform
import subprocess
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
from statistics import mean, stdev

from finsight.data.target_pairs import TargetExample
from finsight.data.target_preparation import LOCK, PLAN, json_bytes
from finsight.sentiment.artifacts import file_sha256, source_manifest, verify_files
from finsight.sentiment.artifacts import sha as sha
from finsight.sentiment.experiments.target_control import (
    FINAL,
    evaluate_cohorts,
    verified_inputs,
)
from finsight.sentiment.models.target import load_fresh, score_examples
from finsight.sentiment.models.transformer import load_saved
from finsight.sentiment.training.config import FineTuneConfig
from finsight.sentiment.training.target import MODES, fit_target, membership_sha256

REPORT = Path("reports/experiments/target-deberta")
TRIALS = Path("reports/experiments/target-deberta-runs")
BASELINES = {
    "saved_deberta": (
        Path("reports/experiments/target-saved-control.json"),
        "3beb4f813d38384370f4ee14b374d4514b7b6dd2db6d5d776d04023b59bcd54b",
    ),
    "tfidf": (
        Path("reports/experiments/target-tfidf-logreg.json"),
        "05a7e2ce603ae002499a6b9c784c6be3cac61cf8e9513cd02f1ee8269e26a962",
    ),
}


def predictions_bytes(rows: list[dict]) -> bytes:
    return b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)


def context_for(mode: str, examples: tuple[TargetExample, ...]) -> dict:
    return {
        "experiment": "target-sentiment-v1",
        "input_mode": mode,
        "plan_sha256": file_sha256(PLAN),
        "lock_sha256": file_sha256(LOCK),
        "data_manifest_sha256": json.loads(LOCK.read_bytes())["data_manifest_sha256"],
        "validation_membership_sha256": membership_sha256(
            [e for e in examples if e.partition == "validation"]
        ),
        "implementation_sha256": source_manifest(),
        "lockfile_sha256": file_sha256(Path("uv.lock")),
    }


def read_trial(
    path: Path,
    data_dir: Path,
    asset: dict,
    config: FineTuneConfig,
    context: dict,
    examples: tuple[TargetExample, ...],
) -> dict:
    """Resume only fully verified completed trials, never an unchecked progress file."""
    body = path.read_bytes()
    result = json.loads(body)
    if (
        result["context"] != context
        or result["configuration"] != asdict(config)
        or result["final_test_evaluated"]
    ):
        raise ValueError("completed trial context/configuration differs")
    root = data_dir / "interim/target-deberta" / sha(body)
    if (root / "run.json").read_bytes() != body:
        raise ValueError("private trial report differs")
    data = (root / "predictions.jsonl").read_bytes()
    if sha(data) != result["predictions_sha256"]:
        raise ValueError("private trial predictions differ")
    rows = [json.loads(line) for line in data.splitlines()]
    validation = tuple(e for e in examples if e.partition == "validation")
    if len(rows) != len(validation):
        raise ValueError("trial validation length differs")
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
        raise ValueError("trial native validation membership/gold differs")
    if (
        evaluate_cohorts(rows, {e.target for e in examples if e.partition == "train"})
        != result["cohorts"]
    ):
        raise ValueError("trial cohort metrics differ")
    directory = data_dir / result["artifact_relative_path"]
    manifest_path = directory / "artifact-manifest.json"
    if file_sha256(manifest_path) != result["artifact_manifest_sha256"]:
        raise ValueError("trial artifact manifest differs")
    manifest = json.loads(manifest_path.read_bytes())
    if (
        manifest["context"] != context
        or manifest["source"] != asset
        or manifest["configuration"] != asdict(config)
    ):
        raise ValueError("trial artifact source/context differs")
    if manifest["training_membership_sha256"] != membership_sha256(
        [e for e in examples if e.partition == "train"]
    ):
        raise ValueError("trial training membership differs")
    verify_files(directory, manifest["files"])
    if manifest["epoch"] != result["fit"]["best_epoch"]:
        raise ValueError("trial selected epoch differs")
    selected = next(
        entry
        for entry in result["fit"]["history"]
        if entry["epoch"] == result["fit"]["best_epoch"]
    )
    if selected["validation"] != result["cohorts"]["all_targets"]["classification"]:
        raise ValueError("trial selected metrics differ")
    return result


def run_trial(
    data_dir: Path, mode: str, seed: int, *, reference: dict | None = None
) -> dict:
    if mode not in MODES or seed not in (42, 43, 44):
        raise ValueError("trial outside fixed comparison")
    plan, asset, examples, _ = verified_inputs(data_dir)
    config = replace(FineTuneConfig.load(Path(plan["training_config"])), seed=seed)
    context = context_for(mode, examples)
    path = TRIALS / mode / f"seed-{seed}.json"
    if path.exists():
        result = read_trial(path, data_dir, asset, config, context, examples)
        print(f"Reusing verified completed {mode} seed {seed}", flush=True)
    else:
        training = tuple(e for e in examples if e.partition == "train")
        validation = tuple(e for e in examples if e.partition == "validation")
        final_sha = file_sha256(FINAL)
        torch, tokenizer, model, initial = load_fresh(asset, config, data_dir)
        if reference and any(
            initial[key] != reference[key] for key in ("state_sha256", "rng_sha256")
        ):
            raise ValueError("matched arms have different initialization")
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
        )
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        directory = fit.pop("artifact_directory")
        torch, tokenizer, model, manifest = load_saved(
            directory, fit["artifact_manifest_sha256"], asset, config
        )
        if manifest["context"] != context:
            raise ValueError("selected checkpoint input context differs")
        rows, profile = score_examples(
            torch, tokenizer, model, validation, mode, config
        )
        cohorts = evaluate_cohorts(rows, {e.target for e in training})
        selected = next(e for e in fit["history"] if e["epoch"] == fit["best_epoch"])
        if cohorts["all_targets"]["classification"] != selected["validation"]:
            raise ValueError("reloaded selected-checkpoint metrics differ")
        body = predictions_bytes(rows)
        verified_inputs(data_dir)
        if file_sha256(FINAL) != final_sha:
            raise ValueError("original final aggregate changed")
        result = {
            "schema_version": 1,
            "experiment": "target-deberta-v1",
            "input_mode": mode,
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
            "final_test_evaluated": False,
            "training_examples": len(training),
            "training_membership_sha256": membership_sha256(training),
            "original_final_aggregate_sha256": final_sha,
            "code_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "working_tree_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], text=True
                ).strip()
            ),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": version("torch"),
                "transformers": version("transformers"),
                "sentencepiece": version("sentencepiece"),
                "cuda_runtime": torch.version.cuda,
            },
        }
        run_body = json_bytes(result)
        root = data_dir / "interim/target-deberta" / sha(run_body)
        root.mkdir(parents=True, exist_ok=False)
        (root / "predictions.jsonl").write_bytes(body)
        (root / "run.json").write_bytes(run_body)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(run_body)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        result = read_trial(path, data_dir, asset, config, context, examples)
    if reference and any(
        result["initialization"][key] != reference[key]
        for key in ("state_sha256", "rng_sha256")
    ):
        raise ValueError("completed matched initialization differs")
    return result


def aggregate_trials(trials: list[dict], cohorts: list[str]) -> dict:
    """Require every seed, matching starts and matching common epoch shuffles."""
    if len(trials) != 6 or {(r["input_mode"], r["seed"]) for r in trials} != {
        (m, s) for m in MODES for s in (42, 43, 44)
    }:
        raise ValueError("all six fixed trials required")
    for seed in (42, 43, 44):
        a, b = [
            next(r for r in trials if r["input_mode"] == mode and r["seed"] == seed)
            for mode in MODES
        ]
        if (
            a["configuration"] != b["configuration"]
            or a["initialization"]["state_sha256"]
            != b["initialization"]["state_sha256"]
            or a["initialization"]["rng_sha256"] != b["initialization"]["rng_sha256"]
        ):
            raise ValueError("matched configuration/initialization differs")
        for first, second in zip(
            a["fit"]["history"], b["fit"]["history"], strict=False
        ):
            if first["training_order_sha256"] != second["training_order_sha256"]:
                raise ValueError("matched epoch shuffles differ")
    result = {}
    for mode in MODES:
        selected = [r for r in trials if r["input_mode"] == mode]
        result[mode] = {}
        for cohort in cohorts:
            result[mode][cohort] = {}
            for metric in ("accuracy", "macro_f1", "nll", "brier", "ece"):
                family = (
                    "classification"
                    if metric in ("accuracy", "macro_f1")
                    else "calibration"
                )
                values = [r["cohorts"][cohort][family][metric] for r in selected]
                result[mode][cohort][metric] = {
                    "mean": mean(values),
                    "sample_std": stdev(values),
                }
    return result


def run_all(data_dir: Path) -> dict:
    if any(REPORT.with_suffix(s).exists() for s in (".json", ".md")):
        raise FileExistsError("comparison already published; inspect saved reports")
    plan, _, _, _ = verified_inputs(data_dir)
    for path, digest in BASELINES.values():
        if file_sha256(path) != digest:
            raise ValueError("baseline report differs")
    trials = []
    for seed in plan["seeds"]:
        first = run_trial(data_dir, MODES[0], seed)
        second = run_trial(data_dir, MODES[1], seed, reference=first["initialization"])
        trials.extend([first, second])
    result = {
        "schema_version": 1,
        "experiment": "target-deberta-comparison-v1",
        "plan_sha256": file_sha256(PLAN),
        "lock_sha256": file_sha256(LOCK),
        "final_test_evaluated": False,
        "hyperparameters_tuned": False,
        "aggregate": aggregate_trials(trials, plan["evaluation_cohorts"]),
        "trials": trials,
        "baseline_reports": {
            name: {"path": path.as_posix(), "sha256": digest}
            for name, (path, digest) in BASELINES.items()
        },
        "scope": "Exploratory development validation; matched "
        "target-conditioned versus headline-only training; raw "
        "probabilities, no temperatures; all seeds reported, no best-seed selection.",
    }
    REPORT.with_suffix(".json").write_bytes(json_bytes(result))
    REPORT.with_suffix(".md").write_bytes(render_report(result).encode())
    return result


def render_report(result: dict) -> str:
    cohorts = (
        "all_targets",
        "single_target",
        "multi_target",
        "mixed_label_headlines",
        "seen_native_target",
        "unseen_native_target",
    )
    lines = [
        "# Matched DeBERTa target-conditioning comparison",
        "",
        "Each arm starts fresh from the pinned original Microsoft DeBERTa-v3 Base.",
        "Both use the same 11,547 native target training annotations "
        "and 1,410 validation annotations.",
        "Seeds 42/43/44; LR 1e-5; FP32; batch 16/32; max five epochs; patience two.",
        "Checkpoint selection uses all-target validation macro F1; earliest exact tie.",
        "",
        "| Cohort | Headline-only F1 % | Target-pair F1 % | Difference (pp) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for cohort in cohorts:
        a, b = [result["aggregate"][m][cohort]["macro_f1"] for m in MODES]
        lines.append(
            f"| {cohort} | {a['mean'] * 100:.2f} "
            f"± {a['sample_std'] * 100:.2f} | {b['mean'] * 100:.2f} "
            f"± {b['sample_std'] * 100:.2f} | "
            f"{(b['mean'] - a['mean']) * 100:+.2f} |"
        )
    lines += [
        "",
        "Values are mean ± sample standard deviation across three "
        "seeds, not confidence intervals.",
        "Metrics weight target annotations equally; cohorts overlap. "
        "Seen/unseen means exact native target string, not verified company identity.",
        "",
        "| Arm | Seed | Selected epoch | All-target F1 % | "
        "Multi-target F1 % | Mixed-label F1 % |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for trial in sorted(result["trials"], key=lambda r: (r["input_mode"], r["seed"])):
        values = [
            trial["cohorts"][c]["classification"]["macro_f1"] * 100
            for c in cohorts[:1] + cohorts[2:4]
        ]
        lines.append(
            f"| {trial['input_mode']} | {trial['seed']} | "
            f"{trial['fit']['best_epoch']} | "
            + " | ".join(f"{v:.2f}" for v in values)
            + " |"
        )
    lines += [
        "",
        "| Arm | Accuracy % | NLL | Brier | ECE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode in MODES:
        m = result["aggregate"][mode]["all_targets"]
        lines.append(
            f"| {mode} | {m['accuracy']['mean'] * 100:.2f} | "
            f"{m['nll']['mean']:.4f} | {m['brier']['mean']:.4f} | "
            f"{m['ece']['mean']:.4f} |"
        )
    architecture = result["trials"][0]["model"]["architecture"]
    lines += [
        "",
        f"Architecture: {architecture['class']}, "
        f"{architecture['layers']} layers, hidden size {architecture['hidden_size']}, "
        f"{architecture['attention_heads']} attention heads, "
        f"{architecture['parameters']:,} parameters.",
        "Six selected safe checkpoints, input context manifests and "
        "private row predictions are retained locally.",
        "Full per-trial class metrics/confusion matrices, calibration "
        "bins, model bytes, timings, optimizer history and provenance "
        "are in the companion JSON.",
        "",
        "## Interpretation boundaries",
        "",
        "Full initial-state/RNG fingerprints and common-epoch "
        "training-order hashes match between arms for every seed.",
        "Both train on the same examples with equal annotation "
        "weights; the input representation is the intended conditioning difference.",
        "The saved Phase 3 control used only 6,299 single-target "
        "training headlines, so comparisons against it also change "
        "training data/update budgets.",
        "TF-IDF baselines use the same native annotation view but a "
        "different architecture and training procedure; their report remains separate.",
        "No masking, alias normalization, new splits, new LR search, "
        "calibration fitting or reserved-test inference occurred.",
        "These are reused validation/epoch-selection results. They do "
        "not establish independent test generalization or reliable "
        "five-company performance.",
        "No seed was selected as a winner. The original final "
        "evaluation and preparation lock remain unchanged.",
        "",
        f"Plan SHA-256: `{result['plan_sha256']}`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--arm", choices=MODES)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44))
    args = parser.parse_args()
    if (args.arm is None) != (args.seed is None):
        parser.error("--arm and --seed must be supplied together")
    if args.arm:
        run_trial(args.data_dir, args.arm, args.seed)
    else:
        run_all(args.data_dir)


if __name__ == "__main__":
    main()
