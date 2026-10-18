"""Matched RoBERTa/FinBERT supplement to the fixed DeBERTa development study."""

import argparse
import gc
import json
import platform
import subprocess
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

from finsight.data.target_pairs import TargetExample, audit_tokenization
from finsight.data.target_preparation import LOCK, PLAN, json_bytes
from finsight.sentiment.artifacts import file_sha256, source_manifest
from finsight.sentiment.experiments.target_control import (
    FINAL,
    evaluate_cohorts,
    verified_inputs,
)
from finsight.sentiment.experiments.target_deberta import (
    aggregate_trials,
    predictions_bytes,
    read_trial,
    sha,
)
from finsight.sentiment.models.finbert import (
    standardize_bert_head as standardize_bert_head,
)
from finsight.sentiment.models.target import load_backbone as load_backbone
from finsight.sentiment.models.target import score_examples
from finsight.sentiment.models.transformer import load_saved
from finsight.sentiment.training.config import FineTuneConfig
from finsight.sentiment.training.target import MODES, fit_target, membership_sha256

SUPPLEMENT = Path("configs/target-backbones-plan.json")
REPORT = Path("reports/experiments/target-backbones")
TRIALS = Path("reports/experiments/target-backbones-runs")
DEBERTA = Path("reports/experiments/target-deberta.json")


def load_plan() -> dict:
    """Keep the historical DeBERTa contract intact and verify supplemental inputs."""
    plan = json.loads(SUPPLEMENT.read_bytes())
    for path, key in (
        (PLAN, "original_plan_sha256"),
        (LOCK, "original_lock_sha256"),
        (DEBERTA, "deberta_report_sha256"),
    ):
        if file_sha256(path) != plan[key]:
            raise ValueError("original experiment lineage differs")
    if (
        file_sha256(Path("reports/experiments/transformer-tuning-runs/selection.json"))
        != plan["prior_selection_sha256"]
    ):
        raise ValueError("prior learning-rate selection differs")
    for entry in plan["models"].values():
        if (
            file_sha256(Path(entry["training_config"]))
            != entry["training_config_sha256"]
        ):
            raise ValueError("training configuration differs")
    return plan


def context_for(model_key: str, mode: str, examples: tuple[TargetExample, ...]) -> dict:
    load_plan()
    return {
        "experiment": "target-backbones-v1",
        "model_key": model_key,
        "input_mode": mode,
        "plan_sha256": file_sha256(SUPPLEMENT),
        "original_plan_sha256": file_sha256(PLAN),
        "lock_sha256": file_sha256(LOCK),
        "data_manifest_sha256": json.loads(LOCK.read_bytes())["data_manifest_sha256"],
        "validation_membership_sha256": membership_sha256(
            [e for e in examples if e.partition == "validation"]
        ),
        "implementation_sha256": source_manifest(),
        "lockfile_sha256": file_sha256(Path("uv.lock")),
    }


def run_trial(
    data_dir: Path,
    model_key: str,
    mode: str,
    seed: int,
    *,
    reference: dict | None = None,
) -> dict:
    # Validate the predeclared arm/seed before touching model weights.
    if mode not in MODES or seed not in (42, 43, 44):
        raise ValueError("trial outside fixed comparison")
    plan = load_plan()
    _, _, examples, _ = verified_inputs(data_dir)
    entry = plan["models"][model_key]
    asset = entry["asset"]
    config = replace(FineTuneConfig.load(Path(entry["training_config"])), seed=seed)
    context = context_for(model_key, mode, examples)
    path = TRIALS / model_key / mode / f"seed-{seed}.json"
    if path.exists():
        result = read_trial(path, data_dir, asset, config, context, examples)
        print(f"Reusing verified completed {mode} seed {seed}", flush=True)
    else:
        training = tuple(e for e in examples if e.partition == "train")
        validation = tuple(e for e in examples if e.partition == "validation")
        final_sha = file_sha256(FINAL)
        torch, tokenizer, model, initial = load_backbone(
            model_key, asset, config, data_dir
        )
        if reference and any(
            initial[key] != reference[key] for key in ("state_sha256", "rng_sha256")
        ):
            raise ValueError("matched arms have different initialization")
        tokenization = audit_tokenization(
            tokenizer, examples, max_length=config.max_length
        )
        print(f"{model_key} {mode} seed {seed}: verified tokenization", flush=True)
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
        # Reload the selected saved checkpoint; reports must describe saved weights.
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
            "experiment": "target-backbones-v1",
            "model_key": model_key,
            "input_mode": mode,
            "seed": seed,
            "context": context,
            "configuration": asdict(config),
            "initialization": initial,
            "tokenization": tokenization,
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


def run_all(data_dir: Path) -> dict:
    if REPORT.with_suffix(".json").exists() or REPORT.with_suffix(".md").exists():
        raise FileExistsError("supplement already published; inspect saved reports")
    plan = load_plan()
    original, _, _, _ = verified_inputs(data_dir)
    models = {}
    for key in plan["models"]:
        trials = []
        for seed in plan["seeds"]:
            first = run_trial(data_dir, key, MODES[0], seed)
            second = run_trial(
                data_dir, key, MODES[1], seed, reference=first["initialization"]
            )
            trials.extend([first, second])
        models[key] = {
            "aggregate": aggregate_trials(trials, original["evaluation_cohorts"]),
            "trials": trials,
        }
    result = {
        "schema_version": 1,
        "experiment": "target-backbones-comparison-v1",
        "plan_sha256": file_sha256(SUPPLEMENT),
        "final_test_evaluated": False,
        "hyperparameters_tuned": False,
        "models": models,
        "deberta_report_sha256": file_sha256(DEBERTA),
        "scope": plan["scope"],
        "limitations": plan["comparison_limitations"],
    }
    REPORT.with_suffix(".json").write_bytes(json_bytes(result))
    REPORT.with_suffix(".md").write_text(render_report(result), encoding="utf-8")
    return result


def render_report(result: dict) -> str:
    deberta = json.loads(DEBERTA.read_bytes())
    models = {"deberta_base": deberta, **result["models"]}
    lines = [
        "# Target conditioning across three backbones",
        "",
        "Development validation only: 11,547 training target annotations; "
        "1,410 validation annotations from 1,053 headlines. Seeds 42/43/44.",
        "",
        "| Backbone | Cohort | Headline F1 % | Target-pair F1 % | Difference pp |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for key, study in models.items():
        for cohort in study["aggregate"][MODES[0]]:
            a, b = [study["aggregate"][m][cohort]["macro_f1"] for m in MODES]
            lines.append(
                f"| {key} | {cohort} | {a['mean'] * 100:.2f} ± "
                f"{a['sample_std'] * 100:.2f} | {b['mean'] * 100:.2f} ± "
                f"{b['sample_std'] * 100:.2f} | {(b['mean'] - a['mean']) * 100:+.2f} |"
            )
    lines += [
        "",
        "Mean ± sample standard deviation across three seeds; "
        "not confidence intervals.",
        "Cohorts overlap; all metrics weight target annotations equally. Seen/unseen "
        "means exact native target string, not verified company identity.",
        "",
        "| Backbone | Arm | Accuracy % | NLL | Brier | ECE |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for key, study in models.items():
        for mode in MODES:
            m = study["aggregate"][mode]["all_targets"]
            lines.append(
                f"| {key} | {mode} | {m['accuracy']['mean'] * 100:.2f} | "
                f"{m['nll']['mean']:.4f} | {m['brier']['mean']:.4f} | "
                f"{m['ece']['mean']:.4f} |"
            )
    lines += [
        "",
        "| Backbone | Parameters | Layers | Hidden size | Heads | "
        "Selected checkpoint bytes |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, study in models.items():
        trial = study["trials"][0]
        a = trial["model"]["architecture"]
        lines.append(
            f"| {key} | {a['parameters']:,} | {a['layers']} | "
            f"{a['hidden_size']} | {a['attention_heads']} | "
            f"{trial['model_file_bytes']:,} |"
        )
    lines += [
        "",
        "| Backbone | Arm | Seed | Selected epoch | All-target F1 % |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for key, study in models.items():
        for trial in study["trials"]:
            f1 = trial["cohorts"]["all_targets"]["classification"]["macro_f1"]
            lines.append(
                f"| {key} | {trial['input_mode']} | {trial['seed']} | "
                f"{trial['fit']['best_epoch']} | {f1 * 100:.2f} |"
            )
    lines += [
        "",
        "All parameters are updated. FP32, deterministic eager attention, "
        "batch 16/32, maximum five epochs, patience two, max length 128; "
        "strict all-target validation macro F1 selects the earliest tied epoch.",
        "Previously selected learning rates: DeBERTa 1e-5, RoBERTa 2e-5, FinBERT 3e-5. "
        "No new rate search. FinBERT retains its original supervised financial head; "
        "the base backbones receive seeded new heads. Different tokenizers, rates and "
        "pretraining limit causal claims across backbones.",
        "Within each backbone, initial state/RNG and common epoch row orders match "
        "between arms. Original source checkpoints are used, "
        "with no sequential adaptation.",
        "Raw probabilities only; no calibration fitting, masking, alias normalization, "
        "five-company filtering, new split or reserved-test scoring.",
        "Reused validation guides epoch selection, so these results do not establish "
        "independent target-test generalization. FinBERT has known "
        "PhraseBank supervision; "
        "pretraining exposure to SEntFiN remains unknown.",
        "Weights and row predictions remain private. Full per-seed histories, class "
        "metrics, confusion matrices, calibration bins, timings, tokenization audits "
        "and artifact checksums are in the companion JSON and per-trial reports.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model", choices=("roberta_base", "finbert"))
    parser.add_argument("--arm", choices=MODES)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44))
    args = parser.parse_args()
    if any(v is not None for v in (args.model, args.arm, args.seed)):
        if not all(v is not None for v in (args.model, args.arm, args.seed)):
            parser.error("--model, --arm and --seed must be supplied together")
        run_trial(args.data_dir, args.model, args.arm, args.seed)
    else:
        run_all(args.data_dir)


if __name__ == "__main__":
    main()
