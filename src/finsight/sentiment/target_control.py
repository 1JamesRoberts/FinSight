"""Evaluate saved headline-only DeBERTa controls on native target validation rows."""

import argparse
import gc
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from statistics import mean, stdev

from finsight.data.benchmark import load_development_records
from finsight.data.target_pairs import TargetExample, build_examples, examples_bytes
from finsight.data.target_preparation import (
    LOCK,
    OUTPUT,
    PLAN,
    RECEIPT,
    json_bytes,
    load_plan,
)
from finsight.sentiment.finbert import file_sha256
from finsight.sentiment.finbert_training import FineTuneConfig, sentfin_partitions
from finsight.sentiment.transformer_analysis import calibration_metrics, checked_run
from finsight.sentiment.transformer_comparison import (
    load_saved,
    score_model,
    verify_files,
)
from finsight.sentiment.validation import classification_metrics

REPORT = Path("reports/experiments/target-saved-control")
FINAL = Path("reports/experiments/phase3-final.json")


def join_predictions(
    examples: tuple[TargetExample, ...], predictions: dict[str, dict]
) -> list[dict]:
    """Replicate one prediction per source headline, preserving each native gold."""
    if not examples or any(e.partition != "validation" for e in examples):
        raise ValueError("nonempty validation examples only")
    if len({e.example_id for e in examples}) != len(examples):
        raise ValueError("duplicate example IDs")
    if set(predictions) != {e.record_id for e in examples}:
        raise ValueError("headline prediction membership differs")
    texts: dict[str, str] = {}
    rows = []
    for e in examples:
        if texts.setdefault(e.record_id, e.headline) != e.headline:
            raise ValueError("headline ID has inconsistent text")
        p = predictions[e.record_id]
        row = {
            **asdict(e),
            "gold": e.label,
            "prediction": p["prediction"],
            "probabilities": dict(p["probabilities"]),
        }
        rows.append(row)
    calibration_metrics(rows)  # Validate named class mapping, argmax and probabilities.
    return rows


def evaluate_cohorts(rows: list[dict], training_targets: set[str]) -> dict:
    """Target-weighted metrics; empty cohorts are explicitly unavailable."""
    cohorts = {
        "all_targets": rows,
        "single_target": [r for r in rows if not r["multi_target"]],
        "multi_target": [r for r in rows if r["multi_target"]],
        "mixed_label_headlines": [r for r in rows if r["mixed_labels"]],
        "seen_native_target": [r for r in rows if r["target"] in training_targets],
        "unseen_native_target": [
            r for r in rows if r["target"] not in training_targets
        ],
    }
    return {
        name: {
            "examples": len(selected),
            "headlines": len({r["record_id"] for r in selected}),
            "duplicate_groups": len({r["group_id"] for r in selected}),
            "classification": classification_metrics(
                [r["gold"] for r in selected], [r["prediction"] for r in selected]
            )
            if selected
            else None,
            "calibration": calibration_metrics(selected) if selected else None,
        }
        for name, selected in cohorts.items()
    }


def verified_inputs(
    data_dir: Path,
) -> tuple[dict, dict, tuple[TargetExample, ...], list[dict]]:
    """Reconstruct development rows and compare exact prepared private bytes."""
    plan, asset = load_plan()
    lock = json.loads(LOCK.read_bytes())
    if (
        lock["plan_sha256"] != file_sha256(PLAN)
        or lock["preparation_report_sha256"] != file_sha256(OUTPUT.with_suffix(".json"))
        or lock["final_test_evaluated"] is not False
    ):
        raise ValueError("preparation lock differs")
    root = data_dir / "interim/target-pairs" / lock["data_manifest_sha256"]
    if file_sha256(root / "manifest.json") != lock["data_manifest_sha256"]:
        raise ValueError("prepared manifest differs")
    manifest = json.loads((root / "manifest.json").read_bytes())
    if manifest["plan_sha256"] != file_sha256(PLAN):
        raise ValueError("prepared view plan differs")
    verify_files(root, manifest["files"])
    records = load_development_records(data_dir, RECEIPT)
    examples = build_examples(records)
    for part in plan["partitions"]:
        if (root / f"{part}.jsonl").read_bytes() != examples_bytes(examples, part):
            raise ValueError(
                "prepared view does not match source development membership"
            )
        selected = [e for e in examples if e.partition == part]
        if (
            len(selected) != plan["expected_examples"][part]
            or len({e.record_id for e in selected}) != plan["expected_headlines"][part]
        ):
            raise ValueError("prepared counts differ")
    return plan, asset, examples, records


def run(data_dir: Path) -> dict:
    """Inference only, local weights only; refuse overwriting published results."""
    if any(REPORT.with_suffix(s).exists() for s in (".json", ".md")):
        raise FileExistsError("saved control report already exists")
    plan, asset, examples, records = verified_inputs(data_dir)
    final_sha = file_sha256(FINAL)
    validation = tuple(e for e in examples if e.partition == "validation")
    training_targets = {e.target for e in examples if e.partition == "train"}
    headlines = {e.record_id: e.headline for e in validation}
    _, single = sentfin_partitions(records)
    receipt = json.loads(RECEIPT.read_bytes())
    results, private = [], {}
    for control in plan["saved_headline_controls"]:
        seed = control["seed"]
        report = json.loads(Path(control["report_path"]).read_bytes())
        config = FineTuneConfig(**report["configuration"])
        if config.seed != seed or config.max_length != plan["max_length"]:
            raise ValueError("saved configuration differs")
        # Validate the old private prediction bytes, source membership and gold.
        old, old_provenance = checked_run(
            {
                **control,
                "model": "deberta_base",
                "learning_rate": config.learning_rate,
                "validation_membership_sha256": report["validation_membership_sha256"],
                "metrics": report["metrics"],
            },
            data_dir,
            single,
            receipt,
        )
        directory = data_dir / control["artifact_relative_path"]
        torch, tokenizer, model, manifest = load_saved(
            directory, control["artifact_manifest_sha256"], asset, config
        )
        print(
            f"Verified seed {seed}; scoring {len(headlines)} validation headlines",
            flush=True,
        )
        predictions, profile = score_model(
            torch, tokenizer, model, list(headlines.values()), config
        )
        named = {
            identity: {"prediction": p.label.value, "probabilities": p.probabilities}
            for identity, p in zip(headlines, predictions, strict=True)
        }
        rows = join_predictions(validation, named)
        # Batch neighbours can slightly change FP32 softmax. Labels must match.
        max_difference = max(
            abs(
                named[r["record_id"]]["probabilities"][label]
                - r["probabilities"][label]
            )
            for r in old
            for label in plan["labels"]
        )
        if (
            any(named[r["record_id"]]["prediction"] != r["prediction"] for r in old)
            or max_difference > 1e-5
        ):
            raise ValueError("single-target reload predictions differ beyond tolerance")
        cohorts = evaluate_cohorts(rows, training_targets)
        if cohorts["single_target"]["classification"] != report["metrics"]:
            raise ValueError("single-target reload metrics differ")
        body = b"".join((json.dumps(r, sort_keys=True) + "\n").encode() for r in rows)
        private[f"seed-{seed}.jsonl"] = body
        results.append(
            {
                "seed": seed,
                "artifact_manifest_sha256": control["artifact_manifest_sha256"],
                "source_report_sha256": control["report_sha256"],
                "configuration": asdict(config),
                "cohorts": cohorts,
                "model": profile,
                "model_file_bytes": sum(
                    f["size_bytes"] for f in manifest["files"].values()
                ),
                "predictions_sha256": hashlib.sha256(body).hexdigest(),
                "single_target_reload": {
                    "headlines": len(old),
                    "all_labels_identical": True,
                    "max_absolute_probability_difference": max_difference,
                    "probability_tolerance": 1e-5,
                    "source": old_provenance,
                },
            }
        )
        score = cohorts["all_targets"]["classification"]["macro_f1"]
        print(
            f"Seed {seed}: all-target macro F1={score:.6f}",
            flush=True,
        )
        del model, tokenizer, predictions
        gc.collect()
        if config.device == "cuda":
            torch.cuda.empty_cache()
    aggregate = {}
    for cohort in plan["evaluation_cohorts"]:
        aggregate[cohort] = {}
        for metric in ("accuracy", "macro_f1", "nll", "brier", "ece"):
            family = (
                "classification"
                if metric in ("accuracy", "macro_f1")
                else "calibration"
            )
            values = [r["cohorts"][cohort][family][metric] for r in results]
            aggregate[cohort][metric] = {
                "mean": mean(values),
                "sample_std": stdev(values),
            }
    verified_inputs(data_dir)
    if file_sha256(FINAL) != final_sha:
        raise ValueError("original final aggregate changed")
    result = {
        "schema_version": 1,
        "experiment": "target-saved-headline-control-v1",
        "arm": "saved_headline_control",
        "models_trained": 0,
        "final_test_evaluated": False,
        "probability_policy": plan["probability_policy"],
        "plan_sha256": file_sha256(PLAN),
        "lock_sha256": file_sha256(LOCK),
        "data_manifest_sha256": json.loads(LOCK.read_bytes())["data_manifest_sha256"],
        "original_final_aggregate_sha256": final_sha,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "implementation_sha256": file_sha256(Path(__file__)),
        "lockfile_sha256": file_sha256(Path("uv.lock")),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": version("transformers"),
            "sentencepiece": version("sentencepiece"),
        },
        "seeds": results,
        "aggregate": aggregate,
        "scope": "Exploratory development validation; same prediction "
        "across every target of a headline; raw probabilities; no "
        "calibration fitting or test scoring. Seen/unseen means exact "
        "native target string, not verified company identity.",
    }
    body = json_bytes(result)
    identity = hashlib.sha256(body).hexdigest()
    root = data_dir / "interim/target-saved-control" / identity
    root.mkdir(parents=True, exist_ok=False)
    for name, content in private.items():
        (root / name).write_bytes(content)
    (root / "run.json").write_bytes(body)
    REPORT.with_suffix(".json").write_bytes(body)
    REPORT.with_suffix(".md").write_bytes(render_report(result).encode())
    return result


def render_report(result: dict) -> str:
    lines = [
        "# Saved DeBERTa headline control: target-level validation",
        "",
        "Three existing seeds (42/43/44), no new training or temperature scaling.",
        "Each model scores 1,053 headlines once, then joins "
        "predictions to 1,410 native targets.",
        "",
        "| Cohort | Targets | Headlines | Accuracy % | Macro F1 % | "
        "NLL | Brier | ECE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in result["aggregate"].items():
        cohort = result["seeds"][0]["cohorts"][name]
        cells = [
            f"{metrics[m]['mean'] * 100:.2f} ± {metrics[m]['sample_std'] * 100:.2f}"
            for m in ("accuracy", "macro_f1")
        ]
        cells += [f"{metrics[m]['mean']:.4f}" for m in ("nll", "brier", "ece")]
        lines.append(
            f"| {name} | {cohort['examples']} | {cohort['headlines']} | "
            + " | ".join(cells)
            + " |"
        )
    lines += [
        "",
        "Accuracy/F1 are mean ± sample standard deviation across "
        "seeds, not confidence intervals.",
        "All metrics weight target annotations equally; cohorts "
        "overlap. Three-class macro F1 includes all classes.",
        "Mixed-label headlines necessarily receive the same predicted "
        "label for conflicting native targets.",
        "Single-target predictions reproduce every earlier label and "
        "metric; raw probability differences are checked at 1e-5 tolerance.",
        "",
        "| Seed | All-target F1 % | Multi-target F1 % | Mixed-label F1 % |",
        "| --- | ---: | ---: | ---: |",
    ]
    for seed in result["seeds"]:
        values = [
            seed["cohorts"][c]["classification"]["macro_f1"] * 100
            for c in ("all_targets", "multi_target", "mixed_label_headlines")
        ]
        lines.append(
            f"| {seed['seed']} | " + " | ".join(f"{v:.2f}" for v in values) + " |"
        )
    architecture = result["seeds"][0]["model"]["architecture"]
    lines += [
        "",
        f"Model: {architecture['class']}, {architecture['layers']} layers, "
        f"hidden size {architecture['hidden_size']}, "
        f"{architecture['attention_heads']} attention heads, "
        f"{architecture['parameters']:,} parameters. FP32, eager attention.",
        "Previously trained on 6,299 single-target headlines; no "
        "target conditioning. Full profiles, per-class metrics, "
        "confusion matrices and calibration bins are in the companion JSON.",
        "",
        "These are reused development validation results, not "
        "independent test evidence. No target-aware improvement claim is established.",
        "The primary future comparison remains freshly trained matched "
        "headline-only versus target-pair models.",
        "Reserved multi-target test data remain unscored; the original "
        "Phase 3 final aggregate is unchanged.",
        "Native text, target names and row predictions remain in "
        "ignored local storage.",
        "",
        f"Plan SHA-256: `{result['plan_sha256']}`.",
        f"Private view: `{result['data_manifest_sha256']}`.",
        f"Execution revision: `{result['code_revision']}`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    run(args.data_dir)


if __name__ == "__main__":
    main()
