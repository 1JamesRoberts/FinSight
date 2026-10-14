"""Render the three saved exploratory runs; never train or run inference."""

import json
from pathlib import Path
from statistics import mean, stdev

from finsight.sentiment.finbert import file_sha256
from finsight.sentiment.proprak_training import PLAN, REPORTS, load_plan
from finsight.sentiment.transformer_analysis import json_body

AUDITED_RUNS = {
    "42": "2cf0e061c80880fe920b0e34922f06274225c6790cd02df2954453ee664b55d4",
    "43": "b4d1705229d6dcbbd4d4033e6ed1b66c68bb78d36857896d0bb184954a61ef26",
    "44": "9b225fbe1f1d5cfbbcad12a6624b9b5a5af660c2a209ce7a01126cdf2eebf210",
}
AUDIT_SHA = "66c6defce98852c279a27231c3e3725b171eee59e260797c8f358421f4dcc788"


def main() -> None:
    plan, _ = load_plan()
    runs = []
    inputs = {}
    for seed in plan["seeds"]:
        path = REPORTS / f"seed-{seed}.json"
        r = json.loads(path.read_bytes())
        if (
            r["seed"] != seed
            or r["plan_sha256"] != file_sha256(PLAN)
            or r["final_test_evaluated"] is not False
            or not r["regression_weights_unchanged"]
        ):
            raise ValueError("run protocol differs")
        inputs[path.as_posix()] = file_sha256(path)
        if inputs[path.as_posix()] != AUDITED_RUNS[str(seed)]:
            raise ValueError("report differs from independently audited outputs")
        runs.append(r)
    if len({r["validation_membership_sha256"] for r in runs}) != 1:
        raise ValueError("validation cohorts differ")
    scores = {}
    for metric in ("accuracy", "macro_f1", "nll", "brier", "ece"):
        values = [
            r["metrics" if metric in ("accuracy", "macro_f1") else "calibration"][
                metric
            ]
            for r in runs
        ]
        scores[metric] = {"mean": mean(values), "sample_std": stdev(values)}
    base = json.loads(Path("reports/experiments/deberta-tuning.json").read_bytes())
    baseline_membership = base["selected"]["deberta_base"][
        "validation_membership_sha256"
    ]
    if baseline_membership != runs[0]["validation_membership_sha256"]:
        raise ValueError("reference cohort differs")
    raw = json.loads(Path("reports/experiments/temperature-scaling.json").read_bytes())
    original = json.loads(
        Path("reports/experiments/proprak-deberta-validation.json").read_bytes()
    )
    linear = json.loads(Path("reports/experiments/tfidf-tuning.json").read_bytes())
    chosen = linear["selected_run_ids"]["sentfin_single_target"]["logistic_regression"]
    lr = next(c["run"] for c in linear["candidates"] if c["run_id"] == chosen)[
        "metrics"
    ]["sentfin_single_target"]["logistic_regression"]
    refs = {
        p: file_sha256(Path(p))
        for p in (
            "reports/experiments/deberta-tuning.json",
            "reports/experiments/temperature-scaling.json",
            "reports/experiments/proprak-deberta-validation.json",
            "reports/experiments/tfidf-tuning.json",
        )
    }
    summary = {
        "schema_version": 1,
        "experiment": "proprak-deberta-adaptation",
        "scope": (
            "Post-final exploratory validation; all three seeds; raw probabilities"
        ),
        "final_test_evaluated": False,
        "training_samples": 6299,
        "validation_samples": 781,
        "plan": plan,
        "plan_sha256": file_sha256(PLAN),
        "inputs": inputs,
        "reference_reports": refs,
        "statistics": scores,
        "verification": {
            "ordered_validation_predictions": 2343,
            "selected_artifacts": 3,
            "source_vs_saved_text_encodings": 21240,
            "encoder_and_classification_updates": True,
            "regression_weights_unchanged": True,
            "original_final_aggregate_unchanged": True,
            "audit_implementation_sha256": AUDIT_SHA,
        },
        "runs": [
            {
                k: r[k]
                for k in (
                    "seed",
                    "metrics",
                    "calibration",
                    "artifact_relative_path",
                    "artifact_file_bytes",
                    "model",
                    "regression_weights_unchanged",
                )
            }
            | {
                "best_epoch": r["fit"]["best_epoch"],
                "artifact_manifest_sha256": r["fit"]["artifact_manifest_sha256"],
                "training_seconds": r["fit"][
                    "training_seconds_including_validation_and_saving"
                ],
            }
            for r in runs
        ],
    }
    output = Path("reports/experiments/proprak-deberta-adaptation")
    output.with_suffix(".json").write_bytes(json_body(summary))
    lines = [
        "# Financial DeBERTa Base: SEntFiN adaptation",
        "",
        "Post-final exploratory development experiment. No final-test inference,",
        (
            "calibration fitting, best-seed selection or changes to the locked "
            "final results."
        ),
        "",
        "## Protocol",
        "",
        "Start each seed from ProPrak01/deberta-v3-financial-sentiment at",
        "`f008a38828d87fbf80bea4c841333d9c493972e7`. Train the encoder and three-class",
        (
            "head on 6,299 single-target SEntFiN training headlines. Freeze "
            "both regression"
        ),
        "head parameters; verify exact equality to source weights after reload.",
        "Use 781 validation headlines (247 negative, 252 neutral, 282 positive).",
        "",
        "Fixed LR 1e-5 inherits the prior Base validation choice; no new LR search.",
        (
            "Seeds 42/43/44; FP32; train/eval batch 16/32; length 128; maximum "
            "five epochs;"
        ),
        "patience two; strict validation macro-F1 improvement and earliest exact tie.",
        "AdamW betas 0.9/0.999, eps 1e-8, decay 0.01 (bias/LayerNorm excluded),",
        "10% linear warm-up/decay and gradient clipping at 1. No class weights.",
        (
            "Budget matches the selected-rate Base runs, not their full "
            "preceding LR search."
        ),
        "",
        "## Results",
        "",
        "| Seed | Selected epoch | Accuracy | Macro F1 | NLL | Brier | ECE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in runs:
        m, c = r["metrics"], r["calibration"]
        lines.append(
            f"| {r['seed']} | {r['fit']['best_epoch']} | {m['accuracy']:.4f} | "
            f"{m['macro_f1']:.4f} | {c['nll']:.4f} | {c['brier']:.4f} | "
            f"{c['ece']:.4f} |"
        )
    lines += ["", "| Metric | Three-seed mean | Sample SD |", "| --- | ---: | ---: |"]
    lines += [
        f"| {k} | {v['mean']:.4f} | {v['sample_std']:.4f} |" for k, v in scores.items()
    ]
    lines += [
        "",
        "## Same-cohort reference results",
        "",
        "| Model | Accuracy | Macro F1 |",
        "| --- | ---: | ---: |",
        (
            f"| Unchanged ProPrak01 financial Base (fixed checkpoint) | "
            f"{original['metrics']['accuracy']:.4f} | "
            f"{original['metrics']['macro_f1']:.4f} |"
        ),
        (
            f"| Adapted ProPrak01 financial Base (three-seed mean) | "
            f"{scores['accuracy']['mean']:.4f} | "
            f"{scores['macro_f1']['mean']:.4f} |"
        ),
        (
            f"| Original Microsoft Base, SEntFiN-trained (three-seed mean) | "
            f"{base['statistics']['deberta_base']['accuracy']['mean']:.4f} | "
            f"{base['statistics']['deberta_base']['macro_f1']['mean']:.4f} |"
        ),
        (
            f"| Selected TF-IDF Logistic Regression | {lr['accuracy']:.4f} | "
            f"{lr['macro_f1']:.4f} |"
        ),
        "",
        (
            "Means average metrics across independent training seeds; they are "
            "not a voting ensemble."
        ),
        (
            "Sample SD measures variation across these seeds, not "
            "test-sampling uncertainty."
        ),
        (
            "These are checkpoint-selection validation scores, not unbiased "
            "test estimates."
        ),
        "",
        "## Raw probability calibration",
        "",
        "Lower is better for NLL, Brier and ECE. No temperatures were fitted here.",
        "Base reference values below also use raw probabilities; do not compare",
        "this model against the older calibrated Base values as an architecture test.",
        "",
        "| Model (three-seed mean) | NLL | Brier | ECE |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| Adapted financial Base | {scores['nll']['mean']:.4f} | "
            f"{scores['brier']['mean']:.4f} | {scores['ece']['mean']:.4f} |"
        ),
        "| SEntFiN-trained original Base | "
        + " | ".join(
            f"{raw['models']['deberta_base']['statistics'][k]['before']['mean']:.4f}"
            for k in ("nll", "brier", "ece")
        )
        + " |",
        "",
        "ECE uses ten equal-width top-confidence bins; multiclass Brier is the",
        "unnormalized sum over classes (range 0–2). Lower ECE alone is not sufficient.",
        "",
        "## Class results",
        "",
        "| Class | Mean F1 | Sample SD |",
        "| --- | ---: | ---: |",
    ]
    for label in ("negative", "neutral", "positive"):
        values = [r["metrics"]["per_class"][label]["f1"] for r in runs]
        lines.append(f"| {label} | {mean(values):.4f} | {stdev(values):.4f} |")
    a = runs[0]["model"]["architecture"]
    lines += [
        "",
        "## Size, speed and saved models",
        "",
        (
            "12 layers, hidden 768, 12 attention heads; masked mean pooling "
            "and dropout 0.1."
        ),
        (
            f"Total parameters: **{a['parameters']:,}**; trainable parameters: "
            f"**{runs[0]['fit']['parameters_updated']:,}**; 769 regression "
            f"parameters frozen."
        ),
        "Classification head 768→3; retained unused regression head 768→1.",
        "The encoder changes during adaptation, so retaining regression weights",
        "does not preserve upstream FiQA performance. These are classification-only",
        "artifacts; continuous regression outputs have not been validated.",
        (
            "Strict reload uses all backbone and both-head tensors. Derived "
            "weights are safetensors."
        ),
        "",
        (
            "| Seed | Training/validation/saving seconds | Inference ms/text | "
            "Artifact MiB | Manifest SHA-256 |"
        ),
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for r in runs:
        seconds = r["fit"]["training_seconds_including_validation_and_saving"]
        lines.append(
            f"| {r['seed']} | "
            f"{seconds:.2f} | {r['model']['amortized_ms_per_text']:.3f} | "
            f"{r['artifact_file_bytes'] / 1024**2:.2f} | "
            f"`{r['fit']['artifact_manifest_sha256']}` |"
        )
    lines += [
        "",
        "Artifact MiB includes model, tokenizer, configuration and manifest files;",
        "it is not runtime RAM. Inference includes tokenization, length audit and",
        "output transfer after warm-up at batch 32 on RTX 4070 SUPER. Timings are",
        "descriptive and not an isolated production benchmark.",
        "",
        (
            "Saved models remain private on Windows under "
            "`data/interim/transformer-finetuned-models/<manifest>`."
        ),
        (
            "All selected models were reloaded and reproduced selected-epoch "
            "validation metrics."
        ),
        "",
        "## Provenance, limitations and reproduction",
        "",
        (
            f"Execution code: `{runs[0]['code_revision']}`; all three seeds use "
            f"this revision."
        ),
        f"Plan SHA-256: `{file_sha256(PLAN)}`.",
        "Per-seed JSON records source files, memberships, artifact identities, epoch",
        (
            "history, trained/frozen parameter counts, predictions SHA-256 and "
            "implementation hashes."
        ),
        "Seed 42 starts from a clean checkout; later seeds may record only preceding",
        "untracked seed reports. The runner rejects any other working-tree change.",
        "",
        "Use the clean execution revision above in a separate checkout with access to",
        "the reviewed local data directory. That revision predates the output reports.",
        "Do not delete tracked reports in this checkout to bypass the safeguards.",
        "Copy the reviewed data view into its ignored data directory, or add",
        "`--data-dir PATH` to the commands below to point to the existing local view.",
        "",
        "```powershell",
        (
            "python -m uv run --locked --extra transformer python -m "
            "finsight.sentiment.proprak_training --seed 42"
        ),
        (
            "python -m uv run --locked --extra transformer python -m "
            "finsight.sentiment.proprak_training --seed 43"
        ),
        (
            "python -m uv run --locked --extra transformer python -m "
            "finsight.sentiment.proprak_training --seed 44"
        ),
        "```",
        "",
        "Completed reports are refused before training. Inspect saved outputs rather",
        "than repeating training. Use a fresh checkout for any deliberate rerun.",
        "This checksummed plan also requires unchanged code during execution; only",
        "the declared seed output files may be untracked. On a checkout containing",
        "these committed reports, run the following to rebuild this summary only:",
        "",
        "```powershell",
        "python -m uv run --locked python scripts/report_proprak_adaptation.py",
        "```",
        "",
        (
            "Upstream training includes PhraseBank/Twitter/FiQA; SEntFiN "
            "exposure is unknown."
        ),
        "The comparison changes financial supervision, pooling architecture and prior",
        (
            "training history together; it does not isolate financial "
            "pretraining causally."
        ),
        "Validation selected epochs for every trained model. We make no statistical",
        "superiority claim and do not choose a seed or replace the final model based",
        (
            "on this exploratory comparison. New confirmatory claims require "
            "fresh untouched"
        ),
        "evaluation data. No weights, text or private predictions are redistributed.",
        "",
        "## Interpretation and verification",
        "",
        "Adaptation raises observed macro F1 by about 9.00 percentage points over",
        "the unchanged checkpoint. Relative to SEntFiN-trained original Base, the",
        "mean difference is only +0.05 percentage points; mean accuracy is identical.",
        "Seed variation is larger (0.62 versus 0.44 percentage-point sample SD).",
        "Raw NLL/ECE are slightly worse; raw Brier is marginally lower. These results",
        "give no compelling reason to replace the existing calibrated Base model.",
        "They do not establish equivalence or statistical superiority. Neutral is",
        "the weakest class on average. No extra LR search or calibration was added.",
        "",
        "The independent saved-output audit verifies all 2,343 ordered validation",
        "predictions, sklearn metrics, independent NumPy NLL/Brier/ECE, earliest",
        "best-epoch selection, progress histories, source/artifact/code hashes,",
        "real encoder/classifier changes and frozen regression tensors. Token IDs",
        "match source for every train/validation text in all three saved tokenizers",
        "(21,240 text encodings). A generic Mistral-regex warning from the pinned",
        "tokenizer reload was retained in the logs; no regex fix was applied.",
        "Token-ID checks verify faithful reproduction of the published tokenizer,",
        "rather than independently validating its upstream design choices. Frozen data",
        "bytes and the original final aggregate are unchanged. Audit code/results",
        "remain private beside the run artifacts under",
        "`data/interim/proprak-deberta-adaptation/audit.py` and `audit.json`.",
        "The generator refuses inputs differing from the audited run hashes.",
        "Ruff lint/format checks and all 390 tests pass on Mac and Windows.",
        "",
    ]
    output.with_suffix(".md").write_bytes("\n".join(lines).encode("utf-8"))


if __name__ == "__main__":
    main()
