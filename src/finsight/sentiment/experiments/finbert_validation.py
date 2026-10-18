"""Score unchanged FinBERT on verified validation cohorts and saved controls."""

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

from finsight.data.benchmark import load_development_records
from finsight.sentiment.artifacts import project_root, source_manifest
from finsight.sentiment.evaluation.cohorts import sentence_and_label, sentence_cohorts
from finsight.sentiment.evaluation.metrics import LABELS, classification_metrics
from finsight.sentiment.models.finbert import FinBertConfig, FinBertSentiment


def _digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def saved_controls(
    baseline: dict[str, Any], receipt: dict[str, Any], cohort: str, membership: str
) -> dict[str, Any]:
    """Require matching frozen lineage/membership before copying baseline scores."""
    controls = {}
    for kind in ("logistic_regression", "linear_svm"):
        identifier = baseline["selected_run_ids"][cohort][kind]
        candidates = [
            item
            for item in baseline["candidates"]
            if item["run_id"] == identifier and item["model_kind"] == kind
        ]
        if len(candidates) != 1:
            raise ValueError("selected baseline run is missing or ambiguous")
        run = candidates[0]["run"]
        if (
            _digest((json.dumps(run, indent=2, sort_keys=True) + "\n").encode())
            != identifier
        ):
            raise ValueError("baseline run checksum mismatch")
        if run["final_test_evaluated"] is not False:
            raise ValueError("baseline must be development-only")
        for key in ("benchmark_manifest_sha256", "parent_split_manifest_sha256"):
            if run[key] != receipt[key]:
                raise ValueError("baseline frozen lineage differs")
        result = run["metrics"][cohort]
        if result["validation_membership_sha256"] != membership:
            raise ValueError("baseline validation membership differs")
        controls[kind] = {"metrics": result[kind], "source_run_id": identifier}
        for control in ("vader", "training_majority"):
            if control in controls and controls[control]["metrics"] != result[control]:
                raise ValueError("saved controls disagree")
            controls[control] = {
                "metrics": result[control],
                "source_run_id": identifier,
            }
    return controls


def run_finbert_validation(
    data_dir: Path,
    receipt_path: Path,
    config_path: Path,
    report_path: Path,
    baseline_path: Path = Path("reports/experiments/tfidf-tuning.json"),
    *,
    cohort: str = "all",
) -> dict[str, Any]:
    """One fixed inference configuration; no fitting, selection, or test scoring."""
    if cohort not in {"all", "phrasebank", "sentfin_single_target"}:
        raise ValueError("unknown sentence cohort")
    config = FinBertConfig.load(config_path)
    records = load_development_records(data_dir, receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    baseline_bytes = baseline_path.read_bytes()
    baseline = json.loads(baseline_bytes)
    selected = {}
    for name, rows in sentence_cohorts(records).items():
        if cohort != "all" and name != cohort:
            continue
        validation = [row for row in rows if row["partition"] == "validation"]
        if not validation:
            raise ValueError("each selected cohort requires validation records")
        membership = _digest(
            json.dumps([row["record_id"] for row in validation]).encode()
        )
        controls = saved_controls(baseline, receipt, name, membership)
        selected[name] = (validation, membership, controls)

    print("Loading pinned pretrained FinBERT (no fine-tuning).", flush=True)
    started = perf_counter()
    model = FinBertSentiment(config, data_dir / "interim/finbert-checkpoint-cache")
    model.synchronize()
    loading_seconds = perf_counter() - started
    model.predict_batch(["The company reported higher earnings."])
    model.reset_peak_memory()
    predictions, metrics = [], {}
    for name, (rows, membership, controls) in selected.items():
        examples = [sentence_and_label(row) for row in rows]
        texts, gold = [text for text, _ in examples], [label for _, label in examples]
        print(f"Scoring {name}: {len(texts)} validation texts.", flush=True)
        model.synchronize()
        started = perf_counter()
        results = model.predict_batch(texts)
        model.synchronize()
        elapsed = perf_counter() - started
        predicted = [result.label.value for result in results]
        scores = classification_metrics(gold, predicted)
        if any(item["metrics"]["samples"] != len(rows) for item in controls.values()):
            raise ValueError("baseline sample count differs")
        for item in controls.values():
            if any(
                item["metrics"]["per_class"][label]["support"]
                != scores["per_class"][label]["support"]
                for label in LABELS
            ):
                raise ValueError("baseline validation label distribution differs")
        lengths = [result.token_count for result in results]
        metrics[name] = {
            "finbert": scores,
            "saved_baselines": controls,
            "validation_membership_sha256": membership,
            "inference_seconds": elapsed,
            "amortized_ms_per_text": elapsed * 1000 / len(rows),
            "texts_per_second": len(rows) / elapsed,
            "batch_size": config.batch_size,
            "batches": (len(rows) + config.batch_size - 1) // config.batch_size,
            "token_lengths_including_special_tokens": {
                "maximum": max(lengths),
                "mean": sum(lengths) / len(lengths),
                "truncated_texts": sum(result.truncated for result in results),
            },
            "evaluation_caveat": "known FinBERT supervision overlap; diagnostic"
            if name == "phrasebank"
            else (
                "development validation; no known accessible-corpus overlap; "
                "pretraining exposure unknown"
            ),
        }
        predictions.extend(
            {
                "record_id": row["record_id"],
                "cohort": name,
                "gold": label,
                "prediction": result.label.value,
                "probabilities": result.probabilities,
                "token_count": result.token_count,
                "truncated": result.truncated,
            }
            for row, label, result in zip(rows, gold, results, strict=True)
        )
    metadata = model.metadata()
    root = project_root()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    body = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in predictions
    ).encode()
    run = {
        "schema_version": 1,
        "experiment": "finbert-pretrained-validation-v1",
        "scope": "Unchanged pretrained checkpoint; verified validation cohorts only",
        "configuration": asdict(config),
        "cohort_selection": cohort,
        "fine_tuned_in_project": False,
        "hyperparameters_tuned": False,
        "final_test_evaluated": False,
        "training_records_scored": False,
        "benchmark_manifest_sha256": receipt["benchmark_manifest_sha256"],
        "parent_split_manifest_sha256": receipt["parent_split_manifest_sha256"],
        "freeze_receipt_sha256": _digest(receipt_path.read_bytes()),
        "config_sha256": _digest(config_path.read_bytes()),
        "baseline_report_sha256": _digest(baseline_bytes),
        "lockfile_sha256": _digest((root / "uv.lock").read_bytes()),
        "code_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "working_tree_dirty": bool(dirty.stdout) if dirty.returncode == 0 else None,
        "implementation_sha256": source_manifest(),
        "predictions_sha256": _digest(body),
        "model": metadata,
        "metrics": metrics,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": version("torch"),
            "transformers": version("transformers"),
        },
        "loading_seconds": loading_seconds,
        "timing": "one pass after synthetic warm-up; tokenization/length audit, "
        "GPU transfer/forward/softmax and CPU output; CUDA synchronized; "
        "excludes loading, metrics, hashing and serialization",
        "baseline_timing_caveat": "Phase 2 CPU timings used batch size one "
        "on another machine; "
        "no direct speed comparison",
    }
    run_body = (json.dumps(run, indent=2, sort_keys=True) + "\n").encode()
    identifier = _digest(run_body)
    directory = data_dir / "interim/finbert-validation" / identifier
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run.json").write_bytes(run_body)
    (directory / "predictions.jsonl").write_bytes(body)
    acquisition = {
        "model_id": config.model_id,
        "revision": config.revision,
        "checkpoint_files": metadata["checkpoint_files"],
        "weights_downloaded": True,
        "weight_license_spdx": None,
        "use_decision": metadata["use_decision"],
        "license_decision_document": (
            "docs/datasets.md#gpu-preparation-asset-review-2026-10-11"
        ),
        "run_id": identifier,
    }
    (directory / "acquisition-receipt.json").write_bytes(
        (json.dumps(acquisition, indent=2, sort_keys=True) + "\n").encode()
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(run, identifier), encoding="utf-8")
    report_path.with_suffix(".json").write_bytes(run_body)
    return run


def render_report(run: dict[str, Any], identifier: str) -> str:
    """Aggregate results only; per-record labels/probabilities remain local."""
    model, configuration = run["model"], run["configuration"]
    architecture = model["architecture"]
    lines = [
        "# Pretrained FinBERT validation",
        "",
        run["scope"],
        "",
        "No project fine-tuning, tuning, calibration, or final-test scoring.",
        "PhraseBank has known FinBERT supervision overlap; its scores are diagnostic.",
        "SEntFiN is development validation; full pretraining exposure is unknown.",
        "",
        "| Cohort | Model | Validation texts | Accuracy | Macro F1 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for name, result in run["metrics"].items():
        comparisons = {
            key: value["metrics"] for key, value in result["saved_baselines"].items()
        }
        comparisons["finbert_pretrained"] = result["finbert"]
        for kind, score in comparisons.items():
            lines.append(
                f"| {name} | {kind} | {score['samples']} | "
                f"{score['accuracy']:.4f} | {score['macro_f1']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Model architecture and size",
            "",
            f"Checkpoint: `{configuration['model_id']}` "
            f"at `{configuration['revision']}`.",
            f"{architecture['class']}: {architecture['layers']} encoder layers, "
            f"hidden size {architecture['hidden_size']}, "
            f"{architecture['attention_heads']} heads, "
            f"feed-forward size {architecture['intermediate_size']}, "
            f"vocabulary {architecture['vocabulary_size']}; three sentiment outputs.",
            f"Parameters: **{architecture['total_parameters']:,}**; "
            f"gradient-enabled: {architecture['gradient_enabled_parameters']:,}.",
            f"FP32 parameter storage: {architecture['parameter_bytes']:,} bytes; "
            f"downloaded checkpoint files: {model['checkpoint_bytes']:,} bytes.",
            "Index mapping: 0=positive, 1=negative, 2=neutral. Softmax probabilities "
            "are uncalibrated; exact ties retain checkpoint index order.",
            "",
            "## Inference settings and timing",
            "",
            f"Device: {model['device']} / {model['gpu']}; "
            f"CUDA {model['cuda_runtime']}. "
            f"Batch size {configuration['batch_size']}; "
            f"max length {configuration['max_length']} including special tokens; "
            "dynamic padding, right truncation, FP32, eager attention, "
            "evaluation mode, no gradients, deterministic algorithms, TF32 disabled.",
            run["timing"] + ".",
            run["baseline_timing_caveat"] + ".",
            "",
            "| Cohort | Seconds | Amortized ms/text | Texts/sec | "
            "Truncated | Max tokens |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, result in run["metrics"].items():
        lengths = result["token_lengths_including_special_tokens"]
        lines.append(
            f"| {name} | {result['inference_seconds']:.3f} | "
            f"{result['amortized_ms_per_text']:.3f} | "
            f"{result['texts_per_second']:.1f} | "
            f"{lengths['truncated_texts']} | {lengths['maximum']} |"
        )
    lines.extend(
        [
            "",
            f"Peak GPU allocated bytes: {model['peak_gpu_allocated_bytes']}; "
            f"reserved bytes: {model['peak_gpu_reserved_bytes']}. "
            "Includes model residency and inference after the warm-up reset.",
            "Loading time (including acquisition/cache lookup): "
            f"{run['loading_seconds']:.3f}s.",
            "",
            "## Class results",
            "",
        ]
    )
    for name, result in run["metrics"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                "| Class | Support | Precision | Recall | F1 |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for label, score in result["finbert"]["per_class"].items():
            lines.append(
                f"| {label} | {score['support']} | {score['precision']:.4f} | "
                f"{score['recall']:.4f} | {score['f1']:.4f} |"
            )
        lines.extend(
            [
                "",
                "Confusion matrix (rows=gold, columns=predicted; "
                "negative/neutral/positive):",
                "",
                "```text",
                *[str(row) for row in result["finbert"]["confusion_matrix"]],
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Reproducibility and use scope",
            "",
            f"Run: `{identifier}`; code revision: `{run['code_revision']}`; "
            f"working tree dirty: `{run['working_tree_dirty']}`.",
            f"Parent split: `{run['parent_split_manifest_sha256']}`.",
            f"Frozen benchmark: `{run['benchmark_manifest_sha256']}` remains unscored.",
            "JSON records configuration, environment, source/checkpoint hashes, "
            "validation membership digests, and saved baseline run IDs. "
            "Local run, acquisition receipt, and predictions are under ignored "
            "data/interim/finbert-validation/.",
            "Weights are cached locally; weight-license scope remains unresolved. "
            "Use follows the user's local noncommercial research decision; "
            "public weight/text redistribution is outside scope.",
            "Historical Phase 1 source acquisition records remain unchanged.",
            "",
        ]
    )
    return "\n".join(lines)
