"""Receipt-bound final evaluation; default commands never score the frozen test."""

import argparse
import json
import math
import os
import platform
import subprocess
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

from finsight.data.benchmark import load_development_records
from finsight.sentiment.artifacts import file_sha256, json_body, sha
from finsight.sentiment.evaluation.calibration import apply_temperature
from finsight.sentiment.evaluation.cohorts import sentence_and_label, sentence_cohorts
from finsight.sentiment.evaluation.final_metrics import (
    bootstrap,
    evaluate_outputs,
    validate_rows,
)
from finsight.sentiment.evaluation.protocol import safe_path, verify_locked_plan
from finsight.sentiment.evaluation.voting import combine
from finsight.sentiment.experiments.ensemble_voting import verify_extension
from finsight.sentiment.models.saved_inference import infer_candidate


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json_body(value))
    temporary.replace(path)


def environment() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("locked evaluation requires CUDA; no CPU fallback")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            p: version(p)
            for p in (
                "scikit-learn",
                "numpy",
                "scipy",
                "vaderSentiment",
                "torch",
                "transformers",
                "sentencepiece",
            )
        },
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }


def code_identity(root: Path) -> dict[str, Any]:
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).strip()
    if dirty:
        raise ValueError("clean committed checkout required for execution receipt")
    return {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "sources": {
            p.relative_to(root).as_posix(): file_sha256(p)
            for p in sorted((root / "src/finsight").rglob("*.py"))
        },
    }


def preflight(root: Path, data_dir: Path) -> dict[str, Any]:
    verify_locked_plan(root, data_dir, full_artifacts=True)
    verify_extension(root, data_dir)
    return json.loads((root / "configs/final-evaluation-plan.json").read_bytes())


def prepare_execution(root: Path, data_dir: Path) -> Path:
    """Full artifact verification and clean-code receipt; no test JSON decoding."""
    preflight(root, data_dir)
    receipt = {
        "schema_version": 1,
        "protocol_id": "phase3-sentfin-single-target-v2",
        "parent_plan_sha256": file_sha256(root / "configs/final-evaluation-plan.json"),
        "extension_sha256": file_sha256(
            root / "configs/final-evaluation-ensembles-v2.json"
        ),
        "code": code_identity(root),
        "environment": environment(),
        "test_records_decoded_at_preparation": False,
    }
    directory = data_dir / "interim/final-evaluation" / sha(json_body(receipt))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "receipt.json"
    if path.exists() and path.read_bytes() != json_body(receipt):
        raise ValueError("existing execution receipt differs")
    path.write_bytes(json_body(receipt))
    return path


def checked_receipt(root: Path, data_dir: Path, path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_bytes())
    if (
        path.parent.name != sha(path.read_bytes())
        or path.resolve()
        != (
            data_dir
            / "interim/final-evaluation"
            / sha(path.read_bytes())
            / "receipt.json"
        ).resolve()
    ):
        raise ValueError(
            "execution receipt must be content-addressed in private storage"
        )
    if (
        receipt["code"] != code_identity(root)
        or receipt["environment"] != environment()
        or receipt["parent_plan_sha256"]
        != file_sha256(root / "configs/final-evaluation-plan.json")
        or receipt["extension_sha256"]
        != file_sha256(root / "configs/final-evaluation-ensembles-v2.json")
    ):
        raise ValueError("execution code/environment/protocol differs from receipt")
    return receipt


def frozen_records(
    root: Path, data_dir: Path, plan: dict[str, Any]
) -> list[dict[str, Any]]:
    """Called only by explicit --run after receipt and full preflight checks."""
    receipt = json.loads((root / "configs/benchmark-freeze.json").read_bytes())
    relative = next(p for p in plan["frozen_files"] if p.endswith("test-records.jsonl"))
    rows = [
        json.loads(line)
        for line in safe_path(data_dir, relative).read_bytes().splitlines()
    ]
    selected = [r for r in rows if len(r["native_record"]["sentiments"]) == 1]
    if any(
        r["partition"] != "test_candidate"
        or r["corpus"] != "sentfin"
        or r["development_only"]
        or r["review_reasons"]
        for r in selected
    ):
        raise ValueError("invalid frozen test scope")
    result = _records(selected)
    expected = receipt["summary"]["cohorts"]["single_target"]
    if (
        len(result) != plan["expected_samples"]
        or dict(Counter(r["gold"] for r in result)) != expected["class_counts"]
    ):
        raise ValueError("frozen cohort size/class support mismatch")
    return result


def _records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = [
        {
            "record_id": r["record_id"],
            "group_id": r["group_id"],
            "text": sentence_and_label(r)[0],
            "gold": sentence_and_label(r)[1],
        }
        for r in rows
    ]
    if (
        not records
        or len({r["record_id"] for r in records}) != len(records)
        or any(not r["text"].strip() for r in records)
    ):
        raise ValueError("unique nonempty evaluation records required")
    return records


def candidates(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = dict(plan["baselines"])
    for model, runs in plan["transformers"].items():
        for r in runs:
            result[f"{model}.{r['seed']}.raw"] = r
    return result


def derived_outputs(
    records: list[dict[str, Any]],
    raw: dict[str, list[dict[str, Any]]],
    plan: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float]]:
    output = dict(raw)
    started = perf_counter()
    for model, runs in plan["transformers"].items():
        for r in runs:
            output[f"{model}.{r['seed']}.calibrated"] = apply_temperature(
                raw[f"{model}.{r['seed']}.raw"], r["calibration"]["temperature"]
            )
    calibration_seconds = perf_counter() - started
    started = perf_counter()
    for seed in (42, 43, 44):
        joined = [
            combine(
                [
                    output[f"{m}.{seed}.calibrated"][i]
                    for m in ("finbert", "roberta_base", "deberta_base")
                ]
            )
            for i in range(len(records))
        ]
        for method in ("soft", "hard"):
            output[f"{method}.{seed}"] = [
                {"record_id": r["record_id"], "gold": r["gold"], **r[method]}
                for r in joined
            ]
    return output, {
        "calibration_adjustment_seconds": calibration_seconds,
        "voting_seconds": perf_counter() - started,
    }


def execute_predictions(
    directory: Path,
    receipt_id: str,
    records: list[dict[str, Any]],
    plan: dict[str, Any],
    data_dir: Path,
    *,
    resume: bool,
    predict: Callable[..., Any] = infer_candidate,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Atomic per-model checkpoints, no silent drops; log every inference attempt."""
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / "progress.json"
    if status_path.exists() and not resume:
        raise ValueError("execution exists; explicit resume required")
    status = (
        json.loads(status_path.read_bytes())
        if status_path.exists()
        else {"receipt_sha256": receipt_id, "completed": {}, "attempts": []}
    )
    if status["receipt_sha256"] != receipt_id:
        raise ValueError("progress receipt mismatch")
    # A killed process may leave this file: inspect/remove only after confirming
    # the previous process is gone. Never automatically race another evaluator.
    lock = directory / "active.lock"
    with lock.open("x") as file:
        file.write(str(os.getpid()))
    try:
        attempt = {
            "started_utc": datetime.now(UTC).isoformat(),
            "status": "running",
            "reused": [],
            "computed": [],
        }
        status["attempts"].append(attempt)
        atomic_json(status_path, status)
        outputs, metadata = {}, {}
        try:
            for key, spec in candidates(plan).items():
                checkpoint = directory / "predictions" / f"{key}.json"
                if key in status["completed"] and not checkpoint.exists():
                    raise ValueError("completed checkpoint missing; refusing rescoring")
                if checkpoint.exists():
                    body = checkpoint.read_bytes()
                    if (
                        key in status["completed"]
                        and sha(body) != status["completed"][key]
                    ):
                        raise ValueError("prediction checkpoint checksum mismatch")
                    saved = json.loads(body)
                    if (
                        saved["receipt_sha256"] != receipt_id
                        or saved["candidate"] != key
                    ):
                        raise ValueError("checkpoint execution identity mismatch")
                    rows, meta = saved["rows"], saved["metadata"]
                    validate_rows(records, rows)
                    attempt["reused"].append(key)
                else:
                    predictions, meta = predict(
                        key.split(".")[0], spec, [r["text"] for r in records], data_dir
                    )
                    if len(predictions) != len(records):
                        raise ValueError(
                            "failed/incomplete predictions; no row drops permitted"
                        )
                    rows = [
                        {**p, "record_id": r["record_id"], "gold": r["gold"]}
                        for r, p in zip(records, predictions, strict=True)
                    ]
                    validate_rows(records, rows)
                    atomic_json(
                        checkpoint,
                        {
                            "receipt_sha256": receipt_id,
                            "candidate": key,
                            "rows": rows,
                            "metadata": meta,
                        },
                    )
                    attempt["computed"].append(key)
                status["completed"][key] = file_sha256(checkpoint)
                outputs[key], metadata[key] = rows, meta
                atomic_json(status_path, status)
            attempt.update(
                status="predictions_complete",
                finished_utc=datetime.now(UTC).isoformat(),
            )
            atomic_json(status_path, status)
            return outputs, metadata
        except BaseException as error:
            attempt.update(
                status="failed",
                error_type=type(error).__name__,
                error=str(error),
                finished_utc=datetime.now(UTC).isoformat(),
            )
            atomic_json(status_path, status)
            raise
    finally:
        lock.unlink()


def summarize(
    records: list[dict[str, Any]],
    raw: dict[str, list[dict[str, Any]]],
    metadata: dict[str, Any],
    plan: dict[str, Any],
    *,
    scope: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output, timing = derived_outputs(records, raw, plan)
    metrics = evaluate_outputs(records, output)
    uncertainty, draws = bootstrap(records, output)
    report = {
        "schema_version": 1,
        "protocol_id": "phase3-sentfin-single-target-v2",
        "scope": scope,
        "final_test_evaluated": scope == "frozen_test",
        "samples": len(records),
        "class_counts": dict(Counter(r["gold"] for r in records)),
        **metrics,
        "uncertainty": uncertainty,
        "inference": metadata,
        "postprocessing_timing": timing,
        "limitations": plan["limitations"],
        "primary_plan_sha256": file_sha256(Path("configs/final-evaluation-plan.json")),
        "extension_sha256": file_sha256(
            Path("configs/final-evaluation-ensembles-v2.json")
        ),
    }
    return report, {"predictions": output, "bootstrap": draws}


@contextmanager
def session_lock(directory: Path) -> Iterator[None]:
    path = directory / "evaluation.lock"
    with path.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        yield
    finally:
        path.unlink()


def run_final(
    root: Path, data_dir: Path, receipt_path: Path, *, resume: bool = False
) -> Path:
    """Explicit final run; all state checks precede the first test decode/inference."""
    plan = preflight(root, data_dir)
    receipt = checked_receipt(root, data_dir, receipt_path)
    directory = receipt_path.parent
    with session_lock(directory):
        aggregate = directory / "aggregate.json"
        status_path = directory / "progress.json"
        status = json.loads(status_path.read_bytes()) if status_path.exists() else {}
        if "final_aggregate_sha256" in status:
            if not resume:
                raise ValueError("finished execution exists; explicit resume required")
            if file_sha256(aggregate) != status["final_aggregate_sha256"]:
                raise ValueError("finished aggregate checksum mismatch")
            report = json.loads(aggregate.read_bytes())
            if (
                file_sha256(directory / "results-private.json")
                != report["private_results_sha256"]
            ):
                raise ValueError("finished private outputs checksum mismatch")
            # Completed runs return without decoding labels or repeating inference.
            return aggregate
        records = frozen_records(root, data_dir, plan)
        raw, metadata = execute_predictions(
            directory, file_sha256(receipt_path), records, plan, data_dir, resume=resume
        )
        try:
            report, private = summarize(
                records, raw, metadata, plan, scope="frozen_test"
            )
            report["execution_receipt_sha256"] = file_sha256(receipt_path)
            report["execution_code_revision"] = receipt["code"]["revision"]
            report["environment"] = receipt["environment"]
            atomic_json(directory / "results-private.json", private)
            report["private_results_sha256"] = file_sha256(
                directory / "results-private.json"
            )
            atomic_json(aggregate, report)
            status = json.loads(status_path.read_bytes())
            status["final_aggregate_sha256"] = file_sha256(aggregate)
            status["attempts"][-1]["status"] = "complete"
            atomic_json(status_path, status)
            return aggregate
        except BaseException as error:
            status = json.loads(status_path.read_bytes())
            status["attempts"][-1].update(
                status="failed_postprocessing",
                error_type=type(error).__name__,
                error=str(error),
            )
            atomic_json(status_path, status)
            raise


def export_results(directory: Path, destination: Path) -> Path:
    """Publish only a verified completed aggregate; never overwrite older results."""
    report_path = directory / "aggregate.json"
    status = json.loads((directory / "progress.json").read_bytes())
    if file_sha256(report_path) != status.get("final_aggregate_sha256"):
        raise ValueError("completed aggregate required for export")
    report = json.loads(report_path.read_bytes())
    if (
        report["scope"] != "frozen_test"
        or report["final_test_evaluated"] is not True
        or report["execution_receipt_sha256"] != file_sha256(directory / "receipt.json")
    ):
        raise ValueError("final aggregate execution identity mismatch")
    if (
        file_sha256(directory / "results-private.json")
        != report["private_results_sha256"]
    ):
        raise ValueError("private output identity mismatch")
    # Aggregate schema is constructed by summarize; rows/bootstrap IDs stay private.
    public_fields = {
        "schema_version",
        "protocol_id",
        "scope",
        "final_test_evaluated",
        "samples",
        "class_counts",
        "runs",
        "statistics",
        "uncertainty",
        "inference",
        "postprocessing_timing",
        "limitations",
        "primary_plan_sha256",
        "extension_sha256",
        "execution_receipt_sha256",
        "execution_code_revision",
        "environment",
        "private_results_sha256",
    }
    if set(report) != public_fields:
        raise ValueError("unexpected public aggregate fields")
    lines = [
        "# Locked Phase 3 final evaluation",
        "",
        f"Protocol: {report['protocol_id']}; {report['samples']} "
        "single-target frozen SEntFiN headlines.",
        "",
        "| Candidate | Mean macro F1 | Mean accuracy | NLL | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in report["statistics"].items():
        cells = [
            f"{metrics[m]['mean']:.4f}" if m in metrics else "N/A"
            for m in ("macro_f1", "accuracy", "nll", "brier", "ece")
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Individual seed metrics, sample SD, class metrics, confusion matrices, "
        "reliability bins, paired group bootstrap intervals and timings are in "
        "JSON. Hard-vote/SVM/VADER probability metrics are unavailable. "
        "Raw/calibrated variants share their classifier predictions.",
        "",
        *report["limitations"],
        "",
    ]
    files = {
        destination.with_suffix(".json"): json_body(report),
        destination.with_suffix(".md"): "\n".join(lines).encode(),
    }
    for path, body in files.items():
        if path.exists() and path.read_bytes() != body:
            raise ValueError("refusing to overwrite a different published result")
    for path, body in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    return destination.with_suffix(".json")


def development_smoke(root: Path, data_dir: Path) -> Path:
    """Real saved-model inference on validation only; never decode test records."""
    plan = preflight(root, data_dir)
    rows = sentence_cohorts(
        load_development_records(data_dir, root / "configs/benchmark-freeze.json")
    )["sentfin_single_target"]
    records = _records([r for r in rows if r["partition"] == "validation"])
    identity = {
        "sources": {
            p.relative_to(root).as_posix(): file_sha256(p)
            for p in sorted((root / "src/finsight").rglob("*.py"))
        },
        "plan": file_sha256(root / "configs/final-evaluation-plan.json"),
        "scope": "validation_smoke",
    }
    directory = (
        data_dir / "interim/final-evaluation-development" / sha(json_body(identity))
    )
    raw, metadata = execute_predictions(
        directory, sha(json_body(identity)), records, plan, data_dir, resume=True
    )
    # Check fresh raw predictions against historical validation outputs.
    parity = {}
    for model, runs in plan["transformers"].items():
        for r in runs:
            prior = json.loads((root / r["report_path"]).read_bytes())
            namespace = (
                "finbert-finetuning" if model == "finbert" else "transformer-comparison"
            )
            reference = [
                json.loads(line)
                for line in (
                    data_dir
                    / "interim"
                    / namespace
                    / r["report_sha256"]
                    / "predictions.jsonl"
                )
                .read_bytes()
                .splitlines()
            ]
            actual = raw[f"{model}.{r['seed']}.raw"]
            if [r["prediction"] for r in actual] != [
                r["prediction"] for r in reference
            ]:
                raise ValueError("saved-model validation label parity failed")
            difference = max(
                abs(a["probabilities"][c] - b["probabilities"][c])
                for a, b in zip(actual, reference, strict=True)
                for c in ("negative", "neutral", "positive")
            )
            if not math.isfinite(difference) or difference > 1e-6:
                raise ValueError("saved-model validation probability parity failed")
            parity[f"{model}.{r['seed']}"] = difference
            if (
                evaluate_outputs(records, {"one": actual}, families={"one": ["one"]})[
                    "runs"
                ]["one"]["classification"]
                != prior["metrics"]
            ):
                raise ValueError("saved-model validation metrics parity failed")
    report, private = summarize(records, raw, metadata, plan, scope="validation_smoke")
    report["implementation_sha256"] = identity["sources"]
    report["smoke_probability_max_abs_difference"] = parity
    atomic_json(directory / "results-private.json", private)
    report["private_results_sha256"] = file_sha256(directory / "results-private.json")
    atomic_json(directory / "aggregate.json", report)
    return directory / "aggregate.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--development-smoke", action="store_true")
    actions.add_argument("--export", action="store_true")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if (args.run or args.export) and not args.receipt:
        parser.error("--run/--export require --receipt")
    if not (args.run or args.export) and (args.receipt or args.resume):
        parser.error("--receipt/--resume only accompany --run")
    if args.export and (not args.destination or args.resume):
        parser.error("--export requires --destination and forbids --resume")
    if not args.export and args.destination:
        parser.error("--destination only accompanies --export")
    if args.prepare:
        result = prepare_execution(Path.cwd(), args.data_dir)
    elif args.development_smoke:
        result = development_smoke(Path.cwd(), args.data_dir)
    elif args.export:
        result = export_results(args.receipt.parent, args.destination)
    else:
        result = run_final(Path.cwd(), args.data_dir, args.receipt, resume=args.resume)
    print(f"Private output: {result}")


if __name__ == "__main__":
    main()
