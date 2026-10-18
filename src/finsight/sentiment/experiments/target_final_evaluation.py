"""Receipt-bound target evaluation; default command only verifies metadata."""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from finsight.sentiment.artifacts import file_sha256, json_body, sha, source_manifest
from finsight.sentiment.evaluation.final_metrics import validate_rows
from finsight.sentiment.evaluation.target_data import (
    development_inputs,
    evaluation_rows,
    frozen_inputs,
)
from finsight.sentiment.evaluation.target_metrics import families, summarize
from finsight.sentiment.evaluation.target_protocol import (
    LOCK,
    PLAN,
    PROTOCOL,
    verify_plan,
)
from finsight.sentiment.experiments.final_evaluation import (
    atomic_json,
    code_identity,
    environment,
    session_lock,
)
from finsight.sentiment.models.target_inference import TargetText, predict

NAMESPACE = "interim/target-final-evaluation"


def candidates(plan: dict) -> dict[str, dict]:
    variants = {}
    for model, runs in plan["transformers"].items():
        for run in runs:
            variants[f"{model}.{run['arm']}.{run['seed']}"] = {
                **run,
                "kind": "transformer",
                "model": model,
            }
    for arm in ("headline_only", "headline_target"):
        variants[f"tfidf.{arm}"] = {
            **plan["baselines"][f"tfidf.{arm}"],
            "kind": "tfidf",
            "arm": "matched_headline_only" if arm == "headline_only" else "target_pair",
        }
    variants["training_prior"] = {
        **plan["baselines"]["training_prior"],
        "kind": "prior",
        "arm": "matched_headline_only",
    }
    if set(variants) != {k for keys in families().values() for k in keys}:
        raise ValueError("locked candidate variants differ")
    return variants


def prepare_execution(root: Path, data_dir: Path) -> Path:
    """Verify every artifact and bind code/environment without decoding the test."""
    verify_plan(root, data_dir, full_artifacts=True)
    _, _, development = development_inputs(root, data_dir)
    receipt = {
        "schema_version": 1,
        "protocol_id": PROTOCOL,
        "plan_sha256": file_sha256(root / PLAN),
        "lock_sha256": file_sha256(root / LOCK),
        "development": development,
        "code": code_identity(root),
        "environment": environment(),
        "test_records_decoded": False,
    }
    body = json_body(receipt)
    directory = data_dir / NAMESPACE / sha(body)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "receipt.json"
    if path.exists() and path.read_bytes() != body:
        raise ValueError("existing receipt differs")
    path.write_bytes(body)
    return path


def checked_receipt(root: Path, data_dir: Path, path: Path) -> dict:
    body = path.read_bytes()
    if path.resolve() != (data_dir / NAMESPACE / sha(body) / "receipt.json").resolve():
        raise ValueError("target receipt must be content-addressed in private storage")
    receipt = json.loads(body)
    _, _, development = development_inputs(root, data_dir)
    if receipt != {
        "schema_version": 1,
        "protocol_id": PROTOCOL,
        "plan_sha256": file_sha256(root / PLAN),
        "lock_sha256": file_sha256(root / LOCK),
        "development": development,
        "code": code_identity(root),
        "environment": environment(),
        "test_records_decoded": False,
    }:
        raise ValueError("target execution context differs from receipt")
    return receipt


def execute_predictions(
    directory: Path,
    receipt_id: str,
    records: list[dict],
    plan: dict,
    data_dir: Path,
    *,
    resume: bool,
    predictor: Any = predict,
) -> tuple[dict, dict]:
    """Cache verified variants; failed predictions are never marked complete."""
    directory.mkdir(parents=True, exist_ok=True)
    progress = directory / "progress.json"
    if progress.exists() and not resume:
        raise ValueError("existing execution requires explicit resume")
    status = (
        json.loads(progress.read_bytes())
        if progress.exists()
        else {
            "receipt_sha256": receipt_id,
            "membership_sha256": sha(json_body(records)),
            "completed": {},
            "attempts": [],
        }
    )
    if status["receipt_sha256"] != receipt_id or status["membership_sha256"] != sha(
        json_body(records)
    ):
        raise ValueError("execution receipt or membership differs")
    attempt = {
        "started_utc": datetime.now(UTC).isoformat(),
        "status": "running",
        "computed": [],
        "reused": [],
    }
    status["attempts"].append(attempt)
    atomic_json(progress, status)
    outputs, metadata = {}, {}
    # Labels remain in evaluation. The predictor receives this separate text-only DTO.
    inputs = [TargetText(r["headline_id"], r["headline"], r["target"]) for r in records]
    try:
        for key, spec in candidates(plan).items():
            path = directory / "predictions" / f"{key}.json"
            identity = {
                "receipt_sha256": receipt_id,
                "membership_sha256": status["membership_sha256"],
                "candidate": key,
                "spec_sha256": sha(json_body(spec)),
            }
            if key in status["completed"] and not path.exists():
                raise ValueError("completed checkpoint missing; refusing rescoring")
            if path.exists():
                body = path.read_bytes()
                if (
                    key not in status["completed"]
                    or sha(body) != status["completed"][key]
                ):
                    raise ValueError("unchecked or corrupt prediction checkpoint")
                saved = json.loads(body)
                if saved["identity"] != identity:
                    raise ValueError("prediction checkpoint identity differs")
                rows, meta = saved["rows"], saved["metadata"]
                attempt["reused"].append(key)
            else:
                predictions, meta = predictor(spec, inputs, data_dir)
                if len(predictions) != len(records) or any(
                    set(p) != {"prediction", "probabilities"} for p in predictions
                ):
                    raise ValueError(
                        "incomplete or unexpected prediction outputs; no silent drops"
                    )
                rows = [
                    {**p, "record_id": r["record_id"], "gold": r["gold"]}
                    for r, p in zip(records, predictions, strict=True)
                ]
                validate_rows(records, rows)
                atomic_json(
                    path, {"identity": identity, "rows": rows, "metadata": meta}
                )
                attempt["computed"].append(key)
            if any(
                set(row) != {"prediction", "probabilities", "record_id", "gold"}
                for row in rows
            ):
                raise ValueError("cached prediction fields differ")
            validate_rows(records, rows)
            status["completed"][key] = file_sha256(path)
            outputs[key], metadata[key] = rows, meta
            atomic_json(progress, status)
        attempt["status"] = "predictions_complete"
        return outputs, metadata
    except BaseException as error:
        attempt.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        attempt["finished_utc"] = datetime.now(UTC).isoformat()
        atomic_json(progress, status)


def run_final(
    root: Path, data_dir: Path, receipt_path: Path, *, resume: bool = False
) -> Path:
    """All artifact, receipt, code and membership gates precede model inference."""
    verify_plan(root, data_dir, full_artifacts=True)
    receipt = checked_receipt(root, data_dir, receipt_path)
    plan = json.loads((root / PLAN).read_bytes())
    directory = receipt_path.parent
    with session_lock(directory):
        aggregate = directory / "aggregate.json"
        progress = directory / "progress.json"
        status = json.loads(progress.read_bytes()) if progress.exists() else {}
        if "final_aggregate_sha256" in status:
            if not resume or file_sha256(aggregate) != status["final_aggregate_sha256"]:
                raise ValueError(
                    "completed evaluation requires verified explicit resume"
                )
            report = json.loads(aggregate.read_bytes())
            if (
                report["execution_receipt_sha256"] != file_sha256(receipt_path)
                or file_sha256(directory / "results-private.json")
                != report["private_results_sha256"]
            ):
                raise ValueError("completed private results or receipt differs")
            return aggregate
        training, validation, _ = development_inputs(root, data_dir)
        records = frozen_inputs(root, data_dir, plan, training + validation)
        outputs, metadata = execute_predictions(
            directory, file_sha256(receipt_path), records, plan, data_dir, resume=resume
        )
        try:
            report, private = summarize(
                records, outputs, {r["target"] for r in training}, plan
            )
            private["predictions"] = outputs
            atomic_json(directory / "results-private.json", private)
            report.update(
                protocol_id=PROTOCOL,
                scope="reserved_multi_target_test",
                cohort_contract=plan["cohorts"],
                test_scored=True,
                plan_sha256=file_sha256(root / PLAN),
                lock_sha256=file_sha256(root / LOCK),
                selected_model=plan["selection"]["selected_model"],
                selection_changed=False,
                execution_receipt_sha256=file_sha256(receipt_path),
                execution_code_revision=receipt["code"]["revision"],
                environment=receipt["environment"],
                model_metadata=metadata,
                private_results_sha256=file_sha256(directory / "results-private.json"),
            )
            atomic_json(aggregate, report)
            status = json.loads(progress.read_bytes())
            status["final_aggregate_sha256"] = file_sha256(aggregate)
            status["attempts"][-1]["status"] = "complete"
            atomic_json(progress, status)
            return aggregate
        except BaseException as error:
            status = json.loads(progress.read_bytes())
            status["attempts"][-1].update(
                status="failed_postprocessing", error_type=type(error).__name__
            )
            atomic_json(progress, status)
            raise


def render_report(report: dict) -> str:
    """Render aggregates only; native names, texts and bootstrap draws stay private."""
    lines = [
        "# Phase 4 reserved multi-target evaluation",
        "",
        f"Protocol: `{PROTOCOL}`.",
        "",
        "One previously evaluated near-duplicate group is excluded: "
        "262 multi-target headlines / 592 annotations remain. "
        "The original frozen benchmark and Phase 3 results are unchanged.",
        "",
        "The validation-selected candidate remains RoBERTa Base; "
        "test results do not change selection.",
        "Raw probabilities are uncalibrated. "
        "Cross-backbone and subgroup comparisons are secondary.",
    ]
    for name, cohort in report["cohorts"].items():
        lines += [
            "",
            f"## {name}",
            "",
            f"Annotations: {cohort['examples']}; "
            f"headlines: {cohort['headlines']}; "
            f"duplicate groups: {cohort['duplicate_groups']}.",
        ]
        if cohort["metrics"] is None:
            lines += ["", "Unavailable: empty declared cohort."]
            continue
        lines += [
            "",
            "| Model / input | Macro F1 mean (%) | Seed SD (pp) | Accuracy (%) |",
            "| --- | ---: | ---: | ---: |",
        ]
        for family, values in cohort["metrics"]["statistics"].items():
            score, accuracy = values["macro_f1"], values["accuracy"]
            sd = (
                "n/a"
                if score["sample_std"] is None
                else f"{100 * score['sample_std']:.2f}"
            )
            lines.append(
                f"| {family} | {100 * score['mean']:.2f} | {sd} | "
                f"{100 * accuracy['mean']:.2f} |"
            )
        lines += [
            "",
            "| Paired contrast | Difference (pp) | Paired 95% interval (pp) |",
            "| --- | ---: | ---: |",
        ]
        for key, difference in cohort["paired_differences"].items():
            low, high = difference["interval_95"]
            lines.append(
                f"| {key} | {100 * difference['macro_f1_difference']:.2f} | "
                f"[{100 * low:.2f}, {100 * high:.2f}] |"
            )
    lines += [
        "",
        "Intervals use 1,000 paired duplicate-group bootstrap draws "
        "and average seed metrics within each draw.",
        "Seed SD describes variation across trained seeds; "
        "it is not a confidence interval.",
        "Native entities are not verified companies; "
        "sparse per-target results remain private.",
        "Checkpoint pretraining exposure is unknown. "
        "These results do not establish five-company generalization.",
        "",
    ]
    return "\n".join(lines)


def export_results(root: Path, directory: Path, destination: Path) -> tuple[Path, Path]:
    """Publish a completed aggregate once; verify every bound digest before writing."""
    report_path = directory / "aggregate.json"
    status = json.loads((directory / "progress.json").read_bytes())
    report = json.loads(report_path.read_bytes())
    if (
        status.get("final_aggregate_sha256") != file_sha256(report_path)
        or report.get("protocol_id") != PROTOCOL
        or report.get("scope") != "reserved_multi_target_test"
        or report.get("test_scored") is not True
        or report["execution_receipt_sha256"] != file_sha256(directory / "receipt.json")
        or report["private_results_sha256"]
        != file_sha256(directory / "results-private.json")
        or report["plan_sha256"] != file_sha256(root / PLAN)
        or report["lock_sha256"] != file_sha256(root / LOCK)
        or report["selection_changed"] is not False
    ):
        raise ValueError("completed export identity/checksums differ")
    outputs = (destination.with_suffix(".json"), destination.with_suffix(".md"))
    if any(p.exists() for p in outputs):
        raise FileExistsError("published target results already exist")
    destination.parent.mkdir(parents=True, exist_ok=True)
    outputs[0].write_bytes(report_path.read_bytes())
    outputs[1].write_text(render_report(report), encoding="utf-8")
    return outputs


def development_smoke(root: Path, data_dir: Path) -> Path:
    """Replay transformer validation outputs and predict sparse controls on CPU."""
    verify_plan(root, data_dir)
    plan = json.loads((root / PLAN).read_bytes())
    training, validation, development = development_inputs(root, data_dir)
    records = evaluation_rows([r for r in validation if r["multi_target"]])
    caches = {}
    for spec in candidates(plan).values():
        if spec["kind"] != "transformer":
            continue
        trial = json.loads((root / spec["report_path"]).read_bytes())
        path = (
            data_dir
            / "interim/target-deberta"
            / spec["report_sha256"]
            / "predictions.jsonl"
        )
        if file_sha256(path) != trial["predictions_sha256"]:
            raise ValueError("development replay prediction bytes differ")
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        if len(rows) != len(validation) or any(
            p["example_id"] != r["example_id"] or p["gold"] != r["label"]
            for r, p in zip(validation, rows, strict=True)
        ):
            raise ValueError("development replay membership/gold differs")
        caches[(spec["model"], spec["arm"], spec["seed"])] = {
            (r["record_id"], r["target"]): {
                "prediction": r["prediction"],
                "probabilities": r["probabilities"],
            }
            for r in rows
        }

    def replay(
        spec: dict, inputs: list[TargetText], data: Path
    ) -> tuple[list[dict], dict]:
        if spec["kind"] != "transformer":
            return predict(spec, inputs, data)
        cached = caches[(spec["model"], spec["arm"], spec["seed"])]
        return [cached[(r.record_id, r.target)] for r in inputs], {
            "mode": "verified_saved_validation_replay",
            "new_transformer_inference": False,
        }

    directory = (
        data_dir
        / "interim/target-evaluation-smoke"
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    )
    outputs, metadata = execute_predictions(
        directory,
        "development-only",
        records,
        plan,
        data_dir,
        resume=False,
        predictor=replay,
    )
    report, private = summarize(records, outputs, {r["target"] for r in training}, plan)
    report.update(
        scope="development_smoke_not_test",
        plan_sha256=file_sha256(root / PLAN),
        lock_sha256=file_sha256(root / LOCK),
        code_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        working_tree_dirty=bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            ).strip()
        ),
        implementation_sha256=source_manifest(),
        test_scored=False,
        transformer_mode="saved_validation_replay",
        sparse_mode="fresh_saved_model_cpu_inference",
        development=development,
        model_metadata=metadata,
    )
    atomic_json(directory / "results-private.json", private)
    atomic_json(directory / "aggregate.json", report)
    return directory / "aggregate.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--run", type=Path, metavar="RECEIPT")
    actions.add_argument("--development-smoke", action="store_true")
    actions.add_argument("--export", type=Path, metavar="RESULT_DIRECTORY")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/experiments/phase4-target-final"),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and not args.run:
        parser.error("--resume requires --run RECEIPT")
    root = Path.cwd()
    if args.prepare:
        print(prepare_execution(root, args.data_dir))
    elif args.run:
        print(run_final(root, args.data_dir, args.run, resume=args.resume))
    elif args.export:
        for path in export_results(root, args.export, args.report_path):
            print(path)
    elif args.development_smoke:
        print(development_smoke(root, args.data_dir))
    else:
        print(json.dumps(verify_plan(root, args.data_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
