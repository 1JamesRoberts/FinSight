"""Verify a locked evaluation plan without loading models or decoding test records."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

from finsight.sentiment.finbert import file_sha256

MODELS = ("finbert", "roberta_base", "deberta_base")
SEEDS = (42, 43, 44)
BASELINES = (
    "training_prior",
    "vader",
    "tfidf_logreg",
    "tfidf_svm",
    "finbert_pretrained",
)


def safe_path(root: Path, relative: str) -> Path:
    """Restrict manifest paths to their declared repository/data root."""
    path = Path(relative)
    if (
        not relative
        or path.is_absolute()
        or ":" in relative
        or "\\" in relative
        or any(part in {"..", "."} for part in relative.split("/"))
    ):
        raise ValueError("unsafe protocol path")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol path escapes root")
    return resolved


def validate_plan(plan: dict[str, Any]) -> None:
    """Reject scope drift, best-seed selection and mismatched scalar calibrators."""
    if (
        plan["schema_version"] != 1
        or plan["protocol_id"] != "phase3-sentfin-single-target-v1"
        or plan["cohort"] != "sentfin_single_target"
        or plan["expected_samples"] != 766
        or plan["class_order"] != ["negative", "neutral", "positive"]
        or plan["primary_metric"] != "macro_f1"
        or plan["seed_policy"]["seeds"] != list(SEEDS)
        or plan["seed_policy"]["select_best_seed"] is not False
        or plan["seed_policy"]["ensemble"] is not False
        or plan["seed_policy"]["retrain_on_train_plus_validation"] is not False
        or plan["inference"]
        != {
            "device": "cuda",
            "batch_size": 32,
            "max_length": 128,
            "precision": "float32",
            "attention": "eager",
            "tf32": False,
            "deterministic": True,
        }
        or plan["uncertainty"]
        != {
            "unit": "duplicate_group",
            "paired": True,
            "replicates": 1000,
            "seed": 42,
            "interval": "percentile_95",
            "seed_aggregation": "mean_metric",
        }
        or set(plan["transformers"]) != set(MODELS)
        or set(plan["baselines"]) != set(BASELINES)
        or plan["final_test_evaluated"] is not False
    ):
        raise ValueError("unsupported locked evaluation scope")
    artifacts = set()
    for model, runs in plan["transformers"].items():
        if [r["seed"] for r in runs] != list(SEEDS):
            raise ValueError("all three seeds required")
        for r in runs:
            if (
                r["configuration"]["seed"] != r["seed"]
                or r["configuration"]["learning_rate"]
                != {"finbert": 3e-5, "roberta_base": 2e-5, "deberta_base": 1e-5}[model]
                or r["configuration"]["max_length"] != 128
                or r["configuration"]["evaluation_batch_size"] != 32
                or r["calibration"]["source_report_sha256"] != r["report_sha256"]
                or type(r["calibration"]["temperature"]) not in (int, float)
                or not math.isfinite(r["calibration"]["temperature"])
                or not 0.05 <= r["calibration"]["temperature"] <= 20
            ):
                raise ValueError("model/seed/calibration identity mismatch")
            artifacts.add(r["artifact"]["relative_path"])
    if len(artifacts) != 9:
        raise ValueError("nine distinct selected checkpoints required")


def verify_artifact(data_dir: Path, artifact: dict[str, Any], *, full: bool) -> int:
    """Check selected manifests; hash weights only when full=True. Never unpickle."""
    path = safe_path(data_dir, artifact["relative_path"])
    if artifact["kind"] == "transformer":
        manifest_path = path / "artifact-manifest.json"
        if file_sha256(manifest_path) != artifact["manifest_sha256"]:
            raise ValueError("artifact manifest checksum mismatch")
        manifest = json.loads(manifest_path.read_bytes())
        if manifest["files"] != artifact["files"]:
            raise ValueError("artifact file inventory mismatch")
        if not full:
            return 0
    elif artifact["kind"] != "pretrained" and artifact["kind"] != "pickle":
        raise ValueError("unknown artifact kind")
    elif not full:
        return 0
    for name, expected in artifact["files"].items():
        if Path(name).name != name or "\\" in name or name in {".", ".."}:
            raise ValueError("unsafe artifact filename")
        file = (
            safe_path(data_dir, artifact["relative_path"] + "/" + name)
            if artifact["kind"] != "pickle"
            else path
        )
        if artifact["kind"] == "pickle" and name != path.name:
            raise ValueError("pickle filename mismatch")
        if (
            file.stat().st_size != expected["size_bytes"]
            or file_sha256(file) != expected["sha256"]
        ):
            raise ValueError("artifact file checksum mismatch")
    return len(artifact["files"])


def verify_locked_plan(
    root: Path, data_dir: Path, *, full_artifacts: bool = False
) -> dict[str, Any]:
    """Audit protocol, source inputs, split and calibrators; do not score any data."""
    plan_path = root / "configs/final-evaluation-plan.json"
    lock = json.loads((root / "configs/final-evaluation-lock.json").read_bytes())
    if (
        lock["schema_version"] != 1
        or lock["protocol_id"] != "phase3-sentfin-single-target-v1"
        or lock["test_evaluation_performed"] is not False
        or lock["plan_sha256"] != file_sha256(plan_path)
    ):
        raise ValueError("locked plan checksum mismatch")
    plan = json.loads(plan_path.read_bytes())
    validate_plan(plan)
    for relative, expected in plan["repository_inputs"].items():
        if file_sha256(safe_path(root, relative)) != expected:
            raise ValueError(f"protocol input checksum mismatch: {relative}")
    for relative, expected in plan["private_inputs"].items():
        if file_sha256(safe_path(data_dir, relative)) != expected:
            raise ValueError(f"private protocol input checksum mismatch: {relative}")
    # Test JSONL is hashed as opaque bytes only; no text, labels or IDs are decoded.
    for relative, expected in plan["frozen_files"].items():
        if file_sha256(safe_path(data_dir, relative)) != expected:
            raise ValueError("frozen benchmark checksum mismatch")
    calibration = json.loads(
        safe_path(data_dir, plan["calibrators_relative_path"]).read_bytes()
    )
    checked_files = 0
    for model, runs in plan["transformers"].items():
        for run in runs:
            if (
                calibration["models"][model][str(run["seed"])]["temperature"]
                != run["calibration"]["temperature"]
            ):
                raise ValueError("saved temperature mismatch")
            source = next(
                i
                for i in calibration["inputs"]
                if i["model"] == model and i["seed"] == run["seed"]
            )
            if source["report_sha256"] != run["report_sha256"]:
                raise ValueError("calibrator checkpoint ancestry mismatch")
            checked_files += verify_artifact(
                data_dir, run["artifact"], full=full_artifacts
            )
    for baseline in plan["baselines"].values():
        if "artifact" in baseline:
            checked_files += verify_artifact(
                data_dir, baseline["artifact"], full=full_artifacts
            )
    return {
        "protocol_id": plan["protocol_id"],
        "plan_sha256": lock["plan_sha256"],
        "transformer_checkpoints": 9,
        "baselines": len(plan["baselines"]),
        "full_artifacts_verified": full_artifacts,
        "artifact_files_hashed": checked_files,
        "test_records_decoded": False,
        "final_test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--full-artifacts", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            verify_locked_plan(
                Path.cwd(), args.data_dir, full_artifacts=args.full_artifacts
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
