"""Explicit report enrichment using trusted local fitted artifacts."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from finsight.sentiment.artifacts import project_root, source_manifest
from finsight.sentiment.evaluation.model_profile import (
    profile_linear_pipeline as profile_linear_pipeline,
)


def enrich_report(report_path: Path, data_dir: Path) -> dict[str, Any]:
    """Append profiles to an existing report; preserve the original local run."""
    from finsight.sentiment.experiments.tfidf_validation import render_report
    from finsight.sentiment.models.svm import TfidfSVM
    from finsight.sentiment.models.tfidf import TfidfSentiment

    aggregate_path = report_path.with_suffix(".json")
    published = json.loads(aggregate_path.read_text(encoding="utf-8"))
    original_id = published.get("report_enrichment", {}).get("original_run_id")
    if original_id is None:
        body = json.dumps(published, indent=2, sort_keys=True) + "\n"
        original_id = hashlib.sha256(body.encode()).hexdigest()
    prefix = published["experiment"].removesuffix("-validation-v1")
    original_path = data_dir / f"interim/{prefix}-validation" / original_id / "run.json"
    original_body = original_path.read_bytes()
    if hashlib.sha256(original_body).hexdigest() != original_id:
        raise ValueError("original run checksum mismatch")
    run = json.loads(original_body)
    kind = run.get("model_kind", "logistic_regression")
    model_class = TfidfSentiment if kind == "logistic_regression" else TfidfSVM
    for result in run["metrics"].values():
        artifact = data_dir / result["artifact_relative_path"]
        model = model_class.load(artifact, expected_sha256=result["artifact_sha256"])
        result["model_profile"] = profile_linear_pipeline(
            model.pipeline, artifact, result["artifact_sha256"]
        )
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
    run["report_enrichment"] = {
        "original_run_id": original_id,
        "code_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "working_tree_dirty": bool(dirty.stdout) if dirty.returncode == 0 else None,
        "implementation_sha256": source_manifest(),
        "scope": "read original fitted artifacts only; no fitting or prediction",
    }
    report_path.write_text(render_report(run, original_id), encoding="utf-8")
    aggregate_path.write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run


def main() -> None:
    from finsight.config import Settings

    parser = argparse.ArgumentParser(description="Add profiles without retraining.")
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--data-dir", type=Path, default=Settings.from_env().data_dir)
    args = parser.parse_args()
    for report in args.reports:
        enrich_report(report, args.data_dir)
        print(f"Model architecture/size added to {report}")


if __name__ == "__main__":
    main()
