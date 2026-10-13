"""Protocol integrity and opaque artifact checks; no real test records or weights."""

import copy
import json
from pathlib import Path

import pytest

from finsight.sentiment.evaluation_protocol import (
    safe_path,
    validate_plan,
    verify_artifact,
    verify_locked_plan,
)
from finsight.sentiment.finbert import file_sha256
from finsight.sentiment.transformer_analysis import json_body


def plan():
    return json.loads(Path("configs/final-evaluation-plan.json").read_bytes())


@pytest.mark.parametrize(
    "change",
    [
        "seed",
        "ensemble",
        "retrain",
        "cohort",
        "temperature",
        "ancestry",
        "missing_seed",
        "shared_weights",
    ],
)
def test_reject_protocol_scope_drift(change):
    value = plan()
    validate_plan(value)
    if change in {"seed", "ensemble", "retrain"}:
        field = {
            "seed": "select_best_seed",
            "ensemble": "ensemble",
            "retrain": "retrain_on_train_plus_validation",
        }[change]
        value["seed_policy"][field] = True
    elif change == "cohort":
        value["cohort"] = "all_targets"
    elif change == "missing_seed":
        value["transformers"]["finbert"].pop()
    elif change == "shared_weights":
        value["transformers"]["finbert"][1]["artifact"] = copy.deepcopy(
            value["transformers"]["finbert"][0]["artifact"]
        )
    else:
        calibration = value["transformers"]["finbert"][0]["calibration"]
        if change == "temperature":
            calibration["temperature"] = float("nan")
        else:
            calibration["source_report_sha256"] = "wrong-checkpoint"
    with pytest.raises(ValueError):
        validate_plan(value)


@pytest.mark.parametrize(
    "relative", ["../outside", "/absolute", "a/../b", "a\\b", "C:/outside", ""]
)
def test_unsafe_paths_rejected(tmp_path, relative):
    with pytest.raises(ValueError):
        safe_path(tmp_path, relative)


def test_manifest_only_mode_does_not_open_weight_files(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    manifest = {"files": {"model.safetensors": {"sha256": "absent", "size_bytes": 123}}}
    (root / "artifact-manifest.json").write_bytes(json_body(manifest))
    artifact = {
        "kind": "transformer",
        "relative_path": "checkpoint",
        "manifest_sha256": file_sha256(root / "artifact-manifest.json"),
        "files": manifest["files"],
    }
    assert verify_artifact(tmp_path, artifact, full=False) == 0
    with pytest.raises(FileNotFoundError):
        verify_artifact(tmp_path, artifact, full=True)
    (root / "artifact-manifest.json").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="manifest"):
        verify_artifact(tmp_path, artifact, full=False)


def test_full_artifact_hashing_and_corruption_without_unpickling(tmp_path):
    file = tmp_path / "model.pkl"
    file.write_bytes(b"opaque model bytes")
    artifact = {
        "kind": "pickle",
        "relative_path": "model.pkl",
        "files": {
            "model.pkl": {
                "sha256": file_sha256(file),
                "size_bytes": file.stat().st_size,
            }
        },
    }
    assert verify_artifact(tmp_path, artifact, full=True) == 1
    file.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_artifact(tmp_path, artifact, full=True)


@pytest.mark.parametrize(
    "corruption", [None, "lock", "source", "frozen", "temperature"]
)
def test_locked_verifier_opaque_test_bytes_and_calibrator_lineage(tmp_path, corruption):
    value = plan()
    root, data = tmp_path / "repo", tmp_path / "data"
    (root / "configs").mkdir(parents=True)
    data.mkdir()
    (root / "source.py").write_bytes(b"source")
    (data / "test-records.jsonl").write_bytes(b"\xff\x00opaque frozen bytes")
    value["repository_inputs"] = {"source.py": file_sha256(root / "source.py")}
    value["frozen_files"] = {
        "test-records.jsonl": file_sha256(data / "test-records.jsonl")
    }
    calibration = {"models": {}, "inputs": []}
    for model, runs in value["transformers"].items():
        calibration["models"][model] = {}
        for run in runs:
            directory = data / run["artifact"]["relative_path"]
            directory.mkdir(parents=True)
            manifest_path = directory / "artifact-manifest.json"
            manifest_path.write_bytes(json_body({"files": run["artifact"]["files"]}))
            run["artifact"]["manifest_sha256"] = file_sha256(manifest_path)
            calibration["models"][model][str(run["seed"])] = {
                "temperature": run["calibration"]["temperature"]
            }
            calibration["inputs"].append(
                {
                    "model": model,
                    "seed": run["seed"],
                    "report_sha256": run["report_sha256"],
                }
            )
    if corruption == "temperature":
        calibration["models"]["finbert"]["42"]["temperature"] += 0.1
    (data / "calibrators.json").write_bytes(json_body(calibration))
    value["calibrators_relative_path"] = "calibrators.json"
    value["private_inputs"] = {
        "calibrators.json": file_sha256(data / "calibrators.json")
    }
    plan_path = root / "configs/final-evaluation-plan.json"
    plan_path.write_bytes(json_body(value))
    lock = {
        "schema_version": 1,
        "protocol_id": value["protocol_id"],
        "test_evaluation_performed": False,
        "plan_sha256": file_sha256(plan_path),
    }
    (root / "configs/final-evaluation-lock.json").write_bytes(json_body(lock))
    if corruption == "lock":
        plan_path.write_bytes(plan_path.read_bytes() + b" ")
    if corruption == "source":
        (root / "source.py").write_bytes(b"changed")
    if corruption == "frozen":
        (data / "test-records.jsonl").write_bytes(b"changed")
    if corruption:
        with pytest.raises(ValueError):
            verify_locked_plan(root, data)
        return
    result = verify_locked_plan(root, data)
    assert result["test_records_decoded"] is False
    assert result["final_test_evaluated"] is False
    assert result["artifact_files_hashed"] == 0
