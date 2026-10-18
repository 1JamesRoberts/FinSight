"""Contract checks use public metadata and synthetic files, never real test data."""

import copy
import json
import shutil
from pathlib import Path

import pytest

from finsight.sentiment.artifacts import json_body, sha
from finsight.sentiment.evaluation.target_protocol import (
    LOCK,
    MODELS,
    PLAN,
    selected_model,
    validate_plan,
    verify_plan,
)
from finsight.sentiment.experiments.target_evaluation_plan import build_plan, write_plan

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def contract() -> dict:
    return json.loads((ROOT / PLAN).read_bytes())


@pytest.fixture
def metadata_checkout(tmp_path: Path, contract: dict) -> tuple[Path, Path]:
    """Reconstruct every manifest solely from already-public run metadata."""
    root, data = tmp_path / "repo", tmp_path / "private-data"
    paths = {PLAN, LOCK, *contract["repository_inputs"]}
    paths.update(
        r["report_path"] for runs in contract["transformers"].values() for r in runs
    )
    for relative in paths:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    assets = json.loads((root / "configs/transformer-assets.json").read_bytes())[
        "models"
    ]
    supplement = json.loads((root / "configs/target-backbones-plan.json").read_bytes())[
        "models"
    ]
    for model, runs in contract["transformers"].items():
        for run in runs:
            trial = json.loads((root / run["report_path"]).read_bytes())
            source = (
                assets[model] if model == "deberta_base" else supplement[model]["asset"]
            )
            manifest = {
                "schema_version": 1,
                "source": source,
                "configuration": trial["configuration"],
                "epoch": run["selected_epoch"],
                "training_membership_sha256": trial["training_membership_sha256"],
                "files": run["artifact"]["files"],
                "context": trial["context"],
            }
            body = json_body(manifest)
            assert sha(body) == run["artifact"]["manifest_sha256"]
            directory = data / run["artifact"]["relative_path"]
            directory.mkdir(parents=True)
            (directory / "artifact-manifest.json").write_bytes(body)
    for relative in contract["frozen_files"]:
        path = data / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"DO NOT DECODE THIS SYNTHETIC RESERVED FILE")
    return root, data


def rewrite_contract(root: Path, plan: dict) -> None:
    body = json_body(plan)
    (root / PLAN).write_bytes(body)
    lock = json.loads((root / LOCK).read_bytes())
    lock["plan_sha256"] = sha(body)
    (root / LOCK).write_bytes(json_body(lock))


def test_selection_uses_unrounded_means_and_fixed_exact_tie_order() -> None:
    assert selected_model(dict.fromkeys(MODELS, 0.8)) == "deberta_base"
    assert (
        selected_model(
            {"deberta_base": 0.8, "roberta_base": 0.80000001, "finbert": 0.7}
        )
        == "roberta_base"
    )
    for scores in ({}, dict.fromkeys(MODELS, float("nan")), dict.fromkeys(MODELS, 2)):
        with pytest.raises(ValueError):
            selected_model(scores)


def test_published_contract_matches_validation_evidence(contract: dict) -> None:
    validate_plan(contract)
    assert contract["selection"]["selected_model"] == "roberta_base"
    assert contract["cohorts"]["expected"]["multi_target"]["target_annotations"] == 594
    assert contract["cohorts"]["expected"]["multi_target"]["headlines"] == 263
    assert len(contract["paired_comparisons"]) == 7


def test_metadata_verification_never_reads_frozen_files(
    metadata_checkout, monkeypatch
) -> None:
    root, data = metadata_checkout
    read = Path.read_bytes

    def guarded(path: Path) -> bytes:
        if "processed/benchmarks" in path.as_posix():
            raise AssertionError("planning must not read reserved files")
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    assert build_plan(root, data) == json.loads((root / PLAN).read_bytes())
    result = verify_plan(root, data)
    assert result["transformer_checkpoints"] == 18
    assert result["full_artifacts_verified"] is False
    assert result["test_scored"] is result["test_records_decoded"] is False


@pytest.mark.parametrize(
    "change",
    [
        "calibration",
        "best_seed",
        "ensemble",
        "missing_seed",
        "duplicate_checkpoint",
        "comparison",
        "candidate",
        "selection_evidence",
        "cohort",
        "unsafe_path",
        "baseline",
        "epoch",
    ],
)
def test_contract_rejects_drift_even_with_updated_plan_hash(
    metadata_checkout, contract, change
) -> None:
    root, data = metadata_checkout
    plan = copy.deepcopy(contract)
    run = plan["transformers"]["deberta_base"][0]
    if change == "calibration":
        plan["inference"]["calibration"] = "fit_on_test"
    elif change == "best_seed":
        plan["seed_policy"]["seeds"] = [42]
    elif change == "ensemble":
        plan["seed_policy"]["ensemble"] = True
    elif change == "missing_seed":
        plan["transformers"]["deberta_base"].pop()
    elif change == "duplicate_checkpoint":
        run["artifact"]["manifest_sha256"] = plan["transformers"]["deberta_base"][1][
            "artifact"
        ]["manifest_sha256"]
    elif change == "comparison":
        plan["paired_comparisons"][0]["left"] = "finbert.target_pair"
    elif change == "candidate":
        plan["selection"]["selected_model"] = "deberta_base"
    elif change == "selection_evidence":
        plan["selection"]["validation_scores"]["roberta_base"] = 0.99
    elif change == "cohort":
        plan["cohorts"]["primary"] = "all"
    elif change == "unsafe_path":
        run["artifact"]["relative_path"] = "../outside"
    elif change == "baseline":
        plan["baselines"]["training_prior"]["counts"]["negative"] += 1
    else:
        run["selected_epoch"] += 1
    rewrite_contract(root, plan)
    with pytest.raises(ValueError):
        verify_plan(root, data)


@pytest.mark.parametrize("change", ["lock", "source", "trial", "manifest"])
def test_corrupt_receipts_are_rejected(metadata_checkout, contract, change) -> None:
    root, data = metadata_checkout
    run = contract["transformers"]["deberta_base"][0]
    path = {
        "lock": root / PLAN,
        "source": root / "configs/benchmark-freeze.json",
        "trial": root / run["report_path"],
        "manifest": data / run["artifact"]["relative_path"] / "artifact-manifest.json",
    }[change]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        verify_plan(root, data)


def test_full_verification_requires_actual_weights(metadata_checkout) -> None:
    root, data = metadata_checkout
    with pytest.raises(FileNotFoundError):
        verify_plan(root, data, full_artifacts=True)


@pytest.mark.parametrize("existing", [PLAN, LOCK])
def test_writer_preserves_partial_existing_contract(
    tmp_path: Path, existing: str
) -> None:
    path = tmp_path / existing
    path.parent.mkdir(parents=True)
    path.write_bytes(b"preserve original")
    with pytest.raises(FileExistsError):
        write_plan(tmp_path, tmp_path / "data")
    assert path.read_bytes() == b"preserve original"
