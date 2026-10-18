"""Behavior and compatibility checks for the responsibility-based package layout."""

import hashlib
import importlib
import pickle
import subprocess
from pathlib import Path

import pytest

from finsight.sentiment.artifacts import (
    historical_context,
    source_manifest,
    verify_historical_sources,
)


@pytest.mark.parametrize(
    ("legacy", "canonical"),
    [
        ("finbert", "models.finbert"),
        ("tfidf", "models.tfidf"),
        ("transformer_comparison", "experiments.sentence_comparison"),
        ("target_backbones", "experiments.target_comparison"),
        ("target_training", "training.target"),
        ("final_evaluation", "experiments.final_evaluation"),
    ],
)
def test_legacy_import_is_same_module(legacy: str, canonical: str) -> None:
    assert importlib.import_module(
        f"finsight.sentiment.{legacy}"
    ) is importlib.import_module(f"finsight.sentiment.{canonical}")


def test_historical_pickle_class_reference_remains_loadable() -> None:
    from finsight.sentiment.models.tfidf import TfidfConfig

    # This is a trusted, synthetic protocol-0 class reference from the old layout.
    assert pickle.loads(b"cfinsight.sentiment.tfidf\nTfidfConfig\n.") is TfidfConfig


def test_current_manifest_includes_extracted_dependencies() -> None:
    manifest = source_manifest()
    for name in (
        "artifacts.py",
        "models/transformer.py",
        "training/config.py",
        "training/sentence.py",
        "training/target.py",
        "evaluation/metrics.py",
        "evaluation/calibration.py",
    ):
        path = Path("src/finsight/sentiment") / name
        assert (
            manifest[path.as_posix()] == hashlib.sha256(path.read_bytes()).hexdigest()
        )


@pytest.fixture
def historical_source(tmp_path: Path) -> tuple[Path, str, dict[str, str]]:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    path = tmp_path / "src/finsight/sentiment/old.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"original implementation\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True
    ).strip()
    path.unlink()  # A moved file must still verify against its historical commit.
    return tmp_path, revision, {"src/finsight/sentiment/old.py": digest}


def test_historical_verification_uses_recorded_commit(historical_source: tuple) -> None:
    root, revision, sources = historical_source
    verify_historical_sources(sources, revision, root=root)
    recorded = {"input_mode": "target_pair", "implementation_sha256": sources}
    expected = {"input_mode": "target_pair", "implementation_sha256": {"new": "code"}}
    assert historical_context(recorded, expected, revision, root=root) == recorded
    with pytest.raises(ValueError, match="context"):
        historical_context(
            recorded, {**expected, "input_mode": "other"}, revision, root=root
        )


@pytest.mark.parametrize("change", ["digest", "path", "revision", "empty"])
def test_historical_verification_fails_closed(
    historical_source: tuple, change: str
) -> None:
    root, revision, sources = historical_source
    if change == "digest":
        sources = {next(iter(sources)): "0" * 64}
    elif change == "path":
        sources = {"../secret": "0" * 64}
    elif change == "revision":
        revision = "HEAD"
    else:
        sources = {}
    with pytest.raises(ValueError):
        verify_historical_sources(sources, revision, root=root)
