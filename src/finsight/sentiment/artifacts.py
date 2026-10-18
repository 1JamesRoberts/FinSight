"""Checksums, immutable checkpoint manifests and save verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any


def file_sha256(path: Path) -> str:
    """Hash checkpoint files without reading all weights into Python memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_finetuned_artifact(directory: Path, manifest_sha256: str) -> dict[str, Any]:
    """Verify saved model/tokenizer bytes and ancestry before optional imports."""
    from finsight.sentiment.models.finbert import MODEL_ID, REVISION

    manifest_path = directory / "artifact-manifest.json"
    if file_sha256(manifest_path) != manifest_sha256:
        raise ValueError("trained artifact manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_bytes())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("model_id") != MODEL_ID
        or manifest.get("revision") != REVISION
    ):
        raise ValueError("trained artifact ancestry differs")
    files = manifest.get("files", {})
    if (
        not {"model.safetensors", "config.json", "tokenizer_config.json", "vocab.txt"}
        <= files.keys()
    ):
        raise ValueError("trained artifact files are incomplete")
    for name, item in files.items():
        if Path(name).name != name or name in {".", ".."} or "\\" in name:
            raise ValueError("unsafe trained artifact path")
        path = directory / name
        if (
            path.stat().st_size != item["size_bytes"]
            or file_sha256(path) != item["sha256"]
        ):
            raise ValueError("trained artifact file checksum mismatch")
    return manifest


def verify_files(directory: Path, files: dict) -> None:
    for name, expected in files.items():
        if Path(name).name != name or "\\" in name or name in {".", ".."}:
            raise ValueError("unsafe checkpoint filename")
        path = directory / name
        if (
            path.stat().st_size != expected["size_bytes"]
            or file_sha256(path) != expected["sha256"]
        ):
            raise ValueError(f"checkpoint checksum differs: {name}")


def save_model(
    model: Any,
    tokenizer: Any,
    data_dir: Path,
    asset: dict,
    config: FineTuneConfig,
    epoch: int,
    membership: str,
    *,
    context: dict | None = None,
) -> tuple[Path, str]:
    root = data_dir / "interim/transformer-finetuned-models"
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=root) as temporary:
        stage = Path(temporary) / "checkpoint"
        model.save_pretrained(stage, safe_serialization=True)
        tokenizer.save_pretrained(stage)
        manifest = {
            "schema_version": 1,
            "source": asset,
            "configuration": asdict(config),
            "epoch": epoch,
            "training_membership_sha256": membership,
            "files": {
                p.name: {"sha256": file_sha256(p), "size_bytes": p.stat().st_size}
                for p in sorted(stage.iterdir())
                if p.is_file()
            },
        }
        if context is not None:
            manifest["context"] = context
        body = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        # The manifest hash names the complete checkpoint, not just its weights.
        identifier = hashlib.sha256(body).hexdigest()
        (stage / "artifact-manifest.json").write_bytes(body)
        destination = root / identifier
        if destination.exists():
            if (destination / "artifact-manifest.json").read_bytes() != body:
                raise ValueError("artifact manifest differs")
            verify_files(destination, manifest["files"])
        else:
            stage.replace(destination)
    return destination, identifier


def save_artifact(
    model: Any,
    tokenizer: Any,
    root: Path,
    config: FineTuneConfig,
    epoch: int,
    training_membership: str,
) -> tuple[Path, str]:
    """Save model/tokenizer as a complete, immutable content-addressed artifact."""
    from finsight.sentiment.models.finbert import MODEL_ID, REVISION

    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=root) as temporary:
        stage = Path(temporary) / "checkpoint"
        model.save_pretrained(stage, safe_serialization=True)
        tokenizer.save_pretrained(stage)
        files = {
            path.name: {"size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in sorted(stage.iterdir())
            if path.is_file()
        }
        manifest = {
            "schema_version": 1,
            "model_id": MODEL_ID,
            "revision": REVISION,
            "strategy": "adapt_existing_sentiment_checkpoint",
            "cohort": "sentfin_single_target",
            "epoch": epoch,
            "training_membership_sha256": training_membership,
            "configuration": asdict(config),
            "files": files,
            "weight_license_spdx": None,
            "use_decision": "user_authorized_local_research_license_scope_unresolved",
        }
        body = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        # The manifest hash names the complete checkpoint, not just its weights.
        identifier = hashlib.sha256(body).hexdigest()
        (stage / "artifact-manifest.json").write_bytes(body)
        destination = root / identifier
        if destination.exists():
            verify_finetuned_artifact(destination, identifier)
        else:
            stage.replace(destination)
    return destination, identifier


if TYPE_CHECKING:
    from finsight.sentiment.training.config import FineTuneConfig


def project_root() -> Path:
    """Source-checkout root for the existing repository-based experiment commands."""
    return Path(__file__).resolve().parents[3]


def source_manifest() -> dict[str, str]:
    """Record all project Python sources, including helpers extracted by refactors."""
    root = project_root()
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted((root / "src/finsight").rglob("*.py"))
    }


def verify_historical_sources(
    sources: dict[str, str], revision: str, *, root: Path | None = None
) -> None:
    """Verify old source hashes against Git objects without checking out code.

    This is for read-only receipt audits. Training resumes still compare against
    current sources and reject a changed implementation.
    """
    import re
    import subprocess
    from pathlib import PurePosixPath

    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("historical source revision must be a full Git commit ID")
    if not sources:
        raise ValueError("historical source manifest is empty")
    for name, digest in sources.items():
        path = PurePosixPath(name)
        if (
            not name.startswith("src/finsight/")
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or path.as_posix() != name
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ValueError("invalid historical source path or checksum")
        result = subprocess.run(
            ["git", "show", f"{revision}:{name}"],
            cwd=root or project_root(),
            capture_output=True,
            check=False,
        )
        if result.returncode or hashlib.sha256(result.stdout).hexdigest() != digest:
            raise ValueError(f"historical source checksum differs: {name}")


def historical_context(
    recorded: dict, expected: dict, revision: str, *, root: Path | None = None
) -> dict:
    """Validate current data/experiment identity and the original implementation."""
    source_key = "implementation_sha256"
    if {k: v for k, v in recorded.items() if k != source_key} != {
        k: v for k, v in expected.items() if k != source_key
    }:
        raise ValueError("historical experiment context differs")
    verify_historical_sources(recorded.get(source_key, {}), revision, root=root)
    return recorded


def sha(body: bytes) -> str:
    """Hash serialized metadata using the same convention as existing receipts."""
    return hashlib.sha256(body).hexdigest()


def json_body(value: Any) -> bytes:
    """Canonical readable JSON bytes used by content-addressed run receipts."""
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
