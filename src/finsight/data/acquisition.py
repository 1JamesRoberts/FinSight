"""Acquire reviewed, pinned files and verify their byte identities."""

import hashlib
import json
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.request import Request, urlopen


class ArtifactIntegrityError(ValueError):
    """Source bytes do not match the reviewed artifact identity."""


def verify_artifact(path: Path, specification: dict[str, Any]) -> str:
    """Verify size, SHA-256, and optionally the repository's Git blob SHA-1."""
    body = path.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    if len(body) != specification["size_bytes"] or digest != specification["sha256"]:
        raise ArtifactIntegrityError("artifact size or SHA-256 mismatch")
    if expected := specification.get("git_blob_sha1"):
        git_digest = hashlib.sha1(
            b"blob " + str(len(body)).encode("ascii") + b"\0" + body
        ).hexdigest()
        if git_digest != expected:
            raise ArtifactIntegrityError("artifact Git blob identity mismatch")
    return digest


def artifact_path(data_dir: Path, specification: dict[str, Any]) -> Path:
    """Resolve a relative manifest path below the selected local data directory."""
    relative = PurePosixPath(specification["local_path"])
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("artifact path must be relative and contain no traversal")
    target = data_dir / relative
    if not target.resolve().is_relative_to(data_dir.resolve()):
        raise ValueError("artifact path escapes the data directory")
    return target


def acquire_file(data_dir: Path, specification: dict[str, Any]) -> Path:
    """Reuse a verified cache or atomically persist a verified HTTPS download.

    Existing corrupted files raise an error instead of being overwritten.
    An interrupted or invalid download never leaves an accepted final artifact.
    """
    target = artifact_path(data_dir, specification)
    if target.exists():
        verify_artifact(target, specification)
        return target
    url = specification["source_url"]
    if not url.startswith("https://"):
        raise ValueError("artifact source must use HTTPS")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        request = Request(url, headers={"User-Agent": "FinSight-research-acquisition"})
        with urlopen(request, timeout=60) as response:
            body = response.read(specification["size_bytes"] + 1)
        with NamedTemporaryFile(dir=target.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(body)
        verify_artifact(temporary, specification)
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def acquire_datasets(manifest_path: Path, data_dir: Path) -> list[dict[str, str]]:
    """Acquire only assets explicitly listed as reviewed local research datasets."""
    manifest = json.loads(manifest_path.read_text())
    receipt = []
    for name in manifest["local_dataset_assets"]:
        asset = manifest["assets"][name]
        if asset["local_research_decision"] != "supported_under_repository_license":
            raise ValueError("dataset is not reviewed for local research")
        for specification in asset["acquisition_files"]:
            path = acquire_file(data_dir, specification)
            receipt.append(
                {
                    "asset": name,
                    "revision": asset["revision"],
                    "source_url": specification["source_url"],
                    "local_path": specification["local_path"],
                    "sha256": verify_artifact(path, specification),
                }
            )
    return receipt
