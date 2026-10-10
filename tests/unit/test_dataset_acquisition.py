"""Verify integrity failures, safe caching, and atomic acquisition without network."""

import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from finsight.data.acquisition import (
    ArtifactIntegrityError,
    acquire_file,
    artifact_path,
    verify_artifact,
)


def specification(body: bytes) -> dict[str, Any]:
    return {
        "size_bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "git_blob_sha1": hashlib.sha1(
            b"blob " + str(len(body)).encode() + b"\0" + body
        ).hexdigest(),
        "source_url": "https://example.invalid/data",
        "local_path": "raw/test.bin",
    }


def test_verified_cache_never_contacts_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"synthetic data"
    spec = specification(body)
    path = artifact_path(tmp_path, spec)
    path.parent.mkdir()
    path.write_bytes(body)

    def fail(*args: Any, **kwargs: Any) -> None:
        pytest.fail("cached acquisition must not use network")

    monkeypatch.setattr("finsight.data.acquisition.urlopen", fail)
    assert acquire_file(tmp_path, spec) == path


@pytest.mark.parametrize("change", ["size_bytes", "sha256", "git_blob_sha1"])
def test_integrity_rejects_each_identity_mismatch(tmp_path: Path, change: str) -> None:
    body = b"synthetic"
    spec = specification(body)
    path = tmp_path / "test.bin"
    path.write_bytes(body)
    spec[change] = 0 if change == "size_bytes" else "0" * len(spec[change])
    with pytest.raises(ArtifactIntegrityError):
        verify_artifact(path, spec)


@pytest.mark.parametrize("relative", ["../escape", "/absolute", "raw/../../escape"])
def test_rejects_path_traversal(tmp_path: Path, relative: str) -> None:
    spec = specification(b"data")
    spec["local_path"] = relative
    with pytest.raises(ValueError):
        artifact_path(tmp_path, spec)


def test_atomic_download_and_bad_download_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"synthetic data"
    spec = specification(body)
    monkeypatch.setattr(
        "finsight.data.acquisition.urlopen", lambda *a, **kw: BytesIO(b"bad")
    )
    with pytest.raises(ArtifactIntegrityError):
        acquire_file(tmp_path, spec)
    assert list((tmp_path / "raw").iterdir()) == []
    monkeypatch.setattr(
        "finsight.data.acquisition.urlopen", lambda *a, **kw: BytesIO(body)
    )
    assert acquire_file(tmp_path, spec).read_bytes() == body


def test_corrupted_cache_is_preserved_for_diagnosis(tmp_path: Path) -> None:
    spec = specification(b"expected")
    path = artifact_path(tmp_path, spec)
    path.parent.mkdir()
    path.write_bytes(b"corrupted")
    with pytest.raises(ArtifactIntegrityError):
        acquire_file(tmp_path, spec)
    assert path.read_bytes() == b"corrupted"
