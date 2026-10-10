"""Verify configuration defaults, isolation, and invalid input handling."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from finsight.config import Settings


def test_defaults() -> None:
    assert Settings.from_env({}) == Settings(data_dir=Path("data"), log_level="INFO")


def test_explicit_environment() -> None:
    settings = Settings.from_env(
        {"FINSIGHT_DATA_DIR": "research/data", "FINSIGHT_LOG_LEVEL": " debug "}
    )
    assert settings.data_dir == Path("research/data")
    assert settings.log_level == "DEBUG"


def test_reads_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINSIGHT_DATA_DIR", "/tmp/finsight-test")
    monkeypatch.setenv("FINSIGHT_LOG_LEVEL", "warning")
    assert Settings.from_env() == Settings(Path("/tmp/finsight-test"), "WARNING")


def test_explicit_mapping_does_not_fall_back_to_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FINSIGHT_DATA_DIR", "unexpected")
    monkeypatch.setenv("FINSIGHT_LOG_LEVEL", "invalid")
    assert Settings.from_env({}) == Settings()


@pytest.mark.parametrize("value", ["", "   "])
def test_rejects_empty_data_dir(value: str) -> None:
    with pytest.raises(ValueError, match="FINSIGHT_DATA_DIR"):
        Settings.from_env({"FINSIGHT_DATA_DIR": value})


@pytest.mark.parametrize("value", ["", "verbose", "WARN"])
def test_rejects_invalid_log_level(value: str) -> None:
    with pytest.raises(ValueError, match="FINSIGHT_LOG_LEVEL"):
        Settings.from_env({"FINSIGHT_LOG_LEVEL": value})


def test_settings_are_immutable() -> None:
    settings = Settings.from_env({})
    with pytest.raises(FrozenInstanceError):
        settings.log_level = "DEBUG"  # type: ignore[misc]


def test_loading_does_not_create_data_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "missing"
    settings = Settings.from_env({"FINSIGHT_DATA_DIR": str(data_dir)})
    assert settings.data_dir == data_dir
    assert not data_dir.exists()
