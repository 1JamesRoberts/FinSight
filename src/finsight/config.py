"""Explicit environment configuration without import-time side effects."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


@dataclass(frozen=True)
class Settings:
    """Local runtime settings; paths are relative to the working directory."""

    data_dir: Path = Path("data")
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        """Read settings on demand, optionally from an explicit environment mapping.

        No files or directories are created. A .env file must be loaded by the
        caller (for example, using uv run --env-file .env).
        """
        source = os.environ if environ is None else environ
        data_dir = source.get("FINSIGHT_DATA_DIR", "data").strip()
        if not data_dir:
            raise ValueError("FINSIGHT_DATA_DIR must not be empty")

        log_level = source.get("FINSIGHT_LOG_LEVEL", "INFO").strip().upper()
        if log_level not in _LOG_LEVELS:
            raise ValueError(
                "FINSIGHT_LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL"
            )

        return cls(data_dir=Path(data_dir), log_level=log_level)
