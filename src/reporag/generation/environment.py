"""Local environment loading for generation-provider CLIs."""

from __future__ import annotations

from os import PathLike

from dotenv import load_dotenv


def load_local_environment(dotenv_path: str | PathLike[str] | None = None) -> bool:
    """Load a local dotenv file without replacing explicit process values."""
    return load_dotenv(dotenv_path=dotenv_path, override=False)
