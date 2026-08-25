"""Local path configuration for paper reproduction scripts."""

from __future__ import annotations

import os
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent
ENV_FILE = WORKSPACE_ROOT / ".env"


def _load_env_file(path: Path = ENV_FILE) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return WORKSPACE_ROOT / path


_load_env_file()


def font_dir() -> str:
    return str(_resolve_path(os.environ.get("PAPER_REPRO_FONT_DIR", "ext-data/fonts")))


def topojson_uri() -> str:
    topojson_dir = _resolve_path(os.environ.get("PAPER_REPRO_TOPOJSON_DIR", "ext-data"))
    return topojson_dir.as_uri() + "/"
