from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from drl.config import Config


def load_config(path: str) -> Config:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    data: dict[str, Any] = raw
    defaults = asdict(Config())
    defaults.update(data)
    return Config(**defaults)

