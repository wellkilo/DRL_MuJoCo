from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from drl.config import Config


def load_config(path: str) -> Config:
    config_path = Path(path)
    data: dict[str, Any] = {}
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text())
        if isinstance(raw, dict):
            data = raw
    defaults = asdict(Config())
    defaults.update(data)
    return Config(**defaults)

