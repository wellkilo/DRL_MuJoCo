from __future__ import annotations

import json
import time
from typing import Any


def log_event(event: str, payload: dict[str, Any]) -> None:
    message = {"event": event, "timestamp": time.time(), **payload}
    print(json.dumps(message, ensure_ascii=False))

