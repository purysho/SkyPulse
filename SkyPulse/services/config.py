from __future__ import annotations

import json
from pathlib import Path

def load_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8"))
