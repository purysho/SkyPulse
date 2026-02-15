from __future__ import annotations

from pathlib import Path
import json


def write_latest(cache_dir: str | Path, payload: dict) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / 'latest.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')


def read_latest(cache_dir: str | Path) -> dict | None:
    p = Path(cache_dir) / 'latest.json'
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding='utf-8'))
