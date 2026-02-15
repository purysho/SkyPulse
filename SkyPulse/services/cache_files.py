from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime, timezone
from typing import Any

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def cache_boundaries(cache_dir: str | Path, candidates: list[dict]) -> Path:
    cache_dir = Path(cache_dir)
    out = cache_dir / "boundaries_latest.json"
    payload = {"generated_at_utc": now_iso(), "candidates": candidates}
    write_json(out, payload)
    return out

def cache_metar_bias(cache_dir: str | Path, summary: dict) -> Path:
    cache_dir = Path(cache_dir)
    out = cache_dir / "metar_bias_latest.json"
    payload = {
        "generated_at_utc": now_iso(),
        "temp_bias_c": summary.get("temp_bias_c", {}),
        "dewpoint_bias_c": summary.get("dewpoint_bias_c", {}),
    }
    write_json(out, payload)
    return out
