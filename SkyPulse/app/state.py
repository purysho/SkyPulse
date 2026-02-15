from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime, timezone

LATEST_FILE = "latest.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_latest(cache_dir: str | Path, payload: dict) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["updated_at_utc"] = _now_iso()
    (cache_dir / LATEST_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_latest(cache_dir: str | Path) -> dict | None:
    p = Path(cache_dir) / LATEST_FILE
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def minutes_since_update(latest: dict | None) -> float | None:
    if not latest or "updated_at_utc" not in latest:
        return None
    try:
        t = datetime.fromisoformat(latest["updated_at_utc"])
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        dt = datetime.now(timezone.utc) - t
        return dt.total_seconds() / 60.0
    except Exception:
        return None


def maps_dir(cache_dir: str | Path) -> Path:
    d = Path(cache_dir) / "maps"
    d.mkdir(parents=True, exist_ok=True)
    return d
