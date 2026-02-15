from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Optional

def cache_path(cache_dir: str | Path, key: str, suffix: str = "") -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    return cache_dir / f"{h}{suffix}"

def read_bytes(cache_dir: str | Path, key: str, suffix: str = "") -> Optional[bytes]:
    p = cache_path(cache_dir, key, suffix)
    return p.read_bytes() if p.exists() else None

def write_bytes(cache_dir: str | Path, key: str, data: bytes, suffix: str = "") -> Path:
    p = cache_path(cache_dir, key, suffix)
    p.write_bytes(data)
    return p
