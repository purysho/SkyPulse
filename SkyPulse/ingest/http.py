from __future__ import annotations
from pathlib import Path
import requests

from .cache import read_bytes, write_bytes

def get(url: str, *, cache_dir: str | Path, cache_key: str, suffix: str = "", timeout: int = 30) -> bytes:
    """HTTP GET with a simple disk cache."""
    cached = read_bytes(cache_dir, cache_key, suffix=suffix)
    if cached is not None:
        return cached
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    data = r.content
    write_bytes(cache_dir, cache_key, data, suffix=suffix)
    return data
