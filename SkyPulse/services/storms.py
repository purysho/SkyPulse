from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from compute.storms import detect_objects, track_objects, to_dicts

TRACK_FILE = "storms_tracks_latest.json"

def _read_prev(cache_dir: str | Path) -> list[dict] | None:
    p = Path(cache_dir) / TRACK_FILE
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("objects", [])
    except Exception:
        return None

def run_storm_detection(cache_dir: str | Path, *, threshold: float = 6.0, min_pixels: int = 12):
    cache_dir = Path(cache_dir)
    npz = cache_dir / "storm_fields_latest.npz"
    if not npz.exists():
        raise FileNotFoundError("storm_fields_latest.npz not found yet. Run an update first.")

    data = np.load(npz)
    lons = data["lons"]
    lats = data["lats"]
    comp = data["composite"]

    # lons/lats saved from subset_to_bbox; may be xarray DataArray -> 1D arrays
    lons_1d = np.array(lons).astype(float).ravel()
    lats_1d = np.array(lats).astype(float).ravel()

    cur = detect_objects(lons_1d, lats_1d, np.array(comp, dtype=float), threshold=threshold, min_pixels=min_pixels)
    prev = _read_prev(cache_dir)
    tracked = track_objects(cur, prev, max_match_km=60.0)

    payload = {
        "updated_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "threshold": threshold,
        "min_pixels": min_pixels,
        "objects": to_dicts(tracked),
    }
    (cache_dir / TRACK_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
