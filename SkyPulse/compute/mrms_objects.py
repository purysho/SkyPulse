from __future__ import annotations

from typing import Any
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from compute.storms import detect_objects, track_objects, to_dicts

MRMS_TRACKS = "mrms_objects_tracks_latest.json"


def detect_and_track_mrms(
    cache_dir: str,
    *,
    threshold: float = 35.0,
    min_pixels: int = 25,
) -> dict[str, Any]:
    """
    Detect objects on decoded MRMS grid saved in mrms_reflectivity_latest.npz.
    Uses the shared blob detector / tracker.
    """
    cache = Path(cache_dir)
    npz = cache / "mrms_reflectivity_latest.npz"
    if not npz.exists():
        raise FileNotFoundError("mrms_reflectivity_latest.npz not found. Run MRMS update first.")

    data = np.load(npz)
    lons = np.array(data["lons"]).astype(float).ravel()
    lats = np.array(data["lats"]).astype(float).ravel()
    field = np.array(data["field"], dtype=float)

    cur_raw = detect_objects(lons, lats, field, threshold=threshold, min_pixels=min_pixels)

    track_path = cache / MRMS_TRACKS
    prev_payload = None
    if track_path.exists():
        try:
            prev_payload = json.loads(track_path.read_text(encoding="utf-8"))
        except Exception:
            prev_payload = None

    prev_objs = (prev_payload or {}).get("objects", [])
    tracked = track_objects(cur_raw, prev_objs, max_match_km=60.0)
    out_objs = to_dicts(tracked)

    payload = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "threshold": float(threshold),
        "min_pixels": int(min_pixels),
        "objects": out_objs,
    }
    track_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
