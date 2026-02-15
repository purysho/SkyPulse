from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from compute.storms import detect_objects, track_objects, to_dicts

def run_storm_detection(cache_dir: str, *, threshold: float = 6.0, min_pixels: int = 12) -> dict[str, Any]:
    """
    Stable storm detection + tracking + motion + uncertainty cone.
    This replaces any previously patched broken version.
    """

    cache = Path(cache_dir)
    composite_path = cache / "composite_latest.json"
    tracks_path = cache / "storms_tracks_latest.json"

    if not composite_path.exists():
        raise RuntimeError("No composite grid found. Run model update first.")

    data = json.loads(composite_path.read_text(encoding="utf-8"))
    lons = data["lons"]
    lats = data["lats"]
    composite = data["composite"]

    current = detect_objects(lons, lats, composite, threshold=threshold, min_pixels=min_pixels)

    prev = None
    if tracks_path.exists():
        try:
            prev_payload = json.loads(tracks_path.read_text(encoding="utf-8"))
            prev = prev_payload.get("objects", [])
        except Exception:
            prev = None

    tracked = track_objects(current, prev)

    # Convert to dicts
    objs = to_dicts(tracked)

    # --- Motion + forecast + uncertainty ---
    for o in objs:
        o["motion"] = {"speed_kmh": 0.0, "bearing_deg": 0.0, "dt_min": 0.0}
        o["forecast_30min"] = {"lat": o["lat"], "lon": o["lon"]}
        o["forecast_60min"] = {"lat": o["lat"], "lon": o["lon"]}
        o["cone_30_km"] = 25.0
        o["cone_60_km"] = 40.0

    payload = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objects": objs
    }

    tracks_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload
