from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from compute.storms import detect_objects, track_objects, to_dicts, haversine_km

TRACK_FILE = "storms_tracks_latest.json"

def _read_prev_payload(cache_dir: str | Path) -> dict | None:
    p = Path(cache_dir) / TRACK_FILE
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dl)
    brng = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return brng

def _dest_point(lat: float, lon: float, bearing_deg: float, dist_km: float) -> tuple[float,float]:
    import math
    R = 6371.0
    br = math.radians(bearing_deg)
    phi1 = math.radians(lat)
    lam1 = math.radians(lon)
    d = dist_km / R
    phi2 = math.asin(math.sin(phi1)*math.cos(d) + math.cos(phi1)*math.sin(d)*math.cos(br))
    lam2 = lam1 + math.atan2(math.sin(br)*math.sin(d)*math.cos(phi1), math.cos(d)-math.sin(phi1)*math.sin(phi2))
    lat2 = math.degrees(phi2)
    lon2 = (math.degrees(lam2) + 540.0) % 360.0 - 180.0
    return lat2, lon2

def run_storm_detection(cache_dir: str | Path, *, threshold: float = 6.0, min_pixels: int = 12):
    """Detect composite-objects, track IDs, and compute motion + 30/60 min extrapolation."""
    cache_dir = Path(cache_dir)
    npz = cache_dir / "storm_fields_latest.npz"
    if not npz.exists():
        raise FileNotFoundError("storm_fields_latest.npz not found yet. Run an update first.")

    data = np.load(npz)
    lons = data["lons"]
    lats = data["lats"]
    comp = data["composite"]

    lons_1d = np.array(lons).astype(float).ravel()
    lats_1d = np.array(lats).astype(float).ravel()

    cur_raw = detect_objects(lons_1d, lats_1d, np.array(comp, dtype=float), threshold=threshold, min_pixels=min_pixels)

    prev_payload = _read_prev_payload(cache_dir)
    prev_objs = (prev_payload or {}).get("objects", [])

    tracked = track_objects(cur_raw, prev_objs, max_match_km=60.0)
    out_objs = to_dicts(tracked)

    # Motion + extrapolation: use previous centroid for same ID if available
    prev_by_id = {str(o.get("id")): o for o in prev_objs if "id" in o and "lat" in o and "lon" in o}
    dt_minutes = None
    if prev_payload and prev_payload.get("updated_at_utc"):
        try:
            from datetime import datetime
            t_prev = datetime.fromisoformat(prev_payload["updated_at_utc"].replace("Z","+00:00"))
            t_now = datetime.now(__import__("datetime").timezone.utc)
            dt_minutes = max((t_now - t_prev).total_seconds() / 60.0, 0.0)
        except Exception:
            dt_minutes = None

    for o in out_objs:
        pid = str(o["id"])
        po = prev_by_id.get(pid)
        if po is None:
            o["motion"] = None
            o["forecast_30min"] = None
            o["forecast_60min"] = None
            continue

        lat1, lon1 = float(po["lat"]), float(po["lon"])
        lat2, lon2 = float(o["lat"]), float(o["lon"])

        dist_km = haversine_km(lat1, lon1, lat2, lon2)
        bearing = _bearing_deg(lat1, lon1, lat2, lon2)

        # speed: if we know dt, else assume 60 min between snapshots
        dt = dt_minutes if (dt_minutes is not None and dt_minutes > 1e-6) else 60.0
        speed_kmh = dist_km / (dt / 60.0) if dt > 0 else None

        o["motion"] = {
            "from": {"lat": lat1, "lon": lon1},
            "to": {"lat": lat2, "lon": lon2},
            "dist_km": round(dist_km, 1),
            "bearing_deg": round(bearing, 0),
            "speed_kmh": None if speed_kmh is None else round(speed_kmh, 1),
            "dt_min": round(dt, 1),
        }

        # Extrapolate 30/60 minutes using speed & bearing (constant motion)
        if speed_kmh is None:
            o["forecast_30min"] = None
            o["forecast_60min"] = None
        else:
            d30 = speed_kmh * 0.5
            d60 = speed_kmh * 1.0
            f30 = _dest_point(lat2, lon2, bearing, d30)
            f60 = _dest_point(lat2, lon2, bearing, d60)
            o["forecast_30min"] = {"lat": round(f30[0], 4), "lon": round(f30[1], 4)}
            o["forecast_60min"] = {"lat": round(f60[0], 4), "lon": round(f60[1], 4)}

    payload = {
        "updated_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "threshold": threshold,
        "min_pixels": min_pixels,
        "objects": out_objs,
    }
    (cache_dir / TRACK_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
