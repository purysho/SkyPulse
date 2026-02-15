from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import numpy as np

from compute.storms import detect_objects, track_objects, to_dicts, haversine_km

TRACK_FILE = "storms_tracks_latest.json"

DT_MIN_CLAMP = 5.0     # minutes
DT_MAX_CLAMP = 120.0   # minutes
EMA_ALPHA = 0.30       # new contribution (0..1)


def _read_prev_payload(cache_dir: Path) -> dict | None:
    p = cache_dir / TRACK_FILE
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _circ_ema_deg(prev_deg: float, new_deg: float, alpha: float) -> float:
    """Exponential smoothing for angles in degrees."""
    import math
    p = math.radians(prev_deg)
    n = math.radians(new_deg)
    # unit vectors
    px, py = math.cos(p), math.sin(p)
    nx, ny = math.cos(n), math.sin(n)
    x = (1 - alpha) * px + alpha * nx
    y = (1 - alpha) * py + alpha * ny
    if abs(x) < 1e-12 and abs(y) < 1e-12:
        return float(new_deg)
    ang = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return float(ang)


def _dest_point(lat: float, lon: float, bearing_deg: float, dist_km: float) -> tuple[float, float]:
    import math
    R = 6371.0
    br = math.radians(bearing_deg)
    phi1 = math.radians(lat)
    lam1 = math.radians(lon)
    d = dist_km / R
    phi2 = math.asin(math.sin(phi1) * math.cos(d) + math.cos(phi1) * math.sin(d) * math.cos(br))
    lam2 = lam1 + math.atan2(math.sin(br) * math.sin(d) * math.cos(phi1), math.cos(d) - math.sin(phi1) * math.sin(phi2))
    lat2 = math.degrees(phi2)
    lon2 = (math.degrees(lam2) + 540.0) % 360.0 - 180.0
    return lat2, lon2


def run_storm_detection(cache_dir: str | Path, *, threshold: float = 6.0, min_pixels: int = 12) -> dict[str, Any]:
    """Detect composite-objects from cached grids, track IDs, compute smoothed motion + forecasts + cone."""
    cache_dir = Path(cache_dir)
    npz = cache_dir / "storm_fields_latest.npz"
    if not npz.exists():
        raise FileNotFoundError("storm_fields_latest.npz not found yet. Run an update first.")

    data = np.load(npz)
    lons = np.array(data["lons"]).astype(float).ravel()
    lats = np.array(data["lats"]).astype(float).ravel()
    comp = np.array(data["composite"], dtype=float)

    cur_raw = detect_objects(lons, lats, comp, threshold=threshold, min_pixels=min_pixels)

    prev_payload = _read_prev_payload(cache_dir)
    prev_objs = (prev_payload or {}).get("objects", [])

    tracked = track_objects(cur_raw, prev_objs, max_match_km=60.0)
    out_objs: list[dict[str, Any]] = to_dicts(tracked)

    prev_by_id = {str(o.get("id")): o for o in prev_objs if isinstance(o, dict) and "id" in o}

    # time delta for speed calculation (minutes) — best-effort
    dt_minutes = None
    if prev_payload and prev_payload.get("updated_at_utc"):
        try:
            t_prev = datetime.fromisoformat(str(prev_payload["updated_at_utc"]).replace("Z", "+00:00"))
            t_now = datetime.now(timezone.utc)
            dt_minutes = max((t_now - t_prev).total_seconds() / 60.0, 0.0)
        except Exception:
            dt_minutes = None

    # clamp dt
    if dt_minutes is not None:
        dt_minutes = max(DT_MIN_CLAMP, min(DT_MAX_CLAMP, dt_minutes))

    for o in out_objs:
        pid = str(o["id"])
        po = prev_by_id.get(pid)

        # defaults if no previous
        o["motion"] = None
        o["forecast_30min"] = None
        o["forecast_60min"] = None
        o["cone_30_km"] = None
        o["cone_60_km"] = None
        o["motion_confidence"] = "low"

        if not po:
            continue

        lat1, lon1 = float(po.get("lat")), float(po.get("lon"))
        lat2, lon2 = float(o.get("lat")), float(o.get("lon"))

        # If dt is unknown, assume 60 min
        dt = dt_minutes if (dt_minutes is not None and dt_minutes > 0) else 60.0

        # Compute raw motion
        dist_km = haversine_km(lat1, lon1, lat2, lon2)
        bearing = _bearing_deg(lat1, lon1, lat2, lon2)
        speed_kmh = dist_km / (dt / 60.0) if dt > 0 else None

        # If the app reran too quickly, reuse previous speed/bearing if present
        prev_motion = (po.get("motion") or {}) if isinstance(po, dict) else {}
        prev_speed = prev_motion.get("speed_kmh")
        prev_bearing = prev_motion.get("bearing_deg")

        if dt_minutes is not None and dt_minutes <= DT_MIN_CLAMP + 1e-6:
            if isinstance(prev_speed, (int, float)) and isinstance(prev_bearing, (int, float)):
                speed_kmh = float(prev_speed)
                bearing = float(prev_bearing)

        # Smooth motion (EMA)
        if isinstance(prev_speed, (int, float)) and speed_kmh is not None:
            speed_kmh = (1.0 - EMA_ALPHA) * float(prev_speed) + EMA_ALPHA * float(speed_kmh)
        if isinstance(prev_bearing, (int, float)) and bearing is not None:
            bearing = _circ_ema_deg(float(prev_bearing), float(bearing), EMA_ALPHA)

        # Confidence (simple)
        conf = "low"
        if dist_km <= 30.0 and 15.0 <= dt <= 90.0:
            conf = "high"
        elif dist_km <= 60.0 and 10.0 <= dt <= 110.0:
            conf = "med"
        o["motion_confidence"] = conf

        o["motion"] = {
            "from": {"lat": lat1, "lon": lon1},
            "to": {"lat": lat2, "lon": lon2},
            "dist_km": round(dist_km, 1),
            "bearing_deg": round(float(bearing), 0) if bearing is not None else None,
            "speed_kmh": None if speed_kmh is None else round(float(speed_kmh), 1),
            "dt_min": round(float(dt), 1),
        }

        if speed_kmh is None or bearing is None:
            continue

        d30 = float(speed_kmh) * 0.5
        d60 = float(speed_kmh) * 1.0
        f30 = _dest_point(lat2, lon2, float(bearing), d30)
        f60 = _dest_point(lat2, lon2, float(bearing), d60)
        o["forecast_30min"] = {"lat": round(f30[0], 4), "lon": round(f30[1], 4)}
        o["forecast_60min"] = {"lat": round(f60[0], 4), "lon": round(f60[1], 4)}

        # cone heuristic (explainable) + widen if confidence low
        area_km2 = float(o.get("area_km2") or 0.0)
        size_term = min((area_km2 ** 0.5) * 0.15, 40.0)
        speed_term = min(float(speed_kmh) * 0.25, 60.0)
        base30 = 20.0 + 0.5 * size_term + 0.5 * speed_term
        base60 = 35.0 + 0.8 * size_term + 0.8 * speed_term

        if conf == "med":
            base30 *= 1.15
            base60 *= 1.15
        elif conf == "low":
            base30 *= 1.35
            base60 *= 1.35

        o["cone_30_km"] = round(base30, 1)
        o["cone_60_km"] = round(base60, 1)

    payload = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "threshold": float(threshold),
        "min_pixels": int(min_pixels),
        "objects": out_objs,
    }

    (cache_dir / TRACK_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
