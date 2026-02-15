from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import math

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2*r*math.asin(math.sqrt(a))

def point_to_segment_km(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Approx distance from point to line segment in km using local equirectangular projection."""
    # Convert degrees to km scaling at mid-lat
    lat0 = math.radians((py + y1 + y2) / 3.0)
    kx = 111.0 * math.cos(lat0)
    ky = 111.0

    P = (px * kx, py * ky)
    A = (x1 * kx, y1 * ky)
    B = (x2 * kx, y2 * ky)

    vx = B[0] - A[0]; vy = B[1] - A[1]
    wx = P[0] - A[0]; wy = P[1] - A[1]
    vv = vx*vx + vy*vy
    if vv <= 1e-9:
        # A==B
        dx = P[0]-A[0]; dy = P[1]-A[1]
        return math.hypot(dx, dy)

    t = (wx*vx + wy*vy) / vv
    t = max(0.0, min(1.0, t))
    projx = A[0] + t*vx
    projy = A[1] + t*vy
    return math.hypot(P[0]-projx, P[1]-projy)

def impact_hits(
    storms: list[dict],
    targets: list[dict],
    *,
    radius_km: float = 50.0,
    use_path: str = "forecast_60min",
) -> list[dict]:
    """Return list of hits {storm_id, target_name, dist_km, eta_min} based on forecast point or path."""
    hits: list[dict] = []
    for s in storms:
        sid = str(s.get("id","?"))
        lat0 = s.get("lat"); lon0 = s.get("lon")
        f = s.get(use_path) or None
        if lat0 is None or lon0 is None or f is None:
            continue
        latf = f.get("lat"); lonf = f.get("lon")
        if latf is None or lonf is None:
            continue
        for t in targets:
            name = t.get("name","(target)")
            tlat = t.get("lat"); tlon = t.get("lon")
            if tlat is None or tlon is None:
                continue
            # distance to segment current->forecast (more realistic than endpoint)
            d = point_to_segment_km(float(tlon), float(tlat), float(lon0), float(lat0), float(lonf), float(latf))
            if d <= radius_km:
                speed = ((s.get("motion") or {}).get("speed_kmh")) or None
                eta = None
                if speed and speed > 1e-6:
                    # crude eta: distance from current centroid to closest approach ~ segment proportion.
                    # We'll just use straight-line to target from current.
                    eta = (haversine_km(float(lat0), float(lon0), float(tlat), float(tlon)) / float(speed)) * 60.0
                hits.append({
                    "storm_id": sid,
                    "target": name,
                    "dist_km": round(float(d), 1),
                    "eta_min": None if eta is None else round(float(eta), 0),
                    "max_composite": s.get("max_composite"),
                    "speed_kmh": speed,
                    "bearing_deg": ((s.get("motion") or {}).get("bearing_deg")),
                })
    # Sort: nearest first then intensity
    hits.sort(key=lambda h: (h["dist_km"], -(h["max_composite"] or 0)))
    return hits
