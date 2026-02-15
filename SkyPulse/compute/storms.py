from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import numpy as np
from scipy import ndimage

@dataclass
class StormObject:
    id: str
    lat: float
    lon: float
    area_km2: float
    max_composite: float
    mean_composite: float

def _pixel_area_km2(lats: np.ndarray, lons: np.ndarray) -> float:
    # crude: use median spacing -> km (lat: 111 km/deg; lon scaled by cos(lat))
    if lats.size < 2 or lons.size < 2:
        return 1.0
    # assume 1D coordinate arrays
    dlat = float(np.nanmedian(np.abs(np.diff(lats))))
    dlon = float(np.nanmedian(np.abs(np.diff(lons))))
    lat0 = float(np.nanmedian(lats))
    km_lat = 111.0 * dlat
    km_lon = 111.0 * np.cos(np.deg2rad(lat0)) * dlon
    return max(km_lat * km_lon, 0.1)

def detect_objects(
    lons_1d: np.ndarray,
    lats_1d: np.ndarray,
    composite_2d: np.ndarray,
    *,
    threshold: float = 6.0,
    min_pixels: int = 12,
) -> list[tuple[float,float,float,float,float]]:
    """Return list of (lat, lon, area_km2, max, mean) for labeled regions."""
    comp = np.array(composite_2d, dtype=float)
    mask = np.isfinite(comp) & (comp >= threshold)
    if not np.any(mask):
        return []

    labeled, n = ndimage.label(mask)
    if n == 0:
        return []

    pix_area = _pixel_area_km2(np.array(lats_1d, dtype=float), np.array(lons_1d, dtype=float))
    objs = []
    for lab in range(1, n+1):
        ys, xs = np.where(labeled == lab)
        if ys.size < min_pixels:
            continue
        vals = comp[ys, xs]
        # centroid in index space -> lat/lon
        y0 = int(np.round(np.mean(ys)))
        x0 = int(np.round(np.mean(xs)))
        lat = float(lats_1d[y0])
        lon = float(lons_1d[x0])
        area = float(ys.size * pix_area)
        vmax = float(np.nanmax(vals))
        vmean = float(np.nanmean(vals))
        objs.append((lat, lon, area, vmax, vmean))

    # sort by max intensity then area
    objs.sort(key=lambda t: (t[3], t[2]), reverse=True)
    return objs

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math
    r = 6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2-lat1)
    dl = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2*r*math.asin(math.sqrt(a))

def track_objects(
    current: list[tuple[float,float,float,float,float]],
    prev: list[dict[str,Any]] | None,
    *,
    max_match_km: float = 60.0,
) -> list[StormObject]:
    """Assign stable IDs by matching to previous objects by nearest centroid."""
    prev = prev or []
    used = set()
    out: list[StormObject] = []

    for i, (lat, lon, area, vmax, vmean) in enumerate(current, start=1):
        best = None
        best_d = None
        for j, pobj in enumerate(prev):
            if j in used:
                continue
            d = haversine_km(lat, lon, float(pobj["lat"]), float(pobj["lon"]))
            if best_d is None or d < best_d:
                best_d = d
                best = (j, pobj)
        if best is not None and best_d is not None and best_d <= max_match_km:
            j, pobj = best
            used.add(j)
            sid = str(pobj.get("id", f"S{i:02d}"))
        else:
            sid = f"S{i:02d}"
        out.append(StormObject(id=sid, lat=lat, lon=lon, area_km2=area, max_composite=vmax, mean_composite=vmean))

    return out

def to_dicts(objs: list[StormObject]) -> list[dict[str,Any]]:
    return [asdict(o) for o in objs]
