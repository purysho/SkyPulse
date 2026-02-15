from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

@dataclass(frozen=True)
class BoundaryCandidate:
    lat: float
    lon: float
    score: float
    kind: str  # "dewpoint", "temp", "windshift"

def _wrap_lon(lon: np.ndarray) -> np.ndarray:
    # keep -180..180
    return ((lon + 180.0) % 360.0) - 180.0

def _local_gradient_score(xy: np.ndarray, val: np.ndarray, k: int = 6) -> np.ndarray:
    """Estimate local gradient magnitude from kNN differences (units per degree)."""
    tree = cKDTree(xy)
    dists, idxs = tree.query(xy, k=min(k, len(xy)))
    # idxs shape (n,k)
    n = xy.shape[0]
    scores = np.zeros(n, dtype=float)
    for i in range(n):
        nbr = idxs[i, 1:]  # exclude self
        if nbr.size == 0:
            continue
        dv = val[nbr] - val[i]
        # distance in degrees (approx)
        dd = np.linalg.norm(xy[nbr] - xy[i], axis=1)
        m = np.isfinite(dv) & (dd > 0)
        if not np.any(m):
            continue
        # robust gradient proxy: median(|dv|/dd)
        scores[i] = float(np.median(np.abs(dv[m]) / dd[m]))
    return scores

def _wind_dir_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    # meteorological direction wind is from; using atan2(-u,-v)
    ang = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
    return ang

def _wind_shift_score(xy: np.ndarray, wind_dir: np.ndarray, k: int = 6) -> np.ndarray:
    tree = cKDTree(xy)
    dists, idxs = tree.query(xy, k=min(k, len(xy)))
    n = xy.shape[0]
    scores = np.zeros(n, dtype=float)
    for i in range(n):
        nbr = idxs[i, 1:]
        if nbr.size == 0:
            continue
        dd = np.linalg.norm(xy[nbr] - xy[i], axis=1)
        # circular diff
        dtheta = np.abs(((wind_dir[nbr] - wind_dir[i] + 180.0) % 360.0) - 180.0)
        m = np.isfinite(dtheta) & (dd > 0)
        if not np.any(m):
            continue
        # degrees of wind shift per degree distance
        scores[i] = float(np.median(dtheta[m] / dd[m]))
    return scores

def compute_boundary_candidates(metar_df: pd.DataFrame, *, top_n: int = 8) -> tuple[pd.DataFrame, list[BoundaryCandidate]]:
    """Return station-level scores and top boundary candidates."""
    df = metar_df.copy()

    # Standardize columns
    lat_col = "lat" if "lat" in df.columns else ("latitude" if "latitude" in df.columns else None)
    lon_col = "lon" if "lon" in df.columns else ("longitude" if "longitude" in df.columns else None)
    if lat_col is None or lon_col is None:
        raise KeyError("METAR dataframe missing lat/lon.")

    df["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    df["lon"] = _wrap_lon(df["lon"].to_numpy(dtype=float))
    df = df.dropna(subset=["lat","lon"]).copy()

    # temp/dewpoint
    tcol = "temp_c" if "temp_c" in df.columns else None
    tdcol = "dewpoint_c" if "dewpoint_c" in df.columns else None

    # wind
    ucol = "u_wind_kt" if "u_wind_kt" in df.columns else None
    vcol = "v_wind_kt" if "v_wind_kt" in df.columns else None
    # many caches use wind_dir_degrees and wind_speed_kt
    wdcol = "wind_dir_degrees" if "wind_dir_degrees" in df.columns else None

    xy = np.column_stack([df["lon"].to_numpy(dtype=float), df["lat"].to_numpy(dtype=float)])

    candidates: list[BoundaryCandidate] = []

    if tdcol:
        td = pd.to_numeric(df[tdcol], errors="coerce").to_numpy(dtype=float)
        dp_score = _local_gradient_score(xy, td)
        df["dewpoint_grad"] = dp_score
        # pick top
        idx = np.argsort(dp_score)[::-1][:top_n]
        for i in idx:
            if np.isfinite(dp_score[i]) and dp_score[i] > 0:
                candidates.append(BoundaryCandidate(lat=float(df.iloc[i]["lat"]), lon=float(df.iloc[i]["lon"]), score=float(dp_score[i]), kind="dewpoint"))

    if tcol:
        tt = pd.to_numeric(df[tcol], errors="coerce").to_numpy(dtype=float)
        t_score = _local_gradient_score(xy, tt)
        df["temp_grad"] = t_score
        idx = np.argsort(t_score)[::-1][:top_n]
        for i in idx:
            if np.isfinite(t_score[i]) and t_score[i] > 0:
                candidates.append(BoundaryCandidate(lat=float(df.iloc[i]["lat"]), lon=float(df.iloc[i]["lon"]), score=float(t_score[i]), kind="temp"))

    # wind shift: prefer explicit wind_dir_degrees
    if wdcol:
        wd = pd.to_numeric(df[wdcol], errors="coerce").to_numpy(dtype=float)
        ws_score = _wind_shift_score(xy, wd)
        df["windshift"] = ws_score
        idx = np.argsort(ws_score)[::-1][:top_n]
        for i in idx:
            if np.isfinite(ws_score[i]) and ws_score[i] > 0:
                candidates.append(BoundaryCandidate(lat=float(df.iloc[i]["lat"]), lon=float(df.iloc[i]["lon"]), score=float(ws_score[i]), kind="windshift"))

    # sort candidates overall
    candidates = sorted(candidates, key=lambda c: c.score, reverse=True)[:top_n]
    return df, candidates

def grid_boundary_field(df_scored: pd.DataFrame, *, bbox: dict, res_deg: float = 0.25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a gridded boundary score (0..1) from station scores."""
    lat_min, lat_max = bbox["lat_min"], bbox["lat_max"]
    lon_min, lon_max = bbox["lon_min"], bbox["lon_max"]

    lats = np.arange(lat_min, lat_max + 1e-6, res_deg)
    lons = np.arange(lon_min, lon_max + 1e-6, res_deg)
    LON, LAT = np.meshgrid(lons, lats)

    score = np.zeros_like(LON, dtype=float)

    # Combine available station scores
    cols = [c for c in ["dewpoint_grad","temp_grad","windshift"] if c in df_scored.columns]
    if not cols:
        return lons, lats, score

    s = df_scored[cols].to_numpy(dtype=float)
    s = np.nan_to_num(s, nan=0.0)
    station_score = s.max(axis=1)

    pts = np.column_stack([df_scored["lon"].to_numpy(dtype=float), df_scored["lat"].to_numpy(dtype=float)])
    tree = cKDTree(pts)

    # For each grid point, take max of nearby stations within radius
    rad = 1.0  # degrees (~100 km)
    for i in range(LON.shape[0]):
        for j in range(LON.shape[1]):
            idxs = tree.query_ball_point([LON[i,j], LAT[i,j]], r=rad)
            if idxs:
                score[i,j] = float(np.max(station_score[idxs]))
    # normalize to 0..1 using p95
    p95 = np.percentile(score, 95) if np.any(score > 0) else 1.0
    if p95 <= 0: p95 = 1.0
    score = np.clip(score / p95, 0.0, 1.0)
    return lons, lats, score
