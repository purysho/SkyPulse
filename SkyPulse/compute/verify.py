from __future__ import annotations

import numpy as np
import pandas as pd

def _pick_col(df: pd.DataFrame, options: list[str]) -> str | None:
    for c in options:
        if c in df.columns:
            return c
    return None

def metar_obs_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize METAR cache dataframe to columns: station, lat, lon, temp_c, dewpoint_c."""
    station = _pick_col(df, ["station_id", "station", "icao", "id"])
    lat = _pick_col(df, ["latitude", "lat"])
    lon = _pick_col(df, ["longitude", "lon"])
    t = _pick_col(df, ["temp_c", "temperature_c", "temp"])
    td = _pick_col(df, ["dewpoint_c", "dew_point_c", "dewpoint"])

    if station is None or lat is None or lon is None:
        raise KeyError(f"METAR cache missing station/lat/lon fields. Columns: {list(df.columns)[:40]}")
    if t is None or td is None:
        # still return location subset; caller can handle missing obs vars
        out = df[[station, lat, lon]].copy()
        out.columns = ["station", "lat", "lon"]
        out["temp_c"] = np.nan
        out["dewpoint_c"] = np.nan
        return out

    out = df[[station, lat, lon, t, td]].copy()
    out.columns = ["station", "lat", "lon", "temp_c", "dewpoint_c"]
    out["temp_c"] = pd.to_numeric(out["temp_c"], errors="coerce")
    out["dewpoint_c"] = pd.to_numeric(out["dewpoint_c"], errors="coerce")
    return out

def model_fields_at_points(ds, *, lon_name: str, lat_name: str, lats: np.ndarray, lons_deg: np.ndarray) -> dict:
    """Interpolate model fields at arrays of points. lons_deg is -180..180."""
    lon360 = np.mod(lons_deg, 360.0)

    out = {}

    # 2m temp / dewpoint variable names vary. Try common ones.
    tmp_candidates = ["tmp2m", "t2m", "tmpsfc", "tmp2m_sfc"]
    dpt_candidates = ["dpt2m", "td2m", "dptsfc", "dpt2m_sfc"]

    tmp = next((v for v in tmp_candidates if v in ds), None)
    dpt = next((v for v in dpt_candidates if v in ds), None)

    if tmp is not None:
        da = ds[tmp]
        if "time" in da.dims:
            da = da.isel(time=0)
        # vectorized interp
        out["model_temp_k"] = da.interp(**{lat_name: (("points",), lats), lon_name: (("points",), lon360)}).values
    else:
        out["model_temp_k"] = None

    if dpt is not None:
        da = ds[dpt]
        if "time" in da.dims:
            da = da.isel(time=0)
        out["model_dewpoint_k"] = da.interp(**{lat_name: (("points",), lats), lon_name: (("points",), lon360)}).values
    else:
        out["model_dewpoint_k"] = None

    return out

def build_bias_table(ds, metar_df: pd.DataFrame, *, lon_name: str, lat_name: str, max_rows: int = 200) -> tuple[pd.DataFrame, dict]:
    """Return (bias_table, summary_stats). Bias = model - obs in °C."""
    obs = metar_obs_fields(metar_df)
    obs = obs.dropna(subset=["lat", "lon"]).copy()
    if len(obs) > max_rows:
        obs = obs.sample(n=max_rows, random_state=42)

    lats = obs["lat"].to_numpy(dtype=float)
    lons = obs["lon"].to_numpy(dtype=float)

    mf = model_fields_at_points(ds, lon_name=lon_name, lat_name=lat_name, lats=lats, lons_deg=lons)

    # Convert K to C where available
    if mf["model_temp_k"] is not None:
        obs["model_temp_c"] = mf["model_temp_k"] - 273.15
    else:
        obs["model_temp_c"] = np.nan
    if mf["model_dewpoint_k"] is not None:
        obs["model_dewpoint_c"] = mf["model_dewpoint_k"] - 273.15
    else:
        obs["model_dewpoint_c"] = np.nan

    obs["temp_bias_c"] = obs["model_temp_c"] - obs["temp_c"]
    obs["dewpoint_bias_c"] = obs["model_dewpoint_c"] - obs["dewpoint_c"]

    # Summary
    summary = {}
    for col, name in [("temp_bias_c", "temp_bias_c"), ("dewpoint_bias_c", "dewpoint_bias_c")]:
        v = obs[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            summary[name] = {
                "median": float(np.median(v)),
                "mean": float(np.mean(v)),
                "p90_abs": float(np.percentile(np.abs(v), 90)),
                "n": int(v.size),
            }
        else:
            summary[name] = {"median": None, "mean": None, "p90_abs": None, "n": 0}

    # Keep tidy columns
    cols = ["station", "lat", "lon", "temp_c", "model_temp_c", "temp_bias_c", "dewpoint_c", "model_dewpoint_c", "dewpoint_bias_c"]
    return obs[cols], summary
