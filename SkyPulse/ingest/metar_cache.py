from __future__ import annotations

from io import BytesIO
import gzip
import requests
import pandas as pd

METARS_CACHE_CSV_GZ = "https://aviationweather.gov/data/cache/metars.cache.csv.gz"

def fetch_metars_cache(timeout: int = 20) -> pd.DataFrame:
    headers = {"User-Agent": "SkyPulse/alpha (educational; streamlit demo)"}
    r = requests.get(METARS_CACHE_CSV_GZ, headers=headers, timeout=timeout)
    r.raise_for_status()
    raw = gzip.decompress(r.content)
    df = pd.read_csv(BytesIO(raw))
    return df

def filter_bbox(df: pd.DataFrame, *, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> pd.DataFrame:
    lat_col = "latitude" if "latitude" in df.columns else ("lat" if "lat" in df.columns else None)
    lon_col = "longitude" if "longitude" in df.columns else ("lon" if "lon" in df.columns else None)
    if lat_col is None or lon_col is None:
        raise KeyError(f"Could not find lat/lon columns in METAR cache. Columns: {list(df.columns)[:30]}")
    sub = df[(df[lat_col] >= lat_min) & (df[lat_col] <= lat_max) & (df[lon_col] >= lon_min) & (df[lon_col] <= lon_max)].copy()
    return sub
