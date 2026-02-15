from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
import requests


@dataclass(frozen=True)
class GfsRun:
    ymd: str          # YYYYMMDD (UTC)
    cycle: str        # "00" | "06" | "12" | "18"
    url: str          # OPeNDAP dataset base URL (no .dds)


def _candidate_dates_utc(days_back: int = 2) -> list[str]:
    now = datetime.now(timezone.utc)
    return [(now - timedelta(days=i)).strftime("%Y%m%d") for i in range(days_back + 1)]


def _exists_dds(dataset_url: str, timeout: int = 12) -> bool:
    # NOMADS OPeNDAP supports a lightweight .dds endpoint for existence checks
    try:
        r = requests.get(dataset_url + ".dds", timeout=timeout)
        return r.status_code == 200 and ("Dataset" in r.text or "Grid" in r.text)
    except Exception:
        return False


def find_latest_gfs_anl_0p25(*, days_back: int = 2) -> GfsRun:
    """
    Find the latest available GFS 0.25° analysis dataset on NOMADS OPeNDAP.

    Dataset pattern (example):
      https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs20260215/gfs_0p25_06z_anl

    We probe UTC dates (today/back) and cycles (18→00) until we find one that exists.
    """
    base = "https://nomads.ncep.noaa.gov/dods/gfs_0p25"
    cycles = ["18", "12", "06", "00"]

    for ymd in _candidate_dates_utc(days_back=days_back):
        for cyc in cycles:
            url = f"{base}/gfs{ymd}/gfs_0p25_{cyc}z_anl"
            if _exists_dds(url):
                return GfsRun(ymd=ymd, cycle=cyc, url=url)

    raise RuntimeError("Could not find a recent GFS 0.25° analysis dataset via NOMADS OPeNDAP.")


def open_gfs_dataset(url: str):
    """Open the OPeNDAP dataset with xarray (netCDF4 backend)."""
    import xarray as xr
    # Using netCDF4 backend is the most reliable on Streamlit Cloud
    return xr.open_dataset(url)

def coord_names(ds):
    """Return (lon_name, lat_name) coordinate names for a dataset."""
    lon = "lon" if "lon" in ds.coords else ("longitude" if "longitude" in ds.coords else None)
    lat = "lat" if "lat" in ds.coords else ("latitude" if "latitude" in ds.coords else None)
    if lon is None or lat is None:
        raise KeyError(f"Could not find lon/lat coords. Coords: {list(ds.coords)}")
    return lon, lat

