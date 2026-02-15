from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ingest.mrms_aws import find_latest_object, download_and_decompress_grib2

MRMS_NPZ = "mrms_reflectivity_latest.npz"
MRMS_META = "mrms_reflectivity_meta.json"


def update_mrms_reflectivity(
    cache_dir: str | Path,
    *,
    region: str = "CONUS",
    product: str = "MergedReflectivityQCComposite_00.50",
    max_age_minutes: int = 180,
) -> dict[str, Any]:
    """
    Fetch latest MRMS product from AWS, attempt to decode with cfgrib, and store as NPZ.
    If cfgrib/eccodes are unavailable (common on Streamlit Cloud), raw GRIB2 is still saved.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    latest = find_latest_object(region, product)
    grib_bytes = download_and_decompress_grib2(latest.url)

    raw_path = cache_dir / f"mrms_raw_{latest.timestamp_utc.strftime('%Y%m%d-%H%M%S')}.grib2"
    raw_path.write_bytes(grib_bytes)

    age_min = abs((datetime.now(timezone.utc) - latest.timestamp_utc).total_seconds()) / 60.0

    meta: dict[str, Any] = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "region": region,
        "product": product,
        "source": "AWS noaa-mrms-pds",
        "key": latest.key,
        "url": latest.url,
        "timestamp_utc": latest.timestamp_utc.isoformat(),
        "age_minutes": round(age_min, 1),
        "raw_grib2_path": raw_path.name,
        "decoded": False,
        "note": "",
    }

    if age_min > max_age_minutes:
        meta["note"] = f"Latest object is older than max_age_minutes={max_age_minutes} (age={age_min:.1f} min)."

    try:
        import xarray as xr
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".grib2") as f:
            f.write(grib_bytes)
            f.flush()
            da = xr.load_dataarray(f.name, engine="cfgrib", decode_timedelta=True)

        lon_name = "longitude" if "longitude" in da.coords else ("lon" if "lon" in da.coords else None)
        lat_name = "latitude" if "latitude" in da.coords else ("lat" if "lat" in da.coords else None)
        if lon_name is None or lat_name is None:
            raise RuntimeError(f"Could not find lon/lat coords in decoded MRMS data. Coords: {list(da.coords)}")

        lons = np.array(da[lon_name].values, dtype=float)
        lats = np.array(da[lat_name].values, dtype=float)
        vals = np.array(da.values, dtype=float)

        # store generic field name (some products are not reflectivity)
        np.savez_compressed(cache_dir / MRMS_NPZ, lons=lons, lats=lats, field=vals)

        meta["decoded"] = True
        meta["note"] = (meta["note"] + " " if meta["note"] else "") + "Decoded using xarray+cfgrib."
    except Exception as e:
        meta["note"] = (meta["note"] + " " if meta["note"] else "") + (
            "Decode failed (likely missing cfgrib/eccodes). Raw GRIB2 saved. "
            f"Error: {e}"
        )

    (cache_dir / MRMS_META).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
