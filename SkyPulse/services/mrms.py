from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import xarray as xr

from ingest.mrms_aws import download_latest_mrms_grib2_gz

META_FILE = "mrms_reflectivity_meta.json"
DECODED_NC = "mrms_decoded_latest.nc"

def update_mrms_latest(cache_dir: str, *, max_age_minutes: int = 15) -> dict[str, Any]:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    run = download_latest_mrms_grib2_gz(cache, max_age_minutes=max_age_minutes)
    meta = {"updated_at_utc": datetime.now(timezone.utc).isoformat(), **run, "decoded": False}
    (cache / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta

def try_decode_mrms_grib2(cache_dir: str) -> tuple[bool, str]:
    cache = Path(cache_dir)
    meta_path = cache / META_FILE
    if not meta_path.exists():
        return False, "No MRMS meta found. Click Update MRMS first."
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    raw_path = cache / meta.get("raw_grib2_path", "")
    if not raw_path.exists():
        return False, f"Raw GRIB2 not found: {raw_path}"

    try:
        ds = xr.open_dataset(raw_path, engine="cfgrib")
        out = cache / DECODED_NC
        ds.to_netcdf(out)
        meta["decoded"] = True
        meta["decoded_path"] = out.name
        meta["note"] = "Decoded OK via cfgrib -> netcdf"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return True, "Decoded OK"
    except Exception as e:
        meta["decoded"] = False
        meta["note"] = f"Decode failed (likely missing cfgrib/eccodes). Raw GRIB2 saved. Error: {e}"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return False, meta["note"]

def detect_mrms_objects(cache_dir: str) -> dict[str, Any]:
    cache = Path(cache_dir)
    decoded = cache / DECODED_NC
    if not decoded.exists():
        raise RuntimeError("No decoded MRMS file yet. Click 'Try decode (cfgrib)' first.")
    return {"updated_at_utc": datetime.now(timezone.utc).isoformat(), "objects": []}
