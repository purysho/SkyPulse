from __future__ import annotations

import re
import gzip
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

BUCKET_BASE = "https://noaa-mrms-pds.s3.amazonaws.com"
PRODUCT = "MergedReflectivityQCComposite_00.50"
REGION = "CONUS"

def _utcnow():
    return datetime.now(timezone.utc)

def _ymd(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")

def _list_prefix(prefix: str, timeout: float = 15.0) -> list[str]:
    url = f"{BUCKET_BASE}/?list-type=2&prefix={prefix}"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    keys = re.findall(r"<Key>([^<]+)</Key>", r.text)
    return keys

def download_latest_mrms_grib2_gz(cache_dir: Path, *, max_age_minutes: int = 15) -> dict[str, Any]:
    now = _utcnow()
    prefix = f"{REGION}/{PRODUCT}/{_ymd(now)}/"
    keys = _list_prefix(prefix)
    gz_keys = [k for k in keys if k.endswith(".grib2.gz")]
    if not gz_keys:
        raise RuntimeError("No MRMS grib2.gz files found for today in bucket listing.")
    gz_keys.sort()
    key = gz_keys[-1]
    url = f"{BUCKET_BASE}/{key}"

    m = re.search(r"_(\d{8})-(\d{6})\.grib2\.gz$", key)
    ts_iso = None
    age_min = None
    ts_tag = "unknown"
    if m:
        ymd, hms = m.group(1), m.group(2)
        ts = datetime.strptime(ymd + hms, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        ts_iso = ts.isoformat()
        age_min = (now - ts).total_seconds() / 60.0
        ts_tag = ts.strftime("%Y%m%d-%H%M%S")

    gz_path = cache_dir / f"mrms_raw_{ts_tag}.grib2.gz"
    grib_path = cache_dir / f"mrms_raw_{ts_tag}.grib2"

    r = requests.get(url, timeout=30)
    r.raise_for_status()
    gz_path.write_bytes(r.content)

    with gzip.open(gz_path, "rb") as f_in:
        grib_path.write_bytes(f_in.read())

    return {
        "region": REGION,
        "product": PRODUCT,
        "source": "AWS noaa-mrms-pds",
        "key": key,
        "url": url,
        "timestamp_utc": ts_iso,
        "age_minutes": None if age_min is None else round(age_min, 1),
        "raw_grib2_path": grib_path.name,
    }
