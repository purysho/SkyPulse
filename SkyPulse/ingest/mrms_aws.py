from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import gzip
import re
import xml.etree.ElementTree as ET

import requests

AWS_BUCKET = "https://noaa-mrms-pds.s3.amazonaws.com"


@dataclass(frozen=True)
class MRMSSelection:
    region: str
    product: str
    datestring: str  # YYYYMMDD


@dataclass(frozen=True)
class MRMSObject:
    key: str
    url: str
    timestamp_utc: datetime


def list_objects(sel: MRMSSelection, *, timeout_s: int = 20) -> list[MRMSObject]:
    """
    List MRMS objects in the NOAA MRMS AWS public bucket for a given region/product/date.
    Uses S3 ListObjectsV2 (XML).
    """
    prefix = f"{sel.region}/{sel.product}/{sel.datestring}/"
    params = {"list-type": "2", "prefix": prefix}
    r = requests.get(AWS_BUCKET, params=params, timeout=timeout_s)
    r.raise_for_status()

    root = ET.fromstring(r.text)
    ns = {"s3": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}
    keys: list[str] = []
    for c in root.findall(".//s3:Contents" if ns else ".//Contents", ns):
        k = c.find("s3:Key" if ns else "Key", ns)
        if k is not None and k.text:
            keys.append(k.text)

    out: list[MRMSObject] = []
    for k in keys:
        m = re.search(r"_(\d{8}-\d{6})\.grib2\.gz$", k)
        if not m:
            continue
        ts = datetime.strptime(m.group(1), "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
        out.append(MRMSObject(key=k, url=f"{AWS_BUCKET}/{k}", timestamp_utc=ts))

    out.sort(key=lambda o: o.timestamp_utc)
    return out


def find_latest_object(
    region: str,
    product: str,
    *,
    max_age_minutes: int = 120,
    now: datetime | None = None,
) -> MRMSObject:
    """
    Find the most recent MRMS object for today (UTC). If none, try yesterday.
    Returns the newest object even if older than max_age_minutes (but flags as old elsewhere).
    """
    now = now or datetime.now(timezone.utc)
    for back_days in (0, 1):
        day = (now - timedelta(days=back_days)).strftime("%Y%m%d")
        sel = MRMSSelection(region=region, product=product, datestring=day)
        objs = list_objects(sel)
        if not objs:
            continue
        return objs[-1]
    raise RuntimeError("No MRMS objects found for today/yesterday for the selected region/product.")


def download_and_decompress_grib2(url: str, *, timeout_s: int = 30) -> bytes:
    """Download a .grib2.gz MRMS file and return decompressed GRIB2 bytes."""
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    return gzip.decompress(r.content)
