from __future__ import annotations

from dataclasses import dataclass
from time import time

# NWS Ridge radar imagery (static images) — no API key
# Docs/entry points are long-lived; images update frequently.
# We'll add cache-busting querystring to ensure Streamlit fetches fresh frames.

@dataclass(frozen=True)
class RadarProduct:
    key: str
    label: str
    url: str

def cache_bust(url: str) -> str:
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}_ts={int(time())}"

def radar_products() -> list[RadarProduct]:
    # CONUS composites (GIF)
    # NWS Ridge "standard" products are widely used in embeds.
    base = "https://radar.weather.gov/ridge/standard"
    return [
        RadarProduct("conus_ref", "CONUS Reflectivity (base)", f"{base}/CONUS_0.gif"),
        RadarProduct("conus_ref_loop", "CONUS Reflectivity Loop", f"{base}/CONUS_loop.gif"),
        RadarProduct("conus_vel", "CONUS Velocity (base)", f"{base}/CONUS_1.gif"),
        RadarProduct("alaska_ref", "Alaska Reflectivity (base)", f"{base}/ALASKA_0.gif"),
        RadarProduct("hawaii_ref", "Hawaii Reflectivity (base)", f"{base}/HAWAII_0.gif"),
        RadarProduct("pr_ref", "Puerto Rico Reflectivity (base)", f"{base}/GUAM_0.gif"),  # Ridge naming varies; keep option
    ]
