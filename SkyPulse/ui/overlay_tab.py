from __future__ import annotations

import base64
from pathlib import Path
import streamlit as st
import json
import streamlit.components.v1 as components

from app.state import maps_dir

# NWS radar base reflectivity tile service (ArcGIS). No key.
RADAR_TILE = "https://idpgis.ncep.noaa.gov/arcgis/rest/services/NWS_Observations/radar_base_reflectivity/MapServer/tile/{z}/{y}/{x}"

def _png_to_data_url(path: Path) -> str | None:
    try:
        b = path.read_bytes()
        return "data:image/png;base64," + base64.b64encode(b).decode("ascii")
    except Exception:
        return None

def render_overlay_tab(cfg: dict, cache_dir: str):
    st.subheader("Overlay (Georegistered)")
    st.caption("Leaflet map with live radar tiles + optional ingredient image overlay (approx georegistered to the domain bbox).")

    bbox = cfg["region"]["bbox"]
    lat_min = float(bbox["lat_min"]); lat_max = float(bbox["lat_max"])
    lon_min = float(bbox["lon_min"]); lon_max = float(bbox["lon_max"])

    center_lat = (lat_min + lat_max) / 2.0
    center_lon = (lon_min + lon_max) / 2.0

    md = maps_dir(cache_dir)
    composite_path = md / "composite_latest.png"
    cape_path = md / "cape_latest.png"
    shear_path = md / "shear_1000_500_latest.png"

    overlay_choice = st.selectbox("Ingredient overlay", ["None", "Composite", "CAPE", "Shear"], index=1)
    overlay_map = {
        "Composite": composite_path,
        "CAPE": cape_path,
        "Shear": shear_path,
    }

    img_url = None
    if overlay_choice != "None":
        p = overlay_map[overlay_choice]
        if p.exists():
            img_url = _png_to_data_url(p)
        else:
            st.info("Overlay image not found yet — run an update in the main page first.")

    opacity = st.slider("Overlay opacity", 0.0, 1.0, 0.55, 0.05)

    lat_span = abs(lat_max - lat_min)
    if lat_span > 35:
        zoom = 4
    elif lat_span > 20:
        zoom = 5
    elif lat_span > 10:
        zoom = 6
    else:
        zoom = 7

    img_js = ""
    if img_url:
        img_js = f"""
        var imgBounds = [[{lat_min}, {lon_min}], [{lat_max}, {lon_max}]];
        var imgOverlay = L.imageOverlay("{img_url}", imgBounds, {{opacity: {opacity}}}).addTo(map);
        """

    html = f"""
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
      integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
      integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>

    <div id="map" style="width: 100%; height: 600px; border-radius: 12px;"></div>
    <script>
      var map = L.map('map').setView([{center_lat}, {center_lon}], {zoom});

      L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
        maxZoom: 10,
        attribution: '&copy; OpenStreetMap contributors'
      }}).addTo(map);

      L.tileLayer('{RADAR_TILE}', {{
        maxZoom: 10,
        opacity: 0.70,
        attribution: 'NWS radar'
      }}).addTo(map);

      L.rectangle([[{lat_min}, {lon_min}], [{lat_max}, {lon_max}]], {{
        color: '#ff7800',
        weight: 2,
        fill: false
      }}).addTo(map);

      {img_js}
    </script>
    """

    components.html(html, height=640, scrolling=False)

    st.info("Next upgrade: ingest gridded radar (MRMS) and compute storm objects/tracks inside the overlay.")
