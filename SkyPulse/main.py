from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import streamlit as st
import streamlit.components.v1 as components

from services.config import load_config
from services.model import update_and_render
from services.storms import run_storm_detection
from app.state import read_latest, minutes_since_update

from ui.maps_tab import render_maps_tab
from ui.mrms_tab import render_mrms_tab
from ui.nowcast_tab import render_nowcast_tab
from ui.signals_tab import render_signals_tab
from ui.radar_tab import render_radar_tab
from ui.overlay_tab import render_overlay_tab
from ui.storms_tab import render_storms_tab
from ui.satellite_tab import render_satellite_tab

CONFIG_PATH = Path(__file__).parent / "app" / "config.json"

st.set_page_config(page_title="SkyPulse (Alpha)", layout="wide")

cfg = load_config(CONFIG_PATH)
cache_dir = cfg.get("cache_dir", "data_cache")

st.title("SkyPulse — Severe Weather & Atmospheric Intelligence (Alpha)")
st.caption("Public-data severe ingredients dashboard: GFS analysis + METAR verification + boundary detection + initiation watch.")

# Controls
st.subheader("Auto Update")
auto_update = st.toggle("Auto-update model & maps when stale", value=True)
auto_rerun = st.toggle("Auto-rerun page (keeps it live)", value=True)
refresh_min = st.number_input(
    "Refresh interval (minutes)",
    min_value=5,
    max_value=180,
    value=int(cfg.get("refresh_minutes", 30)),
    step=5,
)

auto_storms = st.toggle("Auto-detect storm objects after update", value=True)

latest = read_latest(cache_dir)
age = minutes_since_update(latest)

cols = st.columns([1, 1, 2])
with cols[0]:
    st.write("Now (UTC):", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))
with cols[1]:
    st.write("Domain:", cfg["region"]["name"])
with cols[2]:
    st.write("BBox:", cfg["region"]["bbox"])

if latest and age is not None:
    st.info(f"Last update: {latest.get('ymd','')} {latest.get('cycle','')}Z — age {age:.1f} min")
else:
    st.warning("No updates yet.")

def run_update():
    run, _stats = update_and_render(cfg, cache_dir)

    # Auto storm detection (Composite-object tracking)
    if auto_storms:
        try:
            run_storm_detection(cache_dir, threshold=6.0, min_pixels=12)
        except Exception:
            pass

    return run


if auto_update and (age is None or age > float(refresh_min)):
    with st.spinner("Updating to latest GFS analysis and generating maps..."):
        run = run_update()
        st.success(f"Updated: {run['ymd']} {run['cycle']}Z")
        latest = read_latest(cache_dir)

st.subheader("Manual Update")
if st.button("Update now"):
    with st.spinner("Updating..."):
        run = run_update()
        st.success(f"Updated: {run['ymd']} {run['cycle']}Z")
        latest = read_latest(cache_dir)

if auto_rerun:
    components.html(
        f"""<script>setTimeout(function(){{window.location.reload();}}, {int(refresh_min)*60*1000});</script>""",
        height=0,
    )

tab_maps, tab_nowcast, tab_overlay, tab_storms, tab_radar, tab_mrms, tab_sat, tab_signals = st.tabs(["Maps", "Nowcast", "Overlay", "Storms", "Radar", "MRMS", "Satellite", "Signals"])

with tab_maps:
    render_maps_tab(cache_dir)

with tab_nowcast:
    render_nowcast_tab(cfg, cache_dir, latest)

with tab_overlay:
    render_overlay_tab(cfg, cache_dir)

with tab_storms:
    render_storms_tab(cfg, cache_dir)

with tab_radar:
    render_radar_tab()

with tab_mrms:
    render_mrms_tab(cfg, cache_dir)

with tab_sat:
    render_satellite_tab(cfg)

with tab_signals:
    render_signals_tab(cfg, cache_dir, latest)
