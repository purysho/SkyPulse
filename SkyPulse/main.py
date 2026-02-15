from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st
import streamlit.components.v1 as components
import xarray as xr
import numpy as np

from ingest.gfs_opendap import find_latest_gfs_anl_0p25, open_gfs_dataset, coord_names
from app.state import (
    write_latest,
    read_latest,
    minutes_since_update,
    maps_dir,
    write_stats,
    read_stats,
    read_prev_stats,
)
from compute.fields import get_level, bulk_shear_mag
from compute.indices import simple_hail_score
from compute.signals import build_domain_stats, generate_signals
from viz.maps import plot_scalar_field
from viz.render import save_fig

CONFIG_PATH = Path(__file__).parent / "app" / "config.json"

st.set_page_config(page_title="SkyPulse (Alpha)", layout="wide")

@st.cache_data
def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

cfg = load_config()
CACHE_DIR = cfg.get("cache_dir", "data_cache")

def subset_to_bbox(ds, field, lon_name: str, lat_name: str):
    bbox = cfg["region"]["bbox"]
    lon_min = bbox["lon_min"] % 360
    lon_max = bbox["lon_max"] % 360
    lat_min = bbox["lat_min"]
    lat_max = bbox["lat_max"]

    if lon_min <= lon_max:
        f_sub = field.sel(**{lat_name: slice(lat_min, lat_max), lon_name: slice(lon_min, lon_max)})
        lons_sub = ds[lon_name].sel(**{lon_name: slice(lon_min, lon_max)})
    else:
        f_a = field.sel(**{lat_name: slice(lat_min, lat_max), lon_name: slice(lon_min, 359.75)})
        f_b = field.sel(**{lat_name: slice(lat_min, lat_max), lon_name: slice(0, lon_max)})
        f_sub = xr.concat([f_a, f_b], dim=lon_name)
        lons_sub = xr.concat(
            [ds[lon_name].sel(**{lon_name: slice(lon_min, 359.75)}), ds[lon_name].sel(**{lon_name: slice(0, lon_max)})],
            dim=lon_name,
        )

    lats_sub = ds[lat_name].sel(**{lat_name: slice(lat_min, lat_max)})
    lons_plot = ((lons_sub + 180) % 360) - 180
    return f_sub, lons_plot, lats_sub

def composite_score_0_10(cape, shear):
    # Normalize and blend (simple, stable)
    cape_n = np.clip(cape / 3000.0, 0.0, 1.0)
    shear_n = np.clip(shear / 30.0, 0.0, 1.0)
    return 10.0 * 0.5 * (cape_n + shear_n)

def render_maps_and_stats(ds):
    lon_name, lat_name = coord_names(ds)
    out = maps_dir(CACHE_DIR)

    cape = None
    shear = None
    comp = None

    # CAPE
    if "capesfc" in ds:
        cape = ds["capesfc"]
        if "time" in cape.dims:
            cape = cape.isel(time=0)
        cape_sub, lons, lats = subset_to_bbox(ds, cape, lon_name, lat_name)
        fig = plot_scalar_field(lons, lats, cape_sub, title="Surface CAPE", units="J/kg")
        save_fig(fig, out / "cape_latest.png")
    else:
        cape_sub = None

    # Shear proxy (1000 -> 500 hPa)
    if "ugrdprs" in ds and "vgrdprs" in ds:
        u = ds["ugrdprs"]
        v = ds["vgrdprs"]
        if "time" in u.dims:
            u = u.isel(time=0)
            v = v.isel(time=0)
        u_lo = get_level(u, 1000.0); v_lo = get_level(v, 1000.0)
        u_hi = get_level(u, 500.0);  v_hi = get_level(v, 500.0)
        shear = bulk_shear_mag(u_lo, v_lo, u_hi, v_hi)
        shear_sub, lons, lats = subset_to_bbox(ds, shear, lon_name, lat_name)
        fig = plot_scalar_field(lons, lats, shear_sub, title="Bulk shear (1000–500 hPa proxy)", units="m/s")
        save_fig(fig, out / "shear_1000_500_latest.png")
    else:
        shear_sub = None

    # Composite
    if cape_sub is not None and shear_sub is not None:
        comp = composite_score_0_10(cape_sub, shear_sub)
        fig = plot_scalar_field(lons, lats, comp, title="Severe Ingredients Score (Composite)", units="0–10")
        save_fig(fig, out / "composite_latest.png")

        stats = build_domain_stats(cape=cape_sub.values, shear=shear_sub.values, composite=comp.values)
        write_stats(CACHE_DIR, stats)
        return stats

    return None

def do_update():
    run = find_latest_gfs_anl_0p25(days_back=2)
    write_latest(CACHE_DIR, {"source": "NOMADS OPeNDAP GFS 0.25° (analysis)", "ymd": run.ymd, "cycle": run.cycle, "url": run.url})
    ds = open_gfs_dataset(run.url)
    stats = render_maps_and_stats(ds)
    return run, stats

st.title("SkyPulse — Severe Weather & Atmospheric Intelligence (Alpha)")
st.caption("Ingredients-focused dashboard using public model data (GFS analysis).")

# ---- Update controls ----
st.subheader("Auto Update")
auto_update = st.toggle("Auto-update model & maps when stale", value=True)
auto_rerun = st.toggle("Auto-rerun page (keeps it live)", value=True)
refresh_min = st.number_input("Refresh interval (minutes)", min_value=5, max_value=180, value=int(cfg.get("refresh_minutes", 30)), step=5)

latest = read_latest(CACHE_DIR)
age = minutes_since_update(latest)

cols = st.columns([1,1,2])
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

if auto_update and (age is None or age > float(refresh_min)):
    with st.spinner("Updating to latest GFS analysis and generating maps..."):
        run, _ = do_update()
        st.success(f"Updated: {run.ymd} {run.cycle}Z")

st.subheader("Manual Update")
if st.button("Update now"):
    with st.spinner("Updating..."):
        run, _ = do_update()
        st.success(f"Updated: {run.ymd} {run.cycle}Z")

if auto_rerun:
    components.html(
        f"""<script>setTimeout(function(){{window.location.reload();}}, {int(refresh_min)*60*1000});</script>""" ,
        height=0,
    )

# ---- Tabs ----
tab_maps, tab_nowcast, tab_signals = st.tabs(["Maps", "Nowcast", "Signals"])

with tab_maps:
    st.subheader("Map Viewer")
    choice = st.selectbox("Select map", ["Composite", "CAPE", "Shear"])
    md = maps_dir(CACHE_DIR)
    paths = {
        "CAPE": md / "cape_latest.png",
        "Shear": md / "shear_1000_500_latest.png",
        "Composite": md / "composite_latest.png",
    }
    p = paths[choice]
    if p.exists():
        st.image(str(p), caption=f"{choice} (cached)")
    else:
        st.info("Map not available yet — run an update.")

with tab_nowcast:
    st.subheader("Nowcast Card (from model)")
    lat = st.number_input("Latitude", value=33.0, step=0.1, format="%.3f")
    lon = st.number_input("Longitude", value=-97.0, step=0.1, format="%.3f")

    latest = read_latest(CACHE_DIR)
    if latest and "url" in latest:
        try:
            ds = open_gfs_dataset(latest["url"])
            lon_name, lat_name = coord_names(ds)
            lon360 = lon % 360

            cape_val = None
            if "capesfc" in ds:
                cap = ds["capesfc"]
                if "time" in cap.dims:
                    cap = cap.isel(time=0)
                cape_val = float(cap.interp(**{lat_name: lat, lon_name: lon360}).values)

            shear_val = None
            if "ugrdprs" in ds and "vgrdprs" in ds:
                u = ds["ugrdprs"]; v = ds["vgrdprs"]
                if "time" in u.dims:
                    u = u.isel(time=0); v = v.isel(time=0)
                u_lo = get_level(u, 1000.0); v_lo = get_level(v, 1000.0)
                u_hi = get_level(u, 500.0);  v_hi = get_level(v, 500.0)
                shear = bulk_shear_mag(u_lo, v_lo, u_hi, v_hi)
                shear_val = float(shear.interp(**{lat_name: lat, lon_name: lon360}).values)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("CAPE (J/kg)", "N/A" if cape_val is None else f"{cape_val:.0f}")
            with c2:
                st.metric("Shear proxy (m/s)", "N/A" if shear_val is None else f"{shear_val:.1f}")
            with c3:
                if cape_val is not None and shear_val is not None:
                    st.metric("Hail score (0–10)", simple_hail_score(cape_val, shear_val))
                else:
                    st.metric("Hail score (0–10)", "N/A")

            if cape_val is not None and shear_val is not None:
                reasons = []
                if cape_val >= 1500: reasons.append("CAPE ≥ 1500")
                if shear_val >= 20: reasons.append("Shear ≥ 20 m/s")
                st.caption("Triggered thresholds: " + (", ".join(reasons) if reasons else "None"))

        except Exception as e:
            st.error(f"Nowcast failed: {e}")
    else:
        st.info("No dataset yet — run an update first.")

with tab_signals:
    st.subheader("Signals Feed")
    cur = read_stats(CACHE_DIR)
    prev = read_prev_stats(CACHE_DIR)
    if cur is None:
        st.info("Signals will appear after the first update (composite requires both CAPE and shear).")
    else:
        sigs = generate_signals(cur, prev)
        for line in sigs:
            st.write("•", line)

        st.subheader("Domain stats")
        st.json(cur)
