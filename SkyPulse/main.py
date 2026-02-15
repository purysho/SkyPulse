from compute.fields import get_level, bulk_shear_mag
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st
import streamlit.components.v1 as components
import xarray as xr
from ingest.gfs_opendap import find_latest_gfs_anl_0p25, open_gfs_dataset, coord_names
from app.state import write_latest, read_latest, minutes_since_update, maps_dir
CONFIG_PATH = Path(__file__).parent / "app" / "config.json"

st.set_page_config(page_title="SkyPulse (Alpha)", layout="wide")

@st.cache_data
def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

cfg = load_config()
CACHE_DIR = cfg.get("cache_dir", "data_cache")


def subset_to_bbox(ds, field, lon_name: str, lat_name: str):
    """Subset a field to cfg bbox. Returns (field_sub, lons_plot, lats_sub)."""
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
            [
                ds[lon_name].sel(**{lon_name: slice(lon_min, 359.75)}),
                ds[lon_name].sel(**{lon_name: slice(0, lon_max)}),
            ],
            dim=lon_name,
        )

    lats_sub = ds[lat_name].sel(**{lat_name: slice(lat_min, lat_max)})
    lons_plot = ((lons_sub + 180) % 360) - 180
    return f_sub, lons_plot, lats_sub


def render_maps(ds):
    """Render and cache multiple maps as PNGs in data_cache/maps."""
    lon_name, lat_name = coord_names(ds)
    out = maps_dir(CACHE_DIR)

    # 1) Surface CAPE
    if "capesfc" in ds:
        cape = ds["capesfc"]
        if "time" in cape.dims:
            cape = cape.isel(time=0)
        cape_sub, lons, lats = subset_to_bbox(ds, cape, lon_name, lat_name)
        fig = plot_scalar_field(lons, lats, cape_sub, title="Surface CAPE", units="J/kg")
        save_fig(fig, out / "cape_latest.png")

    # 2) Bulk shear proxy (1000 -> 500 hPa) from ugrdprs/vgrdprs
    if "ugrdprs" in ds and "vgrdprs" in ds:
        u = ds["ugrdprs"]
        v = ds["vgrdprs"]
        if "time" in u.dims:
            u = u.isel(time=0)
            v = v.isel(time=0)
        u_lo = get_level(u, 1000.0)
        v_lo = get_level(v, 1000.0)
        u_hi = get_level(u, 500.0)
        v_hi = get_level(v, 500.0)
        shear = bulk_shear_mag(u_lo, v_lo, u_hi, v_hi)
        shear_sub, lons, lats = subset_to_bbox(ds, shear, lon_name, lat_name)
        fig = plot_scalar_field(lons, lats, shear_sub, title="Bulk shear (1000–500 hPa proxy)", units="m/s")
        save_fig(fig, out / "shear_1000_500_latest.png")
st.title("SkyPulse — Severe Weather & Atmospheric Intelligence (Alpha)")
st.caption("Starter scaffold: we’ll wire in live model ingest + maps next.")

colA, colB = st.columns([1, 1])

with colA:
    st.subheader("Domain")
    st.json(cfg["region"])

with colB:
    st.subheader("Status")
    st.write("Now (UTC):", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    st.write("Refresh interval (minutes):", cfg.get("refresh_minutes", 60))
    st.write("Cache dir:", cfg.get("cache_dir", "data_cache"))

st.subheader("Auto Update")
auto_update = st.toggle("Auto-update model & maps when stale", value=True)
auto_rerun = st.toggle("Auto-rerun page (keeps it live)", value=True)
refresh_min = st.number_input(
    "Refresh interval (minutes)",
    min_value=5,
    max_value=180,
    value=int(cfg.get("refresh_minutes", 60)),
    step=5,
)

latest = read_latest(CACHE_DIR)
age = minutes_since_update(latest)

if latest and age is not None:
    st.caption(f"Last update: {latest.get('updated_at_utc','?')}  (age: {age:.1f} min)")
else:
    st.caption("No updates yet.")

def do_update():
    run = find_latest_gfs_anl_0p25(days_back=2)
    write_latest(
        CACHE_DIR,
        {
            "source": "NOMADS OPeNDAP GFS 0.25° (analysis)",
            "ymd": run.ymd,
            "cycle": run.cycle,
            "url": run.url,
        },
    )
    return run

# Auto-update if stale (or missing)
if auto_update and (age is None or age > float(refresh_min)):
    with st.spinner("Updating to latest GFS run..."):
        run = do_update()
        st.success(f"Updated: {run.ymd} {run.cycle}Z")

# Optional auto-rerun (browser refresh) so the page stays live even if no one clicks.
# This is Streamlit Cloud-friendly: it just reloads the page in the user's browser.
if auto_rerun:
    components.html(
        f"""
        <script>
          setTimeout(function() {{
            window.location.reload();
          }}, {int(refresh_min) * 60 * 1000});
        </script>
        """,
        height=0,
    )


st.divider()

st.subheader("Manual Update")
if st.button("Update now"):
    with st.spinner("Updating..."):
        run = do_update()
        st.success(f"Updated: {run.ymd} {run.cycle}Z")


st.subheader("Nowcast Card (from model)")

lat = st.number_input("Latitude", value=33.0, step=0.1, format="%.3f")
lon = st.number_input("Longitude", value=-97.0, step=0.1, format="%.3f")

latest = read_latest(CACHE_DIR)
if latest and "url" in latest:
    try:
        ds = open_gfs_dataset(latest["url"])
        lon_name, lat_name = coord_names(ds)
        lon360 = lon % 360

        # CAPE
        cape_val = None
        if "capesfc" in ds:
            cap = ds["capesfc"]
            if "time" in cap.dims:
                cap = cap.isel(time=0)
            cape_val = float(cap.interp(**{lat_name: lat, lon_name: lon360}).values)

        # Shear proxy (1000->500 hPa)
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
            if cape_val is not None:
                st.metric("CAPE (J/kg)", round(cape_val, 1))
            else:
                st.metric("CAPE (J/kg)", "N/A")
        with c2:
            if shear_val is not None:
                st.metric("Shear proxy (m/s)", round(shear_val, 2))
            else:
                st.metric("Shear proxy (m/s)", "N/A")
        with c3:
            if cape_val is not None and shear_val is not None:
                st.metric("Hail score (0–10)", simple_hail_score(cape_val, shear_val))
            else:
                st.metric("Hail score (0–10)", "N/A")

    except Exception as e:
        st.error(f"Nowcast failed: {e}")
else:
    st.info("No dataset yet — wait for auto-update or press Update now.")

st.subheader("Latest cached maps")
out = maps_dir(CACHE_DIR)
if (out / "cape_latest.png").exists():
    st.image(str(out / "cape_latest.png"), caption="CAPE (cached)")
if (out / "shear_1000_500_latest.png").exists():
    st.image(str(out / "shear_1000_500_latest.png"), caption="Shear proxy (cached)")

st.subheader("Map:st.subheader("Map: Surface CAPE (GFS capesfc)")
latest = read_latest(CACHE_DIR)
if latest and "url" in latest:
    try:
        ds = open_gfs_dataset(latest["url"])

        # NOMADS GFS OPeNDAP uses lon 0..360; lat -90..90.
        # Variable name for surface CAPE in this dataset is typically `capesfc`.
        if "capesfc" not in ds:
            st.warning(f"capesfc not found. Available vars (sample): {list(ds.data_vars)[:25]}")
        else:
            cap = ds["capesfc"]
            # dims often: [time, lat, lon] or [time, y, x]
            if "time" in cap.dims:
                cap = cap.isel(time=0)

            # Coordinate names
            lon_name = "lon" if "lon" in ds.coords else ("longitude" if "longitude" in ds.coords else None)
            lat_name = "lat" if "lat" in ds.coords else ("latitude" if "latitude" in ds.coords else None)
            if lon_name is None or lat_name is None:
                st.error(f"Could not find lon/lat coords. Coords: {list(ds.coords)}")
            else:
                lons = ds[lon_name]
                lats = ds[lat_name]

                # Subset to config bbox (convert bbox lon to 0..360)
                bbox = cfg["region"]["bbox"]
                lon_min = bbox["lon_min"] % 360
                lon_max = bbox["lon_max"] % 360
                lat_min = bbox["lat_min"]
                lat_max = bbox["lat_max"]

                # Handle cases where lon_min > lon_max (crosses 0 meridian)
                if lon_min <= lon_max:
                    cap_sub = cap.sel(**{lat_name: slice(lat_min, lat_max), lon_name: slice(lon_min, lon_max)})
                    lons_sub = lons.sel(**{lon_name: slice(lon_min, lon_max)})
                else:
                    # Rare for US boxes, but handle anyway
                    cap_a = cap.sel(**{lat_name: slice(lat_min, lat_max), lon_name: slice(lon_min, 359.75)})
                    cap_b = cap.sel(**{lat_name: slice(lat_min, lat_max), lon_name: slice(0, lon_max)})
                    cap_sub = xr.concat([cap_a, cap_b], dim=lon_name)
                    lons_sub = xr.concat([lons.sel(**{lon_name: slice(lon_min, 359.75)}), lons.sel(**{lon_name: slice(0, lon_max)})], dim=lon_name)

                lats_sub = lats.sel(**{lat_name: slice(lat_min, lat_max)})

                # Convert lon back to -180..180 for nicer display
                lons_plot = ((lons_sub + 180) % 360) - 180

                fig = plot_scalar_field(lons_plot, lats_sub, cap_sub, title=f"GFS Surface CAPE (analysis) {latest.get('ymd','')} {latest.get('cycle','')}Z", units="J/kg")
                st.pyplot(fig, clear_figure=True)

    except Exception as e:
        st.error(f"Could not open/plot OPeNDAP dataset: {e}")
else:
    st.info("Select the latest dataset above to see the live CAPE map.")
