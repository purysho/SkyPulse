from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import streamlit as st

from services.mrms import update_mrms_reflectivity
from compute.mrms_objects import detect_and_track_mrms
from viz.maps import plot_scalar_field


def render_mrms_tab(cfg: dict, cache_dir: str):
    st.subheader("MRMS (Real Radar Composite) — Live Ingest + Objects")
    st.caption(
        "Fetch near real-time MRMS products from NOAA's public AWS bucket (2-minute cycle). "
        "We attempt to decode the GRIB2 using xarray+cfgrib when available. "
        "If decoding isn't supported on this host, raw GRIB2 is still saved for debugging."
    )

    region = st.selectbox("Region", ["CONUS", "ALASKA", "CARIB", "GUAM", "HAWAII"], index=0)
    product = st.selectbox(
        "Product",
        [
            "MergedReflectivityQCComposite_00.50",
            "ReflectivityAtLowestAltitude_00.50",
            "POSH_00.50",
        ],
        index=0,
    )
    max_age = st.slider("Max age (minutes)", 30, 360, 180, 10)

    if st.button("Update MRMS now"):
        meta = update_mrms_reflectivity(cache_dir, region=region, product=product, max_age_minutes=int(max_age))
        if meta.get("decoded"):
            st.success("MRMS downloaded + decoded ✔")
        else:
            st.warning("MRMS downloaded, but decode failed here (see note).")

    meta_path = Path(cache_dir) / "mrms_reflectivity_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        st.json(meta)

    npz_path = Path(cache_dir) / "mrms_reflectivity_latest.npz"
    if not npz_path.exists():
        st.info("No decoded MRMS grid yet. Click 'Update MRMS now'.")
        return

    data = np.load(npz_path)
    lons = data["lons"]
    lats = data["lats"]
    field = data["field"]

    st.subheader("MRMS Map (decoded)")
    st.pyplot(plot_scalar_field(lons, lats, field, title=f"MRMS {product}", units=""), clear_figure=True)

    st.divider()
    st.subheader("MRMS Object Detection + Tracking")
    thr = st.slider("Threshold (units depend on product)", 0.0, 80.0, 35.0, 1.0)
    minpix = st.slider("Min size (pixels)", 5, 500, 25, 5)

    if st.button("Detect/Track MRMS objects"):
        payload = detect_and_track_mrms(cache_dir, threshold=float(thr), min_pixels=int(minpix))
        st.success(f"Detected {len(payload.get('objects', []))} objects.")

    track_path = Path(cache_dir) / "mrms_objects_tracks_latest.json"
    if track_path.exists():
        payload = json.loads(track_path.read_text(encoding="utf-8"))
        st.write("Updated (UTC):", payload.get("updated_at_utc", ""))
        st.dataframe(payload.get("objects", []), use_container_width=True)
