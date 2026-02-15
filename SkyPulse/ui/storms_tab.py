from __future__ import annotations

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from services.storms import run_storm_detection
from app.state import maps_dir

def render_storms_tab(cfg: dict, cache_dir: str):
    st.subheader("Storm Objects + Tracking (radar-like)")
    st.caption("This is a first-pass 'object' system using the Composite grid (threshold + labeling). Next step is MRMS radar ingest for real storm structure.")

    threshold = st.slider("Object threshold (Composite)", 0.0, 10.0, 6.0, 0.1)
    min_pixels = st.slider("Min object size (pixels)", 3, 200, 12, 1)

    if st.button("Detect & track objects now"):
        try:
            payload = run_storm_detection(cache_dir, threshold=threshold, min_pixels=min_pixels)
            st.success(f"Detected {len(payload.get('objects', []))} objects.")
        except Exception as e:
            st.error(str(e))
            return

    track_file = Path(cache_dir) / "storms_tracks_latest.json"
    if not track_file.exists():
        st.info("No storm objects yet. Click 'Detect & track' after you run an Update.")
        return

    payload = __import__("json").loads(track_file.read_text(encoding="utf-8"))
    st.write("Updated (UTC):", payload.get("updated_at_utc",""))
    objs = payload.get("objects", [])
    if not objs:
        st.info("No objects above threshold in the domain.")
        return

    df = pd.DataFrame(objs)
    df = df.sort_values(["max_composite","area_km2"], ascending=[False, False])
    st.dataframe(df, use_container_width=True)

    # Plot centroids on composite map (approx)
    md = maps_dir(cache_dir)
    comp_png = md / "composite_latest.png"
    if not comp_png.exists():
        st.info("Composite image not found yet.")
        return

    try:
        import PIL.Image as Image
        img = Image.open(comp_png).convert("RGBA")
        fig = plt.figure(figsize=(10,6))
        ax = plt.gca()
        ax.imshow(img)
        ax.axis("off")
        # approximate mapping from lon/lat to image coords using bbox
        bbox = cfg["region"]["bbox"]
        lon_min, lon_max = float(bbox["lon_min"]), float(bbox["lon_max"])
        lat_min, lat_max = float(bbox["lat_min"]), float(bbox["lat_max"])
        w, h = img.size

        def xy(lon, lat):
            x = (lon - lon_min) / (lon_max - lon_min) * w
            y = (lat_max - lat) / (lat_max - lat_min) * h
            return x, y

        for _, r in df.iterrows():
            x, y = xy(float(r["lon"]), float(r["lat"]))
            ax.scatter([x],[y], s=60)
            ax.text(x+6, y+6, str(r["id"]), fontsize=10, weight="bold")
        st.pyplot(fig, clear_figure=True)
        st.caption("Centroids are approximate on the PNG (bbox-linear mapping). Overlay tab is the better geospatial view.")
    except Exception as e:
        st.warning(f"Could not render centroid overlay: {e}")
