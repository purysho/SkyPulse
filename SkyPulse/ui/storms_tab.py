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
    st.caption("Objects = blobs in the Composite grid (threshold + labeling). Tracking = nearest-centroid matching. Motion = centroid-to-centroid between snapshots (constant motion extrapolation).")

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

    # Flatten motion fields for table readability
    def _get(d, path, default=None):
        cur = d
        for k in path:
            if cur is None or k not in cur:
                return default
            cur = cur[k]
        return cur

    df["speed_kmh"] = df["motion"].apply(lambda m: _get(m, ["speed_kmh"]))
    df["bearing_deg"] = df["motion"].apply(lambda m: _get(m, ["bearing_deg"]))
    df["dt_min"] = df["motion"].apply(lambda m: _get(m, ["dt_min"]))
    df["f30_lat"] = df["forecast_30min"].apply(lambda f: _get(f, ["lat"]))
    df["f30_lon"] = df["forecast_30min"].apply(lambda f: _get(f, ["lon"]))
    df["f60_lat"] = df["forecast_60min"].apply(lambda f: _get(f, ["lat"]))
    df["f60_lon"] = df["forecast_60min"].apply(lambda f: _get(f, ["lon"]))

    show_cols = ["id","lat","lon","area_km2","max_composite","mean_composite","speed_kmh","bearing_deg","dt_min","f30_lat","f30_lon","f60_lat","f60_lon"]
    df = df[show_cols].sort_values(["max_composite","area_km2"], ascending=[False, False])
    st.dataframe(df, use_container_width=True)

    st.divider()
    st.subheader("Map view (approx)")
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
            ax.scatter([x],[y], s=70)
            ax.text(x+6, y+6, str(r["id"]), fontsize=10, weight="bold")

            # draw 60-min arrow if forecast exists
            if pd.notna(r["f60_lat"]) and pd.notna(r["f60_lon"]):
                x2, y2 = xy(float(r["f60_lon"]), float(r["f60_lat"]))
                ax.annotate("", xy=(x2, y2), xytext=(x, y), arrowprops=dict(arrowstyle="->", lw=2))
                ax.scatter([x2],[y2], s=40, marker="x")
                ax.text(x2+6, y2+6, f"{r['id']}+60", fontsize=9)

        st.pyplot(fig, clear_figure=True)
        st.caption("Centroids/arrows are approximate on the PNG (bbox-linear mapping). Use Overlay tab for true geospatial context.")
    except Exception as e:
        st.warning(f"Could not render motion overlay: {e}")
