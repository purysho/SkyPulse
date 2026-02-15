from __future__ import annotations

from pathlib import Path
import streamlit as st
from app.state import maps_dir

def render_maps_tab(cache_dir: str):
    st.subheader("Map Viewer")
    md = maps_dir(cache_dir)
    choice = st.selectbox("Select map", ["Composite", "CAPE", "Shear", "Boundary"])
    paths = {
        "CAPE": md / "cape_latest.png",
        "Shear": md / "shear_1000_500_latest.png",
        "Composite": md / "composite_latest.png",
        "Boundary": md / "boundary_latest.png",
    }
    p = paths[choice]
    if p.exists():
        st.image(str(p), caption=f"{choice} (cached)")
    else:
        st.info("Map not available yet — run an update (and for Boundary: open Signals once).")
