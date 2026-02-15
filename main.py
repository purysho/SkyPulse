from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st

from compute.indices import simple_hail_score, simple_tornado_score

CONFIG_PATH = Path(__file__).parent / "app" / "config.json"

st.set_page_config(page_title="SkyPulse (Alpha)", layout="wide")

@st.cache_data
def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

cfg = load_config()

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

st.divider()

st.subheader("Nowcast Card (demo logic)")
st.write("Right now these are manual inputs. Next step: pull GFS/HRRR fields and interpolate values at a clicked point.")

c1, c2, c3 = st.columns(3)
with c1:
    cape = st.number_input("CAPE (J/kg)", min_value=0.0, max_value=8000.0, value=1500.0, step=50.0)
with c2:
    shear06 = st.number_input("0–6 km bulk shear (m/s)", min_value=0.0, max_value=60.0, value=20.0, step=1.0)
with c3:
    lcl = st.number_input("LCL (m)", min_value=0.0, max_value=4000.0, value=1200.0, step=50.0)

c4, c5 = st.columns(2)
with c4:
    srh01 = st.number_input("0–1 km SRH (m²/s²)", min_value=0.0, max_value=600.0, value=100.0, step=10.0)
with c5:
    st.write("Composite signals (toy)")
    st.metric("Hail score (0–10)", simple_hail_score(cape, shear06))
    st.metric("Tornado score (0–10)", simple_tornado_score(cape, srh01, lcl))

st.info("Next bullet: implement model ingest + generate live maps that update automatically (Streamlit Cloud-friendly).")
