from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st
import xarray as xr

from services.mrms import update_mrms_latest, try_decode_mrms_grib2, detect_mrms_objects


def _read_meta(cache_dir: str) -> dict | None:
    p = Path(cache_dir) / "mrms_reflectivity_meta.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def render_mrms_tab(cfg: dict, cache_dir: str):
    st.subheader("MRMS (real radar) — Merged Reflectivity QC Composite")
    st.caption("MRMS ingest is keyless via NOAA AWS Open Data. Decode requires cfgrib + ecCodes.")

    with st.expander("MRMS decode diagnostics (click if decode fails)", expanded=False):
        st.write("xarray version:", getattr(xr, "__version__", "unknown"))
        try:
            engines = xr.backends.list_engines()
            st.write("xarray engines:", sorted(list(engines.keys())))
        except Exception as e:
            st.write("Could not list engines:", str(e))

        try:
            import cfgrib  # noqa
            st.success("cfgrib import: OK")
        except Exception as e:
            st.error(f"cfgrib import failed: {e}")

        try:
            import eccodes  # noqa
            st.success("eccodes (python) import: OK")
        except Exception as e:
            st.error(f"eccodes (python) import failed: {e}")

    colA, colB = st.columns([1, 2])
    with colA:
        max_age = st.slider("Max age (minutes)", 2, 120, 15, 1)
        if st.button("Update MRMS now"):
            meta = update_mrms_latest(cache_dir, max_age_minutes=int(max_age))
            st.success("MRMS downloaded (raw).")
    with colB:
        meta = _read_meta(cache_dir)
        if meta:
            st.json(meta)
        else:
            st.info("No MRMS meta yet — click Update.")

    st.divider()
    st.subheader("Decode + Object detection")

    meta = _read_meta(cache_dir)
    if not meta:
        st.info("Download MRMS first.")
        return

    if st.button("Try decode (cfgrib)"):
        ok, note = try_decode_mrms_grib2(cache_dir)
        if ok:
            st.success("Decoded ✔")
        else:
            st.error(note)

    if st.button("Detect/Track MRMS objects"):
        try:
            payload = detect_mrms_objects(cache_dir)
            st.success(f"Detected {len(payload.get('objects', []))} objects.")
            st.dataframe(payload.get("objects", []), use_container_width=True)
        except Exception as e:
            st.error(str(e))
