from __future__ import annotations

import json
from pathlib import Path
import streamlit as st

from ingest.gfs_opendap import coord_names
from compute.fields import get_level, bulk_shear_mag
from compute.indices import simple_hail_score
from compute.geo import parse_candidates, nearest_candidate
from compute.watch import initiation_watch_score
from services.model import open_latest_dataset

def render_nowcast_tab(cfg: dict, cache_dir: str, latest: dict | None):
    st.subheader("Nowcast Card (from model)")

    lat = st.number_input("Latitude", value=33.0, step=0.1, format="%.3f")
    lon = st.number_input("Longitude", value=-97.0, step=0.1, format="%.3f")

    if not latest or "url" not in latest:
        st.info("No dataset yet — run an update first.")
        return

    try:
        ds = open_latest_dataset(latest)
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
            st.metric("Hail score (0–10)", "N/A" if (cape_val is None or shear_val is None) else simple_hail_score(cape_val, shear_val))

        reasons = []
        if cape_val is not None and cape_val >= 1500: reasons.append("CAPE ≥ 1500")
        if shear_val is not None and shear_val >= 20: reasons.append("Shear ≥ 20 m/s")
        st.caption("Triggered thresholds: " + (", ".join(reasons) if reasons else "None"))

        # Boundary proximity + initiation watch inputs
        dist_km = None
        try:
            bfile = Path(cache_dir) / "boundaries_latest.json"
            if bfile.exists():
                braw = json.loads(bfile.read_text(encoding="utf-8"))
                cands = parse_candidates(braw.get("candidates", []))
                best, dist_km = nearest_candidate(lat, lon, cands)
                if best is not None and dist_km is not None:
                    st.write(f"Nearest boundary candidate: **{best.kind}** at ({best.lat:.2f}, {best.lon:.2f}) — **{dist_km:.0f} km** away.")
                    if dist_km <= 50:
                        st.warning("Near-boundary environment: initiation/organization risk can increase near surface boundaries.")
                    elif dist_km <= 100:
                        st.info("Somewhat near a boundary: monitor for initiation and storm changes near the boundary zone.")
            else:
                st.caption("Boundary proximity: open the Signals tab once to generate boundary candidates.")
        except Exception as e:
            st.caption(f"Boundary proximity unavailable: {e}")

        dew_bias = None
        try:
            bias_file = Path(cache_dir) / "metar_bias_latest.json"
            if bias_file.exists():
                bias = json.loads(bias_file.read_text(encoding="utf-8"))
                dew_bias = (bias.get("dewpoint_bias_c", {}) or {}).get("median")
        except Exception:
            pass

        st.subheader("Initiation Watch (0–10)")
        score, why = initiation_watch_score(
            cape_jkg=cape_val,
            shear_ms=shear_val,
            boundary_dist_km=dist_km,
            dewpoint_bias_c_median=dew_bias,
        )
        if score is None:
            st.info("Initiation Watch needs CAPE + shear.")
        else:
            st.metric("Score", f"{score:.1f}")
            st.caption("Signals: " + ", ".join(why))

    except Exception as e:
        st.error(f"Nowcast failed: {e}")
