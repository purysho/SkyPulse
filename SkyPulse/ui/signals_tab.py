from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st

from app.state import read_stats, read_prev_stats, maps_dir
from compute.signals import generate_signals
from services.model import open_latest_dataset
from services.metar import metar_in_bbox, verify_model_vs_metar
from services.boundary import detect_and_render_boundaries
from services.cache_files import cache_metar_bias, cache_boundaries

def render_signals_tab(cfg: dict, cache_dir: str, latest: dict | None):
    st.subheader("Signals Feed")

    cur = read_stats(cache_dir)
    prev = read_prev_stats(cache_dir)

    if cur is None:
        st.info("Signals will appear after the first update (composite requires both CAPE and shear).")
    else:
        for line in generate_signals(cur, prev):
            st.write("•", line)
        st.subheader("Domain stats")
        st.json(cur)

    st.divider()
    st.subheader("Model vs METAR (quick verification)")
    st.caption("Uses AviationWeather.gov current METAR cache; bias = model − observation.")

    if not latest or "url" not in latest:
        st.info("Run an update first to verify against the model.")
        return

    try:
        ds = open_latest_dataset(latest)
        metar_box = metar_in_bbox(cfg)
        st.write(f"Stations in bbox: {len(metar_box)}")

        table, summary = verify_model_vs_metar(ds, metar_box)

        cA, cB = st.columns(2)
        with cA:
            tb = summary.get("temp_bias_c", {})
            st.metric("Temp bias median (°C)", "N/A" if tb.get("median") is None else f"{tb['median']:+.2f}")
            st.caption(f"n={tb.get('n',0)}, p90(|bias|)={tb.get('p90_abs') if tb.get('p90_abs') is not None else 'N/A'}")
        with cB:
            db = summary.get("dewpoint_bias_c", {})
            st.metric("Dewpoint bias median (°C)", "N/A" if db.get("median") is None else f"{db['median']:+.2f}")
            st.caption(f"n={db.get('n',0)}, p90(|bias|)={db.get('p90_abs') if db.get('p90_abs') is not None else 'N/A'}")

        st.subheader("Worst stations (|temp bias|)")
        show = table.copy()
        show["abs_temp_bias"] = show["temp_bias_c"].abs()
        show = show.sort_values("abs_temp_bias", ascending=False).head(10)
        st.dataframe(show, use_container_width=True)

        # cache bias summary for Nowcast
        cache_metar_bias(cache_dir, summary)

        if db.get("median") is not None and abs(db["median"]) >= 1.5:
            st.warning(
                f"Moisture bias note: model dewpoint median bias is {db['median']:+.2f}°C today — "
                f"composite ingredients may be {'inflated' if db['median']>0 else 'suppressed'}."
            )

    except Exception as e:
        st.error(f"Verification failed: {e}")

    st.divider()
    st.subheader("Boundary detection (from METAR gradients)")
    st.caption("Heuristic boundary likelihood from station-to-station gradients (dewpoint/temp) and wind shifts.")

    try:
        metar_box = metar_in_bbox(cfg)
        candidates = detect_and_render_boundaries(cfg, cache_dir, metar_box)

        if candidates:
            st.write("Top boundary candidates:")
            as_dicts = []
            for c in candidates:
                st.write(f"• {c.kind} boundary near ({c.lat:.2f}, {c.lon:.2f}) — score {c.score:.2f}")
                as_dicts.append({"lat": c.lat, "lon": c.lon, "score": c.score, "kind": c.kind})
            cache_boundaries(cache_dir, as_dicts)
        else:
            st.info("No strong boundary candidates detected.")
            cache_boundaries(cache_dir, [])

        md = maps_dir(cache_dir)
        p = md / "boundary_latest.png"
        if p.exists():
            st.image(str(p), caption="Boundary likelihood map (cached)")

    except Exception as e:
        st.warning(f"Boundary detection unavailable: {e}")
