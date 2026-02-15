from __future__ import annotations

import streamlit as st

from services.radar import radar_products, cache_bust

def render_radar_tab():
    st.subheader("Radar (NWS Ridge)")
    st.caption("Lightweight radar imagery for context (no keys). For true overlays/GeoTIFF ingest, we can add a heavier pipeline next.")

    prods = radar_products()
    labels = [p.label for p in prods]
    idx = st.selectbox("Product", list(range(len(labels))), format_func=lambda i: labels[i], index=0)
    p = prods[idx]

    st.write("Source:", p.url)
    st.image(cache_bust(p.url), caption=p.label, use_container_width=True)

    st.divider()
    st.subheader("Quick view: Radar + Ingredients")
    st.caption("This is side-by-side (not georegistered overlay yet).")
    col1, col2 = st.columns(2)
    with col1:
        st.image(cache_bust(p.url), caption="Radar", use_container_width=True)
    with col2:
        st.info("Go to Maps tab → Composite/CAPE/Shear for ingredients. Next upgrade: georegistered overlay.")
