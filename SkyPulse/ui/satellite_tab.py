from __future__ import annotations

from datetime import datetime, timezone
import time

import streamlit as st

from services.goes import GOESProduct, goes_image_url


def render_satellite_tab(cfg: dict) -> None:
    st.subheader("Satellite Context (GOES)")

    st.caption(
        "Lightweight, Streamlit-Cloud-friendly satellite imagery via NOAA/NESDIS/STAR CDN. "
        "Use GeoColor for a quick 'what's happening now' view, and IR/WV for cloud-top and moisture structure."
    )

    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        sat = st.selectbox(
            "Satellite",
            options=["GOES19 (East)", "GOES18 (West)"],
            index=0,
        )
    with c2:
        domain = st.selectbox(
            "Domain",
            options=["CONUS"],
            index=0,
            help="We start with CONUS for reliability. We can add regional sectors next.",
        )
    with c3:
        product = st.selectbox(
            "Product",
            options=[
                GOESProduct.GEOCOLOR.value,
                GOESProduct.BAND13_IR.value,
                GOESProduct.BAND08_WV_UPPER.value,
                GOESProduct.BAND10_WV_LOWER.value,
            ],
            index=0,
        )

    size = st.radio("Image size", options=["625x375", "1250x750", "2500x1500"], horizontal=True, index=1)

    # cache-busting to keep images fresh even if Streamlit/browser caches
    bust = int(time.time())
    sat_id = "GOES19" if sat.startswith("GOES19") else "GOES18"

    url = goes_image_url(sat_id=sat_id, domain=domain, product=product, size=size, cache_bust=bust)

    st.write("Latest image URL:")
    st.code(url)

    st.image(url, use_container_width=True)

    st.caption(
        f"Rendered at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} • "
        "Image timestamps are embedded on the product itself."
    )
