# SkyPulse (Alpha)

Streamlit Cloud-friendly severe weather *ingredients* dashboard.

## What it does
- Pulls latest GFS 0.25° analysis via NOMADS OPeNDAP (no GRIB dependencies)
- Renders cached maps:
  - Surface CAPE
  - Bulk shear proxy (1000–500 hPa)
  - Composite Severe Ingredients Score (0–10)
- Provides a Nowcast card from the model by lat/lon interpolation
- Generates a signals feed with simple trend detection between updates

## Deploy notes
- Keep Streamlit entry point as `SkyPulse/main.py`
- Recommended Python: 3.11
