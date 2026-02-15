# SkyPulse — Severe Weather & Atmospheric Intelligence (Alpha)

A minimal **Python + Streamlit** starter scaffold for a "Severe Weather Environment Analyzer" using **public weather data**.

## Run locally
```bash
pip install -r requirements.txt
streamlit run main.py
```

## Deploy (recommended): Streamlit Community Cloud
1. Create a **public** GitHub repo (e.g., `SkyPulse`).
2. Upload this project to the repo root.
3. On Streamlit Community Cloud, create a new app from the repo and set:
   - **Main file path:** `main.py`

## Notes
- This starter intentionally avoids heavy geospatial dependencies (e.g., Cartopy) to keep free deployments reliable.
- Next step is wiring in model ingest (GFS/HRRR) + map rendering + click-to-sample "Nowcast Card".
