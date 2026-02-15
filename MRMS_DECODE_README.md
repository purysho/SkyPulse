MRMS GRIB2 decode on Streamlit Community Cloud
================================================

This app uses xarray + cfgrib to decode MRMS GRIB2.

Streamlit Cloud does not include GRIB tooling by default.
To enable decoding:

- requirements.txt includes: cfgrib, eccodes
- packages.txt installs: libeccodes0, libeccodes-data

Both files MUST live at the repository root (same level as SkyPulse/).

After pushing, reboot the app so dependencies reinstall.
