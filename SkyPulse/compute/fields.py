from __future__ import annotations

import numpy as np
import xarray as xr


def bulk_shear_mag(u_lo: xr.DataArray, v_lo: xr.DataArray, u_hi: xr.DataArray, v_hi: xr.DataArray) -> xr.DataArray:
    """Magnitude of bulk shear vector between two levels (m/s)."""
    du = u_hi - u_lo
    dv = v_hi - v_lo
    return np.hypot(du, dv)


def get_level(da: xr.DataArray, lev_hpa: float) -> xr.DataArray:
    """
    Select nearest pressure level (hPa) from a pressure-level field.
    NOMADS commonly uses coordinate names like: lev, isobaric, isobaric1, level.
    """
    for lev_name in ["lev", "isobaric", "isobaric1", "level"]:
        if lev_name in da.coords:
            return da.sel({lev_name: lev_hpa}, method="nearest")
    raise KeyError(f"No pressure level coordinate found. Coords: {list(da.coords)}")
