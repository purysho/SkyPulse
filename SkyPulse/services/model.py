from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import xarray as xr

from ingest.gfs_opendap import find_latest_gfs_anl_0p25, open_gfs_dataset, coord_names
from app.state import write_latest, maps_dir, write_stats
from compute.fields import get_level, bulk_shear_mag
from compute.signals import build_domain_stats
from viz.maps import plot_scalar_field
from viz.render import save_fig

def _subset_to_bbox(cfg: dict, ds: xr.Dataset, field: xr.DataArray, lon_name: str, lat_name: str):
    bbox = cfg["region"]["bbox"]
    lon_min = bbox["lon_min"] % 360
    lon_max = bbox["lon_max"] % 360
    lat_min = bbox["lat_min"]
    lat_max = bbox["lat_max"]

    if lon_min <= lon_max:
        f_sub = field.sel(**{lat_name: slice(lat_min, lat_max), lon_name: slice(lon_min, lon_max)})
        lons_sub = ds[lon_name].sel(**{lon_name: slice(lon_min, lon_max)})
    else:
        f_a = field.sel(**{lat_name: slice(lat_min, lat_max), lon_name: slice(lon_min, 359.75)})
        f_b = field.sel(**{lat_name: slice(lat_min, lat_max), lon_name: slice(0, lon_max)})
        f_sub = xr.concat([f_a, f_b], dim=lon_name)
        lons_sub = xr.concat(
            [ds[lon_name].sel(**{lon_name: slice(lon_min, 359.75)}), ds[lon_name].sel(**{lon_name: slice(0, lon_max)})],
            dim=lon_name,
        )

    lats_sub = ds[lat_name].sel(**{lat_name: slice(lat_min, lat_max)})
    lons_plot = ((lons_sub + 180) % 360) - 180
    return f_sub, lons_plot, lats_sub

def _composite_0_10(cape: xr.DataArray, shear: xr.DataArray) -> xr.DataArray:
    cape_n = np.clip(cape / 3000.0, 0.0, 1.0)
    shear_n = np.clip(shear / 30.0, 0.0, 1.0)
    return 10.0 * 0.5 * (cape_n + shear_n)

def update_and_render(cfg: dict, cache_dir: str) -> tuple[dict, dict | None]:
    """Find latest run, write latest.json, render maps, and write domain stats."""
    run = find_latest_gfs_anl_0p25(days_back=2)
    write_latest(cache_dir, {"source": "NOMADS OPeNDAP GFS 0.25° (analysis)", "ymd": run.ymd, "cycle": run.cycle, "url": run.url})

    ds = open_gfs_dataset(run.url)
    lon_name, lat_name = coord_names(ds)
    out = maps_dir(cache_dir)

    cape_sub = None
    shear_sub = None
    lons = None
    lats = None

    if "capesfc" in ds:
        cape = ds["capesfc"]
        if "time" in cape.dims:
            cape = cape.isel(time=0)
        cape_sub, lons, lats = _subset_to_bbox(cfg, ds, cape, lon_name, lat_name)
        save_fig(plot_scalar_field(lons, lats, cape_sub, title="Surface CAPE", units="J/kg"), out / "cape_latest.png")

    if "ugrdprs" in ds and "vgrdprs" in ds:
        u = ds["ugrdprs"]; v = ds["vgrdprs"]
        if "time" in u.dims:
            u = u.isel(time=0); v = v.isel(time=0)
        u_lo = get_level(u, 1000.0); v_lo = get_level(v, 1000.0)
        u_hi = get_level(u, 500.0);  v_hi = get_level(v, 500.0)
        shear = bulk_shear_mag(u_lo, v_lo, u_hi, v_hi)
        shear_sub, lons, lats = _subset_to_bbox(cfg, ds, shear, lon_name, lat_name)
        save_fig(plot_scalar_field(lons, lats, shear_sub, title="Bulk shear (1000–500 hPa proxy)", units="m/s"), out / "shear_1000_500_latest.png")

    stats = None
    if cape_sub is not None and shear_sub is not None:
        comp = _composite_0_10(cape_sub, shear_sub)
        save_fig(plot_scalar_field(lons, lats, comp, title="Severe Ingredients Score (Composite)", units="0–10"), out / "composite_latest.png")
        # Persist gridded fields for storm-object detection (cloud-friendly, no re-open needed later)
        try:
            import numpy as _np
            _np.savez_compressed(
        Path(cache_dir) / "storm_fields_latest.npz",
        lons=_np.array(lons),
        lats=_np.array(lats),
        cape=_np.array(cape_sub),
        shear=_np.array(shear_sub),
        composite=_np.array(comp),
            )
        except Exception:
            pass

        stats = build_domain_stats(cape=cape_sub.values, shear=shear_sub.values, composite=comp.values)
        write_stats(cache_dir, stats)

    return {"ymd": run.ymd, "cycle": run.cycle, "url": run.url}, stats

def open_latest_dataset(latest: dict):
    return open_gfs_dataset(latest["url"])
