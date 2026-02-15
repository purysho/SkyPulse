from __future__ import annotations

from ingest.metar_cache import fetch_metars_cache, filter_bbox
from compute.verify import build_bias_table
from ingest.gfs_opendap import coord_names

def metar_in_bbox(cfg: dict):
    bbox = cfg["region"]["bbox"]
    metar_all = fetch_metars_cache()
    return filter_bbox(metar_all, lat_min=bbox["lat_min"], lat_max=bbox["lat_max"], lon_min=bbox["lon_min"], lon_max=bbox["lon_max"])

def verify_model_vs_metar(ds, metar_box):
    lon_name, lat_name = coord_names(ds)
    table, summary = build_bias_table(ds, metar_box, lon_name=lon_name, lat_name=lat_name, max_rows=200)
    return table, summary
