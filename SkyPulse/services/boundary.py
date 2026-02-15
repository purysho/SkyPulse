from __future__ import annotations

from compute.boundaries import compute_boundary_candidates, grid_boundary_field
from viz.maps import plot_scalar_field
from viz.render import save_fig
from app.state import maps_dir

def detect_and_render_boundaries(cfg: dict, cache_dir: str, metar_box):
    scored, candidates = compute_boundary_candidates(metar_box, top_n=8)
    lons_g, lats_g, bfield = grid_boundary_field(scored, bbox=cfg["region"]["bbox"], res_deg=0.25)
    md = maps_dir(cache_dir)
    save_fig(plot_scalar_field(lons_g, lats_g, bfield, title="Boundary likelihood (METAR gradients)", units="0–1"), md / "boundary_latest.png")
    return candidates
