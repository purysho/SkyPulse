from __future__ import annotations

import matplotlib.pyplot as plt

def plot_scalar_field(lons, lats, field, *, title: str, units: str = ""):
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    m = ax.pcolormesh(lons, lats, field)
    cbar = plt.colorbar(m, ax=ax, orientation="vertical")
    if units:
        cbar.set_label(units)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    return fig
