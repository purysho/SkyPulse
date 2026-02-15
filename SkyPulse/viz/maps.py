from __future__ import annotations
import matplotlib.pyplot as plt

def plot_scalar_field(lons, lats, field, *, title: str):
    """Basic scalar field plot (no map projection). Cloud-friendly starter."""
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    m = ax.pcolormesh(lons, lats, field)
    plt.colorbar(m, ax=ax, orientation="vertical")
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    return fig
