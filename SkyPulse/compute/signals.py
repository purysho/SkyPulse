from __future__ import annotations

import numpy as np
import json
from pathlib import Path

def generate_signals(ds, cache_dir):
    signals = []

    if "capesfc" in ds:
        cap = ds["capesfc"]
        if "time" in cap.dims:
            cap = cap.isel(time=0)
        max_cape = float(np.nanmax(cap.values))
        if max_cape > 2000:
            signals.append(f"High CAPE detected (max {max_cape:.0f} J/kg).")
        elif max_cape > 1000:
            signals.append(f"Moderate CAPE environment (max {max_cape:.0f} J/kg).")

    if "ugrdprs" in ds and "vgrdprs" in ds:
        signals.append("Deep-layer wind structure present (check shear map).")

    if not signals:
        signals.append("No strong severe ingredients detected.")

    out = Path(cache_dir) / "signals.json"
    out.write_text(json.dumps({"signals": signals}, indent=2), encoding="utf-8")
    return signals
