from __future__ import annotations
import numpy as np

def simple_hail_score(cape_jkg: float, shear_06_ms: float) -> float:
    """Toy composite score for demo purposes (0–10). Replace with real logic later."""
    cape = np.clip(cape_jkg / 3000.0, 0, 1)
    shear = np.clip(shear_06_ms / 40.0, 0, 1)
    return float(np.round(10.0 * (0.65 * cape + 0.35 * shear), 2))

def simple_tornado_score(cape_jkg: float, srh_01_m2s2: float, lcl_m: float) -> float:
    """Toy composite score for demo purposes (0–10)."""
    cape = np.clip(cape_jkg / 2500.0, 0, 1)
    srh = np.clip(srh_01_m2s2 / 200.0, 0, 1)
    lcl = 1.0 - np.clip(lcl_m / 2000.0, 0, 1)  # lower LCL is better
    return float(np.round(10.0 * (0.4 * cape + 0.4 * srh + 0.2 * lcl), 2))
