from __future__ import annotations

def simple_hail_score(cape_jkg: float, shear_ms: float) -> float:
    # toy scoring: cap at 10
    score = 0.0
    score += min(cape_jkg / 300.0, 6.0)     # up to 6 points by CAPE
    score += min(shear_ms / 5.0, 4.0)       # up to 4 points by shear
    return round(min(score, 10.0), 1)

def simple_tornado_score(cape_jkg: float, srh01: float, lcl_m: float) -> float:
    # placeholder; keep for future expansion
    score = 0.0
    score += min(cape_jkg / 500.0, 4.0)
    score += min(srh01 / 50.0, 4.0)
    score += max(0.0, 2.0 - (lcl_m / 1000.0))  # lower LCL better
    return round(min(max(score, 0.0), 10.0), 1)
