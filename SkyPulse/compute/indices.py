from __future__ import annotations

def simple_hail_score(cape_jkg: float, shear_ms: float) -> float:
    score = 0.0
    score += min(cape_jkg / 300.0, 6.0)
    score += min(shear_ms / 5.0, 4.0)
    return round(min(score, 10.0), 1)

def simple_tornado_score(cape_jkg: float, srh01: float, lcl_m: float) -> float:
    score = 0.0
    score += min(cape_jkg / 500.0, 4.0)
    score += min(srh01 / 50.0, 4.0)
    score += max(0.0, 2.0 - (lcl_m / 1000.0))
    return round(min(max(score, 0.0), 10.0), 1)
