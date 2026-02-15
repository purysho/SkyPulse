from __future__ import annotations

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def initiation_watch_score(
    *,
    cape_jkg: float | None,
    shear_ms: float | None,
    boundary_dist_km: float | None,
    dewpoint_bias_c_median: float | None,
) -> tuple[float | None, list[str]]:
    """Score 0–10 for initiation/organization potential using simple heuristics."""
    if cape_jkg is None or shear_ms is None:
        return None, ["Need CAPE + shear to score."]

    reasons: list[str] = []

    cape_n = clamp(cape_jkg / 3000.0, 0.0, 1.0)
    shear_n = clamp(shear_ms / 30.0, 0.0, 1.0)
    base = 10.0 * 0.55 * (0.6 * cape_n + 0.4 * shear_n)

    if cape_jkg >= 2000: reasons.append("High CAPE")
    elif cape_jkg >= 1000: reasons.append("Moderate CAPE")
    else: reasons.append("Low CAPE")

    if shear_ms >= 25: reasons.append("Strong shear")
    elif shear_ms >= 15: reasons.append("Moderate shear")
    else: reasons.append("Weak shear")

    bboost = 0.0
    if boundary_dist_km is not None:
        if boundary_dist_km <= 25:
            bboost = 2.0; reasons.append("Very near boundary")
        elif boundary_dist_km <= 50:
            bboost = 1.3; reasons.append("Near boundary")
        elif boundary_dist_km <= 100:
            bboost = 0.6; reasons.append("Somewhat near boundary")

    conf = 0.0
    if dewpoint_bias_c_median is not None:
        if dewpoint_bias_c_median >= 1.5:
            conf = -1.0; reasons.append("Model moisture high bias")
        elif dewpoint_bias_c_median <= -1.5:
            conf = -0.7; reasons.append("Model moisture low bias")
        else:
            reasons.append("Moisture bias small")

    score = clamp(base + bboost + conf, 0.0, 10.0)
    return round(score, 1), reasons
