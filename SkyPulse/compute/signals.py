from __future__ import annotations

import numpy as np

def _nan(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def field_stats(arr) -> dict:
    v = np.array(arr, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"max": None, "median": None, "p90": None}
    return {
        "max": _nan(np.max(v)),
        "median": _nan(np.median(v)),
        "p90": _nan(np.percentile(v, 90)),
    }

def fraction_above(arr, thr: float) -> float | None:
    v = np.array(arr, dtype=float)
    m = np.isfinite(v)
    if m.sum() == 0:
        return None
    return float((v[m] > thr).sum() / m.sum())

def build_domain_stats(*, cape, shear, composite) -> dict:
    return {
        "cape": field_stats(cape),
        "shear": field_stats(shear),
        "composite": field_stats(composite),
        "fractions": {
            "cape_gt_1500": fraction_above(cape, 1500.0),
            "shear_gt_20": fraction_above(shear, 20.0),
            "composite_gt_6": fraction_above(composite, 6.0),
        },
    }

def _delta(cur: float | None, prev: float | None) -> float | None:
    if cur is None or prev is None:
        return None
    return float(cur - prev)

def generate_signals(cur_stats: dict, prev_stats: dict | None) -> list[str]:
    s = []
    cape_max = cur_stats["cape"]["max"]
    shear_max = cur_stats["shear"]["max"]
    comp_p90 = cur_stats["composite"]["p90"]

    if cape_max is not None:
        if cape_max >= 2500: s.append(f"High instability present (CAPE max {cape_max:.0f} J/kg).")
        elif cape_max >= 1500: s.append(f"Moderate instability (CAPE max {cape_max:.0f} J/kg).")
        else: s.append(f"Low instability overall (CAPE max {cape_max:.0f} J/kg).")

    if shear_max is not None:
        if shear_max >= 25: s.append(f"Strong deep-layer shear pockets (shear max {shear_max:.1f} m/s).")
        elif shear_max >= 15: s.append(f"Moderate deep-layer shear (shear max {shear_max:.1f} m/s).")
        else: s.append(f"Weak deep-layer shear overall (shear max {shear_max:.1f} m/s).")

    frac_overlap = cur_stats["fractions"]["composite_gt_6"]
    if frac_overlap is not None:
        s.append(f"High-ingredient overlap area: {frac_overlap*100:.1f}% of domain (composite > 6).")

    if comp_p90 is not None:
        if comp_p90 >= 7: s.append(f"Composite ingredients are widespread (90th percentile {comp_p90:.1f}/10).")
        elif comp_p90 >= 5: s.append(f"Composite ingredients are localized (90th percentile {comp_p90:.1f}/10).")
        else: s.append(f"Composite ingredients are generally low (90th percentile {comp_p90:.1f}/10).")

    if prev_stats:
        d_cape = _delta(cur_stats["cape"]["median"], prev_stats.get("cape", {}).get("median"))
        d_shear = _delta(cur_stats["shear"]["median"], prev_stats.get("shear", {}).get("median"))
        d_comp = _delta(cur_stats["composite"]["median"], prev_stats.get("composite", {}).get("median"))

        if d_cape is not None and abs(d_cape) >= 50:
            s.append(f"Trend: CAPE median {'up' if d_cape>0 else 'down'} ({d_cape:+.0f} J/kg vs last update).")
        if d_shear is not None and abs(d_shear) >= 0.5:
            s.append(f"Trend: Shear median {'up' if d_shear>0 else 'down'} ({d_shear:+.1f} m/s vs last update).")
        if d_comp is not None and abs(d_comp) >= 0.2:
            s.append(f"Trend: Composite median {'up' if d_comp>0 else 'down'} ({d_comp:+.1f}/10 vs last update).")

    s.append("Note: GFS analysis is coarse resolution; treat this as ingredient guidance, not a warning product.")
    return s
