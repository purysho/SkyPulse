from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Any

@dataclass(frozen=True)
class Candidate:
    lat: float
    lon: float
    score: float
    kind: str

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2*r*math.asin(math.sqrt(a))

def nearest_candidate(lat: float, lon: float, candidates: Iterable[Candidate]) -> tuple[Candidate | None, float | None]:
    best = None; best_d = None
    for c in candidates:
        d = haversine_km(lat, lon, c.lat, c.lon)
        if best_d is None or d < best_d:
            best_d = d; best = c
    return best, best_d

def parse_candidates(raw: Any) -> list[Candidate]:
    out: list[Candidate] = []
    if not raw:
        return out
    for item in raw:
        try:
            out.append(Candidate(lat=float(item["lat"]), lon=float(item["lon"]), score=float(item.get("score", 0.0)), kind=str(item.get("kind","unknown"))))
        except Exception:
            continue
    return out
