from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlencode


class GOESProduct(str, Enum):
    # Processed RGB product
    GEOCOLOR = "GEOCOLOR"

    # Handy baseline channels
    BAND13_IR = "13"  # Clean longwave IR window
    BAND08_WV_UPPER = "08"  # Upper-level water vapor
    BAND10_WV_LOWER = "10"  # Lower-level water vapor


def goes_image_url(
    *,
    sat_id: str = "GOES19",
    domain: str = "CONUS",
    product: str = "GEOCOLOR",
    size: str = "1250x750",
    cache_bust: int | None = None,
) -> str:
    """Return a 'latest image' URL from the NOAA/NESDIS/STAR CDN.

    This is intentionally simple (static JPG) to stay Streamlit-Cloud-friendly.
    Example:
      https://cdn.star.nesdis.noaa.gov/GOES19/ABI/CONUS/GEOCOLOR/1250x750.jpg
      https://cdn.star.nesdis.noaa.gov/GOES19/ABI/CONUS/13/1250x750.jpg
    """
    base = f"https://cdn.star.nesdis.noaa.gov/{sat_id}/ABI/{domain}/{product}/{size}.jpg"
    if cache_bust is None:
        return base
    return base + "?" + urlencode({"t": str(cache_bust)})


# Convenience alias for UI code readability
GOESProduct = GOESProduct
