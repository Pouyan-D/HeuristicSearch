# src/utils_distance.py

import math
from typing import Tuple

EARTH_RADIUS_KM = 6371.0
CORRECTION_FACTOR = 1.28


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    فاصله خط مستقیم (haversine) بین دو نقطه روی زمین بر حسب کیلومتر.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def road_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    فاصله تقریبی خیابانی = haversine * 1.28
    """
    return CORRECTION_FACTOR * haversine_km(lat1, lon1, lat2, lon2)


def distance_between_points(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    ورودی: دو تا tuple مثل (lat, lon)
    خروجی: فاصله خیابانی
    """
    lat1, lon1 = p1
    lat2, lon2 = p2
    return road_distance_km(lat1, lon1, lat2, lon2)
