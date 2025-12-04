# src/utils_distance.py

import math
from typing import Tuple

# شعاع زمین به کیلومتر
EARTH_RADIUS_KM = 6371.0

CORRECTION_FACTOR = 1.28  # فاکتور تبدیل فاصله هوایی به فاصله خیابانی


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    محاسبه فاصله خط مستقیم (Haversine) بین دو نقطه روی کره زمین بر حسب کیلومتر.
    ورودی‌ها: lat/lon به درجه.
    """
    # تبدیل به رادیان
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return EARTH_RADIUS_KM * c


def road_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    فاصله تقریبی خیابانی بر حسب کیلومتر با استفاده از Haversine * 1.28
    """
    return CORRECTION_FACTOR * haversine_km(lat1, lon1, lat2, lon2)


def distance_between_points(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    نسخه راحت‌تر: ورودی دو تا (lat, lon) به صورت تاپل.
    """
    (lat1, lon1) = p1
    (lat2, lon2) = p2
    return road_distance_km(lat1, lon1, lat2, lon2)
