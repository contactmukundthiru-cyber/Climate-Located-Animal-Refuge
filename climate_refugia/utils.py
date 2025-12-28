from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def ensure_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, utc=True, errors="coerce")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def batch_haversine_km(latitudes: np.ndarray, longitudes: np.ndarray, ref_lat: float, ref_lon: float) -> np.ndarray:
    lat_rad = np.radians(latitudes)
    lon_rad = np.radians(longitudes)
    ref_lat_rad = math.radians(ref_lat)
    ref_lon_rad = math.radians(ref_lon)
    dlat = lat_rad - ref_lat_rad
    dlon = lon_rad - ref_lon_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_rad) * math.cos(ref_lat_rad) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rolling_groups(sorted_times: pd.Series, max_gap_seconds: int) -> List[int]:
    group_ids = [0]
    for idx in range(1, len(sorted_times)):
        gap = (sorted_times.iloc[idx] - sorted_times.iloc[idx - 1]).total_seconds()
        group_ids.append(group_ids[-1] + int(gap > max_gap_seconds))
    return group_ids


def parse_time_range(start: str, end: str) -> Tuple[datetime, datetime]:
    start_dt = pd.to_datetime(start, utc=True)
    end_dt = pd.to_datetime(end, utc=True)
    if start_dt > end_dt:
        raise ValueError("start must be before end")
    return start_dt.to_pydatetime(), end_dt.to_pydatetime()


def grid_from_coords(latitudes: Iterable[float], longitudes: Iterable[float]) -> List[Tuple[float, float]]:
    return list({(float(lat), float(lon)) for lat, lon in zip(latitudes, longitudes)})
