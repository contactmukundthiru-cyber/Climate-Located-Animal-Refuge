from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from .utils import EARTH_RADIUS_KM, ensure_datetime, haversine_km


def clean_gps(
    gps_df: pd.DataFrame,
    max_speed_mps: float = 35.0,
    min_fix_interval_s: int = 30,
) -> pd.DataFrame:
    df = gps_df.copy()
    df = df.dropna(subset=["timestamp", "lat", "lon"]).copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["timestamp"] = ensure_datetime(df["timestamp"])
    df = df.dropna(subset=["timestamp"])
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))]
    df = df.sort_values(["individual_id", "timestamp"]).reset_index(drop=True)

    if "speed_mps" not in df.columns:
        df["speed_mps"] = np.nan
    else:
        df["speed_mps"] = pd.to_numeric(df["speed_mps"], errors="coerce")

    def compute_speed(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("timestamp")
        distances_km = [np.nan]
        time_s = [np.nan]
        for idx in range(1, len(group)):
            prev = group.iloc[idx - 1]
            curr = group.iloc[idx]
            dist = haversine_km(prev["lat"], prev["lon"], curr["lat"], curr["lon"])
            dt = (curr["timestamp"] - prev["timestamp"]).total_seconds()
            distances_km.append(dist)
            time_s.append(dt)
        group = group.copy()
        group["_dist_km"] = distances_km
        group["_dt_s"] = time_s
        group.loc[group["_dt_s"] <= 0, "_dt_s"] = np.nan
        group["speed_mps"] = group["speed_mps"].fillna(group["_dist_km"] * 1000 / group["_dt_s"])
        group = group[(group["_dt_s"].isna()) | (group["_dt_s"] >= min_fix_interval_s)]
        return group.drop(columns=["_dist_km", "_dt_s"])

    grouped = []
    for _, group in df.groupby("individual_id", sort=False):
        grouped.append(compute_speed(group))
    df = pd.concat(grouped, ignore_index=True) if grouped else df
    df = df[(df["speed_mps"].isna()) | (df["speed_mps"] <= max_speed_mps)]
    return df.reset_index(drop=True)


def clean_climate(climate_df: pd.DataFrame) -> pd.DataFrame:
    df = climate_df.copy()
    df = df.dropna(subset=["timestamp", "lat", "lon", "temp_c"])
    df["timestamp"] = ensure_datetime(df["timestamp"])
    df["temp_c"] = pd.to_numeric(df["temp_c"], errors="coerce")
    if "humidity" in df.columns:
        df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")
    if "precip_mm" in df.columns:
        df["precip_mm"] = pd.to_numeric(df["precip_mm"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))]
    return df.reset_index(drop=True)


def align_gps_climate(
    gps_df: pd.DataFrame,
    climate_df: pd.DataFrame,
    time_tolerance_minutes: int = 60,
) -> pd.DataFrame:
    gps = gps_df.copy()
    climate = climate_df.copy()

    gps["timestamp"] = ensure_datetime(gps["timestamp"])
    climate["timestamp"] = ensure_datetime(climate["timestamp"])

    climate = climate.dropna(subset=["timestamp", "lat", "lon"])
    gps = gps.dropna(subset=["timestamp", "lat", "lon"])

    climate_grid = climate[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    tree = BallTree(
        np.radians(climate_grid[["lat", "lon"]].to_numpy()),
        metric="haversine",
    )

    gps_coords = np.radians(gps[["lat", "lon"]].to_numpy())
    dist, idx = tree.query(gps_coords, k=1)
    nearest = climate_grid.iloc[idx.flatten()].reset_index(drop=True)
    gps = gps.reset_index(drop=True)
    gps["grid_lat"] = nearest["lat"].to_numpy()
    gps["grid_lon"] = nearest["lon"].to_numpy()
    gps["grid_distance_km"] = dist.flatten() * EARTH_RADIUS_KM
    gps["grid_key"] = list(zip(gps["grid_lat"], gps["grid_lon"]))

    climate["grid_key"] = list(zip(climate["lat"], climate["lon"]))
    gps["grid_key"] = gps["grid_key"].astype(str)
    climate["grid_key"] = climate["grid_key"].astype(str)

    aligned_parts = []
    for grid_key, gps_group in gps.groupby("grid_key"):
        climate_group = climate[climate["grid_key"] == grid_key]
        if climate_group.empty:
            continue
        gps_group = gps_group.sort_values("timestamp")
        climate_group = climate_group.sort_values("timestamp")
        merged = pd.merge_asof(
            gps_group,
            climate_group,
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=time_tolerance_minutes),
            suffixes=("", "_climate"),
        )
        aligned_parts.append(merged)

    if not aligned_parts:
        return pd.DataFrame()

    aligned = pd.concat(aligned_parts, ignore_index=True)
    aligned = aligned.drop(columns=["grid_key"], errors="ignore")
    aligned = aligned.dropna(subset=["temp_c"]).reset_index(drop=True)
    return aligned
