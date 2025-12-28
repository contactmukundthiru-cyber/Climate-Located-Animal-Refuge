from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .clustering import cluster_refugia
from .heat_events import detect_heat_events
from .utils import haversine_km


def heatwave_response_analysis(
    heat_df: pd.DataFrame,
    years: Iterable[int],
    eps_km: float,
    min_samples: int,
) -> pd.DataFrame:
    results = []
    for year in years:
        subset = heat_df[heat_df["timestamp"].dt.year == year]
        _, clusters = cluster_refugia(subset, eps_km=eps_km, min_samples=min_samples)
        clusters["year"] = year
        results.append(clusters)

    if not results:
        return pd.DataFrame()
    combined = pd.concat(results, ignore_index=True)
    return combined


def climate_scenario_shift(
    scenario_a: pd.DataFrame,
    scenario_b: pd.DataFrame,
    probability_threshold: float = 0.7,
) -> pd.DataFrame:
    if scenario_a.empty or scenario_b.empty:
        return pd.DataFrame()

    def centroids(df: pd.DataFrame) -> pd.DataFrame:
        active = df[df["refugia_probability"] >= probability_threshold]
        if active.empty:
            return pd.DataFrame()
        return active.groupby("species").agg(lat=("lat", "mean"), lon=("lon", "mean")).reset_index()

    cent_a = centroids(scenario_a)
    cent_b = centroids(scenario_b)
    merged = cent_a.merge(cent_b, on="species", suffixes=("_a", "_b"))
    if merged.empty:
        return merged

    merged["shift_km"] = merged.apply(
        lambda row: haversine_km(row["lat_a"], row["lon_a"], row["lat_b"], row["lon_b"]), axis=1
    )
    return merged


def model_comparison_empirical_vs_climate(
    heat_df: pd.DataFrame,
    top_cool_percentile: float = 0.1,
) -> Dict[str, float]:
    if heat_df.empty:
        return {"overlap_rate": float("nan")}

    heat_df = heat_df.copy()
    if "cluster_id" not in heat_df.columns:
        return {"overlap_rate": float("nan")}
    threshold = heat_df["temp_c"].quantile(top_cool_percentile)
    climate_refugia = heat_df[heat_df["temp_c"] <= threshold]
    empirical_refugia = heat_df[heat_df["cluster_id"] >= 0]

    if climate_refugia.empty or empirical_refugia.empty:
        return {"overlap_rate": float("nan")}

    climate_points = set(zip(climate_refugia["lat"].round(4), climate_refugia["lon"].round(4)))
    empirical_points = set(zip(empirical_refugia["lat"].round(4), empirical_refugia["lon"].round(4)))
    overlap = climate_points.intersection(empirical_points)
    overlap_rate = len(overlap) / max(1, len(empirical_points))
    return {"overlap_rate": float(overlap_rate)}


def sensitivity_analysis(
    aligned_df: pd.DataFrame,
    thresholds: Dict[str, float],
    default_threshold: float,
    deltas: Iterable[float],
    eps_km: float,
    min_samples: int,
    heat_window_hours: int = 3,
) -> pd.DataFrame:
    results = []
    for delta in deltas:
        adjusted = {species: value + delta for species, value in thresholds.items()}
        temp_threshold = default_threshold + delta
        heat_df, _ = detect_heat_events(
            aligned_df,
            thresholds=adjusted,
            default_threshold=temp_threshold,
            heat_window_hours=heat_window_hours,
        )
        _, clusters = cluster_refugia(heat_df, eps_km=eps_km, min_samples=min_samples)
        results.append({
            "delta_c": delta,
            "num_clusters": int(clusters["cluster_id"].nunique()) if not clusters.empty else 0,
            "num_refugia": int(clusters["is_refugia"].sum()) if not clusters.empty else 0,
        })
    return pd.DataFrame(results)
