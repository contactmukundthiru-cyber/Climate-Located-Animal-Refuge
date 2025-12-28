from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from .utils import EARTH_RADIUS_KM


def cluster_refugia(
    heat_df: pd.DataFrame,
    eps_km: float,
    min_samples: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = heat_df.copy()
    events = df.dropna(subset=["heat_event_id"]).copy()
    if events.empty:
        return df, pd.DataFrame()

    coords = np.radians(events[["lat", "lon"]].to_numpy())
    clustering = DBSCAN(eps=eps_km / EARTH_RADIUS_KM, min_samples=min_samples, metric="haversine")
    labels = clustering.fit_predict(coords)
    events["cluster_id"] = labels
    df.loc[events.index, "cluster_id"] = labels

    clusters = events[events["cluster_id"] >= 0].copy()
    if clusters.empty:
        return df, pd.DataFrame()

    clusters["year"] = clusters["timestamp"].dt.year
    summary = clusters.groupby("cluster_id").agg(
        centroid_lat=("lat", "mean"),
        centroid_lon=("lon", "mean"),
        num_points=("cluster_id", "size"),
        num_individuals=("individual_id", "nunique"),
        num_events=("heat_event_id", "nunique"),
        first_seen=("timestamp", "min"),
        last_seen=("timestamp", "max"),
        years=("year", lambda x: sorted(set(x))),
        species_list=("species", lambda x: sorted(set(x))),
        dominant_species=("species", lambda x: x.value_counts().idxmax()),
    ).reset_index()

    summary["is_refugia"] = (summary["num_individuals"] >= 2) & (summary["num_events"] >= 2)
    return df, summary
